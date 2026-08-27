"""Getting the sliced gcode onto the Flashforge Adventurer 5M Pro.

There are three ways off this Mac and onto the machine, in descending order of
how much this module trusts them:

  1. save_for_transfer()  -- copy the gcode somewhere obvious and tell the user
     how to send it with Orca-Flashforge or a USB stick. Pure filesystem, no
     network, works with the printer unplugged and in another room. This is the
     DEFAULT and the only path this module considers reliable.
  2. HTTP on port 8898   -- the "new" stock-firmware LAN API on 5M / 5M Pro /
     AD5X. Needs a serial number and a check code read off the printer's screen
     (Settings -> Network, LAN mode enabled), supplied via config.FF_SERIAL /
     config.FF_CHECKCODE.
  3. TCP line protocol on port 8899 -- the older '~M601 S1' Flashforge gcode
     socket, still open on the 5M. Needs no credentials, which makes it the
     fallback when no check code has been configured.

HONESTY NOTE -- READ THIS BEFORE TRUSTING ANYTHING BELOW.
    Only save_for_transfer(), the credential/host resolution logic and the
    graceful-degradation behaviour of discover()/status()/upload_gcode() have
    been exercised on this machine. There was no Adventurer 5M Pro on the LAN
    when this module was written, so:
      * discover() has been run, and correctly found nothing;
      * the 276-byte / 140-byte discovery packet PARSERS are implemented from
        the community protocol notes and have never seen a real packet;
      * every request in _http_post / _tcp_session -- /detail, /uploadGcode,
        /printGcode, ~M28/~M29, ~M23 -- is UNVERIFIED against real hardware.
    They are written to the best public description of the protocol (Flashforge
    publishes no spec), and they are written to fail soft. Treat a success
    return from the network paths as "the printer said something that parsed",
    not as "the file is definitely on the printer". Check the printer's screen.

Nothing in here is allowed to take the pipeline down: an absent, unreachable or
uncredentialed printer degrades to option 1 and returns False, never raises.
The single deliberate exception is start_print(), which refuses to do anything
at all unless confirm=True -- starting a print is a physical-world side effect
with a hot nozzle attached to it, and it does not happen by accident.
"""

from __future__ import annotations

import json
import shutil
import socket
import struct
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from . import config

# --- discovery ------------------------------------------------------------
# Modern machines (5M / 5M Pro / AD5X) answer an unsolicited UDP packet on
# either of these; legacy machines (Adventurer 3/4) answer the third. The
# payload is ignored by the printer -- it replies to anything -- but we send
# the historical USR-WIFI232 probe string because it costs nothing and older
# Wi-Fi modules do look for it.
DISCOVERY_TARGETS = (
    ("225.0.0.9", 19000),        # modern, multicast
    ("255.255.255.255", 48899),  # modern, broadcast
    ("225.0.0.9", 8899),         # legacy, multicast
)
DISCOVERY_PROBE = b"www.usr.cn"

# Response layout, from the community protocol notes. Both variants put the
# machine name first; only the modern one carries a serial number, which is
# exactly the value the HTTP API needs.
_RESP_MODERN_MIN = 196
_RESP_LEGACY_LEN = 140
_OFF_NAME = 0x00
_OFF_PORT = 0x84
_OFF_SERIAL = 0x92
_FIELD_LEN = 128

# --- HTTP API -------------------------------------------------------------
# Path names differ across firmware revisions; we try each in turn and take
# the first that answers with parseable JSON rather than betting on one.
_STATUS_PATHS = ("/detail", "/product", "/info")
_UPLOAD_PATH = "/uploadGcode"
_PRINT_PATH = "/printGcode"

HTTP_TIMEOUT_S = 6.0
TCP_TIMEOUT_S = 6.0
UPLOAD_TIMEOUT_S = 120.0


# --------------------------------------------------------------------------
# host / credential resolution
# --------------------------------------------------------------------------
def resolve_host(host: str | None = None) -> str:
    """The printer address to talk to: explicit argument, else config.FF_HOST."""
    return (host or config.FF_HOST or "").strip()


def credentials() -> tuple[str, str]:
    """(serialNumber, checkCode) as configured. Either may be empty."""
    return (config.FF_SERIAL or "").strip(), (config.FF_CHECKCODE or "").strip()


def have_credentials() -> bool:
    """True when both halves of the HTTP API credential pair are present."""
    s, c = credentials()
    return bool(s and c)


def _reason_no_lan() -> str:
    host = resolve_host()
    missing = []
    if not host:
        missing.append("FF_HOST")
    s, c = credentials()
    if not s:
        missing.append("FF_SERIAL")
    if not c:
        missing.append("FF_CHECKCODE")
    if not missing:
        return ""
    return (
        f"LAN printing is not configured: {', '.join(missing)} unset. "
        f"Read the serial number and check code off the printer under "
        f"Settings -> Network (LAN mode on) and export them, or just use "
        f"save_for_transfer()."
    )


# --------------------------------------------------------------------------
# low-level transports -- all of them swallow network errors by design
# --------------------------------------------------------------------------
def _decode_field(buf: bytes) -> str:
    """NUL-terminated, fixed-width string field out of a discovery packet."""
    return buf.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _parse_discovery(data: bytes, addr: tuple[str, int]) -> dict[str, Any] | None:
    """Turn one UDP reply into a printer record, or None if it isn't one.

    UNVERIFIED: written from the published field offsets, never fed a real
    packet. Anything that does not match a known length is reported with the
    raw length so a real reply can be identified from the logs rather than
    silently dropped.
    """
    if len(data) < _RESP_LEGACY_LEN:
        return None
    name = _decode_field(data[_OFF_NAME:_OFF_NAME + _FIELD_LEN])
    port_le, = struct.unpack_from("<H", data, _OFF_PORT)
    port_be, = struct.unpack_from(">H", data, _OFF_PORT)
    # The notes give the offset but not the byte order. 8899 is the documented
    # value, so if either reading hits it, that is the byte order; otherwise
    # take little-endian (what the printer's ARM firmware emits natively) and
    # fall back to the configured port if that is not a usable port number.
    if config.FF_TCP_PORT in (port_le, port_be):
        port = config.FF_TCP_PORT
    elif 1 <= port_le <= 65535:
        port = port_le
    else:
        port = config.FF_TCP_PORT
    rec: dict[str, Any] = {
        "host": addr[0],
        "name": name,
        "command_port": int(port),
        "serial": "",
        "protocol": "legacy" if len(data) == _RESP_LEGACY_LEN else "modern",
        "raw_len": len(data),
        "source": "udp-discovery",
        "verified": False,
    }
    if len(data) >= _RESP_MODERN_MIN and len(data) >= _OFF_SERIAL + _FIELD_LEN:
        rec["serial"] = _decode_field(data[_OFF_SERIAL:_OFF_SERIAL + _FIELD_LEN])
    return rec


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Plain TCP connect test. False for every failure mode, never raises."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_post(
    host: str,
    path: str,
    body: bytes,
    content_type: str,
    timeout: float = HTTP_TIMEOUT_S,
) -> tuple[bool, Any]:
    """POST to the printer's HTTP API. Returns (ok, decoded-json-or-error-str).

    UNVERIFIED against hardware. Never raises: a dead host, a refused
    connection, an HTML error page and a timeout all come back as (False, str).
    """
    url = f"http://{host}:{config.FF_HTTP_PORT}{path}"
    serial, check = credentials()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(body)))
    # Some firmware builds read the credentials from headers as well as the
    # body; sending both is harmless and costs one line.
    if serial:
        req.add_header("serialNumber", serial)
    if check:
        req.add_header("checkCode", check)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} from {path}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return False, f"{type(e).__name__}: {e}"
    try:
        return True, json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, f"non-JSON reply from {path} ({len(raw)} bytes)"


def _http_json(host: str, path: str, extra: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """POST the standard {serialNumber, checkCode, ...} envelope."""
    serial, check = credentials()
    payload: dict[str, Any] = {"serialNumber": serial, "checkCode": check}
    if extra:
        payload.update(extra)
    return _http_post(host, path, json.dumps(payload).encode("utf-8"), "application/json")


def _tcp_session(host: str, commands: list[str], timeout: float = TCP_TIMEOUT_S) -> tuple[bool, str]:
    """Run a '~M...' command sequence on the legacy port and return the text.

    Takes control (~M601 S1) first and releases it (~M602) afterwards, which is
    what the printer expects and what stops the panel staying locked out.
    UNVERIFIED against hardware.
    """
    lines: list[str] = []
    try:
        with socket.create_connection((host, config.FF_TCP_PORT), timeout=timeout) as s:
            s.settimeout(timeout)
            for cmd in ["~M601 S1"] + commands + ["~M602"]:
                s.sendall((cmd + "\r\n").encode("ascii"))
                try:
                    chunk = s.recv(8192)
                except (socket.timeout, TimeoutError):
                    chunk = b""
                lines.append(f"{cmd} -> {chunk.decode('ascii', errors='replace').strip()}")
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "\n".join(lines)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def discover(timeout: float = 3.0) -> list[dict]:
    """Probe the LAN for Flashforge printers. Returns [] when there are none.

    Sends the UDP discovery probe to the documented multicast and broadcast
    targets, collects replies until `timeout` elapses, and additionally does a
    direct TCP probe of config.FF_HOST if one is configured (a printer on a
    different subnet, or behind a router that eats broadcasts, will never show
    up in the UDP sweep but is still perfectly reachable).

    Finding nothing is a normal, non-exceptional outcome and is what this
    returns when the printer is off, absent, or on another network. Every OS
    error is caught -- some networks refuse broadcast outright.

    UNVERIFIED: no Flashforge printer has ever replied to this code, so the
    reply parser is untested. The "found nothing" path is tested.
    """
    found: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + max(0.1, float(timeout))

    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.4)
        sock.bind(("", 0))
        for addr, port in DISCOVERY_TARGETS:
            try:
                sock.sendto(DISCOVERY_PROBE, (addr, port))
            except OSError:
                continue  # e.g. no route for multicast on this interface
        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(4096)
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            rec = _parse_discovery(data, addr)
            if rec is not None:
                found.setdefault(f"{rec['host']}/{rec['serial']}", rec)
    except OSError:
        pass  # a machine that cannot open a UDP socket still gets a clean []
    finally:
        if sock is not None:
            sock.close()

    host = resolve_host()
    if host and not any(r["host"] == host for r in found.values()):
        http_open = _port_open(host, config.FF_HTTP_PORT)
        tcp_open = _port_open(host, config.FF_TCP_PORT)
        if http_open or tcp_open:
            found[f"{host}/config"] = {
                "host": host,
                "name": "(config.FF_HOST)",
                "command_port": config.FF_TCP_PORT if tcp_open else config.FF_HTTP_PORT,
                "serial": credentials()[0],
                "protocol": "modern" if http_open else "legacy",
                "raw_len": 0,
                "source": "config.FF_HOST",
                "verified": True,   # a TCP connect actually succeeded
            }
    return list(found.values())


def status(host: str | None = None) -> dict:
    """Ask the printer how it is. Always returns a dict, never raises.

    The dict always carries "ok" and "reason"; on success it also carries
    "transport" ("http" or "tcp") and "data" (whatever the printer said).
    A printer that is off, absent or uncredentialed is reported as ok=False
    with a reason a human can act on -- it is not an error condition.

    UNVERIFIED against hardware: the request shapes are from community
    documentation, and the reply is passed through unparsed precisely because
    this code has never seen a real one and should not pretend to know its
    schema.
    """
    h = resolve_host(host)
    if not h:
        return {"ok": False, "reason": "no printer host configured (set FF_HOST)",
                "transport": None, "data": None}

    if have_credentials():
        for path in _STATUS_PATHS:
            ok, data = _http_json(h, path)
            if ok:
                return {"ok": True, "reason": f"{path} answered", "transport": "http",
                        "endpoint": path, "data": data, "verified": False}
        http_reason = f"no HTTP endpoint on {h}:{config.FF_HTTP_PORT} answered"
    else:
        http_reason = "no FF_SERIAL/FF_CHECKCODE, skipped HTTP API"

    ok, text = _tcp_session(h, ["~M115", "~M119", "~M105", "~M27"])
    if ok:
        return {"ok": True, "reason": "legacy TCP answered", "transport": "tcp",
                "endpoint": f"{h}:{config.FF_TCP_PORT}", "data": text, "verified": False}
    return {"ok": False, "reason": f"{http_reason}; TCP fallback failed ({text})",
            "transport": None, "data": None}


def upload_gcode(path: str | Path, host: str | None = None) -> bool:
    """Push a gcode file to the printer's storage. Does NOT start a print.

    Returns True only if the printer accepted the transfer, False for every
    other outcome -- no host, no credentials, unreachable, refused, timed out.
    The caller is expected to fall back to save_for_transfer(); this function
    prints the reason it failed rather than raising, because a missing printer
    must not take the pipeline with it.

    A missing *local* file is a caller bug, not a printer problem, and does
    raise FileNotFoundError.

    UNVERIFIED against hardware. Both the multipart field names and the legacy
    ~M28/~M29 sequence come from community reverse-engineering; a True return
    means the printer answered affirmatively, not that the file is confirmed
    good. Check the printer's file list before printing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no gcode file at {p}")

    h = resolve_host(host)
    if not h:
        print(f"[printer] no host configured; not uploading {p.name}. {_reason_no_lan()}")
        return False

    payload = p.read_bytes()

    if have_credentials():
        serial, check = credentials()
        boundary = f"----scan2print{uuid.uuid4().hex}"
        parts: list[bytes] = []
        fields = {
            "serialNumber": serial,
            "checkCode": check,
            "fileSize": str(len(payload)),
            "printNow": "false",             # never start a print from here
            "levelingBeforePrint": "false",
            "flowCalibration": "false",
            "useMatlStation": "false",
            "gcodeToolCnt": "0",
            "materialMappings": "[]",
        }
        for k, v in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                .encode("utf-8")
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="gcodeFile"; '
            f'filename="{p.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            .encode("utf-8")
        )
        parts.append(payload)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        ok, data = _http_post(
            h, _UPLOAD_PATH, b"".join(parts),
            f"multipart/form-data; boundary={boundary}", timeout=UPLOAD_TIMEOUT_S,
        )
        if ok:
            print(f"[printer] uploaded {p.name} ({len(payload)} bytes) to {h}; printer said {data}")
            return True
        print(f"[printer] HTTP upload of {p.name} to {h} failed: {data}")

    # No credentials, or HTTP refused: try the credential-free legacy socket.
    remote = f"0:/user/{p.name}"
    try:
        with socket.create_connection((h, config.FF_TCP_PORT), timeout=TCP_TIMEOUT_S) as s:
            s.settimeout(UPLOAD_TIMEOUT_S)
            s.sendall(b"~M601 S1\r\n")
            s.recv(4096)
            s.sendall(f"~M28 {len(payload)} {remote}\r\n".encode("ascii"))
            s.recv(4096)
            for i in range(0, len(payload), 4096):
                s.sendall(payload[i:i + 4096])
            s.sendall(b"~M29\r\n")
            reply = s.recv(4096).decode("ascii", errors="replace")
            s.sendall(b"~M602\r\n")
    except OSError as e:
        print(f"[printer] no route to {h}:{config.FF_TCP_PORT} ({type(e).__name__}: {e}); "
              f"use save_for_transfer() instead")
        return False
    if "ok" not in reply.lower():
        print(f"[printer] legacy upload of {p.name} was not acknowledged: {reply.strip()!r}")
        return False
    print(f"[printer] uploaded {p.name} ({len(payload)} bytes) to {h} as {remote} "
          f"(legacy socket, UNVERIFIED -- confirm on the printer's screen)")
    return True


def start_print(name: str, host: str | None = None, confirm: bool = False) -> bool:
    """Start printing a file already on the printer. Refuses unless confirm.

    This is the only function in the package that makes something physical
    happen -- a heater turns on and a machine starts moving in a room this
    process cannot see. It therefore does nothing at all unless the caller
    passes confirm=True, and there is deliberately no environment variable or
    config setting that can flip that default.

    Returns True only if the printer acknowledged the start command.

    UNVERIFIED against hardware. The refusal path is tested; the start path
    has never been run, by design.
    """
    if not confirm:
        print(f"[printer] refusing to start a print of {name!r}: start_print() "
              f"requires confirm=True. Nothing was sent to the printer.")
        return False

    h = resolve_host(host)
    if not h:
        print(f"[printer] no host configured; cannot start {name!r}. {_reason_no_lan()}")
        return False

    if have_credentials():
        ok, data = _http_json(h, _PRINT_PATH, {"fileName": name, "levelingBeforePrint": False})
        if ok:
            print(f"[printer] start requested for {name!r} on {h}; printer said {data}")
            return True
        print(f"[printer] HTTP start of {name!r} failed: {data}")

    ok, text = _tcp_session(h, [f"~M23 0:/user/{name}"])
    if ok and "ok" in text.lower():
        print(f"[printer] start requested for {name!r} on {h} via the legacy socket")
        return True
    print(f"[printer] could not start {name!r} on {h}: {text}")
    return False


def save_for_transfer(gcode: str | Path, dest: str | Path) -> Path:
    """Copy sliced gcode somewhere the user can find it, and say what to do.

    This is the default and always-available delivery path. It needs no
    network, no credentials and no printer. `dest` may be a directory (the
    source filename is kept) or a full destination path.

    Verified working on this machine against real sliced gcode from out/.
    """
    src = Path(gcode)
    if not src.is_file():
        raise FileNotFoundError(f"no gcode file at {src}")

    d = Path(dest)
    # A path with no suffix, or one that already exists as a directory, is a
    # directory; anything else is the destination filename.
    if d.is_dir() or not d.suffix:
        d = d / src.name
    d.parent.mkdir(parents=True, exist_ok=True)

    if d.resolve() != src.resolve():
        shutil.copy2(src, d)

    print(transfer_instructions(d))
    return d


def transfer_instructions(gcode: str | Path) -> str:
    """The human-readable 'now do this' block printed by save_for_transfer()."""
    p = Path(gcode)
    size_mb = p.stat().st_size / 1e6 if p.is_file() else 0.0
    lines = [
        "",
        f"  gcode ready: {p}  ({size_mb:.2f} MB)",
        f"  printer:     {config.PRINTER_MODEL}",
        "",
        "  Send it to the printer either way:",
        "    USB  -- copy the file to a USB stick, plug it into the printer's",
        "            front port, and pick it from the printer's file list.",
        "    Orca -- open Orca-Flashforge, File > Import > Import Sliced File,",
        "            choose this file, then use its Print button. Orca already",
        "            knows this machine and does not need a check code.",
        "",
    ]
    if not have_credentials():
        lines += [
            "  Direct LAN upload from scan2print is not configured. To enable it,",
            "  turn on LAN mode on the printer (Settings -> Network), read the",
            "  serial number and check code off that screen, and export:",
            "      export FF_HOST=<printer ip> FF_SERIAL=<serial> FF_CHECKCODE=<code>",
            "  It is optional -- the two routes above are what this tool relies on.",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI -- for poking at a real printer by hand, since the tests cannot
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="scan2print.printer", description=__doc__)
    ap.add_argument("--discover", action="store_true", help="probe the LAN for printers")
    ap.add_argument("--timeout", type=float, default=3.0, help="discovery timeout, seconds")
    ap.add_argument("--status", action="store_true", help="query the configured printer")
    ap.add_argument("--host", default=None, help="override config.FF_HOST")
    ap.add_argument("--save", nargs=2, metavar=("GCODE", "DEST"),
                    help="copy gcode somewhere for manual transfer")
    a = ap.parse_args(argv)

    did = False
    if a.discover:
        found = discover(a.timeout)
        print(f"discover({a.timeout}s): {len(found)} printer(s)")
        for r in found:
            print(f"  {r}")
        if not found:
            print("  (nothing on the LAN -- expected when the printer is off or absent)")
        did = True
    if a.status:
        print(json.dumps(status(a.host), indent=2, default=str))
        did = True
    if a.save:
        save_for_transfer(a.save[0], a.save[1])
        did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
