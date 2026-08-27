import Foundation
import RealityKit

enum CLIError: Error, CustomStringConvertible {
    case usage(String)
    case badPath(String)

    var description: String {
        switch self {
        case .usage(let m): return m
        case .badPath(let m): return m
        }
    }
}

struct Options {
    var input: URL
    var output: URL
    var detail: PhotogrammetrySession.Request.Detail = .full
    var modelExtensions: [String] = ["usdz"]
    var sampleOrdering: PhotogrammetrySession.Configuration.SampleOrdering = .unordered
    var featureSensitivity: PhotogrammetrySession.Configuration.FeatureSensitivity = .high

    static let usage = """
    objcap -- Apple Object Capture wrapper for the scan2print pipeline.

    USAGE:
      objcap --input <image-dir> --output <out-dir> [options]

    REQUIRED:
      --input  <dir>    Directory of input frames (jpg/jpeg/png/heic/heif/tif/tiff).
      --output <dir>    Directory to write model.usdz and poses.json into.
                        Created if it does not exist.

    OPTIONS:
      --detail <level>  preview | reduced | medium | full | raw     (default: full)
      --poses-only      Skip mesh reconstruction; request camera poses only.
                        Equivalent to --format none.
      --format <list>   Comma-separated model file extensions, or 'none'.
                        (default: usdz)  Object Capture picks its writer from
                        the extension and issues one modelFile request each,
                        e.g. --format usdz. VERIFIED by probing this SDK:
                        .usdz is the ONLY accepted extension. usd, usda, usdc,
                        obj, stl, ply, glb, gltf and drc are all rejected with
                        PhotogrammetrySession.Error.invalidOutput, thrown
                        synchronously from process(). Convert to STL/OBJ
                        downstream (trimesh reads usdz via Blender).
      --ordering <o>    unordered | sequential                      (default: unordered)
      --sensitivity <s> normal | high                               (default: high)
      -h, --help        Print this help and exit 0.

    OUTPUT:
      <out-dir>/model.usdz   reconstructed mesh (unless --poses-only / --format none)
      <out-dir>/poses.json   camera poses in the arbitrary-scale RECON frame:
        {
          "reconstruction": "objectcapture",
          "detail": "full",
          "images": [ {"name": "<filename without extension>",
                       "transform": [16 floats, column-major camera-to-world],
                       "center": [x, y, z]} ]
        }
      Extra top-level keys carry bonus data: intrinsics_by_name (9 floats,
      column-major 3x3 K, recovered by Object Capture itself),
      lens_distortion_profiles + lens_distortion_profile_by_name (deduplicated
      radial undistortion tables), sample_id_by_name, unmatched_sample_ids and
      input_images_without_pose. The objects inside "images" are kept to
      exactly name/transform/center so Python can splat them straight into
      ReconPose(**d).

    Progress and diagnostics go to stderr; stdout carries only a one-line JSON
    summary on success. Exit code is non-zero on any failure.
    """

    static func parse(_ argv: [String]) throws -> Options {
        var input: String?
        var output: String?
        var detail: String = "full"
        var format: String?
        var posesOnly = false
        var ordering = "unordered"
        var sensitivity = "high"

        var i = 1
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < argv.count else { throw CLIError.usage("\(flag) requires a value") }
            return argv[i]
        }

        while i < argv.count {
            let a = argv[i]
            switch a {
            case "-h", "--help":
                throw CLIError.usage("")
            case "--input", "-i":   input = try value(a)
            case "--output", "-o":  output = try value(a)
            case "--detail":        detail = try value(a)
            case "--format":        format = try value(a)
            case "--ordering":      ordering = try value(a)
            case "--sensitivity":   sensitivity = try value(a)
            case "--poses-only":    posesOnly = true
            default:
                throw CLIError.usage("unknown argument: \(a)")
            }
            i += 1
        }

        guard let inputPath = input else { throw CLIError.usage("--input is required") }
        guard let outputPath = output else { throw CLIError.usage("--output is required") }

        let detailMap: [String: PhotogrammetrySession.Request.Detail] = [
            "preview": .preview, "reduced": .reduced, "medium": .medium,
            "full": .full, "raw": .raw,
        ]
        guard let detailValue = detailMap[detail.lowercased()] else {
            throw CLIError.usage("--detail must be one of preview|reduced|medium|full|raw, got '\(detail)'")
        }

        if posesOnly, let f = format, f.lowercased() != "none" {
            throw CLIError.usage("--poses-only conflicts with --format \(f)")
        }
        let requestedFormat = posesOnly ? "none" : (format ?? "usdz")
        var exts: [String] = []
        if requestedFormat.lowercased() != "none" {
            exts = requestedFormat.lowercased()
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            guard !exts.isEmpty else {
                throw CLIError.usage("--format got no usable extensions from '\(requestedFormat)'")
            }
            if Set(exts).count != exts.count {
                throw CLIError.usage("--format lists a duplicate extension: '\(requestedFormat)'")
            }
        }

        let orderingMap: [String: PhotogrammetrySession.Configuration.SampleOrdering] = [
            "unordered": .unordered, "sequential": .sequential,
        ]
        guard let ord = orderingMap[ordering.lowercased()] else {
            throw CLIError.usage("--ordering must be unordered|sequential, got '\(ordering)'")
        }

        let sensitivityMap: [String: PhotogrammetrySession.Configuration.FeatureSensitivity] = [
            "normal": .normal, "high": .high,
        ]
        guard let sens = sensitivityMap[sensitivity.lowercased()] else {
            throw CLIError.usage("--sensitivity must be normal|high, got '\(sensitivity)'")
        }

        let inURL = URL(fileURLWithPath: (inputPath as NSString).expandingTildeInPath).standardizedFileURL
        let outURL = URL(fileURLWithPath: (outputPath as NSString).expandingTildeInPath).standardizedFileURL

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: inURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw CLIError.badPath("input is not an existing directory: \(inURL.path)")
        }

        return Options(input: inURL, output: outURL, detail: detailValue,
                       modelExtensions: exts, sampleOrdering: ord, featureSensitivity: sens)
    }
}

extension PhotogrammetrySession.Request.Detail {
    var label: String {
        switch self {
        case .preview: return "preview"
        case .reduced: return "reduced"
        case .medium:  return "medium"
        case .full:    return "full"
        case .raw:     return "raw"
        case .custom:  return "custom"
        @unknown default: return "unknown(\(rawValue))"
        }
    }
}
