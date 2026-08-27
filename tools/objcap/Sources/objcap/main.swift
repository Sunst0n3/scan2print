import Foundation

// Top-level code in Swift 6 is @MainActor-isolated, which is exactly what the
// non-Sendable PhotogrammetrySession and its Outputs sequence need: session
// creation, process(), and output iteration all stay in one isolation domain.

let options: Options
do {
    options = try Options.parse(CommandLine.arguments)
} catch let e as CLIError {
    let msg = e.description
    if msg.isEmpty {
        print(Options.usage)
        exit(0)
    }
    FileHandle.standardError.write((Options.usage + "\n\nerror: " + msg + "\n").data(using: .utf8)!)
    exit(2)
} catch {
    FileHandle.standardError.write(("error: \(error)\n").data(using: .utf8)!)
    exit(2)
}

do {
    let summary = try await runSession(options)
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try enc.encode(summary)
    print(String(data: data, encoding: .utf8)!)
    exit(0)
} catch let e as RunError {
    elog("FATAL: \(e.description)")
    exit(1)
} catch {
    elog("FATAL: \(error.localizedDescription) [\(String(describing: error))]")
    exit(1)
}
