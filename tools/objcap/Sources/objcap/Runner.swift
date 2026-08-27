import Foundation
import RealityKit
import simd

func elog(_ s: String) {
    FileHandle.standardError.write(("objcap: " + s + "\n").data(using: .utf8)!)
}

enum RunError: Error, CustomStringConvertible {
    case notSupported
    case noImages(String)
    case sessionInit(String)
    case requestFailed(String)
    case cancelled
    case noPoses
    case write(String)

    var description: String {
        switch self {
        case .notSupported:
            return "PhotogrammetrySession.isSupported == false on this machine (needs Apple silicon + a supported GPU)."
        case .noImages(let d):
            return "no usable input images found in \(d) (looked for \(imageExtensions.sorted().joined(separator: ", ")))"
        case .sessionInit(let m):
            return "could not create PhotogrammetrySession: \(m)"
        case .requestFailed(let m):
            return "Object Capture request failed: \(m)"
        case .cancelled:
            return "Object Capture cancelled the session before completion"
        case .noPoses:
            return "session completed but never delivered a .poses result -- poses.json cannot be written"
        case .write(let m):
            return "failed writing output: \(m)"
        }
    }
}

struct RunSummary: Encodable {
    let ok: Bool
    let detail: String
    let poses_json: String
    let n_poses: Int
    let n_input_images: Int
    let model_files: [String]
    let name_source: String
    let elapsed_s: Double
}

@MainActor
func runSession(_ opts: Options) async throws -> RunSummary {
    let t0 = Date()

    guard PhotogrammetrySession.isSupported else { throw RunError.notSupported }

    let inputImages = try listInputImages(opts.input)
    guard !inputImages.isEmpty else { throw RunError.noImages(opts.input.path) }

    let limits = PhotogrammetrySession.limits
    elog("input dir      : \(opts.input.path)")
    elog("input images   : \(inputImages.count) (SDK limit \(limits.maximumNumberOfInputImages), max dim \(limits.maximumInputImageDimension))")
    elog("detail         : \(opts.detail.label)")
    elog("first/last     : \(inputImages.first!.lastPathComponent) .. \(inputImages.last!.lastPathComponent)")

    try FileManager.default.createDirectory(at: opts.output, withIntermediateDirectories: true)

    var config = PhotogrammetrySession.Configuration()
    config.sampleOrdering = opts.sampleOrdering
    config.featureSensitivity = opts.featureSensitivity

    let session: PhotogrammetrySession
    do {
        session = try PhotogrammetrySession(input: opts.input, configuration: config)
    } catch {
        throw RunError.sessionInit(String(describing: error))
    }

    var requests: [PhotogrammetrySession.Request] = [.poses]
    let modelURLs: [URL] = opts.modelExtensions.map {
        opts.output.appendingPathComponent("model.\($0)")
    }
    for u in modelURLs {
        requests.append(.modelFile(url: u, detail: opts.detail))
    }
    elog("requests       : \(requests.count) -> .poses" + (modelURLs.isEmpty ? "" : ", " + modelURLs.map { $0.lastPathComponent }.joined(separator: ", ")))

    do {
        try session.process(requests: requests)
    } catch let e as PhotogrammetrySession.Error {
        if case .invalidOutput(let url) = e {
            throw RunError.requestFailed(
                "Object Capture rejected the output file '\(url.lastPathComponent)'. Probing this SDK "
                + "showed .usdz is the ONLY accepted mesh extension (usd/usda/usdc/obj/stl/ply/glb all fail). "
                + "Underlying error: \(String(describing: e))")
        }
        throw RunError.requestFailed("process(requests:) threw: \(String(describing: e))")
    } catch {
        throw RunError.requestFailed("process(requests:) threw: \(String(describing: error))")
    }

    var poses: PhotogrammetrySession.Poses?
    var writtenModels: [String] = []
    var invalidSamples: [Int] = []
    var skippedSamples: [Int] = []
    var lastReported: [String: Int] = [:]
    var lastStage: [String: String] = [:]
    var firstError: String?

    outer: for try await output in session.outputs {
        switch output {
        case .inputComplete:
            elog("stage          : input ingestion complete")

        case .requestProgress(let request, let fractionComplete):
            let key = requestKey(request)
            let pct = Int(fractionComplete * 100)
            if pct >= (lastReported[key] ?? -5) + 5 {
                lastReported[key] = pct
                elog("progress       : \(key) \(pct)%")
            }

        case .requestProgressInfo(let request, let info):
            // ProgressInfo arrives many times a second with an unchanged stage;
            // only the transitions are worth putting on stderr.
            let key = requestKey(request)
            let stage = info.processingStage.map { describeStage($0) } ?? "unreported"
            guard lastStage[key] != stage else { break }
            lastStage[key] = stage
            let eta = info.estimatedRemainingTime.map { " eta≈\(Int($0))s" } ?? ""
            elog("stage          : \(key) \(stage)\(eta)")

        case .requestComplete(let request, let result):
            switch result {
            case .poses(let p):
                poses = p
                elog("poses result   : posesBySample.count = \(p.posesBySample.count)")
            case .modelFile(let url):
                writtenModels.append(url.path)
                elog("model written  : \(url.path)")
            default:
                elog("unused result  : \(requestKey(request))")
            }

        case .requestError(let request, let error):
            let msg = "\(requestKey(request)): \(error.localizedDescription) [\(String(describing: error))]"
            elog("REQUEST ERROR  : \(msg)")
            if firstError == nil { firstError = msg }

        case .invalidSample(let id, let reason):
            invalidSamples.append(id)
            elog("invalid sample : id=\(id) \(reason)")

        case .skippedSample(let id):
            skippedSamples.append(id)
            elog("skipped sample : id=\(id)")

        case .automaticDownsampling:
            elog("note           : Object Capture automatically downsampled the inputs")

        case .stitchingIncomplete:
            elog("WARNING        : stitching incomplete -- reconstruction did not close; poses may cover only part of the track")

        case .processingComplete:
            elog("stage          : processing complete")
            break outer

        case .processingCancelled:
            throw RunError.cancelled

        @unknown default:
            elog("note           : unhandled output case \(String(describing: output))")
        }
    }

    if let e = firstError, poses == nil {
        throw RunError.requestFailed(e)
    }
    guard let poses else { throw RunError.noPoses }

    let (file, summaryNameSource) = buildPosesFile(poses: poses,
                                                   inputImages: inputImages,
                                                   detail: opts.detail)

    let posesURL = opts.output.appendingPathComponent("poses.json")
    do {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        try enc.encode(file).write(to: posesURL, options: .atomic)
    } catch {
        throw RunError.write("\(posesURL.path): \(String(describing: error))")
    }
    elog("poses.json     : \(posesURL.path) (\(file.images.count) cameras)")

    if let e = firstError {
        elog("WARNING        : a request errored but poses were still recovered: \(e)")
    }

    return RunSummary(
        ok: true,
        detail: opts.detail.label,
        poses_json: posesURL.path,
        n_poses: file.images.count,
        n_input_images: inputImages.count,
        model_files: writtenModels,
        name_source: summaryNameSource,
        elapsed_s: Date().timeIntervalSince(t0)
    )
}

/// Join Object Capture's integer sample ids back to input filenames.
///
/// Primary path is `Poses.urlsBySample`, which this SDK does populate when the
/// session was created from a folder URL. The positional fallback exists
/// because that dictionary is documented nowhere and is empty for
/// sample-sequence inputs; ids are assigned in folder-enumeration order, which
/// is the lexicographic order `listInputImages` reproduces.
@MainActor
func buildPosesFile(poses: PhotogrammetrySession.Poses,
                    inputImages: [URL],
                    detail: PhotogrammetrySession.Request.Detail) -> (PosesFile, String) {

    let urls = poses.urlsBySample
    elog("urlsBySample   : \(urls.count) entries" + (urls.isEmpty ? " (EMPTY -- falling back to positional id->filename)" : ""))
    if let sampleID = poses.posesBySample.keys.sorted().first {
        let p = poses.posesBySample[sampleID]!
        elog("probe sample \(sampleID): url=\(urls[sampleID]?.lastPathComponent ?? "<nil>") " +
             "t=(\(p.translation.x), \(p.translation.y), \(p.translation.z)) " +
             "q=(\(p.rotation.vector.x), \(p.rotation.vector.y), \(p.rotation.vector.z), \(p.rotation.vector.w)) " +
             "intrinsics=\(p.intrinsics != nil ? "present" : "nil") " +
             "lensDistortion=\(p.lensDistortionData != nil ? "present" : "nil")")
    }

    let nameSource = urls.isEmpty ? "positional-fallback" : "urlsBySample"

    var entries: [PoseEntry] = []
    var sampleIDByName: [String: Int] = [:]
    var intrinsicsByName: [String: [Double]] = [:]
    var distortionProfiles: [LensDistortionJSON] = []
    var distortionIndexByName: [String: Int] = [:]
    var unmatched: [Int] = []
    var usedNames = Set<String>()

    for sampleID in poses.posesBySample.keys.sorted() {
        guard let pose = poses.posesBySample[sampleID] else { continue }

        var name: String?
        if let u = urls[sampleID] {
            name = stem(u)
        } else if sampleID >= 0 && sampleID < inputImages.count {
            name = stem(inputImages[sampleID])
        }
        guard let resolved = name else {
            unmatched.append(sampleID)
            continue
        }

        let m = pose.transform.matrix
        entries.append(PoseEntry(
            name: resolved,
            transform: columnMajor(m),
            center: [Double(pose.translation.x), Double(pose.translation.y), Double(pose.translation.z)]
        ))
        usedNames.insert(resolved)
        sampleIDByName[resolved] = sampleID
        if let k = pose.intrinsics {
            intrinsicsByName[resolved] = columnMajor3(k)
        }
        if let d = pose.lensDistortionData {
            let profile = LensDistortionJSON(
                center: [Double(d.center.x), Double(d.center.y)],
                radial_lookup_table: d.radialLookupTable.map(Double.init))
            if let existing = distortionProfiles.firstIndex(of: profile) {
                distortionIndexByName[resolved] = existing
            } else {
                distortionProfiles.append(profile)
                distortionIndexByName[resolved] = distortionProfiles.count - 1
            }
        }
    }

    if !unmatched.isEmpty {
        elog("WARNING        : \(unmatched.count) sample ids had no filename mapping: \(unmatched.prefix(10))")
    }
    elog("lens profiles  : \(distortionProfiles.count) unique across \(distortionIndexByName.count) cameras")
    let missing = inputImages.map(stem).filter { !usedNames.contains($0) }
    if !missing.isEmpty {
        elog("note           : \(missing.count)/\(inputImages.count) input images got no pose (Object Capture dropped them)")
    }

    let file = PosesFile(
        reconstruction: "objectcapture",
        detail: detail.label,
        images: entries,
        image_count: entries.count,
        name_source: nameSource,
        sample_id_by_name: sampleIDByName,
        intrinsics_by_name: intrinsicsByName,
        lens_distortion_profiles: distortionProfiles,
        lens_distortion_profile_by_name: distortionIndexByName,
        unmatched_sample_ids: unmatched,
        input_images_without_pose: missing
    )
    return (file, nameSource)
}

func requestKey(_ r: PhotogrammetrySession.Request) -> String {
    switch r {
    case .poses: return "poses"
    case .bounds: return "bounds"
    case .pointCloud: return "pointCloud"
    case .modelEntity: return "modelEntity"
    case .modelFile(let url, _, _): return "modelFile(\(url.lastPathComponent))"
    @unknown default: return "unknown-request"
    }
}

func describeStage(_ s: PhotogrammetrySession.Output.ProcessingStage) -> String {
    switch s {
    case .preProcessing: return "preProcessing"
    case .imageAlignment: return "imageAlignment"
    case .pointCloudGeneration: return "pointCloudGeneration"
    case .meshGeneration: return "meshGeneration"
    case .textureMapping: return "textureMapping"
    case .optimization: return "optimization"
    @unknown default: return "unknownStage"
    }
}
