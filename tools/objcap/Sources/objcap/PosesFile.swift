import Foundation
import RealityKit
import simd

/// One camera in the arbitrary-scale RECON frame.
///
/// Deliberately limited to name/transform/center: the Python side splats these
/// dicts straight into `ReconPose(**d)`, so any extra key here would be a
/// TypeError. Bonus per-camera data lives in side tables on `PosesFile`.
struct PoseEntry: Encodable {
    let name: String
    let transform: [Double]   // 16 floats, column-major, camera-to-world
    let center: [Double]      // translation component, RECON units
}

struct LensDistortionJSON: Encodable, Equatable {
    let center: [Double]              // 2 floats, pixels
    let radial_lookup_table: [Double]
}

struct PosesFile: Encodable {
    let reconstruction: String
    let detail: String
    let images: [PoseEntry]

    // --- additive extras, safe because consumers index data["images"] ---
    let image_count: Int
    let name_source: String
    let sample_id_by_name: [String: Int]
    let intrinsics_by_name: [String: [Double]]        // 9 floats, column-major 3x3
    /// Distortion profiles are deduplicated: every camera in a session shot on
    /// one lens shares a byte-identical 2048-entry radial table, and emitting
    /// it per image inflated poses.json from ~90 KB to 3.2 MB.
    let lens_distortion_profiles: [LensDistortionJSON]
    let lens_distortion_profile_by_name: [String: Int]
    let unmatched_sample_ids: [Int]
    let input_images_without_pose: [String]
}

/// Flatten a float4x4 to 16 doubles in column-major order:
/// [c0.x c0.y c0.z c0.w  c1.x ... c3.w]. This is simd's native storage order
/// and matches what numpy expects from `np.array(t).reshape(4, 4, order="F")`.
func columnMajor(_ m: simd_float4x4) -> [Double] {
    var out = [Double]()
    out.reserveCapacity(16)
    for c in [m.columns.0, m.columns.1, m.columns.2, m.columns.3] {
        out.append(Double(c.x)); out.append(Double(c.y))
        out.append(Double(c.z)); out.append(Double(c.w))
    }
    return out
}

func columnMajor3(_ m: simd_float3x3) -> [Double] {
    var out = [Double]()
    out.reserveCapacity(9)
    for c in [m.columns.0, m.columns.1, m.columns.2] {
        out.append(Double(c.x)); out.append(Double(c.y)); out.append(Double(c.z))
    }
    return out
}

let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "dng"]

/// Input frames in the same lexicographic order Object Capture enumerates a
/// folder in. Used both for reporting and as the fallback id->name mapping.
func listInputImages(_ dir: URL) throws -> [URL] {
    let all = try FileManager.default.contentsOfDirectory(
        at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    return all
        .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
}

func stem(_ url: URL) -> String {
    url.deletingPathExtension().lastPathComponent
}
