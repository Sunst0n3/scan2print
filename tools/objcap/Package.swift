// swift-tools-version: 6.0
import PackageDescription

// Deployment target is macOS 26 on purpose: PhotogrammetrySession.Pose.intrinsics
// and .lensDistortionData are macOS 26.0+, and dropping the availability guards
// keeps the pose-export path free of optional-chaining noise.
let package = Package(
    name: "objcap",
    platforms: [.macOS("26.0")],
    targets: [
        .executableTarget(
            name: "objcap",
            path: "Sources/objcap",
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
