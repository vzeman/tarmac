import ARKit
import CoreMotion
import Flutter

// LiDAR + visual-odometry + gravity-corrected IMU for RoadSurvey Recorder.
//
// Streams ARKit frames (pose + depth + vertical acceleration) at 10 fps via
// a FlutterEventChannel. Falls back to visual-odometry-only on non-LiDAR
// devices so the Dart side always gets pose and gravity-corrected accel.
@objc class LidarPlugin: NSObject {

  static let kMethod = "roadsurvey_recorder/lidar"
  static let kEvents = "roadsurvey_recorder/lidar/frames"

  private var arSession: ARSession?
  private let motionManager = CMMotionManager()
  private var eventSink: FlutterEventSink?

  // Wall-clock anchor: set once on the first ARFrame so utc_ms and pts_ms
  // are consistent with the rest of the sidecar telemetry.
  private var anchorWallMs: Int64 = 0
  private var anchorArTime: Double = -1

  private var lastEmitTime: Double = 0
  private let minIntervalS: Double = 0.10   // 10 fps cap

  private var latestDeviceMotion: CMDeviceMotion?

  // MARK: - Registration

  static func register(with messenger: FlutterBinaryMessenger) {
    let plugin = LidarPlugin()

    let mc = FlutterMethodChannel(name: kMethod, binaryMessenger: messenger)
    mc.setMethodCallHandler { [weak plugin] call, result in
      plugin?.handle(call: call, result: result)
    }

    let ec = FlutterEventChannel(name: kEvents, binaryMessenger: messenger)
    ec.setStreamHandler(plugin)
  }

  // MARK: - Method channel

  private func handle(call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "isLidarAvailable":
      if #available(iOS 14.0, *) {
        result(ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth))
      } else {
        result(false)
      }
    case "isArSupported":
      result(ARWorldTrackingConfiguration.isSupported)
    case "start":
      let args = call.arguments as? [String: Any]
      let captureDepth = (args?["captureDepth"] as? Bool) ?? true
      startSession(captureDepth: captureDepth)
      result(nil)
    case "stop":
      stopSession()
      result(nil)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // MARK: - Session lifecycle

  private func startSession(captureDepth: Bool) {
    stopSession()

    let session = ARSession()
    session.delegate = self
    arSession = session

    let config = ARWorldTrackingConfiguration()
    // Gravity-aligned world: world Y always points up.
    // This makes vert_accel meaningful regardless of phone orientation.
    config.worldAlignment = .gravity

    if captureDepth {
      if #available(iOS 14.0, *),
         ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
        config.frameSemantics = [.smoothedSceneDepth]
      }
    }

    anchorArTime = -1
    anchorWallMs = Int64(Date().timeIntervalSince1970 * 1000)
    lastEmitTime = 0
    session.run(config, options: [.resetTracking, .removeExistingAnchors])

    if motionManager.isDeviceMotionAvailable {
      motionManager.deviceMotionUpdateInterval = 0.01  // 100 Hz
      motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
        self?.latestDeviceMotion = motion
      }
    }
  }

  private func stopSession() {
    arSession?.pause()
    arSession = nil
    motionManager.stopDeviceMotionUpdates()
    latestDeviceMotion = nil
    anchorArTime = -1
  }

  // MARK: - Depth encoding
  // Downsample CVPixelBuffer (kCVPixelFormatType_DepthFloat32) to 32×24
  // Float32, compute centre-patch roughness, return base64 + roughness.

  private func encodeDepth(
    _ pb: CVPixelBuffer
  ) -> (b64: String, w: Int, h: Int, roughness: Double) {
    let tW = 32, tH = 24
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

    let srcW = CVPixelBufferGetWidth(pb)
    let srcH = CVPixelBufferGetHeight(pb)
    guard
      let base = CVPixelBufferGetBaseAddress(pb)
    else { return ("", tW, tH, 0) }
    let src = base.assumingMemoryBound(to: Float32.self)

    var out = [Float32](repeating: 0, count: tW * tH)
    let xScale = Double(srcW) / Double(tW)
    let yScale = Double(srcH) / Double(tH)

    // 16×16 centre patch for roughness
    let pX = tW / 2 - 8, pY = tH / 2 - 8
    var patchVals = [Double]()

    for ty in 0..<tH {
      for tx in 0..<tW {
        let sx = min(Int(Double(tx) * xScale + xScale * 0.5), srcW - 1)
        let sy = min(Int(Double(ty) * yScale + yScale * 0.5), srcH - 1)
        let v = src[sy * srcW + sx]
        let safe: Float32 = (v.isNaN || v.isInfinite || v < 0) ? 0 : v
        out[ty * tW + tx] = safe
        if tx >= pX && tx < pX + 16 && ty >= pY && ty < pY + 16
            && safe > 0.05 && safe < 5.0 {
          patchVals.append(Double(safe))
        }
      }
    }

    var roughness = 0.0
    if patchVals.count > 4 {
      let mean = patchVals.reduce(0, +) / Double(patchVals.count)
      let variance = patchVals.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
        / Double(patchVals.count)
      roughness = variance.squareRoot()
    }

    let data = out.withUnsafeBytes { Data($0) }
    return (data.base64EncodedString(), tW, tH, roughness)
  }
}

// MARK: - ARSessionDelegate

extension LidarPlugin: ARSessionDelegate {
  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    let now = frame.timestamp
    guard now - lastEmitTime >= minIntervalS else { return }
    lastEmitTime = now

    // Anchor wall-clock on first frame
    if anchorArTime < 0 { anchorArTime = now }
    let ptsMs = Int64((now - anchorArTime) * 1000)
    let utcMs = anchorWallMs + ptsMs

    // 4×4 camera-to-world transform, column-major
    let tf = frame.camera.transform
    let pose: [Double] = [
      Double(tf.columns.0.x), Double(tf.columns.0.y),
      Double(tf.columns.0.z), Double(tf.columns.0.w),
      Double(tf.columns.1.x), Double(tf.columns.1.y),
      Double(tf.columns.1.z), Double(tf.columns.1.w),
      Double(tf.columns.2.x), Double(tf.columns.2.y),
      Double(tf.columns.2.z), Double(tf.columns.2.w),
      Double(tf.columns.3.x), Double(tf.columns.3.y),
      Double(tf.columns.3.z), Double(tf.columns.3.w),
    ]

    // Camera intrinsics (focal length + principal point)
    let intr = frame.camera.intrinsics
    let imgSz = frame.camera.imageResolution

    // Gravity-corrected vertical acceleration (world Y = up in .gravity alignment)
    // CMDeviceMotion.userAcceleration is in device frame (gravity already removed).
    // We rotate it into the ARKit world frame using the camera transform's
    // rotation part so vert_accel is always "true up/down" regardless of tilt.
    var vertAccel = 0.0
    if let dm = latestDeviceMotion {
      let ua = dm.userAcceleration
      // Device-frame accel as a column vector
      let dx = Float(ua.x), dy = Float(ua.y), dz = Float(ua.z)
      // Rotate: worldAccel = R * deviceAccel (upper-left 3×3 of camera transform)
      let wy = tf.columns.0.y * dx + tf.columns.1.y * dy + tf.columns.2.y * dz
      vertAccel = Double(wy)  // +ve = upward jolt (pothole), −ve = downward
    }

    // Depth (optional — only present on LiDAR devices with .sceneDepth enabled)
    var depthB64: String? = nil
    var depthW = 0, depthH = 0, roughness = 0.0
    if #available(iOS 14.0, *), let depth = frame.smoothedSceneDepth?.depthMap {
      let enc = encodeDepth(depth)
      depthB64 = enc.b64; depthW = enc.w; depthH = enc.h; roughness = enc.roughness
    }

    // ARKit tracking quality
    let tracking: String
    switch frame.camera.trackingState {
    case .normal: tracking = "normal"
    case .notAvailable: tracking = "notAvailable"
    case .limited(let r):
      switch r {
      case .initializing:       tracking = "limited:initializing"
      case .relocalizing:       tracking = "limited:relocalizing"
      case .excessiveMotion:    tracking = "limited:excessiveMotion"
      case .insufficientFeatures: tracking = "limited:insufficientFeatures"
      @unknown default:         tracking = "limited"
      }
    @unknown default: tracking = "unknown"
    }

    var payload: [String: Any] = [
      "utc_ms":    utcMs,
      "pts_ms":    ptsMs,
      "pose":      pose,
      "fx":        Double(intr.columns.0.x),
      "fy":        Double(intr.columns.1.y),
      "cx":        Double(intr.columns.2.x),
      "cy":        Double(intr.columns.2.y),
      "img_w":     Int(imgSz.width),
      "img_h":     Int(imgSz.height),
      "roughness": roughness,
      "vert_accel": vertAccel,
      "tracking":  tracking,
    ]
    if let b64 = depthB64, !b64.isEmpty {
      payload["depth_f32"] = b64
      payload["depth_w"]   = depthW
      payload["depth_h"]   = depthH
    }

    DispatchQueue.main.async { [weak self] in
      self?.eventSink?(payload)
    }
  }
}

// MARK: - FlutterStreamHandler

extension LidarPlugin: FlutterStreamHandler {
  func onListen(
    withArguments arguments: Any?,
    eventSink events: @escaping FlutterEventSink
  ) -> FlutterError? {
    eventSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    eventSink = nil
    return nil
  }
}
