import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    if let controller = window?.rootViewController as? FlutterViewController {
      let storageChannel = FlutterMethodChannel(
        name: "roadsurvey_recorder/storage",
        binaryMessenger: controller.binaryMessenger
      )
      storageChannel.setMethodCallHandler { call, result in
        guard call.method == "freeBytes" else {
          result(FlutterMethodNotImplemented)
          return
        }

        let args = call.arguments as? [String: Any]
        let path = args?["path"] as? String ?? NSHomeDirectory()
        let url = URL(fileURLWithPath: path)
        do {
          let values = try url.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey
          ])
          result(values.volumeAvailableCapacityForImportantUsage)
        } catch {
          result(FlutterError(code: "storage_unavailable", message: error.localizedDescription, details: nil))
        }
      }
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
