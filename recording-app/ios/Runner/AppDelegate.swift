import Flutter
import UIKit
import UniformTypeIdentifiers

@main
@objc class AppDelegate: FlutterAppDelegate, UIDocumentPickerDelegate {
  private let externalBookmarkKey = "roadsurvey_recorder.externalDirectoryBookmark"
  private var pendingPickerResult: FlutterResult?
  private var activeExternalAccess: [String: ScopedAccess] = [:]

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
      storageChannel.setMethodCallHandler { [weak self] call, result in
        guard let self = self else {
          result(FlutterError(code: "storage_unavailable", message: "App delegate is unavailable.", details: nil))
          return
        }

        switch call.method {
        case "freeBytes":
          let args = call.arguments as? [String: Any]
          let path = args?["path"] as? String ?? NSHomeDirectory()
          let url = URL(fileURLWithPath: path)
          do {
            result(try self.freeBytes(for: url))
          } catch {
            result(self.flutterError(error))
          }
        case "externalFreeBytes":
          do {
            guard let url = try self.resolveExternalDirectory() else {
              result(nil)
              return
            }
            let bytes = try self.withSecurityScopedAccess(url) {
              try self.freeBytes(for: $0)
            }
            result(bytes)
          } catch {
            result(nil)
          }
        case "pickExternalDirectory":
          DispatchQueue.main.async {
            self.pickExternalDirectory(result: result)
          }
        case "isExternalAvailable":
          result(self.isExternalAvailable())
        case "moveFileToExternal":
          let args = call.arguments as? [String: Any]
          guard
            let srcPath = args?["srcPath"] as? String,
            let filename = args?["filename"] as? String
          else {
            result(FlutterError(code: "bad_args", message: "srcPath and filename are required.", details: nil))
            return
          }
          do {
            let destination = try self.copyFileToExternal(srcPath: srcPath, relativePath: filename)
            result(destination.path)
          } catch {
            result(self.flutterError(error))
          }
        case "readExternalText":
          let args = call.arguments as? [String: Any]
          guard let path = args?["path"] as? String else {
            result(FlutterError(code: "bad_args", message: "path is required.", details: nil))
            return
          }
          do {
            result(try self.readExternalText(path: path))
          } catch {
            result(self.flutterError(error))
          }
        case "deleteExternalFile":
          let args = call.arguments as? [String: Any]
          guard let path = args?["path"] as? String else {
            result(FlutterError(code: "bad_args", message: "path is required.", details: nil))
            return
          }
          do {
            result(try self.deleteExternalFile(path: path))
          } catch {
            result(false)
          }
        case "startExternalAccess":
          let args = call.arguments as? [String: Any]
          guard let path = args?["path"] as? String else {
            result(FlutterError(code: "bad_args", message: "path is required.", details: nil))
            return
          }
          do {
            result(try self.startExternalAccess(path: path))
          } catch {
            result(false)
          }
        case "stopExternalAccess":
          let args = call.arguments as? [String: Any]
          guard let path = args?["path"] as? String else {
            result(FlutterError(code: "bad_args", message: "path is required.", details: nil))
            return
          }
          self.stopExternalAccess(path: path)
          result(true)
        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  private func pickExternalDirectory(result: @escaping FlutterResult) {
    guard pendingPickerResult == nil else {
      result(FlutterError(code: "picker_active", message: "A storage picker is already open.", details: nil))
      return
    }
    guard let controller = window?.rootViewController else {
      result(FlutterError(code: "no_controller", message: "No root view controller is available.", details: nil))
      return
    }

    pendingPickerResult = result
    let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.folder])
    picker.delegate = self
    picker.allowsMultipleSelection = false
    controller.present(picker, animated: true)
  }

  func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
    guard let result = pendingPickerResult else {
      return
    }
    pendingPickerResult = nil

    guard let url = urls.first else {
      result(false)
      return
    }

    let accessing = url.startAccessingSecurityScopedResource()
    defer {
      if accessing {
        url.stopAccessingSecurityScopedResource()
      }
    }

    do {
      let bookmark = try url.bookmarkData(
        options: [],
        includingResourceValuesForKeys: nil,
        relativeTo: nil
      )
      UserDefaults.standard.set(bookmark, forKey: externalBookmarkKey)
      result(true)
    } catch {
      result(flutterError(error))
    }
  }

  func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
    pendingPickerResult?(false)
    pendingPickerResult = nil
  }

  private func resolveExternalDirectory() throws -> URL? {
    guard let data = UserDefaults.standard.data(forKey: externalBookmarkKey) else {
      return nil
    }
    var isStale = false
    let url = try URL(
      resolvingBookmarkData: data,
      options: [],
      relativeTo: nil,
      bookmarkDataIsStale: &isStale
    )

    if isStale {
      let accessing = url.startAccessingSecurityScopedResource()
      defer {
        if accessing {
          url.stopAccessingSecurityScopedResource()
        }
      }
      let refreshed = try url.bookmarkData(
        options: [],
        includingResourceValuesForKeys: nil,
        relativeTo: nil
      )
      UserDefaults.standard.set(refreshed, forKey: externalBookmarkKey)
    }

    return url
  }

  private func isExternalAvailable() -> Bool {
    do {
      guard let url = try resolveExternalDirectory() else {
        return false
      }
      return try withSecurityScopedAccess(url) { scopedURL in
        var isDirectory = ObjCBool(false)
        let exists = FileManager.default.fileExists(
          atPath: scopedURL.path,
          isDirectory: &isDirectory
        )
        let reachable = (try? scopedURL.checkResourceIsReachable()) ?? false
        let values = try scopedURL.resourceValues(forKeys: [.isWritableKey])
        return exists && isDirectory.boolValue && reachable && (values.isWritable ?? false)
      }
    } catch {
      return false
    }
  }

  private func copyFileToExternal(srcPath: String, relativePath: String) throws -> URL {
    guard let root = try resolveExternalDirectory() else {
      throw StorageError.externalUnavailable
    }
    let source = URL(fileURLWithPath: srcPath)
    return try withSecurityScopedAccess(root) { scopedRoot in
      guard FileManager.default.fileExists(atPath: source.path) else {
        throw StorageError.sourceMissing
      }
      let destination = try destinationURL(root: scopedRoot, relativePath: relativePath)
      let parent = destination.deletingLastPathComponent()
      try FileManager.default.createDirectory(
        at: parent,
        withIntermediateDirectories: true
      )
      if FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.removeItem(at: destination)
      }
      try FileManager.default.copyItem(at: source, to: destination)
      return destination
    }
  }

  private func readExternalText(path: String) throws -> String {
    guard let root = try resolveExternalDirectory() else {
      throw StorageError.externalUnavailable
    }
    let fileURL = URL(fileURLWithPath: path)
    return try withSecurityScopedAccess(root) { scopedRoot in
      guard isDescendant(fileURL, of: scopedRoot) else {
        throw StorageError.invalidPath
      }
      return try String(contentsOf: fileURL, encoding: .utf8)
    }
  }

  private func deleteExternalFile(path: String) throws -> Bool {
    guard let root = try resolveExternalDirectory() else {
      return false
    }
    let fileURL = URL(fileURLWithPath: path)
    return try withSecurityScopedAccess(root) { scopedRoot in
      guard isDescendant(fileURL, of: scopedRoot) else {
        return false
      }
      if FileManager.default.fileExists(atPath: fileURL.path) {
        try FileManager.default.removeItem(at: fileURL)
      }
      return true
    }
  }

  private func startExternalAccess(path: String) throws -> Bool {
    guard let root = try resolveExternalDirectory() else {
      throw StorageError.externalUnavailable
    }
    let fileURL = URL(fileURLWithPath: path)
    guard isDescendant(fileURL, of: root) else {
      throw StorageError.invalidPath
    }
    if var active = activeExternalAccess[path] {
      active.count += 1
      activeExternalAccess[path] = active
      return true
    }

    let accessing = root.startAccessingSecurityScopedResource()
    let exists = FileManager.default.fileExists(atPath: fileURL.path)
    let reachable = (try? fileURL.checkResourceIsReachable()) ?? exists
    guard exists || reachable else {
      if accessing {
        root.stopAccessingSecurityScopedResource()
      }
      return false
    }
    if accessing {
      activeExternalAccess[path] = ScopedAccess(url: root, count: 1)
    }
    return true
  }

  private func stopExternalAccess(path: String) {
    guard var active = activeExternalAccess[path] else {
      return
    }
    active.count -= 1
    if active.count <= 0 {
      active.url.stopAccessingSecurityScopedResource()
      activeExternalAccess.removeValue(forKey: path)
    } else {
      activeExternalAccess[path] = active
    }
  }

  private func withSecurityScopedAccess<T>(_ url: URL, _ body: (URL) throws -> T) throws -> T {
    let accessing = url.startAccessingSecurityScopedResource()
    defer {
      if accessing {
        url.stopAccessingSecurityScopedResource()
      }
    }
    return try body(url)
  }

  private func destinationURL(root: URL, relativePath: String) throws -> URL {
    let components = try safePathComponents(relativePath)
    var destination = root
    for component in components {
      destination.appendPathComponent(component)
    }
    return destination
  }

  private func safePathComponents(_ relativePath: String) throws -> [String] {
    let components = relativePath.split(separator: "/").map(String.init)
    guard !components.isEmpty else {
      throw StorageError.invalidPath
    }
    for component in components {
      if component.isEmpty || component == "." || component == ".." {
        throw StorageError.invalidPath
      }
    }
    return components
  }

  private func isDescendant(_ url: URL, of root: URL) -> Bool {
    let rootPath = root.standardizedFileURL.path
    let path = url.standardizedFileURL.path
    return path == rootPath || path.hasPrefix(rootPath + "/")
  }

  private func freeBytes(for url: URL) throws -> Int64? {
    let values = try url.resourceValues(forKeys: [
      .volumeAvailableCapacityForImportantUsageKey,
      .volumeAvailableCapacityKey
    ])
    if let importantCapacity = values.volumeAvailableCapacityForImportantUsage {
      return importantCapacity
    }
    if let capacity = values.volumeAvailableCapacity {
      return Int64(capacity)
    }
    return nil
  }

  private func flutterError(_ error: Error) -> FlutterError {
    FlutterError(
      code: "storage_unavailable",
      message: error.localizedDescription,
      details: nil
    )
  }
}

private struct ScopedAccess {
  let url: URL
  var count: Int
}

private enum StorageError: LocalizedError {
  case externalUnavailable
  case invalidPath
  case sourceMissing

  var errorDescription: String? {
    switch self {
    case .externalUnavailable:
      return "External storage is not connected or permission is unavailable."
    case .invalidPath:
      return "The external storage path is invalid."
    case .sourceMissing:
      return "The source file does not exist."
    }
  }
}
