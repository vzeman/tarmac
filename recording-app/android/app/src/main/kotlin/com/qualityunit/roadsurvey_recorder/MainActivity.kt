package com.qualityunit.roadsurvey_recorder

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.os.storage.StorageManager
import android.provider.DocumentsContract
import android.webkit.MimeTypeMap
import androidx.documentfile.provider.DocumentFile
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileInputStream

class MainActivity : FlutterActivity() {
    private val storageChannelName = "roadsurvey_recorder/storage"
    private val externalPrefsName = "roadsurvey_recorder.storage"
    private val externalTreeUriKey = "externalTreeUri"
    private val externalTreeRequestCode = 8704
    private var pendingPickerResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            storageChannelName
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "freeBytes" -> {
                    val path = call.argument<String>("path") ?: filesDir.absolutePath
                    try {
                        result.success(StatFs(path).availableBytes)
                    } catch (error: IllegalArgumentException) {
                        result.error("storage_unavailable", error.localizedMessage, null)
                    }
                }
                "externalFreeBytes" -> result.success(externalFreeBytes())
                "pickExternalDirectory" -> pickExternalDirectory(result)
                "isExternalAvailable" -> result.success(isExternalAvailable())
                "moveFileToExternal" -> {
                    val srcPath = call.argument<String>("srcPath")
                    val filename = call.argument<String>("filename")
                    if (srcPath == null || filename == null) {
                        result.error("bad_args", "srcPath and filename are required.", null)
                        return@setMethodCallHandler
                    }
                    try {
                        result.success(copyFileToExternal(srcPath, filename).toString())
                    } catch (error: Exception) {
                        result.error("storage_unavailable", error.localizedMessage, null)
                    }
                }
                "readExternalText" -> {
                    val path = call.argument<String>("path")
                    if (path == null) {
                        result.error("bad_args", "path is required.", null)
                        return@setMethodCallHandler
                    }
                    try {
                        result.success(readExternalText(path))
                    } catch (error: Exception) {
                        result.error("storage_unavailable", error.localizedMessage, null)
                    }
                }
                "deleteExternalFile" -> {
                    val path = call.argument<String>("path")
                    if (path == null) {
                        result.error("bad_args", "path is required.", null)
                        return@setMethodCallHandler
                    }
                    result.success(deleteExternalFile(path))
                }
                "startExternalAccess" -> {
                    val path = call.argument<String>("path")
                    if (path == null) {
                        result.error("bad_args", "path is required.", null)
                        return@setMethodCallHandler
                    }
                    result.success(canOpenExternalPath(path))
                }
                "stopExternalAccess" -> result.success(true)
                else -> result.notImplemented()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == externalTreeRequestCode) {
            val result = pendingPickerResult
            pendingPickerResult = null
            val uri = data?.data
            if (resultCode == Activity.RESULT_OK && uri != null) {
                val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                try {
                    contentResolver.takePersistableUriPermission(uri, flags)
                    prefs().edit().putString(externalTreeUriKey, uri.toString()).apply()
                    result?.success(true)
                } catch (error: SecurityException) {
                    result?.error("storage_unavailable", error.localizedMessage, null)
                }
            } else {
                result?.success(false)
            }
            return
        }
        super.onActivityResult(requestCode, resultCode, data)
    }

    private fun pickExternalDirectory(result: MethodChannel.Result) {
        if (pendingPickerResult != null) {
            result.error("picker_active", "A storage picker is already open.", null)
            return
        }
        pendingPickerResult = result
        val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
            Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
            Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION or
            Intent.FLAG_GRANT_PREFIX_URI_PERMISSION
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(flags)
        }
        startActivityForResult(intent, externalTreeRequestCode)
    }

    private fun isExternalAvailable(): Boolean {
        val uri = savedTreeUri() ?: return false
        if (!hasPersistedWritePermission(uri)) {
            return false
        }
        return try {
            val tree = DocumentFile.fromTreeUri(this, uri) ?: return false
            tree.exists() && tree.isDirectory && tree.canWrite()
        } catch (_: Exception) {
            false
        }
    }

    private fun copyFileToExternal(srcPath: String, relativePath: String): Uri {
        if (!isExternalAvailable()) {
            throw IllegalStateException("External storage is not connected or writable.")
        }
        val source = File(srcPath)
        if (!source.exists()) {
            throw IllegalStateException("Source file does not exist.")
        }
        val tree = DocumentFile.fromTreeUri(this, savedTreeUri()!!)
            ?: throw IllegalStateException("External storage tree is unavailable.")
        val target = createDestinationFile(tree, relativePath)
        FileInputStream(source).use { input ->
            val output = contentResolver.openOutputStream(target.uri, "w")
                ?: throw IllegalStateException("Could not open external destination.")
            output.use { input.copyTo(it) }
        }
        return target.uri
    }

    private fun createDestinationFile(root: DocumentFile, relativePath: String): DocumentFile {
        val components = safePathComponents(relativePath)
        var directory = root
        for (component in components.dropLast(1)) {
            val existing = directory.findFile(component)
            directory = if (existing != null && existing.isDirectory) {
                existing
            } else {
                existing?.delete()
                directory.createDirectory(component)
                    ?: throw IllegalStateException("Could not create external directory.")
            }
        }
        val filename = components.last()
        directory.findFile(filename)?.delete()
        return directory.createFile(mimeTypeFor(filename), filename)
            ?: throw IllegalStateException("Could not create external file.")
    }

    private fun readExternalText(path: String): String {
        if (!isExternalAvailable()) {
            throw IllegalStateException("External storage is not connected or writable.")
        }
        val uri = Uri.parse(path)
        val input = contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Could not open external file.")
        return input.bufferedReader().use { it.readText() }
    }

    private fun deleteExternalFile(path: String): Boolean {
        if (!isExternalAvailable()) {
            return false
        }
        return try {
            val uri = Uri.parse(path)
            DocumentFile.fromSingleUri(this, uri)?.delete() ?: false
        } catch (_: Exception) {
            false
        }
    }

    private fun canOpenExternalPath(path: String): Boolean {
        return try {
            if (path.startsWith("content://")) {
                contentResolver.openAssetFileDescriptor(Uri.parse(path), "r")?.use {
                    true
                } ?: false
            } else {
                File(path).exists()
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun externalFreeBytes(): Long? {
        if (!isExternalAvailable()) {
            return null
        }
        val uri = savedTreeUri() ?: return null
        return try {
            if (uri.authority != "com.android.externalstorage.documents") {
                return null
            }
            val treeDocumentId = DocumentsContract.getTreeDocumentId(uri)
            val volumeId = treeDocumentId.substringBefore(':')
            val directory = volumeDirectory(volumeId) ?: return null
            StatFs(directory.absolutePath).availableBytes
        } catch (_: Exception) {
            null
        }
    }

    private fun volumeDirectory(volumeId: String): File? {
        if (volumeId.equals("primary", ignoreCase = true)) {
            return Environment.getExternalStorageDirectory()
        }

        val storageManager = getSystemService(StorageManager::class.java)
        for (volume in storageManager.storageVolumes) {
            if (volume.uuid.equals(volumeId, ignoreCase = true)) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    volume.directory?.let { return it }
                }
                return File("/storage/${volume.uuid}")
            }
        }
        return File("/storage/$volumeId").takeIf { it.exists() }
    }

    private fun safePathComponents(relativePath: String): List<String> {
        val components = relativePath.split('/').filter { it.isNotBlank() }
        if (components.isEmpty() || components.any { it == "." || it == ".." }) {
            throw IllegalArgumentException("Invalid external path.")
        }
        return components
    }

    private fun mimeTypeFor(filename: String): String {
        val extension = filename.substringAfterLast('.', "").lowercase()
        if (extension.isEmpty()) {
            return "application/octet-stream"
        }
        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension)
            ?: "application/octet-stream"
    }

    private fun savedTreeUri(): Uri? {
        val raw = prefs().getString(externalTreeUriKey, null) ?: return null
        return Uri.parse(raw)
    }

    private fun hasPersistedWritePermission(uri: Uri): Boolean {
        return contentResolver.persistedUriPermissions.any {
            it.uri == uri && it.isReadPermission && it.isWritePermission
        }
    }

    private fun prefs() = getSharedPreferences(externalPrefsName, MODE_PRIVATE)
}
