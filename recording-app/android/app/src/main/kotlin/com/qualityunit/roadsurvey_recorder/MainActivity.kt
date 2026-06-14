package com.qualityunit.roadsurvey_recorder

import android.os.StatFs
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "roadsurvey_recorder/storage"
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
                else -> result.notImplemented()
            }
        }
    }
}
