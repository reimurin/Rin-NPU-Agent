package com.geniex.demo.model

import android.content.Context
import com.geniex.demo.BuildConfig
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.ModelPullInput
import java.io.File

object TestModelSeeder {
    const val MODEL_NAME = "local/qwen3.5-2b-test"
    private const val ASSET_DIR = "test_model"
    private const val MODEL_FILE = "Qwen3.5-2B-Q4_0.gguf"
    private const val MMPROJ_FILE = "mmproj-F16.gguf"

    suspend fun ensureSeeded(context: Context): String? {
        if (!BuildConfig.TEST_MODEL_SEED) return null
        ModelManagerWrapper.getPaths(MODEL_NAME)?.let { return it.model_dir }

        val seedDir = File(context.filesDir, "debug_seed_qwen35_2b").apply { mkdirs() }
        copyAssetAtomic(context, "$ASSET_DIR/$MODEL_FILE", File(seedDir, MODEL_FILE))
        copyAssetAtomic(context, "$ASSET_DIR/$MMPROJ_FILE", File(seedDir, MMPROJ_FILE))

        var failure: String? = null
        ModelManagerWrapper.pullFlow(
            ModelPullInput(
                model_name = MODEL_NAME,
                hub = HubSource.LOCALFS,
                local_path = seedDir.absolutePath,
            ),
        ).collect { event ->
            if (event is ModelManagerWrapper.PullEvent.Error) {
                failure = event.message
            }
        }
        if (failure != null) error("Test model import failed: $failure")
        val paths = ModelManagerWrapper.getPaths(MODEL_NAME) ?: error("Test model import completed but cache paths are missing")
        seedDir.deleteRecursively()
        return paths.model_dir
    }

    private fun copyAssetAtomic(context: Context, assetPath: String, target: File) {
        val expected = context.assets.openFd(assetPath).use { it.length }
        if (target.isFile && target.length() == expected) return
        val part = File(target.absolutePath + ".part")
        if (part.exists()) part.delete()
        context.assets.open(assetPath, android.content.res.AssetManager.ACCESS_STREAMING).use { input ->
            part.outputStream().buffered(1024 * 1024).use { output -> input.copyTo(output, 1024 * 1024) }
        }
        check(part.length() == expected) { "Asset length mismatch for $assetPath: expected $expected, got ${part.length()}" }
        if (target.exists()) target.delete()
        check(part.renameTo(target)) { "Could not finalize $assetPath" }
    }
}
