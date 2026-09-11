package com.geniex.demo.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import java.io.File
import java.util.Locale

internal data class LoraHistoryEntry(
    val name: String,
    val sha256: String,
    val weight: Double,
    val lastUsedMs: Long,
    val seed: Long,
    val loraCount: Int,
    val mappedModules: Int,
    val ignoredTextEncoderModules: Int,
    val thumbnail: File,
)

internal object LoraHistoryStore {
    private const val DIRECTORY = "lora_history"
    private const val MAX_ITEMS = 300
    private const val THUMB_MAX_EDGE = 420

    fun parseEvidence(json: String): List<JSONObject> {
        val root = JSONObject(json)
        if (!root.optBoolean("active", false)) return emptyList()
        val files = root.optJSONArray("files") ?: return emptyList()
        return (0 until files.length()).mapNotNull { index ->
            files.optJSONObject(index)?.takeIf { item ->
                item.optString("name").isNotBlank() && item.optString("sha256").matches(Regex("[0-9a-fA-F]{64}"))
            }
        }
    }

    fun index(context: Context): Map<String, LoraHistoryEntry> {
        val dir = File(context.filesDir, DIRECTORY)
        if (!dir.isDirectory) return emptyMap()
        val result = linkedMapOf<String, LoraHistoryEntry>()
        dir.listFiles().orEmpty()
            .filter { it.isFile && it.name.endsWith(".json", true) }
            .sortedByDescending { it.lastModified() }
            .take(MAX_ITEMS)
            .forEach { file ->
                runCatching { decode(file, dir) }.getOrNull()?.let { entry ->
                    if (result[entry.name] == null) result[entry.name] = entry
                }
            }
        return result
    }

    fun recordFromDiagnostics(context: Context, base: File, image: File, seed: Long): List<LoraHistoryEntry> {
        if (!image.isFile || image.length() <= 0L) return emptyList()
        val diagnostic = File(base, ".rin_diagnostics/lora_generation_latest.json")
        if (!diagnostic.isFile || diagnostic.length() !in 2..262_144) return emptyList()
        val evidence = parseEvidence(diagnostic.readText(Charsets.UTF_8))
        if (evidence.isEmpty()) return emptyList()

        val dir = File(context.filesDir, DIRECTORY).apply { mkdirs() }
        val thumbBitmap = createThumbnail(image) ?: return emptyList()
        val now = System.currentTimeMillis()
        val total = evidence.size
        val results = mutableListOf<LoraHistoryEntry>()
        try {
            evidence.forEach { item ->
                val sha = item.getString("sha256").lowercase(Locale.ROOT)
                val name = item.getString("name")
                val thumb = File(dir, "$sha.jpg")
                val thumbPending = File(dir, "$sha.jpg.pending")
                thumbPending.outputStream().use { out ->
                    check(thumbBitmap.compress(Bitmap.CompressFormat.JPEG, 86, out)) { "LoRA thumbnail compression failed" }
                }
                if (thumb.exists()) thumb.delete()
                check(thumbPending.renameTo(thumb)) { "Could not publish LoRA thumbnail" }

                val meta = JSONObject()
                    .put("schema", 1)
                    .put("name", name)
                    .put("sha256", sha)
                    .put("weight", item.optDouble("weight", 0.0))
                    .put("last_used_ms", now)
                    .put("seed", seed)
                    .put("lora_count", total)
                    .put("mapped_modules", item.optInt("modules", item.optInt("mapped_modules", 0)))
                    .put("ignored_text_encoder_modules", item.optInt("ignored_text_encoder_modules", 0))
                    .put("thumbnail", thumb.name)
                val metaFile = File(dir, "$sha.json")
                val pending = File(dir, "$sha.json.pending")
                pending.writeText(meta.toString(), Charsets.UTF_8)
                if (metaFile.exists()) metaFile.delete()
                check(pending.renameTo(metaFile)) { "Could not publish LoRA history metadata" }
                results += decode(metaFile, dir)
            }
        } finally {
            thumbBitmap.recycle()
        }
        return results
    }

    private fun decode(file: File, dir: File): LoraHistoryEntry {
        require(file.length() in 2..65_536) { "Invalid LoRA history metadata" }
        val root = JSONObject(file.readText(Charsets.UTF_8))
        require(root.optInt("schema") == 1) { "Unknown LoRA history schema" }
        val sha = root.getString("sha256").lowercase(Locale.ROOT)
        require(sha.matches(Regex("[0-9a-f]{64}"))) { "Invalid LoRA history SHA" }
        val name = root.getString("name")
        LoraTags.format(name)
        val thumbName = root.getString("thumbnail")
        require(thumbName == "$sha.jpg") { "Invalid LoRA history thumbnail" }
        val thumb = File(dir, thumbName).canonicalFile
        require(thumb.parentFile == dir.canonicalFile && thumb.isFile && thumb.length() > 0L) { "Missing LoRA history thumbnail" }
        return LoraHistoryEntry(
            name = name,
            sha256 = sha,
            weight = root.optDouble("weight", 0.0),
            lastUsedMs = root.optLong("last_used_ms", 0L),
            seed = root.optLong("seed", -1L),
            loraCount = root.optInt("lora_count", 1).coerceAtLeast(1),
            mappedModules = root.optInt("mapped_modules", 0).coerceAtLeast(0),
            ignoredTextEncoderModules = root.optInt("ignored_text_encoder_modules", 0).coerceAtLeast(0),
            thumbnail = thumb,
        )
    }

    private fun createThumbnail(source: File): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (bounds.outWidth / sample > THUMB_MAX_EDGE * 2 || bounds.outHeight / sample > THUMB_MAX_EDGE * 2) sample *= 2
        val decoded = BitmapFactory.decodeFile(source.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
        val scale = minOf(1f, THUMB_MAX_EDGE.toFloat() / maxOf(decoded.width, decoded.height).toFloat())
        if (scale >= 0.999f) return decoded
        val scaled = Bitmap.createScaledBitmap(decoded, (decoded.width * scale).toInt().coerceAtLeast(1), (decoded.height * scale).toInt().coerceAtLeast(1), true)
        if (scaled !== decoded) decoded.recycle()
        return scaled
    }
}
