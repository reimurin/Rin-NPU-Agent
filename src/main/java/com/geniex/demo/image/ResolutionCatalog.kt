package com.geniex.demo.image

import org.json.JSONObject
import java.io.File

internal object ResolutionCatalog {
    val contextNames = listOf(
        "unet_encoder_fp16.serialized.bin.bin",
        "unet_decoder_fp16.serialized.bin.bin",
        "vae_decoder.serialized.bin.bin",
    )

    private val unifiedPattern = Regex("^wai-v170-sm8750-lora-r64-(\\d+)x(\\d+)-partitioned-v1$")
    private const val LEGACY_UNIFIED_1024 = "wai-v170-sm8750-lora-r64-1024-partitioned-v1"

    fun parse(name: String): ImageResolution? {
        val match = Regex("^(\\d+)x(\\d+)$").matchEntire(name) ?: return null
        val width = match.groupValues[1].toIntOrNull() ?: return null
        val height = match.groupValues[2].toIntOrNull() ?: return null
        if (width !in 64..8192 || height !in 64..8192 || width % 8 != 0 || height % 8 != 0) return null
        return ImageResolution(width, height)
    }

    fun unifiedModelId(resolution: ImageResolution): String =
        if (resolution == ImageResolution(1024, 1024)) LEGACY_UNIFIED_1024
        else "wai-v170-sm8750-lora-r64-${resolution.width}x${resolution.height}-partitioned-v1"

    fun complete(directory: File): Boolean = contextNames.all { name ->
        File(directory, name).let { it.isFile && it.length() > 0L }
    }

    private fun vaeReady(base: File, resolution: ImageResolution): Boolean {
        val scoped = File(base, "context/${resolution.key}/vae_decoder.serialized.bin.bin")
        if (scoped.isFile && scoped.length() > 0L) return true
        return resolution == ImageResolution(1024, 1024) &&
            File(base, "context/vae_decoder.serialized.bin.bin").let { it.isFile && it.length() > 0L }
    }

    private fun unifiedResolution(name: String): ImageResolution? {
        if (name == LEGACY_UNIFIED_1024) return ImageResolution(1024, 1024)
        val match = unifiedPattern.matchEntire(name) ?: return null
        return parse("${match.groupValues[1]}x${match.groupValues[2]}")
    }

    private fun unifiedComplete(base: File, directory: File, resolution: ImageResolution): Boolean = runCatching {
        val manifestFile = File(directory, "lora_template.json")
        if (!manifestFile.isFile || manifestFile.length() !in 2..8_388_608) return@runCatching false
        val root = JSONObject(manifestFile.readText(Charsets.UTF_8))
        if (root.optInt("schema") != 1 || !root.optBoolean("complete", false) ||
            root.optString("model_id") != unifiedModelId(resolution)
        ) return@runCatching false
        val r = root.optJSONArray("resolution") ?: return@runCatching false
        if (r.length() != 2 || r.optInt(0) != resolution.width || r.optInt(1) != resolution.height ||
            root.optInt("rank_capacity") != 64
        ) return@runCatching false
        val graphs = root.optJSONObject("graphs") ?: return@runCatching false
        for (stage in listOf("encoder", "decoder")) {
            val spec = graphs.optJSONObject(stage) ?: return@runCatching false
            val parts = spec.optJSONArray("parts") ?: return@runCatching false
            if (parts.length() !in 1..8) return@runCatching false
            for (i in 0 until parts.length()) {
                val item = parts.optJSONObject(i) ?: return@runCatching false
                val id = item.optString("id")
                if (id != "${stage}_p$i") return@runCatching false
                val fileName = item.optString("context_file")
                if (fileName != "$id.bin") return@runCatching false
                val expected = item.optLong("context_bytes", -1L)
                val file = File(directory, fileName)
                if (expected <= 0L || !file.isFile || file.length() != expected) return@runCatching false
            }
        }
        vaeReady(base, resolution)
    }.getOrDefault(false)

    private fun safeInside(root: File, relative: String): File? = runCatching {
        if (relative.isBlank() || relative.startsWith('/') || relative.startsWith('\\')) return@runCatching null
        val base = root.canonicalFile
        val file = File(root, relative).canonicalFile
        file.takeIf { it.path == base.path || it.path.startsWith(base.path + File.separator) }
    }.getOrNull()

    private fun sharedFileComplete(folder: File, item: JSONObject?): Boolean {
        if (item == null) return false
        val relative = item.optString("file")
        val expected = item.optLong("bytes", 0L)
        val file = safeInside(folder, relative) ?: return false
        return file.isFile && file.length() > 0L && (expected <= 0L || file.length() == expected)
    }

    private fun scanSharedPack(folder: File): List<ImageResolution> = runCatching {
        val manifest = File(folder, "model_manifest.json")
        if (!manifest.isFile || manifest.length() !in 2..2_097_152) return@runCatching emptyList()
        val root = JSONObject(manifest.readText(Charsets.UTF_8))
        if (root.optInt("schema") != 1 || !root.optBoolean("complete", false) || root.optInt("runtime_abi") != 1) {
            return@runCatching emptyList()
        }
        val target = root.optJSONObject("target") ?: return@runCatching emptyList()
        if (target.optInt("qnn_soc_id") != 69 || target.optInt("dsp_arch") != 79) return@runCatching emptyList()

        val lora = root.optJSONObject("lora") ?: return@runCatching emptyList()
        if (!lora.optBoolean("external_dynamic_ab", false) || lora.optInt("rank_capacity") != 64) {
            return@runCatching emptyList()
        }

        val contexts = root.optJSONObject("contexts") ?: return@runCatching emptyList()
        val keys = listOf("encoder_p0", "encoder_p1", "encoder_p2", "decoder_p0", "decoder_p1", "decoder_p2", "decoder_p3")
        if (keys.any { !sharedFileComplete(folder, contexts.optJSONObject(it)) } || !sharedFileComplete(folder, root.optJSONObject("vae"))) {
            return@runCatching emptyList()
        }

        val array = root.optJSONArray("supported_resolutions") ?: root.optJSONArray("resolutions")
            ?: return@runCatching emptyList()
        buildList {
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                val resolution = ImageResolution(item.optInt("width"), item.optInt("height"))
                val graph = item.optString("graph")
                val template = safeInside(folder, item.optString("template"))
                if (resolution.width in 64..8192 && resolution.height in 64..8192 &&
                    resolution.width % 8 == 0 && resolution.height % 8 == 0 &&
                    graph.matches(Regex("^_[0-9]+x[0-9]+$")) && template?.isFile == true
                ) add(resolution)
            }
        }
    }.getOrDefault(emptyList())

    private fun sharedPackResolutions(base: File): List<ImageResolution> {
        val packs = File(base, "context/model_packs")
        if (!packs.isDirectory) return emptyList()

        val currentFolder = runCatching {
            val pointer = File(packs, "current.json")
            if (!pointer.isFile || pointer.length() !in 2..65_536) return@runCatching null
            val directory = JSONObject(pointer.readText(Charsets.UTF_8)).optString("directory")
            safeInside(packs, directory)?.takeIf { it.isDirectory }
        }.getOrNull()

        return currentFolder?.let(::scanSharedPack) ?: emptyList()
    }

    fun discover(base: File): List<ImageResolution> {
        val root = File(base, "context")
        if (!root.isDirectory) return emptyList()
        val result = mutableSetOf<ImageResolution>()
        result += sharedPackResolutions(base)
        root.listFiles().orEmpty().filter { it.isDirectory }.forEach { dir ->
            parse(dir.name)?.let { if (complete(dir)) result += it }
        }
        if (complete(root)) result += ImageResolution(1024, 1024)
        val unifiedRoot = File(root, "lora")
        unifiedRoot.listFiles().orEmpty().filter { it.isDirectory }.forEach { dir ->
            unifiedResolution(dir.name)?.let { resolution ->
                if (unifiedComplete(base, dir, resolution)) result += resolution
            }
        }
        return result.sortedWith(compareBy<ImageResolution> { it.width.toLong() * it.height }.thenBy { it.width })
    }
}
