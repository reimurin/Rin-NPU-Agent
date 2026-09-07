package com.geniex.demo.image

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal data class LoraEntry(val name: String, val bytes: Long, val readable: Boolean, val detail: String, val compatible:Boolean=false)
internal object LoraCatalog {
    fun directory(base: File): File {
        val root = base.canonicalFile
        val folder = File(root, "Lora").canonicalFile
        require(folder.path.startsWith(root.path + File.separator)) { "LoRA 目录超出运行时范围" }
        require(folder.isDirectory || folder.mkdirs()) { "无法创建 LoRA 目录" }
        return folder
    }
    fun scan(base: File, compatibility:LoraCompatibility?=null): List<LoraEntry> {
        val folder = directory(base)
        return folder.listFiles().orEmpty().filter { it.name.endsWith(".safetensors", true) }
            .sortedBy { it.name.lowercase() }.take(300).map { file ->
                runCatching { inspect(file, folder, compatibility) }.getOrElse {
                    LoraEntry(file.nameWithoutExtension, file.length(), false, it.message ?: "无法读取")
                }
            }
    }
    private fun inspect(file: File, folder: File, compatibility:LoraCompatibility?): LoraEntry {
        require(file.isFile && file.canonicalFile.parentFile == folder.canonicalFile) { "不是目录内的普通文件" }
        LoraTags.format(file.nameWithoutExtension)
        val bytes = file.length(); val modified = file.lastModified()
        val root = RandomAccessFile(file, "r").use { input ->
            require(bytes >= 10) { "文件未下载完整" }
            val prefix = ByteArray(8); input.readFully(prefix)
            val length = ByteBuffer.wrap(prefix).order(ByteOrder.LITTLE_ENDIAN).long
            require(length in 2..8_388_608 && length <= bytes - 8) { "Safetensors 文件头无效" }
            val json = ByteArray(length.toInt()); input.readFully(json)
            Pair(JSONObject(String(json, Charsets.UTF_8)), bytes - length - 8)
        }
        val header = root.first; val dataBytes = root.second
        require(header.length() <= 60_000) { "张量数量超出测试范围" }
        val meta = header.optJSONObject("__metadata__")
        val family = meta?.optString("ss_base_model_version").orEmpty()
        require(family.isBlank() || family.contains("sdxl", true) || family.contains("illustrious", true)) { "不是声明为 SDXL 的 LoRA：$family" }
        val widths = mapOf("F16" to 2L, "BF16" to 2L, "F32" to 4L)
        val records = linkedMapOf<String, Pair<List<Long>, String>>()
        val offsets = mutableListOf<Pair<Long, Long>>()
        header.keys().forEach { name ->
            if (name != "__metadata__") {
                val t = header.getJSONObject(name); val dtype = t.getString("dtype")
                val size = widths[dtype] ?: error("暂不支持权重类型：$dtype")
                val dimensions = t.getJSONArray("shape")
                require(dimensions.length() <= 8) { "张量维度异常" }
                val shape = (0 until dimensions.length()).map { dimensions.getLong(it) }
                require(shape.all { it > 0 }) { "空张量或非法维度" }
                val expected = shape.fold(size) { acc, value -> Math.multiplyExact(acc, value) }
                val range = t.getJSONArray("data_offsets"); require(range.length() == 2)
                val a = range.getLong(0); val b = range.getLong(1)
                require(a >= 0 && b > a && b <= dataBytes && b - a == expected) { "文件未完整下载或张量长度错误" }
                offsets += a to b; records[name] = shape to dtype
            }
        }
        var end = 0L
        offsets.sortedBy { it.first }.forEach { (a, b) -> require(a == end) { "张量数据重叠或缺失" }; end = b }
        require(records.isNotEmpty() && end == dataBytes) { "权重载荷长度错误" }
        val used = mutableSetOf<String>(); val mapped=mutableSetOf<String>(); var pairs = 0; var maxRank = 0L; var textEncoderPairs=0
        records.forEach { (name, value) ->
            val suffixes=mapOf(".lora_down.weight" to ".lora_up.weight",".lora_A.weight" to ".lora_B.weight",".lora_A.default.weight" to ".lora_B.default.weight",".lora.down.weight" to ".lora.up.weight")
            val downSuffix=suffixes.keys.firstOrNull { name.endsWith(it) }
            if (downSuffix != null) {
                val prefix = name.removeSuffix(downSuffix)
                val upName = prefix + suffixes.getValue(downSuffix)
                val up = records[upName]?.first ?: error("缺少配对权重：$upName")
                val down = value.first
                require(down.size in listOf(2, 4) && down.size == up.size && down[0] == up[1]) { "LoRA 秩或矩阵形状不匹配" }
                require(down[0] <= 256) { "LoRA 秩超出当前测试范围" }
                if (down.size == 4) require(down.drop(2) == listOf(1L, 1L) && up.drop(2) == listOf(1L, 1L)) { "空间卷积 LoRA 尚待适配" }
                if(compatibility!=null) {
                    val mappedName=compatibility.check(prefix,down,up)
                    if(mappedName==null) textEncoderPairs++ else require(mapped.add(mappedName)) { "同一 LoRA 重复映射到一层" }
                }
                val alpha = prefix + ".alpha"
                records[alpha]?.let { require(it.first.fold(1L) { acc, x -> Math.multiplyExact(acc, x) } == 1L); used += alpha }
                used += name; used += upName; pairs++; maxRank = maxOf(maxRank, down[0])
            }
        }
        require(pairs > 0 && records.keys.all { it in used }) { "存在尚不支持的 LoRA 类型或额外权重" }
        if(compatibility!=null) require(mapped.isNotEmpty()) { "没有可注入的 UNet LoRA 权重" }
        require(bytes == file.length() && modified == file.lastModified()) { "文件仍在变化，请稍后刷新" }
        val detail="结构已识别 · $pairs 组 / 最大秩 $maxRank\n" + if(compatibility!=null) {
            "UNet ${mapped.size} 组可注入" + if(textEncoderPairs>0) "；文本编码器 $textEncoderPairs 组暂未注入（已显式标记）" else "；无额外文本编码器权重"
        } else "等待 WAI 可更新底模适配，尚未标记为可生成"
        return LoraEntry(file.nameWithoutExtension, bytes, true, detail, compatibility!=null&&mapped.isNotEmpty())
    }
}
