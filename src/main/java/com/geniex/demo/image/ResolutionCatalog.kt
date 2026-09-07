package com.geniex.demo.image

import org.json.JSONObject
import java.io.File

internal object ResolutionCatalog {
    val contextNames=listOf("unet_encoder_fp16.serialized.bin.bin","unet_decoder_fp16.serialized.bin.bin","vae_decoder.serialized.bin.bin")
    private val unifiedPattern=Regex("^wai-v170-sm8750-lora-r64-(\\d+)x(\\d+)-partitioned-v1$")
    private const val LEGACY_UNIFIED_1024="wai-v170-sm8750-lora-r64-1024-partitioned-v1"

    fun parse(name:String):ImageResolution? {
        val match=Regex("^(\\d+)x(\\d+)$").matchEntire(name) ?: return null
        val width=match.groupValues[1].toIntOrNull() ?: return null
        val height=match.groupValues[2].toIntOrNull() ?: return null
        if(width !in 64..8192 || height !in 64..8192 || width%8!=0 || height%8!=0)return null
        return ImageResolution(width,height)
    }

    fun unifiedModelId(resolution:ImageResolution):String =
        if(resolution==ImageResolution(1024,1024)) LEGACY_UNIFIED_1024
        else "wai-v170-sm8750-lora-r64-${resolution.width}x${resolution.height}-partitioned-v1"

    fun complete(directory:File):Boolean=contextNames.all { File(directory,it).let { f->f.isFile&&f.length()>0 } }

    private fun vaeReady(base:File,resolution:ImageResolution):Boolean {
        val scoped=File(base,"context/${resolution.key}/vae_decoder.serialized.bin.bin")
        if(scoped.isFile&&scoped.length()>0)return true
        return resolution==ImageResolution(1024,1024) && File(base,"context/vae_decoder.serialized.bin.bin").let{it.isFile&&it.length()>0}
    }

    private fun unifiedResolution(name:String):ImageResolution? {
        if(name==LEGACY_UNIFIED_1024)return ImageResolution(1024,1024)
        val match=unifiedPattern.matchEntire(name) ?: return null
        return parse("${match.groupValues[1]}x${match.groupValues[2]}")
    }

    private fun unifiedComplete(base:File,directory:File,resolution:ImageResolution):Boolean=runCatching {
        val manifestFile=File(directory,"lora_template.json")
        if(!manifestFile.isFile||manifestFile.length() !in 2..8_388_608)return@runCatching false
        val root=JSONObject(manifestFile.readText(Charsets.UTF_8))
        if(root.optInt("schema")!=1||!root.optBoolean("complete",false)||root.optString("model_id")!=unifiedModelId(resolution))return@runCatching false
        val r=root.optJSONArray("resolution") ?: return@runCatching false
        if(r.length()!=2||r.optInt(0)!=resolution.width||r.optInt(1)!=resolution.height||root.optInt("rank_capacity")!=64)return@runCatching false
        val graphs=root.optJSONObject("graphs") ?: return@runCatching false
        for(stage in listOf("encoder","decoder")) {
            val spec=graphs.optJSONObject(stage) ?: return@runCatching false
            val parts=spec.optJSONArray("parts") ?: return@runCatching false
            if(parts.length() !in 1..8)return@runCatching false
            for(i in 0 until parts.length()) {
                val item=parts.optJSONObject(i) ?: return@runCatching false
                val id=item.optString("id")
                if(id!="${stage}_p$i")return@runCatching false
                val fileName=item.optString("context_file")
                if(fileName!="$id.bin")return@runCatching false
                val expected=item.optLong("context_bytes",-1L)
                val file=File(directory,fileName)
                if(expected<=0L||!file.isFile||file.length()!=expected)return@runCatching false
            }
        }
        vaeReady(base,resolution)
    }.getOrDefault(false)

    fun discover(base:File):List<ImageResolution> {
        val root=File(base,"context");if(!root.isDirectory)return emptyList()
        val result=mutableSetOf<ImageResolution>()
        root.listFiles().orEmpty().filter{it.isDirectory}.forEach {dir->parse(dir.name)?.let{if(complete(dir))result+=it}}
        if(complete(root))result+=ImageResolution(1024,1024)
        val unifiedRoot=File(root,"lora")
        unifiedRoot.listFiles().orEmpty().filter{it.isDirectory}.forEach {dir->
            unifiedResolution(dir.name)?.let {resolution->if(unifiedComplete(base,dir,resolution))result+=resolution}
        }
        return result.sortedWith(compareBy<ImageResolution>{it.width.toLong()*it.height}.thenBy{it.width})
    }
}
