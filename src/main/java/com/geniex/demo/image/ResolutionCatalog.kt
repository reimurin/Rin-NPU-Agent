package com.geniex.demo.image
import java.io.File
internal object ResolutionCatalog {
    val contextNames=listOf("unet_encoder_fp16.serialized.bin.bin","unet_decoder_fp16.serialized.bin.bin","vae_decoder.serialized.bin.bin")
    fun parse(name:String):ImageResolution? {
        val match=Regex("^(\\d+)x(\\d+)$").matchEntire(name) ?: return null
        val width=match.groupValues[1].toIntOrNull() ?: return null
        val height=match.groupValues[2].toIntOrNull() ?: return null
        if(width !in 64..8192 || height !in 64..8192 || width%8!=0 || height%8!=0)return null
        return ImageResolution(width,height)
    }
    fun complete(directory:File):Boolean=contextNames.all { File(directory,it).let { f->f.isFile&&f.length()>0 } }
    fun discover(base:File):List<ImageResolution> {
        val root=File(base,"context");if(!root.isDirectory)return emptyList()
        val result=mutableSetOf<ImageResolution>()
        root.listFiles().orEmpty().filter{it.isDirectory}.forEach {dir->parse(dir.name)?.let{if(complete(dir))result+=it}}
        if(complete(root))result+=ImageResolution(1024,1024)
        return result.sortedWith(compareBy<ImageResolution>{it.width.toLong()*it.height}.thenBy{it.width})
    }
}
