package com.geniex.demo.image

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ResolutionCatalogTest {
    @get:Rule val temp=TemporaryFolder()
    private fun complete(dir:File){dir.mkdirs();ResolutionCatalog.contextNames.forEach{File(dir,it).writeBytes(byteArrayOf(1))}}

    private fun unified(base:File,resolution:ImageResolution,withVae:Boolean=true):File {
        val dir=File(base,"context/lora/${ResolutionCatalog.unifiedModelId(resolution)}").apply{mkdirs()}
        val graphs=JSONObject()
        for((stage,count) in listOf("encoder" to 3,"decoder" to 4)) {
            val parts=JSONArray()
            for(i in 0 until count) {
                val id="${stage}_p$i";File(dir,"$id.bin").writeBytes(byteArrayOf(1))
                parts.put(JSONObject().put("id",id).put("context_file","$id.bin").put("context_bytes",1))
            }
            graphs.put(stage,JSONObject().put("parts",parts))
        }
        val manifest=JSONObject()
            .put("schema",1).put("complete",true).put("model_id",ResolutionCatalog.unifiedModelId(resolution))
            .put("resolution",JSONArray(listOf(resolution.width,resolution.height))).put("rank_capacity",64).put("graphs",graphs)
        File(dir,"lora_template.json").writeText(manifest.toString())
        if(withVae) File(base,"context/${resolution.key}/vae_decoder.serialized.bin.bin").apply{parentFile?.mkdirs();writeBytes(byteArrayOf(1))}
        return dir
    }

    @Test fun legacyIsOnly1024(){val r=temp.newFolder();complete(File(r,"context"));assertEquals(listOf(ImageResolution(1024,1024)),ResolutionCatalog.discover(r))}
    @Test fun scopedDimensionsDiscovered(){val r=temp.newFolder();complete(File(r,"context/832x1216"));assertEquals(listOf(ImageResolution(832,1216)),ResolutionCatalog.discover(r))}
    @Test fun incompleteBucketNotSupported(){val r=temp.newFolder();val d=File(r,"context/832x1216");complete(d);File(d,ResolutionCatalog.contextNames.first()).writeBytes(byteArrayOf());assertTrue(ResolutionCatalog.discover(r).isEmpty())}
    @Test fun duplicate1024Deduped(){val r=temp.newFolder();complete(File(r,"context"));complete(File(r,"context/1024x1024"));assertEquals(1,ResolutionCatalog.discover(r).size)}
    @Test fun malformedAndOversizedDimensionsRejected(){for(s in listOf("0x1024","1025x1024","999999999999999x1024","-1x1024","8193x8193","abc"))assertNull(ResolutionCatalog.parse(s))}
    @Test fun missingModelFolderIsEmpty(){assertTrue(ResolutionCatalog.discover(temp.newFolder()).isEmpty())}
    @Test fun noImplicitRectangleFromSquare(){val r=temp.newFolder();complete(File(r,"context"));assertFalse(ImageResolution(1216,832) in ResolutionCatalog.discover(r))}

    @Test fun unifiedPortraitRequiresMatchingVae(){val r=temp.newFolder();val res=ImageResolution(832,1216);unified(r,res,false);assertTrue(ResolutionCatalog.discover(r).isEmpty());File(r,"context/${res.key}/vae_decoder.serialized.bin.bin").apply{parentFile?.mkdirs();writeBytes(byteArrayOf(1))};assertEquals(listOf(res),ResolutionCatalog.discover(r))}
    @Test fun unifiedIncompletePartRejected(){val r=temp.newFolder();val res=ImageResolution(832,1216);val dir=unified(r,res,true);File(dir,"decoder_p3.bin").writeBytes(byteArrayOf());assertTrue(ResolutionCatalog.discover(r).isEmpty())}
    @Test fun unified1024CanUseLegacyVae(){val r=temp.newFolder();val res=ImageResolution(1024,1024);unified(r,res,false);File(r,"context/vae_decoder.serialized.bin.bin").apply{parentFile?.mkdirs();writeBytes(byteArrayOf(1))};assertEquals(listOf(res),ResolutionCatalog.discover(r))}
    @Test fun unifiedModelIdPreservesHistorical1024(){assertEquals("wai-v170-sm8750-lora-r64-1024-partitioned-v1",ResolutionCatalog.unifiedModelId(ImageResolution(1024,1024)));assertEquals("wai-v170-sm8750-lora-r64-832x1216-partitioned-v1",ResolutionCatalog.unifiedModelId(ImageResolution(832,1216)))}
}
