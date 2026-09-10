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


    private fun sharedPack(base:File,name:String,resolutions:List<ImageResolution>,canonical:Boolean=false):File {
        val folder=File(base,"context/model_packs/$name").apply{mkdirs()}
        val contextDir=File(folder,"contexts").apply{mkdirs()}
        val contexts=JSONObject()
        for(id in listOf("encoder_p0","encoder_p1","encoder_p2","decoder_p0","decoder_p1","decoder_p2","decoder_p3")) {
            File(contextDir,"$id.bin").writeBytes(byteArrayOf(1))
            contexts.put(id,JSONObject().put("file","contexts/$id.bin").put("bytes",1))
        }
        File(contextDir,"vae.bin").writeBytes(byteArrayOf(1))
        val array=JSONArray()
        for(r in resolutions) {
            val template=File(folder,"templates/${r.key}.json");template.parentFile.mkdirs();template.writeText("{}")
            array.put(JSONObject().put("width",r.width).put("height",r.height).put("graph","_${r.key}").put("template","templates/${r.key}.json"))
        }
        val manifest=JSONObject()
            .put("schema",1).put("complete",true).put("runtime_abi",1).put("id",name).put("version","1.0.0")
            .put("target",JSONObject().put("qnn_soc_id",69).put("dsp_arch",79))
            .put("lora",JSONObject().put("external_dynamic_ab",true).put("rank_capacity",64).put("abi_signature","e362"))
            .put("contexts",contexts).put("vae",JSONObject().put("file","contexts/vae.bin").put("bytes",1)).put(if(canonical) "supported_resolutions" else "resolutions",array)
        File(folder,"model_manifest.json").writeText(manifest.toString())
        return folder
    }

    private fun sharedPackV2(base:File,name:String,resolutions:List<ImageResolution>,missingVae:ImageResolution?=null):File {
        val folder=File(base,"context/model_packs/$name").apply{mkdirs()}
        val contextDir=File(folder,"contexts").apply{mkdirs()}
        val contexts=JSONObject()
        for(id in listOf("encoder_p0","encoder_p1","encoder_p2","decoder_p0","decoder_p1","decoder_p2","decoder_p3")) { File(contextDir,"$id.bin").writeBytes(byteArrayOf(1)); contexts.put(id,JSONObject().put("file","contexts/$id.bin").put("bytes",1)) }
        val array=JSONArray()
        for(r in resolutions) {
            val template=File(folder,"templates/${r.key}.json");template.parentFile.mkdirs();template.writeText("{}")
            val vaePath="vae/vae_${r.key}.bin"
            if(r!=missingVae) File(folder,vaePath).apply{parentFile?.mkdirs();writeBytes(byteArrayOf(1))}
            array.put(JSONObject().put("width",r.width).put("height",r.height).put("graph","_${r.key}").put("template","templates/${r.key}.json").put("vae",JSONObject().put("file",vaePath).put("bytes",1).put("sha256","0".repeat(64))).put("vae_graph",""))
        }
        val manifest=JSONObject().put("schema",2).put("complete",true).put("runtime_abi",1).put("id",name).put("version","1.0.1").put("target",JSONObject().put("qnn_soc_id",69).put("dsp_arch",79)).put("lora",JSONObject().put("external_dynamic_ab",true).put("rank_capacity",64).put("abi_signature","e362")).put("contexts",contexts).put("supported_resolutions",array)
        File(folder,"model_manifest.json").writeText(manifest.toString());return folder
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
    @Test fun currentSharedPackDrivesDynamicResolutionChoices(){
        val r=temp.newFolder()
        val old=listOf(ImageResolution(1024,1024),ImageResolution(832,1216),ImageResolution(1216,832),ImageResolution(768,1344),ImageResolution(1344,768))
        val current=listOf(ImageResolution(1024,1024),ImageResolution(832,1216),ImageResolution(1216,832))
        sharedPack(r,"old-five",old);sharedPack(r,"current-three",current)
        val packs=File(r,"context/model_packs")
        File(packs,"current.json").writeText(JSONObject().put("directory","current-three").toString())
        assertEquals(current.toSet(),ResolutionCatalog.discover(r).toSet())
        sharedPack(r,"new-five",old)
        File(packs,"current.json").writeText(JSONObject().put("directory","new-five").toString())
        assertEquals(old.toSet(),ResolutionCatalog.discover(r).toSet())
    }
    @Test fun retainedSharedPackWithoutCurrentPointerIsNotActivated(){
        val r=temp.newFolder();sharedPack(r,"retained-only",listOf(ImageResolution(832,1216)))
        assertTrue(ResolutionCatalog.discover(r).isEmpty())
    }
    @Test fun canonicalSupportedResolutionsDriveSharedPack(){
        val r=temp.newFolder();val current=listOf(ImageResolution(1024,1024),ImageResolution(832,1216),ImageResolution(1216,832))
        sharedPack(r,"canonical-three",current,canonical=true)
        File(r,"context/model_packs/current.json").writeText(JSONObject().put("directory","canonical-three").toString())
        assertEquals(current.toSet(),ResolutionCatalog.discover(r).toSet())
    }
    @Test fun brokenCurrentPackNeverLeaksRetainedOldResolutions(){
        val r=temp.newFolder();val old=ImageResolution(1216,832);val current=ImageResolution(832,1216)
        sharedPack(r,"retained-old",listOf(old));val active=sharedPack(r,"active",listOf(current),canonical=true)
        File(r,"context/model_packs/current.json").writeText(JSONObject().put("directory","active").toString())
        File(active,"contexts/encoder_p0.bin").writeBytes(byteArrayOf())
        assertTrue(ResolutionCatalog.discover(r).isEmpty())
    }
    @Test fun schema2UsesPerResolutionVae(){val r=temp.newFolder();val all=listOf(ImageResolution(1024,1024),ImageResolution(832,1216),ImageResolution(1216,832));sharedPackV2(r,"v101",all);File(r,"context/model_packs/current.json").writeText(JSONObject().put("directory","v101").toString());assertEquals(all.toSet(),ResolutionCatalog.discover(r).toSet())}
    @Test fun schema2MissingOneVaeOnlyRemovesThatResolution(){val r=temp.newFolder();val a=ImageResolution(1024,1024);val b=ImageResolution(832,1216);val c=ImageResolution(1216,832);sharedPackV2(r,"v101-partial",listOf(a,b,c),missingVae=b);File(r,"context/model_packs/current.json").writeText(JSONObject().put("directory","v101-partial").toString());assertEquals(setOf(a,c),ResolutionCatalog.discover(r).toSet())}

}
