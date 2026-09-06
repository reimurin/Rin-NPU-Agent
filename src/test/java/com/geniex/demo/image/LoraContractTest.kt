package com.geniex.demo.image
import org.junit.Test
import org.junit.Assert.*
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.json.JSONObject
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class LoraContractTest {
    @get:Rule val temp = TemporaryFolder()
    @Test fun directWeight() { assertEquals(0.8, LoraTags.parse("<lora:角色:0.8>").selections.single().weight, 0.0) }
    @Test fun twoIndependentWeights() { assertEquals(listOf(0.8,1.1), LoraTags.parse("<lora:a:.8>, <lora:b:1.1>").selections.map { it.weight }) }
    @Test fun defaultWeight() { assertEquals(1.0,LoraTags.parse("<lora:a>").selections.single().weight,0.0) }
    @Test fun zeroWeight() { assertEquals(0.0,LoraTags.parse("<lora:a:0>").selections.single().weight,0.0) }
    @Test fun negativeWeight() { assertEquals(-0.5,LoraTags.parse("<lora:a:-.5>").selections.single().weight,0.0) }
    @Test fun stripControl() { assertEquals("red hair",LoraTags.parse("<lora:a:.8>red hair").text) }
    @Test fun deduplicate() { assertEquals(1,LoraTags.parse("<lora:a:.8><lora:a:0.8>").selections.size) }
    @Test fun filenameAlias() { assertEquals("我的 角色",LoraTags.parse("<lora:我的 角色.safetensors:.8>").selections.single().name) }
    @Test fun invalidTags() {
        for (s in listOf("<lora:a:nan>","<lora:a:3>","<lora:../a:.8>","<lora:a:.8", "<lora:a:0.8><lora:a:1.1>","<lora::1>")) {
            try { LoraTags.parse(s); fail(s) } catch (_: IllegalArgumentException) {}
        }
    }
    @Test fun baseTextUnchanged() { assertEquals("(red hair:1.1), 16:9",LoraTags.parse("(red hair:1.1), 16:9").text) }
    private fun adapter(base: File, family: String = "sdxl_base_v1-0", incomplete: Boolean = false): File {
        val header=JSONObject().put("__metadata__",JSONObject().put("ss_base_model_version",family))
        header.put("m.lora_down.weight",JSONObject().put("dtype","F32").put("shape",JSONArray(listOf(2,4))).put("data_offsets",JSONArray(listOf(0,32))))
        header.put("m.lora_up.weight",JSONObject().put("dtype","F32").put("shape",JSONArray(listOf(4,2))).put("data_offsets",JSONArray(listOf(32,64))))
        val json=header.toString().toByteArray(Charsets.UTF_8);val p=File(LoraCatalog.directory(base),"角色.safetensors")
        p.outputStream().use { it.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(json.size.toLong()).array());it.write(json);it.write(ByteArray(if(incomplete) 60 else 64)) }
        return p
    }
    @Test fun folderUnderRuntime() { val base=temp.newFolder("sdxl_qnn"); assertEquals(File(base,"Lora").canonicalPath,LoraCatalog.directory(base).path) }
    @Test fun emptyCatalog() { assertTrue(LoraCatalog.scan(temp.newFolder()).isEmpty()) }
    @Test fun recognizeCandidate() { val b=temp.newFolder();adapter(b);val result=LoraCatalog.scan(b).single();assertTrue(result.readable);assertTrue(result.detail.contains("等待 WAI")) }
    @Test fun rejectOtherBase() { val b=temp.newFolder();adapter(b,"sd_v1");assertFalse(LoraCatalog.scan(b).single().readable) }
    @Test fun rejectTruncatedPayload() { val b=temp.newFolder();adapter(b,incomplete=true);assertFalse(LoraCatalog.scan(b).single().readable) }
    @Test fun skipDownloadPart() { val b=temp.newFolder();File(LoraCatalog.directory(b),"x.safetensors.part").writeText("part");assertTrue(LoraCatalog.scan(b).isEmpty()) }
    @Test fun rejectCorruptHeader() { val b=temp.newFolder();File(LoraCatalog.directory(b),"bad.safetensors").writeText("not a valid file");assertFalse(LoraCatalog.scan(b).single().readable) }
    @Test fun numericCheckAcceptsReference() { assertTrue(LoraSelfTest.matches(floatArrayOf(1f,2f),floatArrayOf(1f,2f),0.002,0.02)) }
    @Test fun numericCheckRejectsWrongData() { assertFalse(LoraSelfTest.matches(floatArrayOf(4f),floatArrayOf(1f),0.002,0.02)) }
    @Test fun numericCheckRejectsNaN() { assertFalse(LoraSelfTest.matches(floatArrayOf(Float.NaN),floatArrayOf(1f),0.002,0.02)) }
    @Test fun numericCheckRejectsLength() { assertFalse(LoraSelfTest.matches(floatArrayOf(),floatArrayOf(),0.002,0.02)) }
}
