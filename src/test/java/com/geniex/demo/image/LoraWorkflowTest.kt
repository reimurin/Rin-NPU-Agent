package com.geniex.demo.image

import android.app.Activity
import android.app.Application
import android.content.Intent
import android.view.View
import android.widget.EditText
import org.json.JSONObject
import org.json.JSONArray
import org.junit.Test
import org.junit.Assert.*
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34],application=Application::class,qualifiers="w412dp-h892dp-port",shadows=[LoraStartupTest.DeniedStorage::class])
@LooperMode(LooperMode.Mode.PAUSED)
class LoraWorkflowTest {
    @get:Rule val temp=TemporaryFolder()
    @Test fun editorReceivesCurrentPromptAndReturnsWeightedTags() {
        val original="red hair, <lora:角色:0.8>"
        val intent=Intent(RuntimeEnvironment.getApplication(),LoraLabActivity::class.java).putExtra(LoraLabActivity.EXTRA_PROMPT,original)
        val c=Robolectric.buildActivity(LoraLabActivity::class.java,intent).setup().visible();val a=c.get()
        val input=a.window.decorView.findViewWithTag<EditText>("lora_prompt_editor")
        assertEquals(original,input.text.toString());input.setText("red hair, <lora:角色:1.1>")
        a.window.decorView.findViewWithTag<View>("lora_apply_to_prompt").performClick()
        assertEquals(Activity.RESULT_OK,shadowOf(a).resultCode)
        assertEquals("red hair, <lora:角色:1.1>",shadowOf(a).resultIntent.getStringExtra(LoraLabActivity.EXTRA_PROMPT))
        c.pause().stop().destroy()
    }
    @Test fun malformedTagsDoNotFinishEditor() {
        val c=Robolectric.buildActivity(LoraLabActivity::class.java).setup().visible();val a=c.get()
        a.window.decorView.findViewWithTag<EditText>("lora_prompt_editor").setText("<lora:x:nan>")
        a.window.decorView.findViewWithTag<View>("lora_apply_to_prompt").performClick()
        assertFalse(a.isFinishing);c.pause().stop().destroy()
    }
    @Test fun controllerAppliesAcceptedPrompt() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get()
        a.binding.etImagePrompt.setText("original")
        assertTrue(a.image.onActivityResult(ImageModeController.REQUEST_LORA_EDITOR,Activity.RESULT_OK,Intent().putExtra(LoraLabActivity.EXTRA_PROMPT,"new <lora:test:0.8>")))
        assertEquals("new <lora:test:0.8>",a.binding.etImagePrompt.text.toString());c.pause().stop().destroy()
    }
    @Test fun cancelAndMalformedReturnPreserveOriginalPrompt() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get();a.binding.etImagePrompt.setText("keep")
        a.image.onActivityResult(ImageModeController.REQUEST_LORA_EDITOR,Activity.RESULT_CANCELED,Intent().putExtra(LoraLabActivity.EXTRA_PROMPT,"other"))
        assertEquals("keep",a.binding.etImagePrompt.text.toString())
        a.image.onActivityResult(ImageModeController.REQUEST_LORA_EDITOR,Activity.RESULT_OK,Intent().putExtra(LoraLabActivity.EXTRA_PROMPT,"<lora:x:99>"))
        assertEquals("keep",a.binding.etImagePrompt.text.toString());c.pause().stop().destroy()
    }
    private fun index():JSONObject {
        val files=JSONArray();val prefix=LoraModelComponent.INDEX_URL.removeSuffix("component.json")
        val names=(0..2).map{"encoder_p$it.bin"}+(0..3).map{"decoder_p$it.bin"}+"lora_template.json"
        names.forEach {name->
            val part=name+".000";val record=JSONObject().put("name",part).put("url",prefix+part).put("bytes",16).put("sha256","0".repeat(64))
            files.put(JSONObject().put("name",name).put("bytes",16).put("sha256","0".repeat(64)).put("parts",JSONArray().put(record)))
        }
        return JSONObject().put("schema",1).put("model_id",LoraModelComponent.ID).put("soc",69).put("dsp",79).put("files",files)
    }
    @Test fun eightFileComponentDescriptorAccepted(){assertEquals(8,LoraModelComponent.parseIndex(index()).size)}
    @Test fun otherDeviceDescriptorRejected(){val value=index().put("soc",68);assertThrows(IllegalArgumentException::class.java){LoraModelComponent.parseIndex(value)}}
    @Test fun incompleteDescriptorRejected(){val value=index();value.getJSONArray("files").remove(0);assertThrows(IllegalArgumentException::class.java){LoraModelComponent.parseIndex(value)}}
    @Test fun externalDownloadUrlRejected(){val value=index();value.getJSONArray("files").getJSONObject(0).getJSONArray("parts").getJSONObject(0).put("url","https://example.com/file");assertThrows(IllegalArgumentException::class.java){LoraModelComponent.parseIndex(value)}}
    @Test fun invalidPartLengthRejected(){val value=index();value.getJSONArray("files").getJSONObject(0).getJSONArray("parts").getJSONObject(0).put("bytes",15);assertThrows(IllegalArgumentException::class.java){LoraModelComponent.parseIndex(value)}}
    @Test fun componentDirectoryDoesNotReplaceLegacyContexts(){val root=temp.newFolder();File(root,"context").mkdirs();val old=File(root,"context/legacy.bin");old.writeText("keep");assertFalse(LoraModelComponent.installed(root));assertEquals("keep",old.readText())}
    private fun sample(prefix:String,inf:Int,outf:Int,rank:Int=2):File {
        val root=temp.newFolder();val file=File(LoraCatalog.directory(root),"角色.safetensors");val a=rank*inf*4;val b=outf*rank*4
        val header=JSONObject().put("__metadata__",JSONObject().put("ss_base_model_version","sdxl_base_v1-0"))
            .put(prefix+".lora_down.weight",JSONObject().put("dtype","F32").put("shape",JSONArray(listOf(rank,inf))).put("data_offsets",JSONArray(listOf(0,a))))
            .put(prefix+".lora_up.weight",JSONObject().put("dtype","F32").put("shape",JSONArray(listOf(outf,rank))).put("data_offsets",JSONArray(listOf(a,a+b))))
        val bytes=header.toString().toByteArray(Charsets.UTF_8)
        file.outputStream().use{it.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(bytes.size.toLong()).array());it.write(bytes);it.write(ByteArray(a+b))}
        return root
    }
    private fun layer():JSONObject=RuntimeEnvironment.getApplication().assets.open("lora_compatibility.json").bufferedReader().use{JSONObject(it.readText()).getJSONArray("layers").getJSONObject(0)}
    @Test fun exactCompiledAliasIsRecognized() {
        val layer=layer();val root=sample(layer.getJSONArray("aliases").getString(0),layer.getInt("in_features"),layer.getInt("out_features"))
        val entry=LoraCatalog.scan(root,LoraCompatibility.load(RuntimeEnvironment.getApplication())).single();assertTrue(entry.compatible)
    }
    @Test fun unknownLayerIsNotMarkedCompatible() {
        val root=sample("lora_te1_unknown",8,8);val entry=LoraCatalog.scan(root,LoraCompatibility.load(RuntimeEnvironment.getApplication())).single();assertFalse(entry.compatible);assertTrue(entry.detail.contains("未覆盖"))
    }
    @Test fun wrongMatrixDimensionsRejected() {
        val layer=layer();val root=sample(layer.getJSONArray("aliases").getString(0),7,9)
        assertFalse(LoraCatalog.scan(root,LoraCompatibility.load(RuntimeEnvironment.getApplication())).single().compatible)
    }
    @Test fun separateDownloadNotificationIds() {
        fun id(clazz:Class<*>):Int=clazz.getDeclaredField("NOTIFICATION_ID").apply{isAccessible=true}.getInt(null)
        assertNotEquals(id(RuntimeDownloadForegroundService::class.java),id(LoraComponentForegroundService::class.java))
    }
}
