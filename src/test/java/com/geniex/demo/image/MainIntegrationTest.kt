package com.geniex.demo.image
import android.app.Application
import android.os.Bundle
import android.os.Looper
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.EditText
import android.widget.CheckBox
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AlertDialog
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import org.robolectric.shadows.ShadowDialog
import java.time.Duration
import java.io.File
import com.geniex.demo.BuildConfig
import com.geniex.demo.R
import com.geniex.demo.MyApplication
import com.geniex.demo.databinding.ActivityMainBinding
class MainIntegrationActivity:AppCompatActivity() {
    lateinit var binding:ActivityMainBinding
    lateinit var image:ImageModeController
    override fun onCreate(state:Bundle?) {
        setTheme(R.style.Theme_GenieXDemo);super.onCreate(state)
        binding=ActivityMainBinding.inflate(layoutInflater);setContentView(binding.root)
        image=ImageModeController(this,binding){};image.setup()
    }
}
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34],application=Application::class,qualifiers="w412dp-h892dp-port")
@LooperMode(LooperMode.Mode.PAUSED)
class MainIntegrationTest {
    private fun views(v:View):List<View> = listOf(v)+if(v is ViewGroup)(0 until v.childCount).flatMap{views(v.getChildAt(it))}else emptyList()
    private fun idle(){shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(150))}
    @Test fun originalMainLayoutAndLoraEntry() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get()
        val entry=views(a.binding.drawerImageSettingsGroup).filterIsInstance<TextView>().single{it.text.toString()=="LoRA 管理与 NPU 测试"}
        entry.performClick();val intent=shadowOf(a).nextStartedActivity
        assertEquals("com.geniex.demo.image.LoraLabActivity",intent.component?.className)
        assertEquals("com.geniex.demo",intent.component?.packageName)
        c.pause().stop().destroy()
    }
    @Test fun diagnosticsAccessibleWithoutOpeningLab() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get()
        views(a.binding.drawerImageSettingsGroup).filterIsInstance<TextView>().single{it.text.toString()=="分享启动诊断"}.performClick()
        assertEquals(Intent.ACTION_CHOOSER,shadowOf(a).nextStartedActivity.action)
        c.pause().stop().destroy()
    }
    @Test fun selectorDisplaysOnlyInstalledSizes() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get()
        val method=ImageModeController::class.java.getDeclaredMethod("updateResolutionChoices",List::class.java).apply{isAccessible=true}
        method.invoke(a.image,listOf(ImageResolution(1024,1024)));idle()
        assertEquals(1,a.binding.spImageResolution.adapter.count);assertTrue(a.binding.spImageResolution.selectedItem.toString().contains("1024"))
        method.invoke(a.image,listOf(ImageResolution(832,1216),ImageResolution(1024,1024)));idle();assertEquals(2,a.binding.spImageResolution.adapter.count)
        method.invoke(a.image,emptyList<ImageResolution>());idle();assertFalse(a.binding.spImageResolution.isEnabled)
        c.pause().stop().destroy()
    }
    @Test fun originalPresetButtonOpensManagement() {
        val c=Robolectric.buildActivity(MainIntegrationActivity::class.java).setup().visible();val a=c.get()
        a.binding.btnPositivePresetSelect.performClick();idle()
        val dialog=ShadowDialog.getLatestDialog();assertTrue(dialog.isShowing)
        assertTrue(views(dialog.window!!.decorView).filterIsInstance<TextView>().any{it.text.toString().contains("向左滑动")})
        dialog.dismiss();c.pause().stop().destroy()
    }
    @Test fun negativePresetEditorCanChangeDefaultAndContent() {
        val c=Robolectric.buildActivity(PresetTestActivity::class.java).setup();val a=c.get()
        val store=PromptPresetStore(File(a.cacheDir,"negative-test"));val item=store.saveNegative(name="原负面",note="",prompt="blur",makeDefault=true)
        val manager=PresetManagerDialog(a,store,true){}.show();idle()
        manager.recycler.measure(View.MeasureSpec.makeMeasureSpec(1000,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(1500,View.MeasureSpec.EXACTLY));manager.recycler.layout(0,0,1000,1500)
        val row=manager.recycler.findViewHolderForAdapterPosition(0)!!.itemView as PresetSwipeRow
        row.revealActions(true,false);row.findViewWithTag<View>("action_edit").performClick();idle()
        val editor=ShadowDialog.getLatestDialog() as AlertDialog
        editor.window!!.decorView.findViewWithTag<CheckBox>("preset_edit_default").isChecked=false
        editor.window!!.decorView.findViewWithTag<EditText>("preset_edit_prompt").setText("low quality")
        editor.getButton(AlertDialog.BUTTON_POSITIVE).performClick();idle()
        assertNull(store.defaultNegative());assertEquals(item.id,store.listNegative().single().id);assertEquals("low quality",store.listNegative().single().prompt)
        manager.dialog.dismiss();c.pause().stop().destroy()
    }
    @Test fun upgradeIdentityAndInternalLabManifest() {
        val a=RuntimeEnvironment.getApplication();assertEquals("com.geniex.demo",BuildConfig.APPLICATION_ID)
        val pm=a.packageManager
        val launcher=pm.getLaunchIntentForPackage(a.packageName);assertEquals("com.geniex.demo.MainActivity",launcher?.component?.className)
        val lab=pm.getActivityInfo(ComponentName(a.packageName,"com.geniex.demo.image.LoraLabActivity"),0);assertFalse(lab.exported)
    }
    @Test @Config(application=MyApplication::class) fun applicationStartupPreservesLegacyData() {
        val a=RuntimeEnvironment.getApplication() as MyApplication
        val sentinel=File(a.filesDir,"models/preserve.txt");sentinel.parentFile!!.mkdirs();sentinel.writeText("keep")
        a.onCreate();assertEquals("keep",sentinel.readText())
    }
}
