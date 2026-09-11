package com.geniex.demo.image
import android.app.Application
import android.content.Intent
import com.geniex.demo.databinding.ActivityModelUpdateBinding
import com.geniex.demo.R
import androidx.core.view.GravityCompat
import android.view.LayoutInflater
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Looper
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AlertDialog
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import org.robolectric.annotation.LooperMode
import org.robolectric.shadows.ShadowDialog
import java.io.File
import java.time.Duration
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34],application=Application::class,qualifiers="w412dp-h892dp-port-xhdpi",shadows=[LoraStartupTest.DeniedStorage::class])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@LooperMode(LooperMode.Mode.PAUSED)
class RinUiRenderTest {
    private fun layout(view:View,width:Int,height:Int){view.measure(View.MeasureSpec.makeMeasureSpec(width,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(height,View.MeasureSpec.EXACTLY));view.layout(0,0,width,height)}
    private fun capture(view:View,name:String,width:Int=824,height:Int=1400) {
        layout(view,width,height);shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(180))
        val bitmap=Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888);view.draw(Canvas(bitmap))
        val target=File(System.getProperty("rin.render.dir"),"$name.png");target.parentFile!!.mkdirs()
        target.outputStream().use { assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,it)) };bitmap.recycle()
        assertTrue(target.length()>5000)
    }
    @Test fun nativeRinPresetAndEditorRenders() {
        val controller=Robolectric.buildActivity(PresetTestActivity::class.java).setup();val a=controller.get()
        val store=PromptPresetStore(File(a.cacheDir,"render-presets"))
        val first=store.savePositive(name="山间风景",note="晨雾、山峰与自然光",prompt="mountain landscape, morning mist")
        store.setPositivePinned(first.id,true)
        store.savePositive(name="柔和人像",note="窗边自然光",prompt="portrait, soft window light")
        val manager=PresetManagerDialog(a,store,false){}.show()
        capture(manager.dialog.window!!.decorView,"presets_closed")
        layout(manager.recycler,744,840)
        val row=manager.recycler.findViewHolderForAdapterPosition(0)!!.itemView as PresetSwipeRow
        row.revealActions(true,false);capture(manager.dialog.window!!.decorView,"presets_swiped")
        row.findViewWithTag<View>("action_edit").performClick()
        val editor=ShadowDialog.getLatestDialog() as AlertDialog
        capture(editor.window!!.decorView,"preset_editor")
        editor.dismiss();manager.dialog.dismiss();controller.pause().stop().destroy()
    }
    @Test fun nativeLoraManagementPageRenders() {
        val c=Robolectric.buildActivity(LoraLabActivity::class.java).setup().visible()
        capture(c.get().window.decorView,"lora_management",824,1784)
        assertNull(c.get().window.decorView.findViewWithTag<View>("startup_recovery"));c.pause().stop().destroy()
    }
    @Test fun nativeModelUpdateActivityRendersWithoutNetwork() {
        val intent=Intent().putExtra(ModelUpdateActivity.EXTRA_SKIP_AUTO_CHECK,true)
        val c=Robolectric.buildActivity(ModelUpdateActivity::class.java,intent).setup().visible()
        capture(c.get().window.decorView,"alpha6_model_update_activity",824,1784)
        assertEquals(View.VISIBLE,c.get().findViewById<View>(R.id.btn_update_check).visibility)
        assertEquals(View.GONE,c.get().findViewById<View>(R.id.btn_update_ignore).visibility)
        assertEquals(View.VISIBLE,c.get().findViewById<View>(R.id.btn_model_center_lora).visibility)
        assertEquals(View.VISIBLE,c.get().findViewById<View>(R.id.btn_model_center_runtime_check).visibility)
        assertEquals(View.VISIBLE,c.get().findViewById<View>(R.id.btn_model_center_npu_test).visibility)
        assertTrue(c.get().findViewById<View>(R.id.tv_update_last_check).measuredHeight>0)
        c.pause().stop().destroy()
    }
    @Test fun nativeBuild03ModelCenterAndDrawerRenders() {
        val c=Robolectric.buildActivity(PresetTestActivity::class.java).setup().visible();val a=c.get()
        val b=ActivityModelUpdateBinding.inflate(a.layoutInflater);a.setContentView(b.root)
        b.tvUpdateAppVersion.text="Rin NPU Agent 1.6.0-alpha.6"
        b.tvUpdateDevice.text="SM8750 · QNN SoC 69 · V79 · Snapdragon 8 Elite"
        b.tvUpdateLocal.text="当前模型包：1.0.0（本地三分辨率共享权重版本）"
        b.tvUpdateRemote.text="远端模型包：1.1.0"
        b.tvUpdateResolutions.text="更新后可用：1024 × 1024、832 × 1216、1216 × 832、768 × 1344、1344 × 768"
        b.tvUpdateLora.text="动态 LoRA：rank 64 · ABI e362a2fcf8b5c5af928c2db7fd594232720501345c08ef1b5dfce4acb19e7d18"
        b.tvUpdateSize.text="更新大小：5.49 GiB"
        b.tvUpdateSourceUsed.text="自动模式：中国大陆优先魔搭镜像，失败自动回退 GitHub 原始源。"
        b.tvUpdateStatus.text="发现兼容模型更新。这里使用较长中文状态文本验证高 DPI、中文换行、状态信息与按钮之间不会发生重叠或裁切。下载过程中还会显示文件序号、总进度、速度和实际使用的镜像源。"
        b.tvModelCenterLoraStatus.text="已启用：Alyosha 0.6 · 最近使用记录可用"
        b.tvModelCenterAdvancedStatus.text="运行时完整性正常 · NPU 自测与诊断工具集中在高级"
        b.btnUpdateCancel.visibility=View.VISIBLE
        capture(b.root,"alpha8_build03_model_center_top",824,1784)
        layout(b.root,824,1784);shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(100))
        assertTrue(b.btnUpdateCheck.height>=80);assertTrue(b.btnUpdateInstall.height>=80);assertTrue(b.btnUpdateCancel.measuredHeight>=70)
        assertTrue(b.tvUpdateStatus.measuredHeight>0);assertTrue(b.tvUpdateResolutions.measuredHeight>0);assertTrue(b.tvUpdateLora.measuredHeight>0);assertTrue(b.tvUpdateSize.measuredHeight>0)
        b.modelUpdateScroll.fullScroll(View.FOCUS_DOWN);shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(120));capture(b.root,"alpha8_build03_model_center_bottom",824,1784)
        c.pause().stop().destroy()

        val dc=Robolectric.buildActivity(PresetTestActivity::class.java).setup().visible();val da=dc.get()
        val drawer=LayoutInflater.from(da).inflate(R.layout.activity_main,null,false);da.setContentView(drawer)
        layout(drawer,824,1784)
        val drawerLayout=drawer.findViewById<androidx.drawerlayout.widget.DrawerLayout>(R.id.drawer_layout)
        drawer.findViewById<View>(R.id.drawer_chat_settings_group).visibility=View.GONE
        drawer.findViewById<View>(R.id.drawer_image_settings_group).visibility=View.VISIBLE
        val center=drawer.findViewById<View>(R.id.btn_image_model_center)
        assertNotNull(center);drawerLayout.openDrawer(GravityCompat.START);shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(220))
        assertEquals(View.VISIBLE,center.visibility);assertTrue(center.measuredHeight>=70)
        assertTrue(drawer.findViewById<View>(R.id.tv_image_model_status_summary).measuredHeight>0)
        assertTrue(drawer.findViewById<View>(R.id.tv_image_lora_status_summary).measuredHeight>0)
        capture(drawer,"alpha8_build03_drawer_model_center",824,1784)
        dc.pause().stop().destroy()
    }

}
