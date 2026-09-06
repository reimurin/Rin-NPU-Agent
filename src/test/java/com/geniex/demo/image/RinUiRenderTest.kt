package com.geniex.demo.image
import android.app.Application
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
}
