package com.geniex.demo.image
import android.app.Application
import android.os.Bundle
import android.os.Looper
import android.view.View
import android.view.ViewGroup
import android.view.MotionEvent
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AlertDialog
import org.junit.Test
import org.junit.Assert.*
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import org.robolectric.shadows.ShadowDialog
import java.time.Duration
import com.geniex.demo.R
class PresetTestActivity:AppCompatActivity(){override fun onCreate(state:Bundle?){setTheme(R.style.Theme_GenieXDemo);super.onCreate(state)}}
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34],application=Application::class,qualifiers="w412dp-h892dp-port")
@LooperMode(LooperMode.Mode.PAUSED)
class PresetUiTest {
    @get:Rule val temp=TemporaryFolder()
    private fun layout(v:View,w:Int=1000,h:Int=1700){v.measure(View.MeasureSpec.makeMeasureSpec(w,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(h,View.MeasureSpec.EXACTLY));v.layout(0,0,w,h)}
    private fun idle(){shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(200))}
    @Test fun leftSwipeRevealsButDoesNotDelete() {
        val c=Robolectric.buildActivity(PresetTestActivity::class.java).setup();val a=c.get();var deleted=0
        val row=PresetSwipeRow(a);row.bind(PresetListItem("a","长名称预设","备注","cat",false),{},{},{},{deleted++});a.setContentView(row);layout(row,1000,300)
        fun event(action:Int,x:Float)=MotionEvent.obtain(0,100,action,x,100f,0)
        row.onInterceptTouchEvent(event(MotionEvent.ACTION_DOWN,800f))
        val move=event(MotionEvent.ACTION_MOVE,200f);assertTrue(row.onInterceptTouchEvent(move));row.onTouchEvent(move);row.onTouchEvent(event(MotionEvent.ACTION_UP,200f));idle()
        assertTrue(row.isOpen);assertEquals(0,deleted)
        row.findViewWithTag<View>("action_delete").performClick();assertEquals(1,deleted)
        c.pause().stop().destroy()
    }
    @Test fun verticalScrollDoesNotRevealActions() {
        val c=Robolectric.buildActivity(PresetTestActivity::class.java).setup();val row=PresetSwipeRow(c.get());layout(row,1000,300)
        row.onInterceptTouchEvent(MotionEvent.obtain(0,0,MotionEvent.ACTION_DOWN,500f,100f,0))
        assertFalse(row.onInterceptTouchEvent(MotionEvent.obtain(0,50,MotionEvent.ACTION_MOVE,480f,400f,0)));assertFalse(row.isOpen);c.pause().stop().destroy()
    }
    @Test fun managementEditPinDeleteAndCancel() {
        val c=Robolectric.buildActivity(PresetTestActivity::class.java).setup();val a=c.get();val store=PromptPresetStore(temp.newFolder())
        val item=store.savePositive(name="原名称",note="备注",prompt="cat")
        val manager=PresetManagerDialog(a,store,false){}.show();idle();layout(manager.recycler);idle()
        fun row():PresetSwipeRow{layout(manager.recycler);return manager.recycler.findViewHolderForAdapterPosition(0)!!.itemView as PresetSwipeRow}
        row().revealActions(true,false);row().findViewWithTag<View>("action_pin").performClick();idle();assertTrue(store.listPositive().single().pinnedAt>0)
        row().revealActions(true,false);row().findViewWithTag<View>("action_edit").performClick();idle()
        val editor=ShadowDialog.getLatestDialog() as AlertDialog
        editor.findViewById<View>(android.R.id.content)!!.findViewWithTag<EditText>("preset_edit_name").setText("新名称")
        editor.findViewById<View>(android.R.id.content)!!.findViewWithTag<EditText>("preset_edit_prompt").setText("dog")
        editor.getButton(AlertDialog.BUTTON_POSITIVE).performClick();idle()
        assertEquals("dog",store.listPositive().single().prompt);assertEquals(item.id,store.listPositive().single().id);assertTrue(store.listPositive().single().pinnedAt>0)
        row().revealActions(true,false);row().findViewWithTag<View>("action_delete").performClick();idle()
        var confirm=ShadowDialog.getLatestDialog() as AlertDialog;assertEquals(1,store.listPositive().size);confirm.getButton(AlertDialog.BUTTON_NEGATIVE).performClick();idle();assertEquals(1,store.listPositive().size)
        row().revealActions(true,false);row().findViewWithTag<View>("action_delete").performClick();idle();confirm=ShadowDialog.getLatestDialog() as AlertDialog
        confirm.getButton(AlertDialog.BUTTON_POSITIVE).performClick();idle();assertTrue(store.listPositive().isEmpty());manager.dialog.dismiss();c.pause().stop().destroy()
    }
}
