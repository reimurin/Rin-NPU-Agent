package com.geniex.demo.image

import android.app.Application
import android.graphics.Rect
import android.graphics.RectF
import android.os.Looper
import android.view.View
import android.widget.LinearLayout
import com.google.android.material.textfield.TextInputLayout
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import org.robolectric.annotation.LooperMode
import java.time.Duration

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34], application=Application::class, qualifiers="w412dp-h892dp-port-xhdpi")
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@LooperMode(LooperMode.Mode.PAUSED)
class RinInputLayoutTest {
    private fun verify(minLines:Int, maxLines:Int) {
        val controller=Robolectric.buildActivity(PresetTestActivity::class.java).setup()
        val activity=controller.get()
        try {
            val root=LinearLayout(activity).apply {
                orientation=LinearLayout.VERTICAL
                isFocusableInTouchMode=true
                setPadding(36,24,36,24)
            }
            val (field,edit)=RinControls.input(activity,"预设名称","山间风景","regression_input",minLines,maxLines)
            field.isHintAnimationEnabled=false
            root.addView(field,LinearLayout.LayoutParams(-1,-2))
            activity.setContentView(root)
            for(focused in listOf(false,true)) {
                if(focused) edit.requestFocus() else root.requestFocus()
                root.measure(View.MeasureSpec.makeMeasureSpec(824,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(1500,View.MeasureSpec.EXACTLY))
                root.layout(0,0,824,1500)
                shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(300))
                root.measure(View.MeasureSpec.makeMeasureSpec(824,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(1500,View.MeasureSpec.EXACTLY))
                root.layout(0,0,824,1500)
                // Fixed Material 1.12.0: inspect its actual collapsed label bounds.
                val helperField=TextInputLayout::class.java.getDeclaredField("collapsingTextHelper").apply { isAccessible=true }
                val helper=helperField.get(field)
                val labelBounds=RectF()
                helper.javaClass.getMethod("getCollapsedTextActualBounds",RectF::class.java,Int::class.javaPrimitiveType,Int::class.javaPrimitiveType)
                    .invoke(helper,labelBounds,edit.width,edit.gravity)
                val editRect=Rect()
                edit.getDrawingRect(editRect)
                field.offsetDescendantRectToMyCoords(edit,editRect)
                val valueTop=editRect.top+edit.baseline+edit.paint.fontMetrics.ascent
                val minimumGap=2*activity.resources.displayMetrics.density
                assertFalse("Collapsed label must be measurable",labelBounds.isEmpty)
                assertTrue("Hint overlaps value: focused=$focused lines=$minLines hint=$labelBounds valueTop=$valueTop",valueTop>=labelBounds.bottom+minimumGap)
            }
        } finally { controller.pause().stop().destroy() }
    }
    @Test fun singleLineLabelAndValueDoNotOverlap()=verify(1,1)
    @Test fun multilineLabelAndValueDoNotOverlap()=verify(3,6)
}
