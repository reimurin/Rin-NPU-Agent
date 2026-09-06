package com.geniex.demo.image
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import android.app.Application
import android.os.Environment
import org.robolectric.annotation.Implements
import org.robolectric.annotation.Implementation
import org.robolectric.shadows.ShadowEnvironment
import android.widget.TextView
import com.google.android.material.textfield.TextInputLayout
import com.google.android.material.textfield.TextInputEditText
import android.view.View
import android.view.ViewGroup
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class, qualifiers = "w412dp-h892dp-port")
@LooperMode(LooperMode.Mode.PAUSED)
class LoraStartupTest {
    @Implements(Environment::class)
    class DeniedStorage:ShadowEnvironment() { companion object { @JvmStatic @Implementation fun isExternalStorageManager()=false } }
    @Implements(Environment::class)
    class GrantedStorage:ShadowEnvironment() { companion object { @JvmStatic @Implementation fun isExternalStorageManager()=true } }
    @Implements(Environment::class)
    class BrokenStorage:ShadowEnvironment() { companion object { @JvmStatic @Implementation fun isExternalStorageManager():Boolean=throw IllegalStateException("volume unavailable") } }
    @Test @Config(shadows=[DeniedStorage::class]) fun opensWhenPermissionDenied(){opensWithoutStoragePermissionOrNpu()}
    @Test @Config(shadows=[GrantedStorage::class]) fun opensWhenPermissionGranted(){opensWithoutStoragePermissionOrNpu()}
    @Test @Config(shadows=[BrokenStorage::class]) fun opensWhenStorageServiceThrows(){opensWithoutStoragePermissionOrNpu()}
    @Test fun oldGenericLayoutParamsReproduceFirstFrameCrash() {
        val controller=Robolectric.buildActivity(PresetTestActivity::class.java).setup()
        val activity=controller.get();val field=TextInputLayout(activity)
        field.addView(TextInputEditText(field.context),ViewGroup.LayoutParams(-1,-2))
        try {
            field.measure(View.MeasureSpec.makeMeasureSpec(1000,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(500,View.MeasureSpec.AT_MOST))
            fail("Old layout code should reproduce ClassCastException")
        } catch (_:ClassCastException) {}
        controller.pause().stop().destroy()
    }
    @Test fun permissionQueryFailureReturnsFalse(){assertFalse(StorageAccess.granted{throw SecurityException("test")});assertTrue(StorageAccess.lastError!!.contains("SecurityException"))}

    private fun texts(view: View): List<String> = when(view) {
        is ViewGroup -> (0 until view.childCount).flatMap { texts(view.getChildAt(it)) }
        is TextView -> listOf(view.text.toString())
        else -> emptyList()
    }
    @Test fun opensWithoutStoragePermissionOrNpu() {
        val controller = Robolectric.buildActivity(LoraLabActivity::class.java)
        controller.create().start().resume().visible()
        assertTrue(texts(controller.get().window.decorView).any { it.contains("LoRA 实验室") })
        assertNull(controller.get().window.decorView.findViewWithTag<View>("startup_recovery"))
        controller.pause().stop().destroy()
    }
}
