package com.geniex.demo.image
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.view.LayoutInflater
import android.content.res.ColorStateList
import androidx.core.content.ContextCompat
import com.geniex.demo.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputLayout
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.checkbox.MaterialCheckBox

internal object RinControls {
    enum class Tone { PRIMARY, SECONDARY, GHOST, DANGER }
    fun button(context:Context,tone:Tone=Tone.SECONDARY):MaterialButton {
        val layout=when(tone) {
            Tone.PRIMARY->R.layout.rin_component_button_primary
            Tone.SECONDARY->R.layout.rin_component_button_secondary
            Tone.GHOST->R.layout.rin_component_button_ghost
            Tone.DANGER->R.layout.rin_component_button_danger
        }
        return LayoutInflater.from(context).inflate(layout,null,false) as MaterialButton
    }
    fun dialog(context:Context):MaterialAlertDialogBuilder {
        val density=context.resources.displayMetrics.density
        val background=GradientDrawable().apply {
            cornerRadius=24*density
            setColor(ContextCompat.getColor(context,R.color.rin_surface))
            setStroke(density.toInt().coerceAtLeast(1),ContextCompat.getColor(context,R.color.rin_outline))
        }
        return MaterialAlertDialogBuilder(context,R.style.ThemeOverlay_Rin_Dialog).setBackground(background)
    }
    fun input(context:Context,hint:String,value:String,tag:String,minLines:Int=1,maxLines:Int=1):Pair<TextInputLayout,TextInputEditText> {
        val field=LayoutInflater.from(context).inflate(R.layout.rin_component_text_input,null,false) as TextInputLayout
        field.hint=hint
        val edit=field.editText as TextInputEditText
        edit.tag=tag;edit.minLines=minLines;edit.maxLines=maxLines
        if(maxLines==1)edit.setSingleLine()
        edit.setText(value)
        return field to edit
    }
    fun check(context:Context):MaterialCheckBox = MaterialCheckBox(context).apply {
        setUseMaterialThemeColors(false)
        buttonTintList=ColorStateList(arrayOf(intArrayOf(android.R.attr.state_checked),intArrayOf()),intArrayOf(ContextCompat.getColor(context,R.color.rin_primary),ContextCompat.getColor(context,R.color.rin_text_secondary)))
        setTextColor(ContextCompat.getColor(context,R.color.rin_text_primary))
        textSize=14f
    }
}
