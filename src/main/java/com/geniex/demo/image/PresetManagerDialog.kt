package com.geniex.demo.image
import android.app.Activity
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.geniex.demo.R

internal data class PresetListItem(val id:String,val name:String,val note:String,val prompt:String,val pinned:Boolean,val isDefault:Boolean=false)
internal class PresetManagerDialog(private val activity:Activity,private val store:PromptPresetStore,private val negative:Boolean,private val selected:(PresetListItem)->Unit) {
    private var items=emptyList<PresetListItem>();private var opened:PresetSwipeRow?=null
    private val empty=TextView(activity).apply { textSize=15f;setPadding(dp(12),dp(20),dp(12),dp(20)) }
    internal val recycler=RecyclerView(activity).apply { layoutManager=LinearLayoutManager(activity);itemAnimator=null;contentDescription="已保存的预设" }
    private val adapter=object:RecyclerView.Adapter<Holder>() {
        override fun getItemCount()=items.size
        override fun onCreateViewHolder(parent:ViewGroup,viewType:Int):Holder {
            val row=PresetSwipeRow(activity).apply {
                layoutParams=RecyclerView.LayoutParams(-1,-2).apply { bottomMargin=dp(8) }
                onOpened={ current->if(opened!==current)opened?.revealActions(false);opened=current }
            }
            return Holder(row)
        }
        override fun onBindViewHolder(holder:Holder,position:Int) { val item=items[position];holder.row.bind(item,{choose(item)},{togglePin(item)},{edit(item)},{confirmDelete(item)}) }
        override fun onViewRecycled(holder:Holder) { if(opened===holder.row)opened=null;holder.row.revealActions(false,false);super.onViewRecycled(holder) }
    }
    internal val dialog:AlertDialog
    init {
        val root=LinearLayout(activity).apply { orientation=LinearLayout.VERTICAL;setPadding(dp(16),dp(6),dp(16),0) }
        root.addView(TextView(activity).apply { text="点选使用；向左滑动可置顶、编辑或删除。也可点 ⋮ 展开。";textSize=14f;setPadding(0,dp(4),0,dp(12)) },LinearLayout.LayoutParams(-1,-2))
        root.addView(empty,LinearLayout.LayoutParams(-1,-2));recycler.adapter=adapter
        root.addView(recycler,LinearLayout.LayoutParams(-1,(activity.resources.displayMetrics.heightPixels*0.55).toInt().coerceAtLeast(dp(200))))
        dialog=AlertDialog.Builder(activity).setTitle(if(negative)"负面提示词预设" else "正向提示词预设").setView(root).setNegativeButton("关闭",null).create()
        dialog.setOnDismissListener { opened?.revealActions(false,false);opened=null }
    }
    fun show():PresetManagerDialog { reload();dialog.show();return this }
    private fun reload() {
        opened=null
        runCatching {
            items=if(negative)store.listNegative().map{PresetListItem(it.id,it.name,it.note,it.prompt,it.pinnedAt>0,it.isDefault)}
                else store.listPositive().map{PresetListItem(it.id,it.name,it.note,it.prompt,it.pinnedAt>0)}
            empty.text="还没有保存的预设。请先在生图页保存提示词。"
        }.onFailure {items=emptyList();empty.text=it.message ?: "读取失败，原文件未改动"}
        empty.visibility=if(items.isEmpty())View.VISIBLE else View.GONE;recycler.visibility=if(items.isEmpty())View.GONE else View.VISIBLE;adapter.notifyDataSetChanged()
    }
    private fun choose(item:PresetListItem) {
        runCatching {if(negative)check(store.touchNegative(item.id)!=null) else check(store.touchPositive(item.id)!=null);selected(item);dialog.dismiss()}.onFailure {error(it);reload()}
    }
    private fun togglePin(item:PresetListItem) {
        runCatching {check(if(negative)store.setNegativePinned(item.id,!item.pinned) else store.setPositivePinned(item.id,!item.pinned));reload()}.onFailure(::error)
    }
    private fun confirmDelete(item:PresetListItem) {
        AlertDialog.Builder(activity).setTitle("删除预设？").setMessage("删除“${item.name}”"+(if(item.isDefault)"后将不再有这条默认负面预设。" else "？"))
            .setNegativeButton("取消",null).setPositiveButton("删除") {_,_->runCatching {if(negative)store.deleteNegative(item.id) else store.deletePositive(item.id);reload()}.onFailure(::error)}.show()
    }
    private fun edit(item:PresetListItem) {
        val name=EditText(activity).apply{hint="预设名称";setSingleLine();setText(item.name);tag="preset_edit_name"}
        val note=EditText(activity).apply{hint="备注";minLines=2;maxLines=5;setText(item.note);tag="preset_edit_note"}
        val prompt=EditText(activity).apply{hint="提示词";minLines=4;maxLines=10;setText(item.prompt);tag="preset_edit_prompt"}
        val default=CheckBox(activity).apply{text="设为默认负面预设";isChecked=item.isDefault;tag="preset_edit_default"}
        val root=LinearLayout(activity).apply {
            orientation=LinearLayout.VERTICAL;setPadding(dp(18),dp(8),dp(18),dp(8))
            listOf(name,note,prompt).forEach{input->input.setBackgroundResource(R.drawable.rin_bg_control);addView(input,LinearLayout.LayoutParams(-1,-2).apply{topMargin=dp(10)})}
            if(negative)addView(default,LinearLayout.LayoutParams(-1,-2))
        }
        val editor=AlertDialog.Builder(activity).setTitle("编辑预设").setView(ScrollView(activity).apply{addView(root)}).setNegativeButton("取消",null).setPositiveButton("保存",null).create()
        editor.setOnShowListener {
            editor.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                if(name.text.isBlank()){name.error="请输入名称";return@setOnClickListener}
                if(prompt.text.isBlank()){prompt.error="请输入提示词";return@setOnClickListener}
                runCatching {
                    if(negative)store.saveNegative(item.id,name.text.toString(),note.text.toString(),prompt.text.toString(),default.isChecked)
                    else store.savePositive(item.id,name.text.toString(),note.text.toString(),prompt.text.toString())
                    editor.dismiss();reload()
                }.onFailure{error(it)}
            }
        }
        editor.show()
    }
    private fun error(t:Throwable){Toast.makeText(activity,t.message ?: "操作未完成",Toast.LENGTH_LONG).show()}
    private fun dp(v:Int)=(v*activity.resources.displayMetrics.density).toInt()
    private class Holder(val row:PresetSwipeRow):RecyclerView.ViewHolder(row)
}
