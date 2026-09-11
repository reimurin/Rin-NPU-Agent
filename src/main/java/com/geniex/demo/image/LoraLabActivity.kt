package com.geniex.demo.image

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.geniex.demo.BuildConfig
import com.geniex.demo.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class LoraLabActivity:AppCompatActivity() {
    private var pageReady=false
    private lateinit var content:LinearLayout
    private lateinit var catalog:LinearLayout
    private lateinit var prompt:TextInputEditText
    private lateinit var tagResult:TextView
    private val scanning=AtomicBoolean(false)
    private val base:File get()=File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),"sdxl_qnn")
    private fun dp(v:Int)=(v*resources.displayMetrics.density).toInt()
    private fun label(text:String,size:Float=15f,parent:LinearLayout=content):TextView {
        val view=TextView(this).apply {
            this.text=text;textSize=size;setTextIsSelectable(true);setPadding(0,dp(7),0,dp(7))
            setTextColor(ContextCompat.getColor(context,if(size<15)R.color.rin_text_secondary else R.color.rin_text_primary))
        }
        parent.addView(view,LinearLayout.LayoutParams(-1,-2));return view
    }
    private fun button(text:String,parent:LinearLayout=content,tone:RinControls.Tone=RinControls.Tone.SECONDARY,action:()->Unit):MaterialButton {
        val view=RinControls.button(this,tone).apply {this.text=text;isAllCaps=false;minHeight=dp(48);setOnClickListener{action()}}
        parent.addView(view,LinearLayout.LayoutParams(-1,-2).apply{bottomMargin=dp(9)});return view
    }
    override fun onCreate(savedInstanceState:Bundle?) {
        super.onCreate(savedInstanceState);StartupDiagnostics.install(applicationContext)
        try{buildPage(savedInstanceState);pageReady=true}
        catch(e:Exception){StartupDiagnostics.record(this,"LoraLabActivity.buildPage",e);showRecovery(e)}
    }
    private fun buildPage(saved:Bundle?) {
        title="LoRA 管理"
        content=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(dp(20),dp(16),dp(20),dp(32))}
        setContentView(ScrollView(this).apply{isFillViewport=true;addView(content,ViewGroup.LayoutParams(-1,-2))})
        button("返回生图页面",tone=RinControls.Tone.GHOST){finish()}
        label("LoRA 实验室",26f);label("${BuildConfig.VERSION_NAME} · 原应用内 LoRA 管理",14f)
        label("支持兼容的 SDXL / Illustrious UNet 线性 LoRA；同一层的多个 LoRA 合计秩不超过 64。已知文本编码器权重会明确显示当前是否注入，其他未覆盖层仍会报错，不会悄悄忽略。")
        button("授予文件访问权限") {
            if(StorageAccess.granted())refreshCatalog()
            else runCatching{startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,Uri.parse("package:$packageName")))}
                .onFailure{startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))}
        }
        label("提示词与直接权重",21f)
        val initial=saved?.getString(EXTRA_PROMPT) ?: intent.getStringExtra(EXTRA_PROMPT) ?: getPreferences(MODE_PRIVATE).getString("tags","").orEmpty()
        val (field,edit)=RinControls.input(this,"例如 <lora:角色名称:0.8>",initial,"lora_prompt_editor",3,8)
        prompt=edit;content.addView(field,LinearLayout.LayoutParams(-1,-2))
        tagResult=label("末尾直接写 :0.8、:1.1；:0 表示关闭。标签从文本编码中分离，普通词组的 embedding 加权尚未接入。",14f)
        button("检查标签") {
            runCatching{LoraTags.parse(prompt.text?.toString().orEmpty())}.onSuccess { parsed->
                tagResult.text=if(parsed.selections.isEmpty())"没有 LoRA 标签，将使用统一底模。" else parsed.selections.joinToString("\n"){
                    "${it.name}：${it.weight}"+if(it.weight==0.0)"（关闭）" else "（已插入；生成时校验文件和组件）"
                }
                refreshCatalog()
            }.onFailure{tagResult.text=it.message ?: "标签格式错误"}
        }
        button("应用并返回生图",tone=RinControls.Tone.PRIMARY) {
            val text=prompt.text?.toString().orEmpty()
            runCatching{LoraTags.parse(text)}.onSuccess {
                setResult(Activity.RESULT_OK,Intent().putExtra(EXTRA_PROMPT,text));finish()
            }.onFailure{tagResult.text=it.message ?: "标签格式错误"}
        }.tag="lora_apply_to_prompt"
        label("本地 LoRA 文件夹",21f);label(File(base,"Lora").absolutePath,14f)
        label("将 .safetensors 放入此目录。卡片会检测当前提示词：未插入显示“插入”，已插入显示“弹出 · 当前权重”。如果这个 LoRA 成功生成过图片，还会显示最近一次效果图、权重、时间和 Seed。弹出只移除标签，不删除 LoRA 文件。",14f)
        button("刷新 LoRA 列表"){refreshCatalog()}
        catalog=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL};content.addView(catalog,LinearLayout.LayoutParams(-1,-2))
    }
    override fun onResume(){super.onResume();if(pageReady)refreshCatalog()}
    override fun onPause(){if(::prompt.isInitialized)getPreferences(MODE_PRIVATE).edit().putString("tags",prompt.text?.toString().orEmpty()).apply();super.onPause()}
    override fun onSaveInstanceState(outState:Bundle){if(::prompt.isInitialized)outState.putString(EXTRA_PROMPT,prompt.text?.toString().orEmpty());super.onSaveInstanceState(outState)}
    private fun refreshCatalog() {
        if(!StorageAccess.granted()){catalog.removeAllViews();label("等待文件访问权限。",parent=catalog);return}
        if(!scanning.compareAndSet(false,true))return
        thread(name="rin-lora-catalog") {
            val result=runCatching{LoraCatalog.scan(base,LoraCompatibility.load(applicationContext))}
            val history=runCatching{LoraHistoryStore.index(applicationContext)}.getOrDefault(emptyMap())
            runOnUiThread {
                scanning.set(false)
                if(!pageReady||isFinishing||isDestroyed)return@runOnUiThread
                catalog.removeAllViews()
                result.onSuccess {entries->
                    if(entries.isEmpty())label("目录已创建，暂未发现 LoRA 文件。",parent=catalog)
                    entries.forEach {entry->
                        val card=CardView(this).apply {radius=dp(16).toFloat();cardElevation=dp(2).toFloat();useCompatPadding=true}
                        val body=LinearLayout(this).apply {orientation=LinearLayout.VERTICAL;setPadding(dp(16),dp(12),dp(16),dp(12))}
                        card.addView(body,ViewGroup.LayoutParams(-1,-2));catalog.addView(card,LinearLayout.LayoutParams(-1,-2).apply{bottomMargin=dp(12)})
                        label(entry.name,18f,body)
                        history[entry.name]?.let {used->
                            BitmapFactory.decodeFile(used.thumbnail.absolutePath)?.let {bitmap->
                                body.addView(ImageView(this).apply {setImageBitmap(bitmap);adjustViewBounds=true;scaleType=ImageView.ScaleType.CENTER_CROP;contentDescription="${entry.name} 最近一次 LoRA 成图"},LinearLayout.LayoutParams(-1,dp(260)).apply{topMargin=dp(6);bottomMargin=dp(8)})
                            }
                            val time=SimpleDateFormat("MM-dd HH:mm",Locale.getDefault()).format(Date(used.lastUsedMs))
                            val multi=if(used.loraCount>1)" · 本图含 ${used.loraCount} 个 LoRA" else ""
                            label("上次使用：$time · 权重 ${used.weight} · Seed ${used.seed}$multi",14f,body)
                            val te=if(used.ignoredTextEncoderModules>0)" · TE ${used.ignoredTextEncoderModules} 组未注入" else ""
                            label("最近实机记录：UNet ${used.mappedModules} 组已注入$te",14f,body)
                        } ?: label("尚无使用记录；首次成功成图后会在这里显示最近效果图。",14f,body)
                        label("${entry.bytes/1024/1024} MiB · ${entry.detail}",14f,body)
                        if(entry.compatible) {
                            val current=prompt.text?.toString().orEmpty()
                            val selection=runCatching{LoraTags.parse(current).selections.firstOrNull{it.name==entry.name}}.getOrNull()
                            val inserted=selection!=null
                            val actionText=if(inserted)"弹出 ${entry.name} · ${selection!!.weight}" else "插入 ${entry.name} 标签"
                            button(actionText,body,if(inserted)RinControls.Tone.GHOST else RinControls.Tone.SECONDARY) {
                                runCatching {
                                    val old=prompt.text?.toString().orEmpty()
                                    if(LoraTags.contains(old,entry.name)) {
                                        prompt.setText(LoraTags.remove(old,entry.name))
                                        tagResult.text="已弹出 ${entry.name}；只移除了提示词标签，本地 LoRA 文件仍保留。"
                                    } else {
                                        prompt.setText(old.trimEnd()+(if(old.isBlank())"" else ", ")+LoraTags.format(entry.name))
                                        tagResult.text="已插入 ${entry.name} · 0.8；可直接修改末尾权重，然后应用回生图页。"
                                    }
                                    refreshCatalog()
                                }.onFailure{tagResult.text=it.message ?: "请先修正已有标签"}
                            }
                        }
                    }
                }.onFailure{label(it.message ?: "扫描失败",parent=catalog)}
            }
        }
    }
    private fun shareFile(file:File) {
        if(!file.isFile){Toast.makeText(this,"暂无这类报告",Toast.LENGTH_SHORT).show();return}
        runCatching {
            val exported=File(cacheDir,"lora_reports/latest.json").apply{parentFile?.mkdirs()};file.copyTo(exported,overwrite=true)
            val uri=FileProvider.getUriForFile(this,"$packageName.fileprovider",exported)
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply{type="application/json";putExtra(Intent.EXTRA_STREAM,uri);addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)},"分享 LoRA 报告"))
        }.onFailure{Toast.makeText(this,it.message ?: "报告分享失败",Toast.LENGTH_LONG).show()}
    }
    private fun showRecovery(error:Exception) {
        pageReady=false
        val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(dp(20),dp(30),dp(20),dp(20))}
        root.addView(TextView(this).apply{text="LoRA 页面暂未打开，错误已保存。\n${error.javaClass.simpleName}: ${error.message}";textSize=16f;tag="startup_recovery"})
        root.addView(Button(this).apply{text="分享启动诊断";setOnClickListener{StartupDiagnostics.share(this@LoraLabActivity)}})
        root.addView(Button(this).apply{text="返回生图页面";setOnClickListener{finish()}});setContentView(root)
    }
    companion object {const val EXTRA_PROMPT="rin_lora_prompt"}
}
