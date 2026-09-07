package com.geniex.demo.image

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.geniex.demo.BuildConfig
import com.geniex.demo.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class LoraLabActivity:AppCompatActivity() {
    private var pageReady=false
    private lateinit var content:LinearLayout
    private lateinit var catalog:LinearLayout
    private lateinit var prompt:TextInputEditText
    private lateinit var tagResult:TextView
    private lateinit var componentStatus:TextView
    private lateinit var status:TextView
    private lateinit var startButton:MaterialButton
    private lateinit var installButton:MaterialButton
    private lateinit var pauseButton:MaterialButton
    private val scanning=AtomicBoolean(false)
    private val handler=Handler(Looper.getMainLooper())
    private val base:File get()=File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),"sdxl_qnn")
    private val refreshState=object:Runnable {
        override fun run() {
            if(pageReady&&!isFinishing&&!isDestroyed) {
                status.text=LoraSelfTest.lastMessage
                componentStatus.text=LoraModelComponent.status
                startButton.isEnabled=!LoraSelfTest.busy.get()&&!LoraModelComponent.busy.get()
                installButton.isEnabled=!LoraModelComponent.busy.get()&&!LoraSelfTest.busy.get()
                pauseButton.isEnabled=LoraModelComponent.busy.get()
                handler.postDelayed(this,700)
            }
        }
    }
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
        label("支持兼容的 SDXL UNet 线性 LoRA；同一层的多个 LoRA 合计秩不超过 64。含文本编码器、空间卷积或其他未覆盖权重的文件会显示原因，不会悄悄忽略。")
        button("授予文件访问权限") {
            if(StorageAccess.granted())refreshCatalog()
            else runCatching{startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,Uri.parse("package:$packageName")))}
                .onFailure{startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))}
        }
        label("统一 WAI 模型（普通生图 + LoRA）",21f)
        label("SM8750 · 原生 1024 × 1024\n升级载荷 7,662,304,830 bytes（7.14 GiB）；升级阶段临时峰值约 9.50 GiB。激活后普通生图与 LoRA 共用同一套七段 UNet。旧两段 UNet 共 5,256,428,704 bytes（4.90 GiB）先保留作回滚点；实机质量与速度通过后再释放。按当前 CLIP-G 修复基线估算，清理旧 UNet 后整个 runtime 约 9.17 GiB。",14f)
        componentStatus=label(LoraModelComponent.status)
        installButton=button("安装 / 继续升级统一模型") {
            if(!StorageAccess.granted()){LoraModelComponent.status="请先授予文件访问权限";return@button}
            if(LoraSelfTest.busy.get()){LoraModelComponent.status="请等待 NPU 自测结束";return@button}
            RinControls.dialog(this).setTitle("升级到支持 LoRA 的统一 WAI 模型")
                .setMessage("下载并校验 7.14 GiB 的新统一 UNet 到 staging；不会提前删除旧模型。alpha.4 实机验收通过后再释放旧 UNet。")
                .setNegativeButton("取消",null).setPositiveButton("开始 / 继续") {_,_->
                    LoraModelComponent.install(this,base) { message->runOnUiThread {
                        if(pageReady&&!isFinishing&&!isDestroyed)componentStatus.text=message
                    }}
                }.show()
        }
        pauseButton=button("暂停组件下载"){LoraModelComponent.cancel()}
        label("提示词与直接权重",21f)
        val initial=saved?.getString(EXTRA_PROMPT) ?: intent.getStringExtra(EXTRA_PROMPT) ?: getPreferences(MODE_PRIVATE).getString("tags","").orEmpty()
        val (field,edit)=RinControls.input(this,"例如 <lora:角色名称:0.8>",initial,"lora_prompt_editor",3,8)
        prompt=edit;content.addView(field,LinearLayout.LayoutParams(-1,-2))
        tagResult=label("末尾直接写 :0.8、:1.1；:0 表示关闭。标签从文本编码中分离，普通词组的 embedding 加权尚未接入。",14f)
        button("检查标签") {
            runCatching{LoraTags.parse(prompt.text?.toString().orEmpty())}.onSuccess { parsed->
                tagResult.text=if(parsed.selections.isEmpty())"没有 LoRA 标签，将使用原底模。" else parsed.selections.joinToString("\n"){
                    "${it.name}：${it.weight}"+if(it.weight==0.0)"（关闭）" else "（语法通过，生成时校验文件和组件）"
                }
            }.onFailure{tagResult.text=it.message ?: "标签格式错误"}
        }
        button("应用并返回生图",tone=RinControls.Tone.PRIMARY) {
            val text=prompt.text?.toString().orEmpty()
            runCatching{LoraTags.parse(text)}.onSuccess {
                setResult(Activity.RESULT_OK,Intent().putExtra(EXTRA_PROMPT,text));finish()
            }.onFailure{tagResult.text=it.message ?: "标签格式错误"}
        }.tag="lora_apply_to_prompt"
        label("本地 LoRA 文件夹",21f);label(File(base,"Lora").absolutePath,14f)
        label("将 .safetensors 放入此目录。打开页面时自动检查文件完整性、层名、矩阵形状和秩；插入后可直接编辑权重。",14f)
        button("刷新 LoRA 列表"){refreshCatalog()}
        catalog=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL};content.addView(catalog,LinearLayout.LayoutParams(-1,-2))
        val advanced=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;visibility=View.GONE}
        button("展开 / 收起高级自测") {advanced.visibility=if(advanced.visibility==View.VISIBLE)View.GONE else View.VISIBLE}
        content.addView(advanced,LinearLayout.LayoutParams(-1,-2))
        label("NPU 适配器自测",21f,advanced)
        label("以下使用内置小模型，只用于诊断接口；已经通过时无需反复运行，不能代替真实 WAI 成图验收。",14f,advanced)
        status=label(LoraSelfTest.lastMessage,parent=advanced)
        startButton=button("开始动态 LoRA NPU 自测",advanced) {
            if(LoraModelComponent.busy.get()){Toast.makeText(this,"请先完成或暂停组件下载",Toast.LENGTH_SHORT).show();return@button}
            LoraSelfTest.start(this){message->runOnUiThread{if(pageReady&&!isFinishing&&!isDestroyed)status.text=message}}
        }
        button("对照测试：预编译适配器",advanced) {
            if(!LoraModelComponent.busy.get())LoraSelfTest.start(this,dynamic=false){message->runOnUiThread{if(pageReady&&!isFinishing&&!isDestroyed)status.text=message}}
        }
        button("分享自测报告",advanced){shareFile(File(cacheDir,"lora_selftest_latest.json"))}
        button("分享最近 LoRA 生图记录",advanced){shareFile(File(base,".rin_diagnostics/lora_generation_latest.json"))}
        button("分享启动异常记录",advanced){StartupDiagnostics.share(this)}
    }
    override fun onResume(){super.onResume();if(pageReady){refreshCatalog();handler.post(refreshState)}}
    override fun onPause(){handler.removeCallbacks(refreshState);if(::prompt.isInitialized)getPreferences(MODE_PRIVATE).edit().putString("tags",prompt.text?.toString().orEmpty()).apply();super.onPause()}
    override fun onSaveInstanceState(outState:Bundle){if(::prompt.isInitialized)outState.putString(EXTRA_PROMPT,prompt.text?.toString().orEmpty());super.onSaveInstanceState(outState)}
    private fun refreshCatalog() {
        if(!StorageAccess.granted()){catalog.removeAllViews();label("等待文件访问权限。",parent=catalog);return}
        if(!scanning.compareAndSet(false,true))return
        thread(name="rin-lora-catalog") {
            val result=runCatching{LoraCatalog.scan(base,LoraCompatibility.load(applicationContext))}
            val installed=LoraModelComponent.installed(base)
            runOnUiThread {
                scanning.set(false)
                if(!pageReady||isFinishing||isDestroyed)return@runOnUiThread
                if(!LoraModelComponent.busy.get())LoraModelComponent.status=if(installed)"统一 WAI 模型已安装；旧 UNet 暂保留回滚，alpha.4 激活后普通生图与 LoRA 共用七段 UNet。" else "尚未升级统一 WAI 模型；当前旧普通 UNet 仍可正常使用。"
                catalog.removeAllViews()
                result.onSuccess {entries->
                    if(entries.isEmpty())label("目录已创建，暂未发现 LoRA 文件。",parent=catalog)
                    entries.forEach {entry->
                        label(entry.name,18f,catalog);label("${entry.bytes/1024/1024} MiB · ${entry.detail}",14f,catalog)
                        if(entry.compatible)button("插入 ${entry.name} 标签",catalog) {
                            runCatching {
                                val old=prompt.text?.toString().orEmpty();val parsed=LoraTags.parse(old)
                                if(parsed.selections.none{it.name==entry.name})prompt.setText(old.trimEnd()+(if(old.isBlank())"" else ", ")+LoraTags.format(entry.name))
                                tagResult.text="已准备 ${entry.name}，可直接修改末尾权重，然后应用回生图页。"
                            }.onFailure{tagResult.text=it.message ?: "请先修正已有标签"}
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
