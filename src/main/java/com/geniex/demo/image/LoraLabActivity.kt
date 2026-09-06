package com.geniex.demo.image

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Environment
import android.provider.Settings
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.geniex.demo.BuildConfig
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class LoraLabActivity : AppCompatActivity() {
    private lateinit var content: LinearLayout
    private lateinit var status: TextView
    private lateinit var catalog: LinearLayout
    private lateinit var prompt: TextInputEditText
    private lateinit var tagResult: TextView
    private lateinit var startButton: MaterialButton
    private val scanning = AtomicBoolean(false)
    private val handler = Handler(Looper.getMainLooper())
    private val refreshState = object : Runnable {
        override fun run() {
            if (!isFinishing && !isDestroyed) {
                status.text = LoraSelfTest.lastMessage
                startButton.isEnabled = !LoraSelfTest.busy.get()
                handler.postDelayed(this, 600)
            }
        }
    }
    private val base: File get() = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "sdxl_qnn")
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun label(text: String, size: Float = 15f, parent: LinearLayout = content): TextView {
        val view = TextView(this).apply { this.text = text; textSize = size; setTextIsSelectable(true); setPadding(0, dp(7), 0, dp(7)) }
        parent.addView(view, LinearLayout.LayoutParams(-1, -2))
        return view
    }
    private fun button(text: String, parent: LinearLayout = content, action: () -> Unit): MaterialButton {
        val view = MaterialButton(this).apply { this.text = text; isAllCaps = false; minHeight = dp(48); setOnClickListener { action() } }
        parent.addView(view, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
        return view
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "LoRA 1.6 测试"
        content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), dp(16), dp(20), dp(32)) }
        setContentView(ScrollView(this).apply { isFillViewport = true; addView(content, ViewGroup.LayoutParams(-1, -2)) })
        label("LoRA 实验室", 26f)
        label("${BuildConfig.VERSION_NAME} · 与 1.5.11 正式版并列安装", 14f)
        label("优先验证动态 A/B 权重输入，再用适配器分段作对照。请先暂停正式版的生图任务。自测使用内置小模型，不生成图片、不改已有生图模型。")
        button("授予文件访问权限") {
            if (Environment.isExternalStorageManager()) refreshCatalog()
            else runCatching { startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName"))) }
                .onFailure { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
        }
        label("NPU 适配器自测", 21f)
        label("同一模型连续检查：原始 → 强度 0.8 → 1.1 → 更换权重 → 恢复 → 归零。结果与参考数值对比，不只检查是否报错。")
        status = label(LoraSelfTest.lastMessage)
        startButton = button("开始动态 LoRA NPU 自测") {
            if (!Environment.isExternalStorageManager()) { status.text = "请先授予文件访问权限"; return@button }
            val started = LoraSelfTest.start(this) { message ->
                runOnUiThread { if (!isFinishing && !isDestroyed) { status.text = message; startButton.isEnabled = !LoraSelfTest.busy.get() } }
            }
            if (!started) status.text = "已有自测正在运行，请等待结果。"
            startButton.isEnabled = !LoraSelfTest.busy.get()
        }
        button("对照测试：预编译适配器") {
            if (!LoraSelfTest.start(this, dynamic = false) { message ->
                runOnUiThread { if (!isFinishing && !isDestroyed) status.text = message }
            }) status.text = "已有测试正在运行。"
        }
        button("分享自测报告") { shareReport() }
        label("直接权重标签", 21f)
        val field = TextInputLayout(this).apply { hint = "例如 <lora:角色名称:0.8>" }
        prompt = TextInputEditText(field.context).apply { minLines = 2; maxLines = 6; setText(getPreferences(MODE_PRIVATE).getString("tags", "<lora:角色名称:0.8>")) }
        field.addView(prompt, ViewGroup.LayoutParams(-1, -2)); content.addView(field, LinearLayout.LayoutParams(-1, -2))
        tagResult = label("此处检查 LoRA 名称和强度，不会把标签直接交给 CLIP。普通词组的 embedding 加权仍待接入。")
        button("检查标签") {
            runCatching { LoraTags.parse(prompt.text?.toString().orEmpty()) }.onSuccess { parsed ->
                tagResult.text = if (parsed.selections.isEmpty()) "未发现 LoRA 标签。" else parsed.selections.joinToString("\n") {
                    "${it.name}：${it.weight}" + if (it.weight == 0.0) "（关闭）" else "（语法通过，等待 WAI 适配）"
                }
            }.onFailure { tagResult.text = it.message ?: "标签格式错误" }
        }
        label("LoRA 文件夹", 21f)
        label(File(base, "Lora").absolutePath, 14f)
        label("放入 SDXL .safetensors 后自动检测。结构可读不等于已经能在 WAI 上调用；尚不兼容的文件会显示原因。")
        button("刷新 LoRA 列表") { refreshCatalog() }
        catalog = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        content.addView(catalog, LinearLayout.LayoutParams(-1, -2))
    }
    override fun onResume() {
        super.onResume()
        status.text = LoraSelfTest.lastMessage
        startButton.isEnabled = !LoraSelfTest.busy.get()
        refreshCatalog()
        handler.post(refreshState)
    }
    override fun onPause() {
        handler.removeCallbacks(refreshState)
        getPreferences(MODE_PRIVATE).edit().putString("tags", prompt.text?.toString().orEmpty()).apply()
        super.onPause()
    }
    private fun refreshCatalog() {
        if (!Environment.isExternalStorageManager()) { catalog.removeAllViews(); label("等待文件访问权限。", parent = catalog); return }
        if (!scanning.compareAndSet(false, true)) return
        thread(name = "rin-lora-catalog") {
            val result = runCatching { LoraCatalog.scan(base) }
            runOnUiThread {
                scanning.set(false)
                if (isFinishing || isDestroyed) return@runOnUiThread
                catalog.removeAllViews()
                result.onSuccess { entries ->
                    if (entries.isEmpty()) label("文件夹已创建，暂未发现 LoRA。", parent = catalog)
                    entries.forEach { entry ->
                        label(entry.name, 18f, catalog)
                        label("${entry.bytes / 1024 / 1024} MiB · ${entry.detail}", 14f, catalog)
                        if (entry.readable) button("插入 ${entry.name} 标签", catalog) {
                            val value = LoraTags.format(entry.name)
                            val old = prompt.text?.toString().orEmpty().trim()
                            if (!old.contains(value)) prompt.setText(if (old.isBlank()) value else "$old, $value")
                            tagResult.text = "已插入，可直接编辑末尾权重。当前测试版尚未接入 WAI 生成。"
                        }
                    }
                }.onFailure { label(it.message ?: "扫描失败", parent = catalog) }
            }
        }
    }
    private fun shareReport() {
        val file = File(cacheDir, "lora_selftest_latest.json")
        if (!file.isFile) { status.text = "请先运行一次自测。"; return }
        runCatching {
            val exported = File(cacheDir, "lora_reports/latest.json").apply { parentFile?.mkdirs() }
            file.copyTo(exported, overwrite = true)
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", exported)
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "application/json"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }, "分享 LoRA 自测报告"))
        }.onFailure {
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"; putExtra(Intent.EXTRA_TEXT, file.readText())
            }, "分享 LoRA 自测报告"))
        }
    }
}
