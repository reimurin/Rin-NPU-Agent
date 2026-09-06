package com.geniex.demo.image

import android.content.Context
import android.os.Build
import android.os.Environment
import com.geniex.demo.BuildConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.abs

internal object LoraSelfTest {
    val busy = AtomicBoolean(false)
    @Volatile var lastMessage = "尚未运行 NPU 自测"
    @Volatile var lastReport = ""
    internal fun matches(actual: FloatArray, expected: FloatArray, atol: Double, rtol: Double): Boolean =
        actual.isNotEmpty() && actual.size == expected.size && atol >= 0 && rtol >= 0 &&
        actual.indices.all { actual[it].isFinite() && expected[it].isFinite() && abs(actual[it] - expected[it]) <= atol + rtol * abs(expected[it]) }
    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream -> val b = ByteArray(65536); while (true) { val n = stream.read(b); if (n < 0) break; digest.update(b, 0, n) } }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    internal fun floats(file: File, count: Int): FloatArray {
        require(file.isFile && file.length() == count * 4L) { "自测输出缺失或长度错误：${file.name}" }
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(count) { buffer.float.also { value -> require(value.isFinite()) { "输出含非有限值" } } }
    }
    fun start(context: Context, dynamic: Boolean = true, callback: (String) -> Unit): Boolean {
        if (!busy.compareAndSet(false, true)) return false
        val app = context.applicationContext
        val route = if (dynamic) "dynamic_weights" else "binary_sections"
        val assetPrefix = if (dynamic) "lora_dynamic_selftest" else "lora_selftest"
        lastMessage = "正在验证素材并准备 NPU…"; callback(lastMessage)
        thread(name = "rin-lora-npu-selftest") {
            val report = JSONObject().put("version", BuildConfig.VERSION_NAME).put("soc", Build.SOC_MODEL)
                .put("scope", "synthetic LoRA same-context switch test; not WAI compatibility").put("route", route)
            var diag: File? = null
            try {
                require(Environment.isExternalStorageManager()) { "请先授予所有文件访问权限" }
                require(Build.SOC_MODEL.contains("SM8750", true)) { "本自测素材只针对 SM8750" }
                val base = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "sdxl_qnn")
                LoraCatalog.directory(base)
                diag = File(base, ".rin_diagnostics").apply { mkdirs() }
                val root = File(app.cacheDir, "lora_selftest/${UUID.randomUUID()}").apply { mkdirs() }
                val assets = File(root, "assets").apply { mkdirs() }
                val manifest = app.assets.open("$assetPrefix/manifest.json").bufferedReader().use { JSONObject(it.readText()) }
                val files = manifest.getJSONArray("assets")
                for (i in 0 until files.length()) {
                    val spec = files.getJSONObject(i); val name = spec.getString("name")
                    require(!name.contains('/') && !name.contains('\\') && name != "..")
                    val target = File(assets, name)
                    app.assets.open("$assetPrefix/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
                    require(target.length() == spec.getLong("bytes") && hash(target) == spec.getString("sha256")) { "自测素材校验失败：$name" }
                }
                val cases = manifest.getJSONArray("cases")
                for (i in 0 until cases.length()) {
                    val name = cases.getJSONObject(i).getString("name")
                    val folder = File(root, name).apply { mkdirs() }
                    val inputs = cases.getJSONObject(i).optJSONObject("inputs") ?: JSONObject()
                        .put("sample", "sample.raw").put("lora_alpha", "$name.alpha.raw")
                    val entries = inputs.keys().asSequence().toList().sorted().map { key ->
                        require(Regex("[A-Za-z_][A-Za-z_0-9]*").matches(key))
                        val filename = inputs.getString(key)
                        require(!filename.contains('/') && !filename.contains('\\') && filename != "..")
                        "$key:=${File(assets, filename).absolutePath}"
                    }
                    File(folder, "inputs.txt").writeText(entries.joinToString(" ") + "\n")
                }
                val nativeDir = File(app.applicationInfo.nativeLibraryDir)
                report.put("preload", QnnInProcessBridgeServer(app, base, diag).configureAndPreload())
                lastMessage = (if (dynamic) "动态权重" else "适配器分段") + "：正在 NPU 上执行 7 组切换与恢复检查…"; callback(lastMessage)
                val output = File(root, "output").apply { mkdirs() }
                val raw = QnnInProcessNative.runLoraSequence(
                    File(nativeDir, "libQnnHtp.so").absolutePath,
                    File(nativeDir, "libQnnSystem.so").absolutePath,
                    File(assets, manifest.optString("context", "tiny.bin")).absolutePath,
                    File(root, "base0/inputs.txt").absolutePath,
                    output.absolutePath, false, false, root.absolutePath, !dynamic)
                val native = JSONObject(raw); report.put("native", native)
                require(native.optBoolean("ok")) { "${native.optString("stage")}: ${native.optString("detail")}" }
                val values = linkedMapOf<String, FloatArray>(); val checks = JSONArray(); var ok = true
                for (i in 0 until cases.length()) {
                    val spec = cases.getJSONObject(i); val name = spec.getString("name"); val count = spec.getInt("elements")
                    val actual = floats(File(output, "$name/Result_0/output.raw"), count)
                    val expected = floats(File(assets, "$name.expected.raw"), count)
                    val maxError = actual.indices.maxOf { abs(actual[it] - expected[it]).toDouble() }
                    val passed = matches(actual, expected, spec.getDouble("atol"), spec.getDouble("rtol"))
                    values[name] = actual; ok = ok && passed
                    checks.put(JSONObject().put("case", name).put("pass", passed).put("max_abs_error", maxError))
                }
                fun diff(a: String, b: String): Double = values.getValue(a).indices.maxOf { abs(values.getValue(a)[it] - values.getValue(b)[it]).toDouble() }
                val transitions = JSONObject()
                    .put("strength_changes_output", diff("original08", "original11") > 0.0001)
                    .put("adapter_changes_output", diff("original08", "changed08") > 0.0001)
                    .put("restore_original", diff("original08", "restored08") < 0.0005)
                    .put("zero_strength_restores_base", diff("base0", "restored0") < 0.0005)
                    .put("zero_adapter_restores_base", diff("base0", "zero11") < 0.0005)
                transitions.keys().forEach { ok = ok && transitions.getBoolean(it) }
                report.put("checks", checks).put("transitions", transitions).put("passed", ok)
                report.put("test_assets", manifest).put("work_dir", root.absolutePath)
                lastMessage = if (ok) "NPU 自测通过：7/7 数值检查，适配器切换与归零恢复均通过。\n这只证明底层接口，不代表任意 WAI LoRA 已适配。" else "NPU 执行完成，但数值或恢复检查未通过。请导出报告。"
            } catch (t: Throwable) {
                report.put("passed", false).put("error", "${t.javaClass.simpleName}: ${t.message}")
                lastMessage = "自测未通过：${t.message ?: t.javaClass.simpleName}"
            } finally {
                lastReport = report.toString(2)
                runCatching { diag?.let { File(it, "lora_selftest_latest.json").writeText(lastReport) } }
                runCatching { File(app.cacheDir, "lora_selftest_latest.json").writeText(lastReport) }
                runCatching { File(app.cacheDir, "lora_selftest_${route}.json").writeText(lastReport) }
                runCatching { diag?.let { File(it, "lora_selftest_${route}.json").writeText(lastReport) } }
                busy.set(false)
                callback(lastMessage)
            }
        }
        return true
    }
}
