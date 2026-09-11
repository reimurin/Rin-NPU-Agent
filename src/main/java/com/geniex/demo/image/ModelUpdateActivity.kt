package com.geniex.demo.image

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.text.format.DateFormat
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.geniex.demo.BuildConfig
import com.geniex.demo.R
import com.geniex.demo.databinding.ActivityModelUpdateBinding
import com.gyf.immersionbar.ktx.immersionBar
import java.io.File
import java.util.Locale

class ModelUpdateActivity : AppCompatActivity() {
    private lateinit var binding: ActivityModelUpdateBinding
    private lateinit var manager: ModelPackUpdateManager
    private var pendingRelease: ModelPackRelease? = null
    private var sourceSpinnerReady = false
    private var autoInstallAfterCheck = false
    private var installInProgress = false
    private val imageRuntime by lazy { ImageGenerationRuntime(applicationContext) }
    private val runtimeInstaller by lazy { GitHubImageRuntimeInstaller(this, imageRuntime) }
    private var currentPrompt = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityModelUpdateBinding.inflate(layoutInflater)
        setContentView(binding.root)
        immersionBar { statusBarColorInt(getColor(R.color.rin_bg)); statusBarDarkFont(true) }
        manager = ModelPackUpdateManager(this)
        autoInstallAfterCheck = intent.getBooleanExtra(EXTRA_AUTO_INSTALL, false)
        currentPrompt = intent.getStringExtra(EXTRA_PROMPT).orEmpty()
        setupUi()
        renderLocal()
        renderLastCheck()
        if (!intent.getBooleanExtra(EXTRA_SKIP_AUTO_CHECK, false)) checkNow()
    }

    private fun setupUi() {
        binding.btnUpdateBack.setOnClickListener { finish() }
        binding.tvUpdateAppVersion.text = getString(R.string.model_update_app_version, BuildConfig.VERSION_NAME)
        binding.tvUpdateDevice.text = manager.deviceTarget().label

        val labels = listOf(
            getString(R.string.model_source_auto),
            getString(R.string.model_source_github),
            getString(R.string.model_source_china),
        )
        binding.spUpdateSource.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val values = ModelSourcePreference.entries
        binding.spUpdateSource.setSelection(values.indexOf(manager.sourcePreference()).coerceAtLeast(0), false)
        binding.spUpdateSource.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!sourceSpinnerReady) { sourceSpinnerReady = true; return }
                values.getOrNull(position)?.let(manager::setSourcePreference)
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }
        binding.btnUpdateCheck.setOnClickListener { checkNow() }
        binding.btnUpdateInstall.setOnClickListener { pendingRelease?.let(::startInstall) }
        binding.btnUpdateIgnore.setOnClickListener {
            pendingRelease?.let {
                manager.ignoreVersion(it)
                binding.tvUpdateStatus.text = getString(R.string.model_update_ignored, it.version)
                binding.btnUpdateIgnore.visibility = View.GONE
            }
        }
        binding.btnUpdateCancel.setOnClickListener {
            val wasInstall = installInProgress
            manager.cancel()
            installInProgress = false
            RuntimeDownloadForegroundService.stop(this)
            setBusy(false)
            binding.tvUpdateStatus.setText(if (wasInstall) R.string.model_update_cancelled else R.string.model_update_check_cancelled)
        }
        binding.btnModelCenterLora.setOnClickListener {
            if (manager.isBusy() || runtimeInstaller.isBusy() || LoraSelfTest.busy.get()) {
                Toast.makeText(this, "请等待当前模型任务完成", Toast.LENGTH_SHORT).show()
            } else {
                startActivityForResult(
                    Intent(this, LoraLabActivity::class.java).putExtra(LoraLabActivity.EXTRA_PROMPT, currentPrompt),
                    REQUEST_LORA_EDITOR,
                )
            }
        }
        binding.btnModelCenterRuntimeCheck.setOnClickListener { inspectImageRuntime(showToast = true) }
        binding.btnModelCenterRuntimeRepair.setOnClickListener { startRuntimeRepair() }
        binding.btnModelCenterRuntimeCancel.setOnClickListener {
            runtimeInstaller.cancel()
            RuntimeDownloadForegroundService.stop(this)
            setRuntimeBusy(false)
            binding.tvModelCenterAdvancedStatus.text = "基础运行时下载已暂停；已下载分卷会保留用于续传。"
        }
        binding.btnModelCenterNpuTest.setOnClickListener {
            if (manager.isBusy() || runtimeInstaller.isBusy()) {
                Toast.makeText(this, "请先完成当前模型下载或检查", Toast.LENGTH_SHORT).show()
            } else if (!LoraSelfTest.busy.compareAndSet(false, false)) {
                Toast.makeText(this, "NPU 自测正在运行", Toast.LENGTH_SHORT).show()
            } else {
                binding.tvModelCenterAdvancedStatus.text = "正在运行动态 LoRA NPU 自测…"
                LoraSelfTest.start(this) { message -> runOnUiThread {
                    if (!isFinishing && !isDestroyed) binding.tvModelCenterAdvancedStatus.text = message
                } }
            }
        }
        binding.btnModelCenterShareNpuReport.setOnClickListener { shareDiagnostic(File(cacheDir, "lora_selftest_latest.json"), "NPU 自测报告") }
        binding.btnModelCenterShareLoraReport.setOnClickListener {
            shareDiagnostic(File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "sdxl_qnn/.rin_diagnostics/lora_generation_latest.json"), "LoRA 生图诊断")
        }
        binding.btnModelCenterShareDiagnostics.setOnClickListener { StartupDiagnostics.share(this) }
        renderLoraSummary()
        inspectImageRuntime(showToast = false)
    }

    private fun renderLoraSummary() {
        val selections = runCatching { LoraTags.parse(currentPrompt).selections }.getOrDefault(emptyList())
        binding.tvModelCenterLoraStatus.text = if (selections.isEmpty()) {
            "当前提示词未启用 LoRA。"
        } else {
            "当前：" + selections.joinToString(" · ") { "${it.name} ${it.weight}" }
        }
    }

    private fun inspectImageRuntime(showToast: Boolean) {
        binding.tvModelCenterAdvancedStatus.text = "正在检查基础运行时完整性…"
        Thread({
            val result = runCatching { imageRuntime.inspect() }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                result.onSuccess { status ->
                    val detail = when {
                        status.ready -> "基础运行时完整 · 原生尺寸：" + status.availableResolutions.joinToString { it.key } + if (status.previewSupported) " · 实时预览可用" else " · 实时预览不可用"
                        !status.storagePermission -> "需要文件访问权限后才能检查基础运行时。"
                        status.pythonPath == null || !status.pythonDependenciesOk -> status.detail ?: "Python / NumPy / Pillow 运行时不完整。"
                        else -> "基础运行时不完整：" + status.missingCommonFiles.take(4).joinToString(", ")
                    }
                    binding.tvModelCenterAdvancedStatus.text = detail
                    if (showToast) Toast.makeText(this, detail, Toast.LENGTH_LONG).show()
                }.onFailure { error ->
                    StartupDiagnostics.record(this, "ModelUpdateActivity.inspectImageRuntime", error)
                    val detail = "运行时检查失败：${error.message ?: error.javaClass.simpleName}"
                    binding.tvModelCenterAdvancedStatus.text = detail
                    if (showToast) Toast.makeText(this, detail, Toast.LENGTH_LONG).show()
                }
            }
        }, "rin-model-center-runtime-check").start()
    }

    private fun requestStorageAccessForRuntime() {
        runCatching {
            startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName")))
        }.onFailure { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
    }

    private fun startRuntimeRepair() {
        if (runtimeInstaller.isBusy()) return
        if (!imageRuntime.hasStorageAccess()) {
            requestStorageAccessForRuntime()
            binding.tvModelCenterAdvancedStatus.text = "授予文件访问权限后再次点击修复 / 安装。"
            return
        }
        setRuntimeBusy(true)
        RuntimeDownloadForegroundService.start(this, "正在修复基础运行时")
        val started = runtimeInstaller.install(::handleRuntimeInstallEvent)
        if (!started) {
            RuntimeDownloadForegroundService.stop(this)
            setRuntimeBusy(false)
            Toast.makeText(this, "已有基础运行时任务正在执行", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleRuntimeInstallEvent(event: ImageInstallEvent) {
        runOnUiThread {
            when (event) {
                ImageInstallEvent.Checking -> {
                    binding.progressModelCenterRuntime.visibility = View.VISIBLE
                    binding.progressModelCenterRuntime.isIndeterminate = true
                    binding.tvModelCenterAdvancedStatus.text = "正在检查基础运行时…"
                }
                is ImageInstallEvent.Unavailable -> {
                    setRuntimeBusy(false); RuntimeDownloadForegroundService.stop(this)
                    binding.tvModelCenterAdvancedStatus.text = event.message
                }
                is ImageInstallEvent.Downloading -> {
                    binding.progressModelCenterRuntime.visibility = View.VISIBLE
                    binding.progressModelCenterRuntime.isIndeterminate = false
                    binding.progressModelCenterRuntime.setProgressCompat(event.percent.coerceIn(0, 100), true)
                    binding.tvModelCenterAdvancedStatus.text = "下载 ${event.partIndex}/${event.partCount} · ${event.percent}% · ${formatSpeed(event.bytesPerSecond)}"
                    RuntimeDownloadForegroundService.update(this, event.percent, binding.tvModelCenterAdvancedStatus.text.toString())
                }
                is ImageInstallEvent.BrowserWaiting -> binding.tvModelCenterAdvancedStatus.text = "等待浏览器下载的运行时文件…"
                ImageInstallEvent.Verifying -> binding.tvModelCenterAdvancedStatus.text = "正在校验基础运行时…"
                ImageInstallEvent.Extracting -> binding.tvModelCenterAdvancedStatus.text = "正在安装基础运行时…"
                is ImageInstallEvent.Complete -> {
                    setRuntimeBusy(false); RuntimeDownloadForegroundService.stop(this)
                    binding.progressModelCenterRuntime.visibility = View.GONE
                    binding.tvModelCenterAdvancedStatus.text = "基础运行时已修复 / 安装完成：${event.version}"
                    inspectImageRuntime(showToast = false)
                }
                is ImageInstallEvent.Failure -> {
                    setRuntimeBusy(false); RuntimeDownloadForegroundService.stop(this)
                    binding.progressModelCenterRuntime.visibility = View.GONE
                    binding.tvModelCenterAdvancedStatus.text = "基础运行时修复失败：${event.message}"
                }
            }
        }
    }

    private fun setRuntimeBusy(value: Boolean) {
        binding.btnModelCenterRuntimeCheck.isEnabled = !value
        binding.btnModelCenterRuntimeRepair.isEnabled = !value
        binding.btnModelCenterNpuTest.isEnabled = !value
        binding.btnModelCenterRuntimeCancel.visibility = if (value) View.VISIBLE else View.GONE
        if (value) {
            binding.progressModelCenterRuntime.visibility = View.VISIBLE
            binding.progressModelCenterRuntime.isIndeterminate = true
        }
    }

    private fun shareDiagnostic(file: File, title: String) {
        if (!file.isFile) {
            Toast.makeText(this, "暂无$title", Toast.LENGTH_SHORT).show()
            return
        }
        runCatching {
            val exported = File(cacheDir, "model_center_reports/${file.name}").apply { parentFile?.mkdirs() }
            file.copyTo(exported, overwrite = true)
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", exported)
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }, "分享$title"))
        }.onFailure { Toast.makeText(this, it.message ?: "分享失败", Toast.LENGTH_LONG).show() }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_LORA_EDITOR && resultCode == RESULT_OK) {
            data?.getStringExtra(LoraLabActivity.EXTRA_PROMPT)?.let { currentPrompt = it }
            renderLoraSummary()
            setResult(RESULT_OK, Intent().putExtra(LoraLabActivity.EXTRA_PROMPT, currentPrompt))
        }
    }

    private fun renderLocal() {
        val local = manager.currentPack()
        binding.tvUpdateRemoteResolutions.text = ""
        binding.tvUpdateRemoteLora.text = ""
        binding.tvUpdateSize.text = ""
        if (local == null) {
            binding.tvUpdateLocal.setText(R.string.model_update_local_none)
            binding.tvUpdateResolutions.setText(R.string.model_update_resolution_none)
            binding.tvUpdateLora.setText(R.string.model_update_lora_unknown)
        } else {
            binding.tvUpdateLocal.text = getString(R.string.model_update_local_version, local.version)
            binding.tvUpdateResolutions.text = getString(R.string.model_update_resolutions, local.resolutions.joinToString { "${it.width} × ${it.height}" })
            binding.tvUpdateLora.text = getString(R.string.model_update_lora_abi, local.loraAbi.take(20) + if (local.loraAbi.length > 20) "…" else "")
        }
    }

    private fun renderLastCheck() {
        val at = manager.lastCheckAt()
        binding.tvUpdateLastCheck.text = if (at > 0L) getString(R.string.model_update_last_check, DateFormat.format("yyyy-MM-dd HH:mm", at))
        else getString(R.string.model_update_last_check_none)
    }

    private fun checkNow() {
        if (manager.isBusy()) return
        pendingRelease = null
        installInProgress = false
        binding.btnUpdateInstall.isEnabled = false
        binding.btnUpdateIgnore.visibility = View.GONE
        binding.tvUpdateRemote.setText(R.string.model_update_remote_checking)
        binding.tvUpdateStatus.setText(R.string.model_update_checking)
        binding.progressModelUpdate.visibility = View.VISIBLE
        binding.progressModelUpdate.isIndeterminate = true
        binding.btnUpdateCheck.isEnabled = false
        binding.spUpdateSource.isEnabled = false
        binding.btnUpdateCancel.visibility = View.VISIBLE
        val started = manager.checkAsync { result -> runOnUiThread { if (!isFinishing && !isDestroyed) renderCheck(result) } }
        if (!started) {
            binding.btnUpdateCheck.isEnabled = true
            binding.spUpdateSource.isEnabled = true
            binding.btnUpdateCancel.visibility = View.GONE
            binding.progressModelUpdate.visibility = View.GONE
        }
    }

    private fun renderCheck(result: ModelUpdateCheck) {
        binding.btnUpdateCheck.isEnabled = true
        binding.spUpdateSource.isEnabled = true
        binding.btnUpdateCancel.visibility = View.GONE
        binding.progressModelUpdate.visibility = View.GONE
        renderLastCheck()
        binding.tvUpdateDevice.text = result.device.label
        when (result) {
            is ModelUpdateCheck.UpToDate -> {
                pendingRelease = null
                binding.tvUpdateRemote.text = getString(R.string.model_update_remote_version, result.remote.version)
                binding.tvUpdateStatus.setText(R.string.model_update_up_to_date)
                binding.btnUpdateInstall.isEnabled = false
                binding.btnUpdateIgnore.visibility = View.GONE
                showRemoteCapabilities(result.remote, result.source)
            }
            is ModelUpdateCheck.UpdateAvailable -> {
                pendingRelease = result.remote
                binding.tvUpdateRemote.text = getString(R.string.model_update_remote_version, result.remote.version)
                val localVersion = result.local?.version ?: getString(R.string.model_update_not_installed)
                binding.tvUpdateStatus.text = getString(R.string.model_update_available, localVersion, result.remote.version)
                binding.btnUpdateInstall.isEnabled = true
                binding.btnUpdateIgnore.visibility = View.VISIBLE
                showRemoteCapabilities(result.remote, result.source)
                if (autoInstallAfterCheck) { autoInstallAfterCheck = false; startInstall(result.remote) }
            }
            is ModelUpdateCheck.Ignored -> {
                pendingRelease = result.remote
                binding.tvUpdateRemote.text = getString(R.string.model_update_remote_version, result.remote.version)
                binding.tvUpdateStatus.text = getString(R.string.model_update_ignored, result.remote.version)
                binding.btnUpdateInstall.isEnabled = true
                binding.btnUpdateIgnore.visibility = View.GONE
                showRemoteCapabilities(result.remote, result.source)
            }
            is ModelUpdateCheck.Cancelled -> {
                pendingRelease = null
                binding.tvUpdateRemote.setText(R.string.model_update_remote_cancelled)
                binding.tvUpdateStatus.setText(R.string.model_update_check_cancelled)
                binding.btnUpdateInstall.isEnabled = false
                binding.btnUpdateIgnore.visibility = View.GONE
            }
            is ModelUpdateCheck.AppUpdateRequired -> {
                pendingRelease = result.remote
                binding.tvUpdateRemote.text = getString(R.string.model_update_remote_version, result.remote.version)
                binding.tvUpdateStatus.text = getString(R.string.model_update_app_required, result.remote.version)
                binding.btnUpdateInstall.isEnabled = false
                binding.btnUpdateIgnore.visibility = View.VISIBLE
                showRemoteCapabilities(result.remote, result.source)
            }
            is ModelUpdateCheck.Unsupported -> {
                pendingRelease = null
                binding.tvUpdateRemote.setText(R.string.model_update_remote_none)
                binding.tvUpdateStatus.text = result.message
                binding.btnUpdateInstall.isEnabled = false
                binding.btnUpdateIgnore.visibility = View.GONE
            }
            is ModelUpdateCheck.Error -> {
                pendingRelease = null
                binding.tvUpdateRemote.setText(R.string.model_update_remote_unavailable)
                binding.tvUpdateStatus.text = getString(R.string.model_update_check_failed, result.message)
                binding.btnUpdateInstall.isEnabled = false
                binding.btnUpdateIgnore.visibility = View.GONE
            }
        }
    }

    private fun showRemoteCapabilities(release: ModelPackRelease, source: ModelSourcePreference) {
        binding.tvUpdateRemoteResolutions.text = getString(R.string.model_update_remote_resolutions, release.resolutions.joinToString { "${it.width} × ${it.height}" })
        binding.tvUpdateRemoteLora.text = getString(R.string.model_update_lora_rank, release.loraRank, release.loraAbi.take(20) + if (release.loraAbi.length > 20) "…" else "")
        binding.tvUpdateSize.text = if (release.totalDownloadBytes > 0L) getString(R.string.model_update_size, formatBytes(release.totalDownloadBytes)) else getString(R.string.model_update_size_unknown)
        binding.tvUpdateSourceUsed.text = getString(R.string.model_update_source_used, sourceLabel(source))
        val note = if (resources.configuration.locales[0].language.equals("zh", true)) release.notesZh else release.notesEn
        if (note.isNotBlank()) binding.tvUpdateStatus.text = binding.tvUpdateStatus.text.toString() + "\n" + note
    }

    private fun startInstall(release: ModelPackRelease) {
        if (manager.isBusy()) return
        installInProgress = true
        setBusy(true)
        binding.progressModelUpdate.visibility = View.VISIBLE
        binding.progressModelUpdate.isIndeterminate = true
        RuntimeDownloadForegroundService.start(this, getString(R.string.model_update_downloading))
        val started = manager.installAsync(release) { event -> runOnUiThread { if (!isDestroyed) handleInstallEvent(event) } }
        if (!started) {
            installInProgress = false
            RuntimeDownloadForegroundService.stop(this)
            setBusy(false)
        }
    }

    private fun handleInstallEvent(event: ModelInstallEvent) {
        when (event) {
            is ModelInstallEvent.Starting -> {
                binding.tvUpdateStatus.text = getString(R.string.model_update_starting, event.release.version)
            }
            is ModelInstallEvent.Downloading -> {
                val percent = if (event.total > 0L) ((event.done * 100L) / event.total).toInt().coerceIn(0, 100) else 0
                binding.progressModelUpdate.isIndeterminate = false
                binding.progressModelUpdate.setProgressCompat(percent, true)
                val text = getString(
                    R.string.model_update_download_progress,
                    event.index, event.count, event.name,
                    formatBytes(event.done), formatBytes(event.total), formatSpeed(event.speed), sourceLabel(event.source),
                )
                binding.tvUpdateStatus.text = text
                RuntimeDownloadForegroundService.update(this, percent, text)
            }
            is ModelInstallEvent.Verifying -> {
                binding.progressModelUpdate.isIndeterminate = true
                binding.tvUpdateStatus.text = getString(R.string.model_update_verifying, event.name)
            }
            is ModelInstallEvent.Switching -> {
                binding.progressModelUpdate.isIndeterminate = true
                binding.tvUpdateStatus.text = getString(R.string.model_update_switching, event.version)
            }
            is ModelInstallEvent.Complete -> {
                installInProgress = false
                RuntimeDownloadForegroundService.stop(this)
                setBusy(false)
                binding.progressModelUpdate.visibility = View.GONE
                pendingRelease = null
                renderLocal()
                binding.tvUpdateRemote.text = getString(R.string.model_update_remote_version, event.pack.version)
                binding.tvUpdateStatus.setText(R.string.model_update_complete)
                Toast.makeText(this, R.string.model_update_complete, Toast.LENGTH_LONG).show()
                setResult(RESULT_OK)
            }
            is ModelInstallEvent.Failure -> {
                installInProgress = false
                RuntimeDownloadForegroundService.stop(this)
                setBusy(false)
                binding.progressModelUpdate.visibility = View.GONE
                binding.tvUpdateStatus.text = getString(R.string.model_update_install_failed, event.message)
                Toast.makeText(this, binding.tvUpdateStatus.text, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setBusy(value: Boolean) {
        binding.btnUpdateCheck.isEnabled = !value
        binding.btnUpdateInstall.isEnabled = !value && pendingRelease != null
        binding.btnUpdateIgnore.isEnabled = !value
        binding.spUpdateSource.isEnabled = !value
        binding.btnUpdateCancel.visibility = if (value) View.VISIBLE else View.GONE
    }

    private fun sourceLabel(source: ModelSourcePreference): String = when (source) {
        ModelSourcePreference.AUTO -> getString(R.string.model_source_auto)
        ModelSourcePreference.GITHUB -> getString(R.string.model_source_github)
        ModelSourcePreference.CHINA_MIRROR -> getString(R.string.model_source_china)
    }

    private fun formatBytes(bytes: Long): String {
        val v = bytes.coerceAtLeast(0).toDouble()
        return when {
            v >= 1073741824.0 -> String.format(Locale.US, "%.2f GiB", v / 1073741824.0)
            v >= 1048576.0 -> String.format(Locale.US, "%.1f MiB", v / 1048576.0)
            v >= 1024.0 -> String.format(Locale.US, "%.1f KiB", v / 1024.0)
            else -> "${bytes.coerceAtLeast(0)} B"
        }
    }

    private fun formatSpeed(bytesPerSecond: Long): String = if (bytesPerSecond > 0L) formatBytes(bytesPerSecond) + "/s" else "—"

    override fun onDestroy() {
        if (isFinishing && manager.isBusy()) manager.cancel()
        if (isFinishing && runtimeInstaller.isBusy()) runtimeInstaller.cancel()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_AUTO_INSTALL = "rin_model_update_auto_install"
        const val EXTRA_PROMPT = LoraLabActivity.EXTRA_PROMPT
        private const val REQUEST_LORA_EDITOR = 16605
        const val EXTRA_SKIP_AUTO_CHECK = "rin_model_update_skip_auto_check"
    }
}
