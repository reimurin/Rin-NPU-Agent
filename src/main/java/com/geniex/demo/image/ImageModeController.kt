package com.geniex.demo.image

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.view.View
import android.widget.ArrayAdapter
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.core.widget.doAfterTextChanged
import com.geniex.demo.R
import com.geniex.demo.databinding.ActivityMainBinding
import java.io.File
import java.io.FileInputStream
import java.util.Locale

internal fun resolveImageGenerationSeed(
    intent: Intent,
    allowTestOverride: Boolean = false,
    now: () -> Long = { System.currentTimeMillis() },
): Long {
    val override = if (allowTestOverride) intent.getLongExtra(ImageModeController.EXTRA_TEST_SEED, -1L) else -1L
    return if (override in 0L..0x7fffffffL) override else (now() and 0x7fffffffL)
}

internal data class ImageTestAutomation(val prompt: String?, val autorun: Boolean)

internal fun resolveImageTestAutomation(intent: Intent): ImageTestAutomation {
    val prompt = intent.getStringExtra(ImageModeController.EXTRA_TEST_PROMPT)?.takeIf { it.isNotBlank() }
    return ImageTestAutomation(prompt, prompt != null && intent.getBooleanExtra(ImageModeController.EXTRA_TEST_AUTORUN, false))
}

class ImageModeController(
    private val activity: Activity,
    private val binding: ActivityMainBinding,
    private val onNeedMemoryForGeneration: ((() -> Unit) -> Unit),
) {
    enum class AppMode { CHAT, IMAGE }

    private val prefs = activity.getSharedPreferences("rin_image_ui", Activity.MODE_PRIVATE)
    private val presetStore = PromptPresetStore(activity)
    private val runtime = ImageGenerationRuntime(activity.applicationContext)
    private val installer = GitHubImageRuntimeInstaller(activity, runtime)
    private val generationListener: (ImageGenerationEvent) -> Unit = { event -> handleRuntimeEvent(event) }
    private var currentMode = runCatching {
        AppMode.valueOf(prefs.getString(KEY_MODE, AppMode.CHAT.name) ?: AppMode.CHAT.name)
    }.getOrDefault(AppMode.CHAT)
    private var lastGeneratedFile: File? = null
    private var lastRuntimeStatus: ImageRuntimeStatus? = null
    private var activeGenerationSeed: Long = -1L
    @Volatile private var autoContextRepairRequested = false
    private var resolutions = emptyList<ImageResolution>()
    private var uiActive = false

    fun setup() {
        setupSpinners()
        setupPromptState()
        setupListeners()
        val test = resolveImageTestAutomation(activity.intent)
        val automationSeed = if (test.autorun) resolveImageGenerationSeed(activity.intent, allowTestOverride = true) else null
        test.prompt?.let { binding.etImagePrompt.setText(it) }
        applyMode(if (test.autorun) AppMode.IMAGE else currentMode, persist = false)
        ImageGenerationSession.attach(generationListener, replay = false)
        ImageGenerationSession.currentRequest()?.let { activeGenerationSeed = it.seed }
        if (ImageGenerationSession.isRunning()) setGenerating(true)
        if (test.autorun) {
            activity.window.decorView.postDelayed({
                if (!activity.isFinishing && !activity.isDestroyed && !ImageGenerationSession.isRunning()) startGeneration(automationSeed)
                activity.intent.removeExtra(EXTRA_TEST_SEED)
                activity.intent.removeExtra(EXTRA_TEST_PROMPT)
                activity.intent.removeExtra(EXTRA_TEST_AUTORUN)
            }, 2500L)
        }
    }

    fun onResume() {
        uiActive = true
        refreshModelCenterSummary()
        renderSessionSnapshot(ImageGenerationSession.snapshot())
        if (currentMode == AppMode.IMAGE) refreshRuntimeStatus(false)
        if (prefs.getBoolean(KEY_BROWSER_PENDING, false) && !installer.isBusy()) resumeBrowserImport()
    }

    fun onPause() { uiActive = false }

    fun dispose() {
        uiActive = false
        ImageGenerationSession.detach(generationListener)
        installer.cancel()
        RuntimeDownloadForegroundService.stop(activity)
    }

    fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode == REQUEST_LORA_EDITOR || requestCode == REQUEST_MODEL_CENTER) {
            if (resultCode == Activity.RESULT_OK) {
                data?.getStringExtra(LoraLabActivity.EXTRA_PROMPT)?.let { text ->
                    runCatching { LoraTags.parse(text) }.onSuccess { binding.etImagePrompt.setText(text) }
                        .onFailure { Toast.makeText(activity,it.message ?: "LoRA 标签无效",Toast.LENGTH_LONG).show() }
                }
            }
            return true
        }

        if (requestCode != REQUEST_RUNTIME_DOWNLOAD_DIR) return false
        if (resultCode == Activity.RESULT_OK) {
            val uri = data?.data
            if (uri != null) {
                val readFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
                runCatching { activity.contentResolver.takePersistableUriPermission(uri, readFlags) }
                prefs.edit().putString(KEY_BROWSER_TREE_URI, uri.toString()).putBoolean(KEY_BROWSER_PENDING, true).apply()
                openBrowserReleasePage()
            }
        }
        return true
    }

    private fun setupSpinners() {
        updateResolutionChoices(emptyList())
        binding.spImageResolution.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                resolutions.getOrNull(position)?.let { prefs.edit().putString(KEY_RESOLUTION, it.key).apply() }
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }
        binding.spImageProfile.adapter = ArrayAdapter(
            activity,
            android.R.layout.simple_spinner_item,
            listOf(activity.getString(R.string.profile_stable_8)),
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        binding.spImageProfile.isEnabled = false
    }

    private fun updateResolutionChoices(installed: List<ImageResolution>) {
        if (resolutions == installed && binding.spImageResolution.adapter != null) return
        val preferred = prefs.getString(KEY_RESOLUTION, "1024x1024")
        resolutions = installed
        val labels = if (installed.isEmpty()) listOf("等待检测已安装的原生尺寸") else installed.map { "${it.width} × ${it.height} · 原生" }
        binding.spImageResolution.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        binding.spImageResolution.setSelection(installed.indexOfFirst { it.key == preferred }.coerceAtLeast(0))
        binding.spImageResolution.isEnabled = installed.isNotEmpty() && !ImageGenerationSession.isRunning()
        if (installed.isNotEmpty() && installed.none { it.key == preferred } && preferred != "1024x1024") {
            Toast.makeText(activity, "原选择尺寸未安装；已显示可用的原生尺寸，其他尺寸需对应模型包。", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupPromptState() {
        val locked = prefs.getBoolean(KEY_NEGATIVE_LOCKED, false)
        binding.switchLockNegative.isChecked = locked
        binding.etImageNegativePrompt.isEnabled = !locked
        binding.switchLockNegative.setOnCheckedChangeListener { _, value ->
            prefs.edit().putBoolean(KEY_NEGATIVE_LOCKED, value).apply()
            binding.etImageNegativePrompt.isEnabled = !value
        }
        if (binding.etImageNegativePrompt.text.isNullOrBlank()) {
            runCatching {
                presetStore.defaultNegative()?.let {
                    binding.etImageNegativePrompt.setText(it.prompt)
                    presetStore.touchNegative(it.id)
                }
            }.onFailure { Toast.makeText(activity, it.message, Toast.LENGTH_LONG).show() }
        }
        binding.switchLivePreview.isChecked = prefs.getBoolean(KEY_LIVE_PREVIEW, true)
        binding.switchLivePreview.setOnCheckedChangeListener { _, value ->
            prefs.edit().putBoolean(KEY_LIVE_PREVIEW, value).apply()
        }
        binding.etImagePrompt.doAfterTextChanged { refreshModelCenterSummary() }
        refreshModelCenterSummary()
    }

    private fun setupListeners() {
        binding.btnModeSwitch.setOnClickListener { showModeDialog() }
        binding.btnPositivePresetSave.setOnClickListener { savePositivePreset() }
        binding.btnNegativePresetSave.setOnClickListener { saveNegativePreset() }
        binding.btnPositivePresetSelect.setOnClickListener { selectPositivePreset() }
        binding.btnNegativePresetSelect.setOnClickListener { selectNegativePreset() }
        binding.btnImageInstallRuntime.setOnClickListener {
            if (runtime.hasStorageAccess()) showRuntimeInstallOptions() else requestAllFilesAccess()
        }
        binding.btnImageCheckRuntime.setOnClickListener {
            if (runtime.hasStorageAccess()) refreshRuntimeStatus(true) else requestAllFilesAccess()
        }
        binding.btnImageModelCenter.setOnClickListener {
            if (ImageGenerationSession.isRunning()) {
                Toast.makeText(activity, "请等待当前生图完成后再打开模型管理", Toast.LENGTH_LONG).show()
            } else if (installer.isBusy()) {
                Toast.makeText(activity, "请先完成或暂停当前模型下载", Toast.LENGTH_LONG).show()
            } else {
                activity.startActivityForResult(
                    Intent(activity, ModelUpdateActivity::class.java)
                        .putExtra(ModelUpdateActivity.EXTRA_PROMPT, binding.etImagePrompt.text?.toString().orEmpty()),
                    REQUEST_MODEL_CENTER,
                )
            }
        }
        binding.btnImageGenerate.setOnClickListener { startGeneration() }
        binding.btnImageStop.setOnClickListener { ImageGenerationSession.stop(activity) }
        binding.btnImageSave.setOnClickListener { saveLastImageToGallery() }
    }

    private fun showModeDialog() {
        val items = arrayOf(activity.getString(R.string.mode_chat), activity.getString(R.string.mode_image))
        RinControls.dialog(activity)
            .setTitle(R.string.select_mode)
            .setSingleChoiceItems(items, if (currentMode == AppMode.CHAT) 0 else 1) { dialog, which ->
                applyMode(if (which == 0) AppMode.CHAT else AppMode.IMAGE)
                dialog.dismiss()
                binding.drawerLayout.closeDrawer(GravityCompat.START)
            }
            .show()
    }

    private fun applyMode(mode: AppMode, persist: Boolean = true) {
        currentMode = mode
        if (persist) prefs.edit().putString(KEY_MODE, mode.name).apply()
        val image = mode == AppMode.IMAGE
        binding.chatModeContainer.visibility = if (image) View.GONE else View.VISIBLE
        binding.imageModeContainer.visibility = if (image) View.VISIBLE else View.GONE
        binding.drawerChatSettingsGroup.visibility = if (image) View.GONE else View.VISIBLE
        binding.drawerImageSettingsGroup.visibility = if (image) View.VISIBLE else View.GONE
        binding.btnClearHistory.visibility = if (image) View.GONE else View.VISIBLE
        binding.tvTopMode.setText(if (image) R.string.mode_image else R.string.mode_chat)
        binding.btnModeSwitch.setText(if (image) R.string.mode_image else R.string.mode_chat)
        binding.btnModeSwitch.icon = ContextCompat.getDrawable(activity, if (image) R.drawable.ic_image_mode_24 else R.drawable.ic_chat_mode_24)
        if (image) { refreshRuntimeStatus(false); refreshModelCenterSummary() }
    }

    private fun refreshModelCenterSummary() {
        val pack = runCatching { ModelPackUpdateManager(activity).currentPack() }.getOrNull()
        binding.tvImageModelStatusSummary.text = if (pack == null) {
            "基础模型：未安装 model-pack"
        } else {
            val sizes = pack.resolutions.joinToString(" / ") { "${it.width}×${it.height}" }
            "基础模型：${pack.version}" + if (sizes.isBlank()) "" else " · $sizes"
        }
        val selected = runCatching { LoraTags.parse(binding.etImagePrompt.text?.toString().orEmpty()).selections }.getOrDefault(emptyList())
        binding.tvImageLoraStatusSummary.text = if (selected.isEmpty()) {
            "LoRA：未启用"
        } else {
            "LoRA：" + selected.joinToString(" · ") { "${it.name} ${it.weight}" }
        }
    }
    private fun inspectOrReport(): ImageRuntimeStatus? = try {
        runtime.inspect()
    } catch (e:Exception) {
        StartupDiagnostics.record(activity,"ImageGenerationRuntime.inspect",e)
        activity.runOnUiThread {
            if (!activity.isFinishing && !activity.isDestroyed) {
                binding.tvImageRuntimeStatus.text="运行时暂不可访问：${e.message}"
                updateResolutionChoices(emptyList())
            }
        }
        null
    }

    private fun refreshRuntimeStatus(showToast: Boolean) {
        binding.tvImageRuntimeStatus.setText(R.string.runtime_checking)
        Thread({
            val status = inspectOrReport() ?: return@Thread
            activity.runOnUiThread {
                lastRuntimeStatus = status
                updateResolutionChoices(status.availableResolutions)
                binding.tvImageRuntimeStatus.text = runtimeStatusText(status)
                if (runtime.hasKnownBadClipG(status.baseDir) && !installer.isBusy() && !autoContextRepairRequested) {
                    autoContextRepairRequested = true
                    binding.tvImageRuntimeStatus.text = "检测到旧版 CLIP-G 上下文，正在自动修复…"
                    startAcceleratedRuntimeInstall()
                    return@runOnUiThread
                }
                binding.tvImageRuntimeStatus.setTextColor(
                    ContextCompat.getColor(activity, if (status.ready) R.color.rin_success else R.color.rin_warning),
                )
                binding.switchLivePreview.isEnabled = status.ready && status.previewSupported
                if (!status.previewSupported && binding.switchLivePreview.isChecked) binding.switchLivePreview.isChecked = false
                if (showToast) Toast.makeText(activity, runtimeStatusText(status), Toast.LENGTH_LONG).show()
            }
        }, "rin-image-runtime-check").start()
    }

    private fun showRuntimeInstallOptions() {
        val current = inspectOrReport() ?: return
        if (current.ready) {
            Toast.makeText(activity, R.string.image_runtime_ready, Toast.LENGTH_SHORT).show()
            return
        }
        val items = arrayOf(activity.getString(R.string.runtime_download_mode_app), activity.getString(R.string.runtime_download_mode_browser))
        RinControls.dialog(activity)
            .setTitle(R.string.runtime_download_mode_title)
            .setItems(items) { _, which -> if (which == 0) startAcceleratedRuntimeInstall() else startBrowserDownloadFlow() }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun startAcceleratedRuntimeInstall() {
        binding.btnImageInstallRuntime.isEnabled = false
        binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
        binding.imageRuntimeInstallProgress.isIndeterminate = true
        RuntimeDownloadForegroundService.start(activity)
        val started = installer.install(::handleInstallEvent)
        if (!started) {
            RuntimeDownloadForegroundService.stop(activity)
            binding.btnImageInstallRuntime.isEnabled = true
            Toast.makeText(activity, R.string.image_runtime_install_running, Toast.LENGTH_SHORT).show()
        }
    }

    private fun startBrowserDownloadFlow() {
        prefs.edit().putBoolean(KEY_BROWSER_PENDING, true).apply()
        openBrowserReleasePage()
    }

    private fun openBrowserReleasePage() {
        binding.tvImageRuntimeStatus.setText(R.string.runtime_browser_opened)
        runCatching { activity.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(GitHubImageRuntimeInstaller.RELEASE_PAGE_URL))) }
            .onFailure { Toast.makeText(activity, it.message ?: "Browser launch failed", Toast.LENGTH_LONG).show() }
    }

    private fun resumeBrowserImport() {
        if (!runtime.hasStorageAccess() || installer.isBusy()) return
        binding.btnImageInstallRuntime.isEnabled = false
        binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
        binding.imageRuntimeInstallProgress.isIndeterminate = true
        RuntimeDownloadForegroundService.start(activity, activity.getString(R.string.runtime_browser_scanning))
        val started = installer.importFromPublicDownloads(::handleInstallEvent)
        if (!started) {
            RuntimeDownloadForegroundService.stop(activity)
            binding.btnImageInstallRuntime.isEnabled = true
        }
    }

    private fun handleInstallEvent(event: ImageInstallEvent) {
        activity.runOnUiThread {
            when (event) {
                ImageInstallEvent.Checking -> {
                    binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_checking)
                    RuntimeDownloadForegroundService.update(activity, 0, binding.tvImageRuntimeStatus.text.toString())
                }
                is ImageInstallEvent.Unavailable -> {
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    binding.tvImageRuntimeStatus.text = event.message
                    RuntimeDownloadForegroundService.stop(activity)
                    Toast.makeText(activity, event.message, Toast.LENGTH_LONG).show()
                }
                is ImageInstallEvent.Downloading -> {
                    binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
                    binding.imageRuntimeInstallProgress.isIndeterminate = false
                    binding.imageRuntimeInstallProgress.setProgressCompat(event.percent, true)
                    val text = if (event.partCount > 0) {
                        activity.getString(
                            R.string.image_runtime_install_downloading_detail,
                            event.partIndex, event.partCount,
                            formatBytes(event.partDone), formatBytes(event.partTotal),
                            formatSpeed(event.bytesPerSecond), formatEta(event.etaSeconds), event.threads,
                        )
                    } else {
                        activity.getString(R.string.image_runtime_install_downloading, event.percent)
                    }
                    binding.tvImageRuntimeStatus.text = text
                    RuntimeDownloadForegroundService.update(activity, event.percent, text)
                }
                is ImageInstallEvent.BrowserWaiting -> {
                    val percent = if (event.total > 0L) ((event.done * 100L) / event.total).toInt().coerceIn(0, 100) else 0
                    binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
                    binding.imageRuntimeInstallProgress.isIndeterminate = false
                    binding.imageRuntimeInstallProgress.setProgressCompat(percent, true)
                    binding.tvImageRuntimeStatus.text = activity.getString(
                        R.string.runtime_browser_waiting,
                        event.foundParts, event.totalParts, formatBytes(event.done), formatBytes(event.total),
                    )
                    binding.btnImageInstallRuntime.isEnabled = true
                    RuntimeDownloadForegroundService.stop(activity)
                }
                ImageInstallEvent.Verifying -> {
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_verifying)
                    RuntimeDownloadForegroundService.update(activity, binding.imageRuntimeInstallProgress.progress, binding.tvImageRuntimeStatus.text.toString())
                }
                ImageInstallEvent.Extracting -> {
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_extracting)
                    RuntimeDownloadForegroundService.update(activity, 100, binding.tvImageRuntimeStatus.text.toString())
                }
                is ImageInstallEvent.Complete -> {
                    autoContextRepairRequested = false
                    prefs.edit().putBoolean(KEY_BROWSER_PENDING, false).apply()
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    RuntimeDownloadForegroundService.stop(activity)
                    Toast.makeText(activity, activity.getString(R.string.image_runtime_install_complete, event.version), Toast.LENGTH_LONG).show()
                    refreshRuntimeStatus(false)
                }
                is ImageInstallEvent.Failure -> {
                    autoContextRepairRequested = false
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    binding.tvImageRuntimeStatus.text = activity.getString(R.string.image_runtime_install_failed, event.message)
                    RuntimeDownloadForegroundService.stop(activity)
                    Toast.makeText(activity, binding.tvImageRuntimeStatus.text, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun formatBytes(bytes: Long): String {
        val value = bytes.coerceAtLeast(0L).toDouble()
        return when {
            value >= 1073741824.0 -> String.format(Locale.US, "%.2f GiB", value / 1073741824.0)
            value >= 1048576.0 -> String.format(Locale.US, "%.1f MiB", value / 1048576.0)
            value >= 1024.0 -> String.format(Locale.US, "%.1f KiB", value / 1024.0)
            else -> "${bytes.coerceAtLeast(0L)} B"
        }
    }

    private fun formatSpeed(bytesPerSecond: Long): String = if (bytesPerSecond > 0L) formatBytes(bytesPerSecond) + "/s" else "—"

    private fun formatEta(seconds: Long): String = when {
        seconds < 0L -> "—"
        seconds < 60L -> "${seconds}s"
        seconds < 3600L -> String.format(Locale.US, "%d:%02d", seconds / 60L, seconds % 60L)
        else -> String.format(Locale.US, "%d:%02d:%02d", seconds / 3600L, (seconds % 3600L) / 60L, seconds % 60L)
    }

    private fun runtimeStatusText(status: ImageRuntimeStatus): String = when {
        !status.storagePermission -> activity.getString(R.string.runtime_permission_needed)
        status.pythonPath == null -> activity.getString(R.string.runtime_python_missing)
        !status.pythonDependenciesOk -> status.detail ?: (activity.getString(R.string.runtime_python_missing) + " · numpy/Pillow")
        status.missingCommonFiles.isNotEmpty() || status.availableResolutions.isEmpty() ->
            activity.getString(R.string.runtime_context_missing) + " · " + status.missingCommonFiles.take(3).joinToString(", ")
        else -> activity.getString(R.string.image_runtime_ready) + " · " + status.availableResolutions.joinToString { it.key } + "（原生；其他尺寸需要对应模型包）" +
            if (status.previewSupported) " · ${activity.getString(R.string.runtime_preview_enabled)}" else " · ${activity.getString(R.string.runtime_preview_disabled)}"
    }

    private fun requestAllFilesAccess() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        val appIntent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${activity.packageName}"))
        runCatching { activity.startActivity(appIntent) }
            .onFailure { activity.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
    }

    private fun startGeneration(seedOverride: Long? = null) {
        val prompt = binding.etImagePrompt.text?.toString().orEmpty().trim()
        if (prompt.isEmpty()) {
            binding.etImagePrompt.error = activity.getString(R.string.positive_prompt)
            return
        }
        if (!runtime.hasStorageAccess()) {
            requestAllFilesAccess()
            return
        }
        if (ImageGenerationSession.isRunning() || LoraSelfTest.busy.get()) {
            Toast.makeText(activity,"已有生图或 LoRA 自测正在执行，请稍候",Toast.LENGTH_SHORT).show();return
        }
        val resolution = resolutions.getOrNull(binding.spImageResolution.selectedItemPosition)
        if (resolution == null) { refreshRuntimeStatus(true);return }
        val negative = binding.etImageNegativePrompt.text?.toString().orEmpty()
        Thread({
            val status = inspectOrReport() ?: return@Thread
            activity.runOnUiThread {
                lastRuntimeStatus = status
                updateResolutionChoices(status.availableResolutions)
                binding.tvImageRuntimeStatus.text = runtimeStatusText(status)
                if (!status.ready) {
                    Toast.makeText(activity, R.string.image_mode_requires_runtime, Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                if (!status.supports(resolution)) {
                    Toast.makeText(activity, "${resolution.key} 尚未安装对应的原生模型。请从已安装尺寸中选择。", Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                onNeedMemoryForGeneration {
                    lastGeneratedFile = null
                    binding.ivImageGenerationPreview.setImageDrawable(null)
                    binding.imagePreviewPlaceholder.visibility = View.VISIBLE
                    binding.imageGenerationProgress.progress = 0
                    binding.tvImageGenerationTiming.visibility = View.GONE
                    setGenerating(true)
                    val generationSeed = seedOverride ?: resolveImageGenerationSeed(activity.intent)
                    activeGenerationSeed = generationSeed
                    val started = ImageGenerationSession.start(
                        activity,
                        ImageGenerationRequest(
                            prompt = prompt,
                            negativePrompt = negative,
                            resolution = resolution,
                            steps = 8,
                            cfg = 3.5f,
                            seed = generationSeed,
                            livePreview = binding.switchLivePreview.isChecked && status.previewSupported,
                            progressiveCfg = true,
                        ),
                    )
                    if (!started) { activeGenerationSeed = -1L; setGenerating(false) }
                }
            }
        }, "rin-image-preflight").start()
    }

    private fun renderSessionSnapshot(snapshot: ImageGenerationSessionSnapshot) {
        if (!uiActive) return
        snapshot.progress?.let { handleRuntimeEvent(it, replay = true) }
        snapshot.latestPreview?.let { handleRuntimeEvent(it, replay = true) }
        snapshot.warning?.let { handleRuntimeEvent(it, replay = true) }
        snapshot.terminal?.let { handleRuntimeEvent(it, replay = true) }
        if (snapshot.running) {
            activeGenerationSeed = snapshot.request?.seed ?: activeGenerationSeed
            setGenerating(true)
        }
    }

    private fun handleRuntimeEvent(event: ImageGenerationEvent, replay: Boolean = false) {
        if (!uiActive) return
        activity.runOnUiThread {
            when (event) {
                is ImageGenerationEvent.Progress -> {
                    if (event.stage != ImageGenerationEvent.Stage.COMPLETE) setGenerating(true)
                    activeGenerationSeed = ImageGenerationSession.currentRequest()?.seed ?: activeGenerationSeed
                    binding.imageGenerationProgress.setProgressCompat(event.percent.coerceIn(0, 100), true)
                    when (event.stage) {
                        ImageGenerationEvent.Stage.PREPARING -> binding.tvImageGenerationStatus.setText(R.string.runtime_stage_start)
                        ImageGenerationEvent.Stage.CLIP -> binding.tvImageGenerationStatus.setText(R.string.runtime_stage_clip)
                        ImageGenerationEvent.Stage.DENOISING -> {
                            binding.tvImageGenerationStatus.text = activity.getString(R.string.runtime_stage_unet, event.step, event.totalSteps)
                            binding.tvImageGenerationStep.text = "${event.step}/${event.totalSteps}"
                        }
                        ImageGenerationEvent.Stage.VAE -> binding.tvImageGenerationStatus.setText(R.string.runtime_stage_vae)
                        ImageGenerationEvent.Stage.SAVING -> binding.tvImageGenerationStatus.setText(R.string.runtime_stage_saved)
                        ImageGenerationEvent.Stage.COMPLETE -> binding.tvImageGenerationStatus.setText(R.string.image_status_idle)
                    }
                }
                is ImageGenerationEvent.Preview -> showImage(event.file)
                is ImageGenerationEvent.Warning -> {
                    binding.tvImageGenerationTiming.visibility = View.VISIBLE
                    binding.tvImageGenerationTiming.text = event.message
                }
                is ImageGenerationEvent.Complete -> {
                    lastGeneratedFile = event.file
                    showImage(event.file)
                    activeGenerationSeed = -1L
                    binding.imageGenerationProgress.setProgressCompat(100, true)
                    binding.tvImageGenerationStep.text = "8/8"
                    binding.tvImageGenerationStatus.setText(R.string.image_status_idle)
                    event.totalSeconds?.let {
                        binding.tvImageGenerationTiming.visibility = View.VISIBLE
                        binding.tvImageGenerationTiming.text = activity.getString(
                            R.string.runtime_generation_complete,
                            String.format(Locale.US, "%.1f", it),
                        )
                    }
                    setGenerating(false)
                    binding.btnImageSave.visibility = View.VISIBLE
                }
                is ImageGenerationEvent.Failure -> {
                    activeGenerationSeed = -1L
                    setGenerating(false)
                    binding.tvImageGenerationTiming.visibility = View.VISIBLE
                    binding.tvImageGenerationTiming.text = activity.getString(R.string.runtime_generate_failed, event.message)
                    if (!replay) Toast.makeText(activity, activity.getString(R.string.runtime_generate_failed, event.message), Toast.LENGTH_LONG).show()
                }
                ImageGenerationEvent.Cancelled -> {
                    activeGenerationSeed = -1L
                    setGenerating(false)
                    binding.tvImageGenerationStatus.text = activity.getString(R.string.image_status_idle)
                    binding.tvImageGenerationTiming.visibility = View.VISIBLE
                    binding.tvImageGenerationTiming.text = activity.getString(R.string.image_generation_stopped)
                }
            }
        }
    }

    private fun showImage(file: File) {
        if (!file.isFile || file.length() == 0L) return
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sample = 1
        while (bounds.outWidth / sample > 1200 || bounds.outHeight / sample > 1200) sample *= 2
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return
        binding.ivImageGenerationPreview.setImageBitmap(bitmap)
        binding.imagePreviewPlaceholder.visibility = View.GONE
    }

    private fun setGenerating(value: Boolean) {
        binding.btnImageGenerate.isEnabled = !value
        binding.btnImageStop.visibility = if (value) View.VISIBLE else View.GONE
        if (value) binding.btnImageSave.visibility = View.GONE
        binding.spImageResolution.isEnabled = !value && resolutions.isNotEmpty()
        binding.switchLivePreview.isEnabled = !value && lastRuntimeStatus?.previewSupported == true
        binding.btnPositivePresetSave.isEnabled = !value
        binding.btnNegativePresetSave.isEnabled = !value
    }

    private fun saveLastImageToGallery() {
        val source = lastGeneratedFile?.takeIf { it.isFile } ?: return
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "Rin_${System.currentTimeMillis()}.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Rin NPU Agent")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        runCatching {
            val uri = activity.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("MediaStore insert failed")
            activity.contentResolver.openOutputStream(uri)?.use { out ->
                FileInputStream(source).use { input -> input.copyTo(out) }
            } ?: error("Could not open MediaStore output")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                activity.contentResolver.update(uri, values, null, null)
            }
        }.onSuccess { Toast.makeText(activity, R.string.image_saved_gallery, Toast.LENGTH_SHORT).show() }
            .onFailure { Toast.makeText(activity, R.string.image_save_failed, Toast.LENGTH_LONG).show() }
    }

    private fun savePositivePreset() {
        val prompt = binding.etImagePrompt.text?.toString().orEmpty().trim()
        if (prompt.isEmpty()) return
        val name = EditText(activity).apply { hint = activity.getString(R.string.preset_name); setBackgroundResource(R.drawable.rin_bg_control) }
        val note = EditText(activity).apply { hint = activity.getString(R.string.preset_note_hint); minLines = 2; setBackgroundResource(R.drawable.rin_bg_control) }
        val root = editorLayout(name, note, null)
        RinControls.dialog(activity).setTitle(R.string.positive_preset_title).setView(root)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.save_positive_preset) { _, _ ->
                runCatching { presetStore.savePositive(name = name.text.toString(), note = note.text.toString(), prompt = prompt) }
                    .onSuccess { Toast.makeText(activity, R.string.preset_saved, Toast.LENGTH_SHORT).show() }
                    .onFailure { Toast.makeText(activity, it.message, Toast.LENGTH_LONG).show() }
            }.show()
    }

    private fun saveNegativePreset() {
        val prompt = binding.etImageNegativePrompt.text?.toString().orEmpty().trim()
        if (prompt.isEmpty()) return
        val name = EditText(activity).apply { hint = activity.getString(R.string.preset_name); setBackgroundResource(R.drawable.rin_bg_control) }
        val note = EditText(activity).apply { hint = activity.getString(R.string.preset_note_hint); minLines = 2; setBackgroundResource(R.drawable.rin_bg_control) }
        val makeDefault = CheckBox(activity).apply { text = activity.getString(R.string.set_default_negative) }
        val root = editorLayout(name, note, makeDefault)
        RinControls.dialog(activity).setTitle(R.string.negative_preset_title).setView(root)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.save_negative_preset) { _, _ ->
                runCatching {
                    presetStore.saveNegative(
                        name = name.text.toString(),
                        note = note.text.toString(),
                        prompt = prompt,
                        makeDefault = makeDefault.isChecked,
                    )
                }.onSuccess { Toast.makeText(activity, R.string.preset_saved, Toast.LENGTH_SHORT).show() }
                    .onFailure { Toast.makeText(activity, it.message, Toast.LENGTH_LONG).show() }
            }.show()
    }

    private fun editorLayout(name: EditText, note: EditText, extra: View?): LinearLayout = LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(20), dp(10), dp(20), 0)
        addView(name, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)))
        addView(note, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(10) })
        extra?.let { addView(it, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) }) }
    }

    private fun selectPositivePreset() {
        PresetManagerDialog(activity,presetStore,false) { item -> binding.etImagePrompt.setText(item.prompt) }.show()
    }
    private fun selectNegativePreset() {
        PresetManagerDialog(activity,presetStore,true) { item ->
            if (!binding.switchLockNegative.isChecked) binding.etImageNegativePrompt.setText(item.prompt)
            else Toast.makeText(activity,"负面提示词已锁定，请解锁后应用预设",Toast.LENGTH_SHORT).show()
        }.show()
    }

    private fun dp(value: Int): Int = (value * activity.resources.displayMetrics.density).toInt()

    companion object {
        const val REQUEST_LORA_EDITOR = 16603
        const val REQUEST_MODEL_CENTER = 16604
        const val EXTRA_TEST_SEED = "rin_test_seed"
        const val EXTRA_TEST_PROMPT = "rin_test_prompt"
        const val EXTRA_TEST_AUTORUN = "rin_test_autorun"
        private const val KEY_MODE = "mode"
        private const val KEY_RESOLUTION = "resolution"
        private const val KEY_NEGATIVE_LOCKED = "negative_locked"
        private const val KEY_LIVE_PREVIEW = "live_preview"
        private const val KEY_BROWSER_TREE_URI = "browser_runtime_tree_uri"
        private const val KEY_BROWSER_PENDING = "browser_runtime_pending"
        const val REQUEST_RUNTIME_DOWNLOAD_DIR = 3302
    }
}

