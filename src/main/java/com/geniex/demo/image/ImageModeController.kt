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
import com.geniex.demo.R
import com.geniex.demo.databinding.ActivityMainBinding
import java.io.File
import java.io.FileInputStream
import java.util.Locale

class ImageModeController(
    private val activity: Activity,
    private val binding: ActivityMainBinding,
    private val onNeedMemoryForGeneration: ((() -> Unit) -> Unit),
) {
    enum class AppMode { CHAT, IMAGE }

    private val prefs = activity.getSharedPreferences("rin_image_ui", Activity.MODE_PRIVATE)
    private val presetStore = PromptPresetStore(activity)
    private val runtime = ImageGenerationRuntime(activity)
    private val installer = GitHubImageRuntimeInstaller(activity, runtime)
    private var currentMode = runCatching {
        AppMode.valueOf(prefs.getString(KEY_MODE, AppMode.CHAT.name) ?: AppMode.CHAT.name)
    }.getOrDefault(AppMode.CHAT)
    private var lastGeneratedFile: File? = null
    private var lastRuntimeStatus: ImageRuntimeStatus? = null
    private val resolutions = ImageGenerationRuntime.UI_RESOLUTIONS

    fun setup() {
        setupSpinners()
        setupPromptState()
        setupListeners()
        applyMode(currentMode, persist = false)
    }

    fun onResume() {
        if (currentMode == AppMode.IMAGE) refreshRuntimeStatus(false)
    }

    fun dispose() {
        installer.cancel()
        runtime.stop()
    }

    private fun setupSpinners() {
        val labels = listOf(
            activity.getString(R.string.preset_1024),
            activity.getString(R.string.preset_1216_832),
            activity.getString(R.string.preset_832_1216),
            activity.getString(R.string.preset_1344_768),
            activity.getString(R.string.preset_768_1344),
        )
        binding.spImageResolution.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val selected = resolutions.indexOfFirst { it.key == prefs.getString(KEY_RESOLUTION, "1024x1024") }.coerceAtLeast(0)
        binding.spImageResolution.setSelection(selected)
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

    private fun setupPromptState() {
        val locked = prefs.getBoolean(KEY_NEGATIVE_LOCKED, false)
        binding.switchLockNegative.isChecked = locked
        binding.etImageNegativePrompt.isEnabled = !locked
        binding.switchLockNegative.setOnCheckedChangeListener { _, value ->
            prefs.edit().putBoolean(KEY_NEGATIVE_LOCKED, value).apply()
            binding.etImageNegativePrompt.isEnabled = !value
        }
        if (binding.etImageNegativePrompt.text.isNullOrBlank()) {
            presetStore.defaultNegative()?.let {
                binding.etImageNegativePrompt.setText(it.prompt)
                presetStore.touchNegative(it.id)
            }
        }
        binding.switchLivePreview.isChecked = prefs.getBoolean(KEY_LIVE_PREVIEW, true)
        binding.switchLivePreview.setOnCheckedChangeListener { _, value ->
            prefs.edit().putBoolean(KEY_LIVE_PREVIEW, value).apply()
        }
    }

    private fun setupListeners() {
        binding.btnModeSwitch.setOnClickListener { showModeDialog() }
        binding.btnPositivePresetSave.setOnClickListener { savePositivePreset() }
        binding.btnNegativePresetSave.setOnClickListener { saveNegativePreset() }
        binding.btnPositivePresetSelect.setOnClickListener { selectPositivePreset() }
        binding.btnNegativePresetSelect.setOnClickListener { selectNegativePreset() }
        binding.btnImageInstallRuntime.setOnClickListener {
            if (runtime.hasStorageAccess()) startRuntimeInstall() else requestAllFilesAccess()
        }
        binding.btnImageCheckRuntime.setOnClickListener {
            if (runtime.hasStorageAccess()) refreshRuntimeStatus(true) else requestAllFilesAccess()
        }
        binding.btnImageGenerate.setOnClickListener { startGeneration() }
        binding.btnImageStop.setOnClickListener {
            runtime.stop()
            setGenerating(false)
            binding.tvImageGenerationStatus.text = activity.getString(R.string.image_status_idle)
        }
        binding.btnImageSave.setOnClickListener { saveLastImageToGallery() }
    }

    private fun showModeDialog() {
        val items = arrayOf(activity.getString(R.string.mode_chat), activity.getString(R.string.mode_image))
        AlertDialog.Builder(activity)
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
        if (image) refreshRuntimeStatus(false)
    }

    private fun refreshRuntimeStatus(showToast: Boolean) {
        binding.tvImageRuntimeStatus.setText(R.string.runtime_checking)
        Thread({
            val status = runtime.inspect()
            activity.runOnUiThread {
                lastRuntimeStatus = status
                binding.tvImageRuntimeStatus.text = runtimeStatusText(status)
                binding.tvImageRuntimeStatus.setTextColor(
                    ContextCompat.getColor(activity, if (status.ready) R.color.rin_success else R.color.rin_warning),
                )
                binding.switchLivePreview.isEnabled = status.ready && status.previewSupported
                if (!status.previewSupported && binding.switchLivePreview.isChecked) binding.switchLivePreview.isChecked = false
                if (showToast) Toast.makeText(activity, runtimeStatusText(status), Toast.LENGTH_LONG).show()
            }
        }, "rin-image-runtime-check").start()
    }

    private fun startRuntimeInstall() {
        val current = runtime.inspect()
        if (current.ready) {
            Toast.makeText(activity, R.string.image_runtime_ready, Toast.LENGTH_SHORT).show()
            return
        }
        binding.btnImageInstallRuntime.isEnabled = false
        binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
        binding.imageRuntimeInstallProgress.isIndeterminate = true
        val started = installer.install(::handleInstallEvent)
        if (!started) {
            binding.btnImageInstallRuntime.isEnabled = true
            Toast.makeText(activity, R.string.image_runtime_install_running, Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleInstallEvent(event: ImageInstallEvent) {
        activity.runOnUiThread {
            when (event) {
                ImageInstallEvent.Checking -> {
                    binding.imageRuntimeInstallProgress.visibility = View.VISIBLE
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_checking)
                }
                is ImageInstallEvent.Unavailable -> {
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    binding.tvImageRuntimeStatus.text = event.message
                    Toast.makeText(activity, event.message, Toast.LENGTH_LONG).show()
                }
                is ImageInstallEvent.Downloading -> {
                    binding.imageRuntimeInstallProgress.isIndeterminate = false
                    binding.imageRuntimeInstallProgress.setProgressCompat(event.percent, true)
                    binding.tvImageRuntimeStatus.text = activity.getString(R.string.image_runtime_install_downloading, event.percent)
                }
                ImageInstallEvent.Verifying -> {
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_verifying)
                }
                ImageInstallEvent.Extracting -> {
                    binding.imageRuntimeInstallProgress.isIndeterminate = true
                    binding.tvImageRuntimeStatus.setText(R.string.image_runtime_install_extracting)
                }
                is ImageInstallEvent.Complete -> {
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    Toast.makeText(activity, activity.getString(R.string.image_runtime_install_complete, event.version), Toast.LENGTH_LONG).show()
                    refreshRuntimeStatus(false)
                }
                is ImageInstallEvent.Failure -> {
                    binding.imageRuntimeInstallProgress.visibility = View.GONE
                    binding.btnImageInstallRuntime.isEnabled = true
                    binding.tvImageRuntimeStatus.text = activity.getString(R.string.image_runtime_install_failed, event.message)
                    Toast.makeText(activity, binding.tvImageRuntimeStatus.text, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun runtimeStatusText(status: ImageRuntimeStatus): String = when {
        !status.storagePermission -> activity.getString(R.string.runtime_permission_needed)
        status.pythonPath == null -> activity.getString(R.string.runtime_python_missing)
        !status.pythonDependenciesOk -> activity.getString(R.string.runtime_python_missing) + " · numpy/Pillow"
        status.missingCommonFiles.isNotEmpty() || status.availableResolutions.isEmpty() ->
            activity.getString(R.string.runtime_context_missing) + " · " + status.missingCommonFiles.take(3).joinToString(", ")
        else -> activity.getString(R.string.image_runtime_ready) + " · " + status.availableResolutions.joinToString { it.key } +
            if (status.previewSupported) " · ${activity.getString(R.string.runtime_preview_enabled)}" else " · ${activity.getString(R.string.runtime_preview_disabled)}"
    }

    private fun requestAllFilesAccess() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        val appIntent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${activity.packageName}"))
        runCatching { activity.startActivity(appIntent) }
            .onFailure { activity.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
    }

    private fun startGeneration() {
        val prompt = binding.etImagePrompt.text?.toString().orEmpty().trim()
        if (prompt.isEmpty()) {
            binding.etImagePrompt.error = activity.getString(R.string.positive_prompt)
            return
        }
        if (!runtime.hasStorageAccess()) {
            requestAllFilesAccess()
            return
        }
        if (runtime.isRunning()) return
        val resolution = resolutions.getOrElse(binding.spImageResolution.selectedItemPosition) { resolutions.first() }
        val negative = binding.etImageNegativePrompt.text?.toString().orEmpty()
        Thread({
            val status = runtime.inspect()
            activity.runOnUiThread {
                lastRuntimeStatus = status
                binding.tvImageRuntimeStatus.text = runtimeStatusText(status)
                if (!status.ready) {
                    Toast.makeText(activity, R.string.image_mode_requires_runtime, Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                if (!status.supports(resolution)) {
                    Toast.makeText(activity, "${resolution.key}: ${activity.getString(R.string.runtime_context_missing)}", Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                onNeedMemoryForGeneration {
                    lastGeneratedFile = null
                    binding.ivImageGenerationPreview.setImageDrawable(null)
                    binding.imagePreviewPlaceholder.visibility = View.VISIBLE
                    binding.imageGenerationProgress.progress = 0
                    binding.tvImageGenerationTiming.visibility = View.GONE
                    setGenerating(true)
                    val started = runtime.generate(
                        ImageGenerationRequest(
                            prompt = prompt,
                            negativePrompt = negative,
                            resolution = resolution,
                            steps = 8,
                            cfg = 3.5f,
                            livePreview = binding.switchLivePreview.isChecked && status.previewSupported,
                            progressiveCfg = true,
                        ),
                        callback = ::handleRuntimeEvent,
                    )
                    if (!started) setGenerating(false)
                }
            }
        }, "rin-image-preflight").start()
    }

    private fun handleRuntimeEvent(event: ImageGenerationEvent) {
        activity.runOnUiThread {
            when (event) {
                is ImageGenerationEvent.Progress -> {
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
                    setGenerating(false)
                    binding.tvImageGenerationTiming.visibility = View.VISIBLE
                    binding.tvImageGenerationTiming.text = activity.getString(R.string.runtime_generate_failed, event.message)
                    Toast.makeText(activity, activity.getString(R.string.runtime_generate_failed, event.message), Toast.LENGTH_LONG).show()
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
        binding.spImageResolution.isEnabled = !value
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
        AlertDialog.Builder(activity).setTitle(R.string.positive_preset_title).setView(root)
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
        AlertDialog.Builder(activity).setTitle(R.string.negative_preset_title).setView(root)
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
        val items = presetStore.listPositive()
        if (items.isEmpty()) { Toast.makeText(activity, R.string.preset_empty, Toast.LENGTH_SHORT).show(); return }
        val labels = items.map { listOf(it.name, it.note).filter(String::isNotBlank).joinToString("\n") }.toTypedArray()
        AlertDialog.Builder(activity).setTitle(R.string.select_positive_preset).setItems(labels) { _, which ->
            items.getOrNull(which)?.let { binding.etImagePrompt.setText(it.prompt); presetStore.touchPositive(it.id) }
        }.setNegativeButton(android.R.string.cancel, null).show()
    }

    private fun selectNegativePreset() {
        val items = presetStore.listNegative()
        if (items.isEmpty()) { Toast.makeText(activity, R.string.preset_empty, Toast.LENGTH_SHORT).show(); return }
        val labels = items.map {
            (if (it.isDefault) "★ " else "") + listOf(it.name, it.note).filter(String::isNotBlank).joinToString("\n")
        }.toTypedArray()
        AlertDialog.Builder(activity).setTitle(R.string.select_negative_preset).setItems(labels) { _, which ->
            items.getOrNull(which)?.let {
                if (!binding.switchLockNegative.isChecked) binding.etImageNegativePrompt.setText(it.prompt)
                presetStore.touchNegative(it.id)
            }
        }.setNegativeButton(android.R.string.cancel, null).show()
    }

    private fun dp(value: Int): Int = (value * activity.resources.displayMetrics.density).toInt()

    companion object {
        private const val KEY_MODE = "mode"
        private const val KEY_RESOLUTION = "resolution"
        private const val KEY_NEGATIVE_LOCKED = "negative_locked"
        private const val KEY_LIVE_PREVIEW = "live_preview"
    }
}
