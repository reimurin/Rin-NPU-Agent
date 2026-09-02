package com.geniex.demo.image

import android.content.Context
import android.os.Build
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.roundToInt


data class ImageResolution(
    val width: Int,
    val height: Int,
) {
    val key: String get() = "${width}x${height}"
}

data class ImageRuntimeStatus(
    val baseDir: File,
    val storagePermission: Boolean,
    val pythonPath: String?,
    val pythonDependenciesOk: Boolean,
    val missingCommonFiles: List<String>,
    val availableResolutions: List<ImageResolution>,
    val previewSupported: Boolean,
    val detail: String? = null,
) {
    val ready: Boolean
        get() = storagePermission && pythonPath != null && pythonDependenciesOk && missingCommonFiles.isEmpty() && availableResolutions.isNotEmpty()

    fun supports(resolution: ImageResolution): Boolean = availableResolutions.any { it == resolution }
}

data class ImageGenerationRequest(
    val prompt: String,
    val negativePrompt: String,
    val resolution: ImageResolution = ImageResolution(1024, 1024),
    val steps: Int = 8,
    val cfg: Float = 3.5f,
    val seed: Long = (System.currentTimeMillis() and 0x7fffffff),
    val livePreview: Boolean = true,
    val progressiveCfg: Boolean = true,
    val loraSlot: String? = null,
)

sealed class ImageGenerationEvent {
    data class Progress(
        val percent: Int,
        val stage: Stage,
        val step: Int = 0,
        val totalSteps: Int = 0,
        val rawLine: String = "",
    ) : ImageGenerationEvent()

    data class Preview(val file: File, val step: Int = 0, val totalSteps: Int = 0) : ImageGenerationEvent()
    data class Warning(val message: String) : ImageGenerationEvent()
    data class Complete(val file: File, val totalSeconds: Double?) : ImageGenerationEvent()
    data class Failure(val message: String, val exitCode: Int? = null) : ImageGenerationEvent()

    enum class Stage { PREPARING, CLIP, DENOISING, VAE, SAVING, COMPLETE }
}

class ImageGenerationRuntime(private val context: Context) {
    private val running = AtomicBoolean(false)
    @Volatile private var process: Process? = null
    @Volatile private var previewThread: Thread? = null

    val defaultBaseDir: File
        get() = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "sdxl_qnn")

    fun hasStorageAccess(): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) Environment.isExternalStorageManager() else true

    fun inspect(baseDir: File = defaultBaseDir): ImageRuntimeStatus {
        val permission = hasStorageAccess()
        if (!permission) {
            return ImageRuntimeStatus(
                baseDir = baseDir,
                storagePermission = false,
                pythonPath = null,
                pythonDependenciesOk = false,
                missingCommonFiles = listOf("storage permission"),
                availableResolutions = emptyList(),
                previewSupported = false,
                detail = "All-files access has not been granted",
            )
        }

        val missing = mutableListOf<String>()
        val common = listOf(
            "context/clip_l.serialized.bin.bin",
            "context/clip_g.serialized.bin.bin",
            "phone_gen/tokenizer/vocab.json",
            "phone_gen/tokenizer/merges.txt",
            "bin/qnn-net-run",
        )
        common.filterTo(missing) { !File(baseDir, it).isFile }
        if (!File(baseDir, "lib").isDirectory) missing += "lib/"

        val python = findPython(baseDir)
        val pyDeps = python?.let { checkPythonDependencies(it) } == true
        val resolutions = discoverResolutions(baseDir)
        val previewSupported =
            File(baseDir, "phone_gen/taesd_decoder.onnx").isFile ||
                File(baseDir, "context/taesd_decoder.serialized.bin.bin").isFile ||
                resolutions.any { File(baseDir, "context/${it.key}/taesd_decoder.serialized.bin.bin").isFile }

        val detail = when {
            !baseDir.isDirectory -> "Model directory does not exist"
            python == null -> "Python 3 runtime not found"
            !pyDeps -> "Python runtime is missing numpy and/or Pillow"
            missing.isNotEmpty() -> "Missing ${missing.size} runtime files"
            resolutions.isEmpty() -> "No complete SDXL resolution context was found"
            else -> null
        }

        return ImageRuntimeStatus(
            baseDir = baseDir,
            storagePermission = permission,
            pythonPath = python,
            pythonDependenciesOk = pyDeps,
            missingCommonFiles = missing,
            availableResolutions = resolutions,
            previewSupported = previewSupported,
            detail = detail,
        )
    }

    fun generate(
        request: ImageGenerationRequest,
        baseDir: File = defaultBaseDir,
        callback: (ImageGenerationEvent) -> Unit,
    ): Boolean {
        if (!running.compareAndSet(false, true)) return false
        thread(name = "rin-sdxl-generation") {
            var totalSeconds: Double? = null
            var savedFile: File? = null
            try {
                val status = inspect(baseDir)
                if (!status.ready) {
                    callback(ImageGenerationEvent.Failure(status.detail ?: "Image runtime is not ready"))
                    return@thread
                }
                if (!status.supports(request.resolution)) {
                    callback(ImageGenerationEvent.Failure("Resolution ${request.resolution.key} is not installed"))
                    return@thread
                }

                val python = status.pythonPath ?: error("Python runtime is unavailable")
                val driverDir = ensureDriverScripts()
                val driver = File(driverDir, "phone_generate.py")
                val outputDir = File(context.cacheDir, "sdxl_generated").apply { mkdirs() }
                val workDir = File(context.cacheDir, "sdxl_qnn_work").apply { mkdirs() }
                val previewFile = File(outputDir, "preview_current.png")
                previewFile.delete()
                val name = "rin_${System.currentTimeMillis()}"

                val args = mutableListOf(
                    python,
                    driver.absolutePath,
                    request.prompt.take(MAX_PROMPT_CHARS),
                    "--seed", request.seed.toString(),
                    "--steps", request.steps.coerceIn(1, 20).toString(),
                    "--cfg", request.cfg.coerceIn(1f, 7f).toString(),
                    "--width", request.resolution.width.toString(),
                    "--height", request.resolution.height.toString(),
                    "--name", name,
                )
                if (request.negativePrompt.isNotBlank()) {
                    args += listOf("--neg", request.negativePrompt.take(MAX_PROMPT_CHARS))
                }
                if (request.progressiveCfg) args += "--prog-cfg"
                request.loraSlot?.takeIf { it.isNotBlank() }?.let { args += listOf("--lora-slot", it.take(80)) }
                if (request.livePreview && status.previewSupported) args += "--preview"

                callback(ImageGenerationEvent.Progress(2, ImageGenerationEvent.Stage.PREPARING))
                val builder = ProcessBuilder(args)
                    .directory(baseDir)
                    .redirectErrorStream(true)
                configureEnvironment(builder.environment(), baseDir, workDir, outputDir, previewFile, request.livePreview && status.previewSupported)
                configurePythonRuntimeEnvironment(builder.environment(), python)

                val started = builder.start()
                process = started
                if (request.livePreview && status.previewSupported) {
                    startPreviewWatcher(started, previewFile, workDir, callback)
                }

                val unetRegex = Regex("\\[UNet\\s+(\\d+)/(\\d+)]")
                val previewRegex = Regex("\\[PREVIEW\\s+(\\d+)/(\\d+)]")
                val totalRegex = Regex("^Total:\\s*([0-9.]+)s", RegexOption.IGNORE_CASE)
                val savedRegex = Regex("^Saved:\\s*(.+)$", RegexOption.IGNORE_CASE)
                var lastErrorLine: String? = null

                started.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        Log.d(TAG, line)
                        when {
                            line.startsWith("[CLIP", ignoreCase = true) -> {
                                callback(ImageGenerationEvent.Progress(8, ImageGenerationEvent.Stage.CLIP, rawLine = line))
                            }
                            unetRegex.containsMatchIn(line) -> {
                                val m = unetRegex.find(line)!!
                                val step = m.groupValues[1].toIntOrNull() ?: 0
                                val total = m.groupValues[2].toIntOrNull() ?: request.steps
                                val p = 10 + ((step.toDouble() / total.coerceAtLeast(1)) * 75.0).roundToInt()
                                callback(ImageGenerationEvent.Progress(p.coerceIn(10, 85), ImageGenerationEvent.Stage.DENOISING, step, total, line))
                            }
                            previewRegex.containsMatchIn(line) -> {
                                val m = previewRegex.find(line)!!
                                callback(ImageGenerationEvent.Progress(
                                    percent = 10 + (((m.groupValues[1].toIntOrNull() ?: 0).toDouble() / (m.groupValues[2].toIntOrNull() ?: request.steps).coerceAtLeast(1)) * 75.0).roundToInt(),
                                    stage = ImageGenerationEvent.Stage.DENOISING,
                                    step = m.groupValues[1].toIntOrNull() ?: 0,
                                    totalSteps = m.groupValues[2].toIntOrNull() ?: request.steps,
                                    rawLine = line,
                                ))
                            }
                            line.startsWith("[VAE", ignoreCase = true) -> {
                                callback(ImageGenerationEvent.Progress(90, ImageGenerationEvent.Stage.VAE, rawLine = line))
                            }
                            line.startsWith("TAESD_WARNING:", ignoreCase = true) -> {
                                callback(ImageGenerationEvent.Warning(line.substringAfter(':').trim()))
                            }
                            savedRegex.containsMatchIn(line) -> {
                                val raw = savedRegex.find(line)!!.groupValues[1].trim()
                                savedFile = File(raw).let { if (it.isAbsolute) it else File(outputDir, raw) }
                                callback(ImageGenerationEvent.Progress(98, ImageGenerationEvent.Stage.SAVING, rawLine = line))
                            }
                            totalRegex.containsMatchIn(line) -> {
                                totalSeconds = totalRegex.find(line)?.groupValues?.getOrNull(1)?.toDoubleOrNull()
                            }
                            line.contains("error", ignoreCase = true) || line.contains("failed", ignoreCase = true) -> {
                                lastErrorLine = line.take(2_000)
                            }
                        }
                    }
                }

                val exit = started.waitFor()
                stopPreviewWatcher()
                val finalFile = savedFile?.takeIf { it.isFile && it.length() > 0 }
                    ?: outputDir.listFiles()?.filter { it.isFile && it.extension.equals("png", true) && it.name != previewFile.name }
                        ?.maxByOrNull { it.lastModified() }
                if (exit == 0 && finalFile != null) {
                    callback(ImageGenerationEvent.Progress(100, ImageGenerationEvent.Stage.COMPLETE))
                    callback(ImageGenerationEvent.Complete(finalFile, totalSeconds))
                } else {
                    callback(ImageGenerationEvent.Failure(lastErrorLine ?: "Generator exited with code $exit", exit))
                }
            } catch (t: Throwable) {
                callback(ImageGenerationEvent.Failure(t.message ?: t.javaClass.simpleName))
            } finally {
                stopPreviewWatcher()
                process = null
                running.set(false)
            }
        }
        return true
    }

    fun stop() {
        running.set(false)
        stopPreviewWatcher()
        runCatching { process?.destroy() }
        try {
            Thread.sleep(150)
        } catch (_: InterruptedException) {
        }
        runCatching { if (process?.isAlive == true) process?.destroyForcibly() }
        process = null
    }

    fun isRunning(): Boolean = running.get()

    private fun ensureDriverScripts(): File {
        val dir = File(context.filesDir, "sdxl_runtime_driver").apply { mkdirs() }
        for (name in DRIVER_FILES) {
            val target = File(dir, name)
            context.assets.open("sdxl_runtime/$name").use { input ->
                FileOutputStream(target, false).use { output -> input.copyTo(output, 256 * 1024) }
            }
            target.setReadable(true, true)
        }
        return dir
    }

    private fun findPython(baseDir: File): String? {
        val candidates = listOf(
            File(context.filesDir, "py_runtime/usr/bin/python3"),
            File(context.filesDir, "sdxl_runtime/py_runtime/usr/bin/python3"),
            File(baseDir, "py_runtime/usr/bin/python3"),
            File(baseDir, "python/usr/bin/python3"),
            File(baseDir, "bin/python3"),
        )
        candidates.firstOrNull { it.isFile && it.canExecute() }?.let { return it.absolutePath }

        val shellProbe = runCatching {
            val p = ProcessBuilder("/system/bin/sh", "-c", "command -v python3 || command -v python").redirectErrorStream(true).start()
            val result = p.inputStream.bufferedReader().readLine()?.trim()
            p.waitFor()
            result?.takeIf { p.exitValue() == 0 && it.isNotBlank() && File(it).canExecute() }
        }.getOrNull()
        return shellProbe
    }

    private fun checkPythonDependencies(python: String): Boolean = runCatching {
        val builder = ProcessBuilder(python, "-c", "import numpy, PIL; print('ok')").redirectErrorStream(true)
        configurePythonRuntimeEnvironment(builder.environment(), python)
        val p = builder.start()
        val output = p.inputStream.bufferedReader().readText()
        val exit = p.waitFor()
        exit == 0 && output.contains("ok")
    }.getOrDefault(false)

    private fun configurePythonRuntimeEnvironment(env: MutableMap<String, String>, python: String) {
        val usr = File(python).parentFile?.parentFile ?: return
        val bin = File(usr, "bin")
        val lib = File(usr, "lib")
        if (!bin.isDirectory || !lib.isDirectory) return
        env["PYTHONHOME"] = usr.absolutePath
        env["PATH"] = listOf(bin.absolutePath, env["PATH"].orEmpty()).filter { it.isNotBlank() }.joinToString(":")
        env["LD_LIBRARY_PATH"] = listOf(lib.absolutePath, env["LD_LIBRARY_PATH"].orEmpty()).filter { it.isNotBlank() }.joinToString(":")
    }

    private fun discoverResolutions(baseDir: File): List<ImageResolution> {
        val root = File(baseDir, "context")
        if (!root.isDirectory) return emptyList()
        val result = mutableSetOf<ImageResolution>()
        root.listFiles()?.filter { it.isDirectory }?.forEach { dir ->
            val m = Regex("^(\\d+)x(\\d+)$").matchEntire(dir.name) ?: return@forEach
            val resolution = ImageResolution(m.groupValues[1].toInt(), m.groupValues[2].toInt())
            if (hasResolutionContexts(dir)) result += resolution
        }
        if (hasResolutionContexts(root)) result += ImageResolution(1024, 1024)
        return result.sortedWith(compareBy<ImageResolution> { it.width * it.height }.thenBy { it.width })
    }

    private fun hasResolutionContexts(dir: File): Boolean = listOf(
        "unet_encoder_fp16.serialized.bin.bin",
        "unet_decoder_fp16.serialized.bin.bin",
        "vae_decoder.serialized.bin.bin",
    ).all { File(dir, it).isFile }

    private fun configureEnvironment(
        env: MutableMap<String, String>,
        baseDir: File,
        workDir: File,
        outputDir: File,
        previewFile: File,
        previewEnabled: Boolean,
    ) {
        val lib = File(baseDir, "lib").absolutePath
        val bin = File(baseDir, "bin").absolutePath
        val model = File(baseDir, "model").absolutePath
        val oldLd = env["LD_LIBRARY_PATH"].orEmpty()
        env["MODEL_TO_NPU_MODEL_FAMILY"] = "sdxl"
        env["MODEL_TO_NPU_BASE"] = baseDir.absolutePath
        env["SDXL_QNN_BASE"] = baseDir.absolutePath
        env["MODEL_TO_NPU_WORK_DIR"] = workDir.absolutePath
        env["MODEL_TO_NPU_OUTPUT_DIR"] = outputDir.absolutePath
        env["MODEL_TO_NPU_PREVIEW_PNG"] = previewFile.absolutePath
        env["PYTHONPATH"] = File(context.filesDir, "sdxl_runtime_driver").absolutePath
        env["SDXL_QNN_USE_MMAP"] = "1"
        env["SDXL_QNN_LOG_LEVEL"] = "warn"
        env["SDXL_QNN_PERF_PROFILE"] = "burst"
        env["SDXL_QNN_SHARED_SERVER"] = "1"
        env["SDXL_QNN_ASYNC_PREP"] = "1"
        env["SDXL_QNN_PRESTAGE_RUNTIME"] = "1"
        env["SDXL_QNN_CLIP_CACHE"] = "1"
        env["SDXL_QNN_PREWARM_ALL_CONTEXTS"] = "0"
        env["SDXL_QNN_PREWARM_PREVIEW"] = "0"
        env["SDXL_QNN_PREVIEW_STRIDE"] = if (previewEnabled) "2" else "auto"
        env["SDXL_SHOW_TEMP"] = "0"
        env["LD_LIBRARY_PATH"] = listOf(lib, bin, model, oldLd).filter { it.isNotBlank() }.joinToString(":")
        env["ADSP_LIBRARY_PATH"] = "$lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"
    }

    private fun startPreviewWatcher(
        activeProcess: Process,
        previewFile: File,
        workDir: File,
        callback: (ImageGenerationEvent) -> Unit,
    ) {
        stopPreviewWatcher()
        previewThread = thread(name = "rin-sdxl-preview") {
            var lastModified = 0L
            var lastSize = 0L
            while (activeProcess.isAlive && running.get()) {
                try {
                    if (previewFile.isFile) {
                        val modified = previewFile.lastModified()
                        val size = previewFile.length()
                        if (size > 512 && (modified != lastModified || size != lastSize)) {
                            Thread.sleep(120)
                            val stableSize = previewFile.length()
                            if (stableSize == size) {
                                val snapshot = File(workDir, "preview_snapshot_${modified}.png")
                                previewFile.copyTo(snapshot, overwrite = true)
                                callback(ImageGenerationEvent.Preview(snapshot))
                                lastModified = modified
                                lastSize = size
                                workDir.listFiles()?.filter { it.name.startsWith("preview_snapshot_") && it != snapshot }
                                    ?.sortedByDescending { it.lastModified() }?.drop(3)?.forEach { it.delete() }
                            }
                        }
                    }
                    Thread.sleep(650)
                } catch (_: InterruptedException) {
                    return@thread
                } catch (t: Throwable) {
                    Log.w(TAG, "preview polling failed", t)
                    Thread.sleep(900)
                }
            }
        }
    }

    private fun stopPreviewWatcher() {
        previewThread?.interrupt()
        previewThread = null
    }

    companion object {
        private const val TAG = "RinImageRuntime"
        private const val MAX_PROMPT_CHARS = 100_000
        private val DRIVER_FILES = listOf("phone_generate.py", "phone_runtime_accel.py")
        val UI_RESOLUTIONS = listOf(
            ImageResolution(1024, 1024),
            ImageResolution(1216, 832),
            ImageResolution(832, 1216),
            ImageResolution(1344, 768),
            ImageResolution(768, 1344),
        )
    }
}
