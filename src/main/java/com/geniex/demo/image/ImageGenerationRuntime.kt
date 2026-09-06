package com.geniex.demo.image

import android.content.Context
import com.geniex.demo.BuildConfig
import android.os.Build
import android.os.Environment
import android.system.Os
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.TimeUnit
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
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) StorageAccess.granted() else true

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
        if (hasKnownBadClipG(baseDir)) missing += "context/clip_g.serialized.bin.bin (known-bad context)"
        if (!File(baseDir, "lib").isDirectory) missing += "lib/"

        val python = findPython(baseDir)
        val pythonProbe = python?.let { checkPythonDependencies(it) }
        val pyDeps = pythonProbe?.ok == true
        val resolutions = discoverResolutions(baseDir)
        val previewSupported =
            File(baseDir, "phone_gen/taesd_decoder.onnx").isFile ||
                File(baseDir, "context/taesd_decoder.serialized.bin.bin").isFile ||
                resolutions.any { File(baseDir, "context/${it.key}/taesd_decoder.serialized.bin.bin").isFile }

        val detail = when {
            !baseDir.isDirectory -> "Model directory does not exist"
            python == null -> "Python 3 runtime not found"
            !pyDeps -> "Python runtime probe failed: ${pythonProbe?.detail ?: "unknown Python startup/import error"}"
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

    fun hasKnownBadClipG(baseDir: File = defaultBaseDir): Boolean {
        val target = File(baseDir, LEGACY_BAD_CLIP_G_RELATIVE)
        if (!target.isFile || target.length() != LEGACY_BAD_CLIP_G_BYTES) return false
        return runCatching { sha256File(target).equals(LEGACY_BAD_CLIP_G_SHA256, ignoreCase = true) }.getOrDefault(false)
    }

    private fun sha256File(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun generate(
        request: ImageGenerationRequest,
        baseDir: File = defaultBaseDir,
        callback: (ImageGenerationEvent) -> Unit,
    ): Boolean {
        if (LoraSelfTest.busy.get() || !running.compareAndSet(false, true)) return false
        thread(name = "rin-sdxl-generation") {
            var totalSeconds: Double? = null
            var savedFile: File? = null
            var generationLog: File? = null
            var qnnBridge: QnnInProcessBridgeServer? = null
            val diagnosticTail = mutableListOf<String>()
            try {
                val status = inspect(baseDir)
                if (!status.ready) {
                    callback(ImageGenerationEvent.Failure(status.detail ?: "Image runtime is not ready"))
                    return@thread
                }
                if (!status.supports(request.resolution)) {
                    callback(ImageGenerationEvent.Failure("原生尺寸 ${request.resolution.key} 尚未安装对应模型。现有 context 不会自动改变宽高。"))
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
                val diagDir = File(baseDir, ".rin_diagnostics").apply { mkdirs() }
                generationLog = File(diagDir, "generation_latest.log")
                qnnBridge = QnnInProcessBridgeServer(context, baseDir, diagDir).start()
                val platformProbe = runQnnPlatformProbe(baseDir)
                runCatching {
                    generationLog?.writeText(
                        "Rin NPU generation diagnostic\n" +
                            "version=${BuildConfig.VERSION_NAME}\n" +
                            "time=${System.currentTimeMillis()}\n" +
                            "base=${baseDir.absolutePath}\n" +
                            "nativeLib=${context.applicationInfo.nativeLibraryDir}\n" +
                            "executionMode=inprocess-jni\n" +
                            "bridgePort=${qnnBridge?.port ?: 0}\n" +
                            "bridgePreload=${qnnBridge?.preloadSummary ?: "unavailable"}\n" +
                            "legacyPlatformProbePassed=${platformProbe.passed}\n" +
                            "legacyPlatformProbeExit=${platformProbe.exitCode}\n" +
                            "legacyPlatformProbeSummary=${platformProbe.summary}\n" +
                            "legacyPlatformProbeLog=${platformProbe.logFile.absolutePath}\n\n"
                    )
                }

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
                configureEnvironment(
                    builder.environment(),
                    baseDir,
                    workDir,
                    outputDir,
                    previewFile,
                    request.livePreview && status.previewSupported,
                    qnnBridge?.port ?: 0,
                )
                configurePythonRuntimeEnvironment(builder.environment(), python)

                val started = builder.start()
                process = started
                if (request.livePreview && status.previewSupported) {
                    startPreviewWatcher(started, previewFile, workDir, callback)
                }

                val unetRegex = Regex("\\[UNet\\s+(\\d+)/(\\d+)]")
                val unetStartRegex = Regex("\\[UNet START\\s+(\\d+)/(\\d+)]")
                val previewRegex = Regex("\\[PREVIEW\\s+(\\d+)/(\\d+)]")
                val totalRegex = Regex("^Total:\\s*([0-9.]+)s", RegexOption.IGNORE_CASE)
                val savedRegex = Regex("^Saved:\\s*(.+)$", RegexOption.IGNORE_CASE)
                var lastErrorLine: String? = null

                started.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        Log.d(TAG, line)
                        runCatching { generationLog?.appendText(line + "\n") }
                        diagnosticTail += line.take(6_000)
                        if (diagnosticTail.size > 80) diagnosticTail.removeAt(0)
                        when {
                            line.startsWith("[CLIP", ignoreCase = true) -> {
                                callback(ImageGenerationEvent.Progress(8, ImageGenerationEvent.Stage.CLIP, rawLine = line))
                            }
                            unetStartRegex.containsMatchIn(line) -> {
                                val m = unetStartRegex.find(line)!!
                                val step = m.groupValues[1].toIntOrNull() ?: 1
                                val total = m.groupValues[2].toIntOrNull() ?: request.steps
                                val p = 10 + (((step - 1).coerceAtLeast(0).toDouble() / total.coerceAtLeast(1)) * 75.0).roundToInt()
                                callback(ImageGenerationEvent.Progress(p.coerceIn(10, 85), ImageGenerationEvent.Stage.DENOISING, step, total, line))
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
                val expectedFile = File(outputDir, "$name.png")
                val finalFile = expectedFile.takeIf { it.isFile && it.length() > 0L &&
                    (savedFile == null || savedFile?.absolutePath == it.absolutePath) }
                if (exit == 0 && finalFile != null) {
                    callback(ImageGenerationEvent.Progress(100, ImageGenerationEvent.Stage.COMPLETE))
                    callback(ImageGenerationEvent.Complete(finalFile, totalSeconds))
                } else {
                    val tail = diagnosticTail.takeLast(18).joinToString(" | ").takeLast(7_000)
                    val detail = lastErrorLine ?: tail.ifBlank { "Generator exited with code $exit" }
                    callback(ImageGenerationEvent.Failure("$detail · 日志: ${generationLog?.absolutePath ?: "unavailable"}", exit))
                }
            } catch (t: Throwable) {
                callback(ImageGenerationEvent.Failure("${t.message ?: t.javaClass.simpleName} · 日志: ${generationLog?.absolutePath ?: "unavailable"}"))
            } finally {
                runCatching { qnnBridge?.close() }
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

    private data class PythonProbe(val ok: Boolean, val detail: String)

    private data class QnnPlatformProbe(
        val passed: Boolean,
        val exitCode: Int,
        val summary: String,
        val logFile: File,
    )

    @Synchronized
    private fun ensureQnnProbeAssets(): File {
        val dir = File(context.filesDir, "qnn_probe/dsp").apply { mkdirs() }
        val target = File(dir, "libCalculator_skel.so")
        context.assets.open("sdxl_runtime/qnn_probe/libCalculator_skel.so").use { input ->
            FileOutputStream(target, false).use { output -> input.copyTo(output, 128 * 1024) }
        }
        target.setReadable(true, true)
        return dir
    }

    private fun vendorLibraryProbe(name: String): String = runCatching {
        System.loadLibrary(name)
        "$name=load-ok"
    }.getOrElse { "$name=load-failed:${it.javaClass.simpleName}:${it.message ?: "unknown"}" }

    private fun fastRpcNodeSummary(): String = listOf(
        "/dev/fastrpc-cdsp",
        "/dev/fastrpc-cdsp-secure",
        "/dev/fastrpc-adsp",
    ).joinToString(" | ") { path ->
        val f = File(path)
        "$path exists=${f.exists()} read=${f.canRead()} write=${f.canWrite()}"
    }

    private fun runQnnPlatformProbe(baseDir: File): QnnPlatformProbe {
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val validator = File(nativeDir, "librinqnnplatformvalidator.so")
        val diagDir = File(baseDir, ".rin_diagnostics").apply { mkdirs() }
        val logFile = File(diagDir, "platform_validator.log")
        val probeDspDir = ensureQnnProbeAssets()
        if (!validator.isFile) {
            val msg = "qnn-platform-validator missing: ${validator.absolutePath}"
            runCatching { logFile.writeText(msg + "\n") }
            return QnnPlatformProbe(false, -1, msg, logFile)
        }
        val builder = ProcessBuilder(
            validator.absolutePath,
            "--backend", "dsp",
            "--testBackend",
            "--coreVersion",
            "--libVersion",
        ).directory(baseDir).redirectErrorStream(true)
        val env = builder.environment()
        val oldLd = env["LD_LIBRARY_PATH"].orEmpty()
        env["LD_LIBRARY_PATH"] = listOf(nativeDir.absolutePath, oldLd).filter { it.isNotBlank() }.joinToString(":")
        env["ADSP_LIBRARY_PATH"] = listOf(
            probeDspDir.absolutePath,
            nativeDir.absolutePath,
            "/odm/lib/rfsa/adsp/aiboost/signed",
            "/odm/lib/rfsa/adsp",
            "/vendor/lib64/rfs/dsp",
            "/vendor/lib/rfsa/adsp",
            "/vendor/dsp",
        ).joinToString(";")
        val header = buildString {
            appendLine("Rin QNN platform probe")
            appendLine("version=${BuildConfig.VERSION_NAME}")
            appendLine("validator=${validator.absolutePath}")
            appendLine("nativeLib=${nativeDir.absolutePath}")
            appendLine("probeDsp=${probeDspDir.absolutePath}")
            appendLine("LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")
            appendLine("ADSP_LIBRARY_PATH=${env["ADSP_LIBRARY_PATH"]}")
            appendLine(vendorLibraryProbe("cdsprpc"))
            appendLine(vendorLibraryProbe("adsprpc"))
            appendLine(fastRpcNodeSummary())
        }
        return runCatching {
            val p = builder.start()
            val output = StringBuilder()
            val reader = thread(name = "rin-qnn-validator-reader") {
                p.inputStream.bufferedReader().forEachLine { output.appendLine(it) }
            }
            val finished = p.waitFor(20, TimeUnit.SECONDS)
            if (!finished) p.destroyForcibly()
            reader.join(2_000)
            val exit = if (finished) p.exitValue() else -9
            val body = output.toString()
            val passed = exit == 0 && body.contains("Unit Test", ignoreCase = true) && body.contains("Passed", ignoreCase = true)
            val interesting = body.lineSequence().filter {
                it.contains("Backend", true) || it.contains("Core Version", true) ||
                    it.contains("Unit Test", true) || it.contains("ERROR", true) ||
                    it.contains("Failed", true) || it.contains("FastRPC", true)
            }.toList().takeLast(20).joinToString(" | ").ifBlank { body.lineSequence().toList().takeLast(12).joinToString(" | ") }
            logFile.writeText(header + "exit=$exit\n--- output ---\n" + body)
            QnnPlatformProbe(passed, exit, interesting.takeLast(4_000), logFile)
        }.getOrElse {
            val msg = "${it.javaClass.simpleName}: ${it.message ?: "validator startup failed"}"
            runCatching { logFile.writeText(header + "probe_exception=$msg\n") }
            QnnPlatformProbe(false, -2, msg, logFile)
        }
    }

    @Synchronized
    private fun ensurePrivatePythonNativeLinks() {
        val root = File(context.filesDir, "py_runtime")
        if (!root.isDirectory) return
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        if (!nativeDir.isDirectory) return
        runCatching {
            val manifest = context.assets.open("sdxl_runtime/python_native_map.json")
                .bufferedReader().use { JSONObject(it.readText()) }
            val entries = manifest.getJSONArray("entries")
            val rootPrefix = root.absoluteFile.path.trimEnd(File.separatorChar) + File.separator
            for (i in 0 until entries.length()) {
                val item = entries.getJSONObject(i)
                val relative = item.getString("relative")
                val nativeName = item.getString("native")
                val target = File(root, relative.replace('/', File.separatorChar)).absoluteFile
                require(target.path.startsWith(rootPrefix)) { "Unsafe Python native path: $relative" }
                val nativeFile = File(nativeDir, nativeName)
                require(nativeFile.isFile) { "APK Python native library missing: $nativeName" }
                target.parentFile?.mkdirs()
                val current = runCatching { Os.readlink(target.absolutePath) }.getOrNull()
                if (current == nativeFile.absolutePath) continue
                if (current != null || target.exists()) {
                    require(target.delete()) { "Could not replace Python native file: $relative" }
                }
                Os.symlink(nativeFile.absolutePath, target.absolutePath)
            }
        }.onFailure { Log.e(TAG, "Python native-link repair failed", it) }
    }

    private fun ensureRelocatablePythonLauncher(): File? {
        val usr = File(context.filesDir, "py_runtime/usr")
        val stdlib = File(usr, "lib/python3.13")
        if (!stdlib.isDirectory) return null
        ensurePrivatePythonNativeLinks()
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val launcher = File(nativeDir, "librinpythonlauncher.so")
        val libpython = File(nativeDir, "libpython3.13.so")
        if (!launcher.isFile || !libpython.isFile) return null
        return launcher
    }

    private fun findPython(baseDir: File): String? {
        ensureRelocatablePythonLauncher()?.let { return it.absolutePath }
        val shellProbe = runCatching {
            val p = ProcessBuilder("/system/bin/sh", "-c", "command -v python3 || command -v python").redirectErrorStream(true).start()
            val result = p.inputStream.bufferedReader().readLine()?.trim()
            p.waitFor()
            result?.takeIf { p.exitValue() == 0 && it.isNotBlank() && File(it).canExecute() }
        }.getOrNull()
        return shellProbe
    }

    private fun checkPythonDependencies(python: String): PythonProbe = runCatching {
        val builder = ProcessBuilder(
            python,
            "-c",
            "import sys; import numpy, PIL; print('RIN_PY_OK', sys.prefix, numpy.__version__, PIL.__version__)",
        ).redirectErrorStream(true)
        configurePythonRuntimeEnvironment(builder.environment(), python)
        val p = builder.start()
        val output = p.inputStream.bufferedReader().readText().trim()
        val finished = p.waitFor(20, TimeUnit.SECONDS)
        if (!finished) {
            p.destroyForcibly()
            PythonProbe(false, "Python probe timed out after 20s")
        } else {
            val exit = p.exitValue()
            if (exit == 0 && output.contains("RIN_PY_OK")) {
                PythonProbe(true, output.take(1200))
            } else {
                PythonProbe(false, "exit=$exit · ${output.ifBlank { "no output" }.take(1600)}")
            }
        }
    }.getOrElse { error ->
        PythonProbe(false, "${error.javaClass.simpleName}: ${error.message ?: "startup failed"}")
    }

    private fun configurePythonRuntimeEnvironment(env: MutableMap<String, String>, python: String) {
        val executable = File(python)
        val relocatable = executable.name == "librinpythonlauncher.so"
        val usr = if (relocatable) {
            File(context.filesDir, "py_runtime/usr")
        } else {
            executable.parentFile?.parentFile ?: return
        }
        val bin = File(usr, "bin")
        val lib = File(usr, "lib")
        val pyLib = File(lib, "python3.13")
        val dynload = File(pyLib, "lib-dynload")
        val sitePackages = File(pyLib, "site-packages")
        if (!lib.isDirectory || !pyLib.isDirectory) return
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        if (relocatable) {
            val libpython = File(nativeDir, "libpython3.13.so")
            if (libpython.isFile) env["RIN_PYTHON_LIB"] = libpython.absolutePath
        }
        env["PYTHONHOME"] = usr.absolutePath
        val oldPythonPath = env["PYTHONPATH"].orEmpty()
        env["PYTHONPATH"] = listOf(
            pyLib.absolutePath,
            dynload.absolutePath,
            sitePackages.absolutePath,
            oldPythonPath,
        ).filter { it.isNotBlank() }.joinToString(":")
        env["PATH"] = listOf(bin.absolutePath, env["PATH"].orEmpty()).filter { it.isNotBlank() }.joinToString(":")
        env["LD_LIBRARY_PATH"] = listOf(
            if (relocatable) nativeDir.absolutePath else "",
            lib.absolutePath,
            dynload.absolutePath,
            env["LD_LIBRARY_PATH"].orEmpty(),
        ).filter { it.isNotBlank() }.joinToString(":")
        env["HOME"] = context.filesDir.absolutePath
        env["TMPDIR"] = context.cacheDir.absolutePath
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUTF8"] = "1"
    }

    private fun discoverResolutions(baseDir: File): List<ImageResolution> = ResolutionCatalog.discover(baseDir)

    private fun configureEnvironment(
        env: MutableMap<String, String>,
        baseDir: File,
        workDir: File,
        outputDir: File,
        previewFile: File,
        previewEnabled: Boolean,
        bridgePort: Int,
    ) {
        val lib = File(baseDir, "lib").absolutePath
        val bin = File(baseDir, "bin").absolutePath
        val model = File(baseDir, "model").absolutePath
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        val qnnNetRun = File(nativeLibDir, "librinqnnnetrun.so")
        if (qnnNetRun.isFile) {
            env["SDXL_QNN_NET_RUN"] = qnnNetRun.absolutePath
            env["SDXL_QNN_BIN_DIR"] = nativeLibDir
            env["SDXL_QNN_LIB_DIR"] = nativeLibDir
            env["SDXL_QNN_SYSTEM_LIB"] = File(nativeLibDir, "libQnnSystem.so").absolutePath
        }
        val oldLd = env["LD_LIBRARY_PATH"].orEmpty()
        env["MODEL_TO_NPU_MODEL_FAMILY"] = "sdxl"
        env["MODEL_TO_NPU_BASE"] = baseDir.absolutePath
        env["SDXL_QNN_BASE"] = baseDir.absolutePath
        env["MODEL_TO_NPU_WORK_DIR"] = workDir.absolutePath
        env["MODEL_TO_NPU_OUTPUT_DIR"] = outputDir.absolutePath
        env["MODEL_TO_NPU_PREVIEW_PNG"] = previewFile.absolutePath
        env["PYTHONPATH"] = File(context.filesDir, "sdxl_runtime_driver").absolutePath
        env["SDXL_QNN_USE_MMAP"] = "1"
        env["SDXL_QNN_LOG_LEVEL"] = "verbose"
        env["SDXL_QNN_STDOUT_ECHO"] = "1"
        env["SDXL_QNN_TRY_SIGNED_PD"] = "1"
        env["SDXL_QNN_SIGNED_PD_PLATFORM_OPTIONS"] = "unsignedPD:OFF"
        env["SDXL_QNN_USE_SERVER"] = "0"
        env["SDXL_QNN_USE_DAEMON"] = "0"
        env["SDXL_QNN_SHARED_SERVER"] = "0"
        env["SDXL_QNN_TRACE_ALL_STEPS"] = "1"
        env["SDXL_QNN_DIAG_DIR"] = File(baseDir, ".rin_diagnostics").apply { mkdirs() }.absolutePath
        if (bridgePort > 0) {
            env["SDXL_QNN_BRIDGE_PORT"] = bridgePort.toString()
            env["SDXL_QNN_BRIDGE_REQUIRED"] = "1"
            env["SDXL_QNN_BRIDGE_TIMEOUT_SEC"] = "180"
        }
        env["SDXL_QNN_PERF_PROFILE"] = "burst"
        env["SDXL_QNN_ASYNC_PREP"] = "1"
        env["SDXL_QNN_PRESTAGE_RUNTIME"] = "1"
        env["SDXL_QNN_CLIP_CACHE"] = "1"
        env["SDXL_QNN_PREWARM_ALL_CONTEXTS"] = "0"
        env["SDXL_QNN_PREWARM_PREVIEW"] = "0"
        env["SDXL_QNN_PREVIEW_STRIDE"] = if (previewEnabled) "2" else "auto"
        env["SDXL_SHOW_TEMP"] = "0"
        env["LD_LIBRARY_PATH"] = listOf(nativeLibDir, lib, bin, model, oldLd).filter { it.isNotBlank() }.joinToString(":")
        env["ADSP_LIBRARY_PATH"] = "$nativeLibDir;$lib;/odm/lib/rfsa/adsp;/odm/lib/rfsa/adsp/aiboost/signed;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"
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
        private const val LEGACY_BAD_CLIP_G_RELATIVE = "context/clip_g.serialized.bin.bin"
        private const val LEGACY_BAD_CLIP_G_BYTES = 42_991_616L
        private const val LEGACY_BAD_CLIP_G_SHA256 = "604c9dd468ca74553138502d1e32b18129b6671cc91c68f0213403ae769bc2d4"
        private val DRIVER_FILES = listOf("phone_generate.py", "phone_runtime_accel.py", "rin_tensor_io.py", "rin_lora.py", "rin_lora_module_banks.py", "rin_lora_partitioned.py")
        val UI_RESOLUTIONS = listOf(
            ImageResolution(1024, 1024),
            ImageResolution(1216, 832),
            ImageResolution(832, 1216),
            ImageResolution(1344, 768),
            ImageResolution(768, 1344),
        )
    }
}
