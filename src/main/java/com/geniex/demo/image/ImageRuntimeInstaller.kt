package com.geniex.demo.image

import android.content.Context
import android.os.StatFs
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.ZipInputStream
import kotlin.concurrent.thread

sealed class ImageInstallEvent {
    data object Checking : ImageInstallEvent()
    data class Unavailable(val message: String) : ImageInstallEvent()
    data class Downloading(
        val percent: Int,
        val done: Long,
        val total: Long,
        val partIndex: Int = 0,
        val partCount: Int = 0,
        val partDone: Long = 0L,
        val partTotal: Long = 0L,
        val bytesPerSecond: Long = 0L,
        val etaSeconds: Long = -1L,
        val threads: Int = 1,
    ) : ImageInstallEvent()
    data class BrowserWaiting(
        val foundParts: Int,
        val totalParts: Int,
        val done: Long,
        val total: Long,
    ) : ImageInstallEvent()
    data object Verifying : ImageInstallEvent()
    data object Extracting : ImageInstallEvent()
    data class Complete(val version: String) : ImageInstallEvent()
    data class Failure(val message: String) : ImageInstallEvent()
}

class ImageRuntimeInstaller(
    private val context: Context,
    private val runtime: ImageGenerationRuntime,
) {
    private val cancelled = AtomicBoolean(false)
    @Volatile private var worker: Thread? = null

    fun install(callback: (ImageInstallEvent) -> Unit): Boolean {
        if (worker?.isAlive == true) return false
        cancelled.set(false)
        worker = thread(name = "rin-image-runtime-installer") {
            try {
                callback(ImageInstallEvent.Checking)
                if (!runtime.hasStorageAccess()) error("Storage permission is required")
                val base = runtime.defaultBaseDir.apply { mkdirs() }
                val manifest = fetchManifest()
                if (!manifest.optBoolean("available", false)) {
                    val language = context.resources.configuration.locales[0]?.language.orEmpty()
                    val key = if (language.equals("zh", ignoreCase = true)) "message_zh" else "message"
                    val fallback = manifest.optString("message", "Image runtime package is not published yet")
                    callback(ImageInstallEvent.Unavailable(manifest.optString(key, fallback)))
                    return@thread
                }
                val version = manifest.getString("version")
                val packageUrl = manifest.getString("url")
                val expectedSha = manifest.getString("sha256").lowercase()
                val expectedBytes = manifest.optLong("bytes", 0L)
                require(packageUrl.startsWith("https://")) { "Only HTTPS runtime packages are allowed" }
                require(expectedSha.matches(Regex("^[0-9a-f]{64}$"))) { "Invalid package SHA-256" }
                if (expectedBytes > 0) {
                    val free = StatFs(base.absolutePath).availableBytes
                    require(free > expectedBytes * 2L) { "Not enough free storage for the image runtime package" }
                }
                val installDir = File(base, ".rin_install").apply { mkdirs() }
                val part = File(installDir, "runtime-$version.zip.part")
                download(URL(packageUrl), part, expectedBytes, callback)
                checkCancelled()
                callback(ImageInstallEvent.Verifying)
                val actualSha = sha256(part)
                require(actualSha.equals(expectedSha, ignoreCase = true)) { "Runtime package SHA-256 mismatch" }
                callback(ImageInstallEvent.Extracting)
                extractSafely(part, base)
                restoreRuntimePermissions(base)
                installPrivatePythonRuntime(base)
                File(base, ".rin_runtime_version").writeText(version)
                part.delete()
                val status = runtime.inspect(base)
                if (!status.ready) error(status.detail ?: "Runtime package extracted, but required files are still missing")
                callback(ImageInstallEvent.Complete(version))
            } catch (t: Throwable) {
                if (!cancelled.get()) callback(ImageInstallEvent.Failure(t.message ?: t.javaClass.simpleName))
            } finally {
                worker = null
            }
        }
        return true
    }

    fun cancel() {
        cancelled.set(true)
        worker?.interrupt()
        worker = null
    }

    private fun fetchManifest(): JSONObject {
        val conn = open(URL(MANIFEST_URL), 0)
        try {
            require(conn.responseCode in 200..299) { "Runtime manifest HTTP ${conn.responseCode}" }
            return JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
        } finally {
            conn.disconnect()
        }
    }

    private fun download(url: URL, target: File, expectedBytes: Long, callback: (ImageInstallEvent) -> Unit) {
        var existing = if (target.isFile) target.length() else 0L
        var conn = open(url, existing)
        if (existing > 0 && conn.responseCode != HttpURLConnection.HTTP_PARTIAL) {
            conn.disconnect()
            target.delete()
            existing = 0L
            conn = open(url, 0)
        }
        require(conn.responseCode == HttpURLConnection.HTTP_OK || conn.responseCode == HttpURLConnection.HTTP_PARTIAL) {
            "Runtime package HTTP ${conn.responseCode}"
        }
        val contentLength = conn.contentLengthLong.coerceAtLeast(0L)
        val total = if (expectedBytes > 0) expectedBytes else existing + contentLength
        FileOutputStream(target, existing > 0).use { out ->
            BufferedInputStream(conn.inputStream, 1024 * 1024).use { input ->
                val buf = ByteArray(1024 * 1024)
                var done = existing
                var lastPct = -1
                while (true) {
                    checkCancelled()
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    done += n
                    val pct = if (total > 0) ((done * 100L) / total).toInt().coerceIn(0, 100) else 0
                    if (pct != lastPct) {
                        lastPct = pct
                        callback(ImageInstallEvent.Downloading(pct, done, total))
                    }
                }
                out.fd.sync()
            }
        }
        conn.disconnect()
        if (expectedBytes > 0) require(target.length() == expectedBytes) { "Downloaded package size mismatch" }
    }

    private fun extractSafely(zipFile: File, destination: File) {
        val root = destination.canonicalFile
        ZipInputStream(BufferedInputStream(FileInputStream(zipFile), 1024 * 1024)).use { zis ->
            while (true) {
                checkCancelled()
                val entry = zis.nextEntry ?: break
                val out = File(destination, entry.name).canonicalFile
                require(out.path == root.path || out.path.startsWith(root.path + File.separator)) { "Unsafe ZIP entry" }
                if (entry.isDirectory) {
                    out.mkdirs()
                } else {
                    out.parentFile?.mkdirs()
                    FileOutputStream(out).use { fos -> zis.copyTo(fos, 1024 * 1024) }
                }
                zis.closeEntry()
            }
        }
    }

    private fun installPrivatePythonRuntime(base: File) {
        val source = File(base, "py_runtime")
        if (!source.isDirectory) return
        val target = File(context.filesDir, "py_runtime")
        val staging = File(context.filesDir, ".py_runtime_install")
        val backup = File(context.filesDir, ".py_runtime_previous")
        staging.deleteRecursively()
        backup.deleteRecursively()
        check(source.copyRecursively(staging, overwrite = true)) { "Failed to stage private Python runtime" }
        require(File(staging, "usr/bin/python3").isFile) { "Staged Python runtime is missing usr/bin/python3" }
        if (target.exists()) {
            check(target.renameTo(backup)) { "Failed to rotate existing private Python runtime" }
        }
        if (!staging.renameTo(target)) {
            target.deleteRecursively()
            if (!staging.copyRecursively(target, overwrite = true)) {
                target.deleteRecursively()
                if (backup.exists()) backup.renameTo(target)
                error("Failed to install private Python runtime")
            }
            staging.deleteRecursively()
        }
        File(target, "usr/bin").listFiles()?.filter { it.isFile }?.forEach { it.setExecutable(true, false) }
        require(File(target, "usr/bin/python3").isFile) { "Private Python runtime install completed without python3" }
        backup.deleteRecursively()
    }

    private fun restoreRuntimePermissions(base: File) {
        listOf(
            File(base, "bin"),
            File(base, "py_runtime/usr/bin"),
            File(base, "python/usr/bin"),
        ).forEach { dir -> dir.listFiles()?.filter { it.isFile }?.forEach { it.setExecutable(true, false) } }
    }

    private fun open(url: URL, rangeStart: Long): HttpURLConnection = (url.openConnection() as HttpURLConnection).apply {
        connectTimeout = 15_000
        readTimeout = 60_000
        instanceFollowRedirects = true
        setRequestProperty("User-Agent", "Rin-NPU-Agent/1.5.2")
        if (rangeStart > 0) setRequestProperty("Range", "bytes=$rangeStart-")
        connect()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun checkCancelled() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Install cancelled")
    }

    companion object {
        const val MANIFEST_URL = "https://reimurin.com/files/rin-image-runtime/manifest.json"
    }
}
