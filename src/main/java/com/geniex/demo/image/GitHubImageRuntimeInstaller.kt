package com.geniex.demo.image

import android.content.Context
import android.os.Build
import android.os.StatFs
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.ZipInputStream
import kotlin.concurrent.thread

class GitHubImageRuntimeInstaller(
    private val context: Context,
    private val runtime: ImageGenerationRuntime,
) {
    private data class PackagePart(
        val name: String,
        val url: String,
        val sha256: String,
        val bytes: Long,
    )

    private data class PackageSpec(
        val id: String,
        val version: String,
        val archiveSha256: String,
        val archiveBytes: Long,
        val parts: List<PackagePart>,
    )

    private val cancelled = AtomicBoolean(false)
    @Volatile private var worker: Thread? = null

    fun install(callback: (ImageInstallEvent) -> Unit): Boolean {
        if (worker?.isAlive == true) return false
        cancelled.set(false)
        worker = thread(name = "rin-github-image-installer") {
            try {
                callback(ImageInstallEvent.Checking)
                if (!runtime.hasStorageAccess()) error("Storage permission is required")
                val base = runtime.defaultBaseDir.apply { mkdirs() }
                val manifest = fetchManifest()
                val packageJson = selectPublishedPackage(manifest)
                if (packageJson == null) {
                    callback(ImageInstallEvent.Unavailable(unavailableMessage(manifest)))
                    return@thread
                }
                val spec = parsePackage(packageJson)
                val totalDownloadBytes = spec.parts.sumOf { it.bytes }
                val workingBytes = maxOf(spec.archiveBytes, totalDownloadBytes)
                if (workingBytes > 0) {
                    val free = StatFs(base.absolutePath).availableBytes
                    require(free > (workingBytes * 32L) / 10L) {
                        "Not enough free storage for the image runtime package"
                    }
                }

                val installDir = File(base, ".rin_install").apply { mkdirs() }
                var completedBytes = 0L
                val downloadedParts = mutableListOf<File>()
                spec.parts.forEachIndexed { index, partSpec ->
                    checkCancelled()
                    val safeName = partSpec.name.ifBlank { "part-${index + 1}.bin" }
                    require(!safeName.contains('/') && !safeName.contains('\\')) { "Invalid package part name" }
                    val preferred = File(installDir, safeName + ".part")
                    val partFile = obtainPart(
                        preferred,
                        partSpec,
                        completedBytes,
                        totalDownloadBytes,
                        callback,
                    )
                    callback(ImageInstallEvent.Verifying)
                    require(sha256(partFile).equals(partSpec.sha256, ignoreCase = true)) {
                        "Package part SHA-256 mismatch: ${partSpec.name}"
                    }
                    downloadedParts += partFile
                    completedBytes += if (partSpec.bytes > 0) partSpec.bytes else partFile.length()
                    callback(
                        ImageInstallEvent.Downloading(
                            if (totalDownloadBytes > 0) ((completedBytes * 100L) / totalDownloadBytes).toInt().coerceIn(0, 100) else 100,
                            completedBytes,
                            totalDownloadBytes,
                        ),
                    )
                }

                checkCancelled()
                val archive = File(
                    installDir,
                    "runtime-${spec.version}-${System.currentTimeMillis()}.zip.part",
                )
                Files.newOutputStream(
                    archive.toPath(),
                    StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE,
                ).use { out ->
                    downloadedParts.forEach { file ->
                        FileInputStream(file).use { input -> input.copyTo(out, 1024 * 1024) }
                    }
                }
                if (spec.archiveBytes > 0) require(archive.length() == spec.archiveBytes) { "Runtime archive size mismatch" }
                callback(ImageInstallEvent.Verifying)
                require(sha256(archive).equals(spec.archiveSha256, ignoreCase = true)) { "Runtime archive SHA-256 mismatch" }

                callback(ImageInstallEvent.Extracting)
                extractSafelyCreateNew(archive, base)
                restoreRuntimePermissions(base)
                val marker = File(base, ".rin_runtime_version_${sanitizeToken(spec.version)}")
                if (!marker.exists()) {
                    Files.newOutputStream(
                        marker.toPath(),
                        StandardOpenOption.CREATE_NEW,
                        StandardOpenOption.WRITE,
                    ).use { output -> output.write(spec.version.toByteArray(Charsets.UTF_8)) }
                }

                val status = runtime.inspect(base)
                if (!status.ready) error(status.detail ?: "Runtime package extracted, but required files are still missing")
                callback(ImageInstallEvent.Complete(spec.version))
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

    private fun selectPublishedPackage(manifest: JSONObject): JSONObject? {
        val packages = manifest.optJSONArray("packages") ?: return null
        val compatible = mutableListOf<JSONObject>()
        for (i in 0 until packages.length()) {
            val item = packages.optJSONObject(i) ?: continue
            if (deviceMatches(item)) compatible += item
        }
        return compatible.firstOrNull {
            it.optString("status", "published").equals("published", ignoreCase = true)
        }
    }

    private fun deviceMatches(item: JSONObject): Boolean {
        val requiredAbis = jsonStrings(item.optJSONArray("abis"))
        val deviceAbis = Build.SUPPORTED_ABIS.map { it.lowercase() }
        val abiOk = requiredAbis.isEmpty() || requiredAbis.any { it == "*" || deviceAbis.contains(it.lowercase()) }
        if (!abiOk) return false

        val requiredChipsets = jsonStrings(item.optJSONArray("chipsets"))
        if (requiredChipsets.isEmpty() || requiredChipsets.any { it == "*" }) return true
        val deviceTokens = listOf(Build.SOC_MODEL, Build.HARDWARE, Build.BOARD)
            .filter { it.isNotBlank() }
            .map { it.lowercase() }
        return requiredChipsets.any { required ->
            val needle = required.lowercase()
            deviceTokens.any { actual -> actual.contains(needle) || needle.contains(actual) }
        }
    }

    private fun parsePackage(item: JSONObject): PackageSpec {
        val version = item.getString("version")
        val archive = item.getJSONObject("archive")
        val archiveSha = archive.getString("sha256").lowercase()
        require(archiveSha.matches(Regex("^[0-9a-f]{64}$"))) { "Invalid archive SHA-256" }
        val partsJson = item.getJSONArray("parts")
        require(partsJson.length() > 0) { "Runtime package has no downloadable parts" }
        val parts = (0 until partsJson.length()).map { index ->
            val part = partsJson.getJSONObject(index)
            val url = part.getString("url")
            val sha = part.getString("sha256").lowercase()
            require(url.startsWith("https://")) { "Only HTTPS runtime packages are allowed" }
            require(sha.matches(Regex("^[0-9a-f]{64}$"))) { "Invalid part SHA-256" }
            PackagePart(
                name = part.optString("name", "part-${index + 1}.bin"),
                url = url,
                sha256 = sha,
                bytes = part.optLong("bytes", 0L),
            )
        }
        return PackageSpec(
            id = item.optString("id", version),
            version = version,
            archiveSha256 = archiveSha,
            archiveBytes = archive.optLong("bytes", parts.sumOf { it.bytes }),
            parts = parts,
        )
    }

    private fun unavailableMessage(manifest: JSONObject): String {
        val packages = manifest.optJSONArray("packages")
        var matching: JSONObject? = null
        if (packages != null) {
            for (i in 0 until packages.length()) {
                val item = packages.optJSONObject(i) ?: continue
                if (deviceMatches(item)) {
                    matching = item
                    break
                }
            }
        }
        val source = matching ?: manifest
        val language = context.resources.configuration.locales[0]?.language.orEmpty()
        val key = if (language.equals("zh", ignoreCase = true)) "message_zh" else "message"
        val fallback = if (matching != null) {
            "A matching NPU image package is being prepared for this device"
        } else {
            "No matching NPU image package is listed for this device"
        }
        return source.optString(key, manifest.optString(key, fallback))
    }

    private fun jsonStrings(array: JSONArray?): List<String> {
        if (array == null) return emptyList()
        val values = mutableListOf<String>()
        for (i in 0 until array.length()) {
            val value = array.optString(i, "").trim()
            if (value.isNotEmpty()) values += value
        }
        return values
    }

    private fun obtainPart(
        preferred: File,
        spec: PackagePart,
        completedBefore: Long,
        totalAll: Long,
        callback: (ImageInstallEvent) -> Unit,
    ): File {
        if (preferred.isFile && spec.bytes > 0 && preferred.length() == spec.bytes &&
            sha256(preferred).equals(spec.sha256, ignoreCase = true)
        ) return preferred

        var target = preferred
        var existing = if (target.isFile) target.length() else 0L
        if (spec.bytes > 0 && existing >= spec.bytes) {
            target = uniqueSibling(preferred)
            existing = 0L
        }
        var conn = open(URL(spec.url), existing)
        if (existing > 0 && conn.responseCode != HttpURLConnection.HTTP_PARTIAL) {
            conn.disconnect()
            target = uniqueSibling(preferred)
            existing = 0L
            conn = open(URL(spec.url), 0)
        }
        require(conn.responseCode == HttpURLConnection.HTTP_OK || conn.responseCode == HttpURLConnection.HTTP_PARTIAL) {
            "Runtime package HTTP ${conn.responseCode}"
        }
        val contentLength = conn.contentLengthLong.coerceAtLeast(0L)
        val partTotal = if (spec.bytes > 0) spec.bytes else existing + contentLength
        val options = if (existing > 0) {
            arrayOf(StandardOpenOption.APPEND, StandardOpenOption.WRITE)
        } else {
            arrayOf(StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)
        }
        Files.newOutputStream(target.toPath(), *options).use { out ->
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
                    val overallDone = completedBefore + done
                    val overallTotal = if (totalAll > 0) totalAll else completedBefore + partTotal
                    val pct = if (overallTotal > 0) ((overallDone * 100L) / overallTotal).toInt().coerceIn(0, 100) else 0
                    if (pct != lastPct) {
                        lastPct = pct
                        callback(ImageInstallEvent.Downloading(pct, overallDone, overallTotal))
                    }
                }
            }
        }
        conn.disconnect()
        if (spec.bytes > 0) require(target.length() == spec.bytes) { "Downloaded package part size mismatch" }
        return target
    }

    private fun uniqueSibling(preferred: File): File = File(
        preferred.parentFile,
        preferred.name + ".retry-" + System.currentTimeMillis(),
    )

    private fun extractSafelyCreateNew(zipFile: File, destination: File) {
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
                    require(!out.exists()) { "Runtime target already exists: ${entry.name}" }
                    out.parentFile?.mkdirs()
                    Files.newOutputStream(
                        out.toPath(),
                        StandardOpenOption.CREATE_NEW,
                        StandardOpenOption.WRITE,
                    ).use { output -> zis.copyTo(output, 1024 * 1024) }
                }
                zis.closeEntry()
            }
        }
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
        setRequestProperty("User-Agent", "Rin-NPU-Agent/1.5.1")
        setRequestProperty("Cache-Control", "no-cache")
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

    private fun sanitizeToken(value: String): String = value.replace(Regex("[^0-9A-Za-z_.-]+"), "_")

    private fun checkCancelled() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Install cancelled")
    }

    companion object {
        const val MANIFEST_URL = "https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json"
    }
}
