package com.geniex.demo.image

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.StatFs
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.io.FileOutputStream
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

    private data class ContextPatch(
        val id: String,
        val target: String,
        val bytes: Long,
        val sha256: String,
        val applyIfSha256: Set<String>,
        val parts: List<PackagePart>,
    )

    private data class PackageSpec(
        val id: String,
        val version: String,
        val archiveSha256: String,
        val archiveBytes: Long,
        val parts: List<PackagePart>,
        val contextPatches: List<ContextPatch>,
    )

    private data class StreamPart(
        val name: String,
        val bytes: Long,
        val openStream: () -> InputStream,
        val cleanup: () -> Unit = {},
    )

    private val cancelled = AtomicBoolean(false)
    private val downloader = RuntimePartDownloader(cancelled)
    @Volatile private var worker: Thread? = null

    fun isBusy(): Boolean = worker?.isAlive == true

    fun install(callback: (ImageInstallEvent) -> Unit): Boolean {
        if (isBusy()) return false
        cancelled.set(false)
        worker = thread(name = "rin-github-image-installer") {
            try {
                callback(ImageInstallEvent.Checking)
                if (!runtime.hasStorageAccess()) error("Storage permission is required")
                val base = runtime.defaultBaseDir.apply { mkdirs() }
                val spec = loadCompatibleSpec()
                repairContextPatches(spec, base, callback)

                val existing = runtime.inspect(base)
                if (modelPayloadComplete(existing)) {
                    callback(ImageInstallEvent.Verifying)
                    restoreRuntimePermissions(base)
                    if (!existing.ready && File(base, "py_runtime").isDirectory) {
                        installPrivatePythonRuntime(base)
                    }
                    val repaired = runtime.inspect(base)
                    if (repaired.ready) {
                        cleanupCachedParts(spec, base)
                        callback(ImageInstallEvent.Complete(spec.version))
                        return@thread
                    }
                    error(repaired.detail ?: "Existing model payload is complete, but Python runtime repair failed")
                }

                val totalDownloadBytes = spec.parts.sumOf { it.bytes.coerceAtLeast(0L) }
                ensureInstallSpace(base, totalDownloadBytes, spec.archiveBytes)

                val installDir = File(base, ".rin_install").apply { mkdirs() }
                var completedBytes = 0L
                val downloadedParts = mutableListOf<File>()
                spec.parts.forEachIndexed { index, partSpec ->
                    checkCancelled()
                    val safeName = partSpec.name.ifBlank { "part-${index + 1}.bin" }
                    require(!safeName.contains('/') && !safeName.contains('\\')) { "Invalid package part name" }
                    val preferred = File(installDir, safeName + ".part")
                    val partFile = obtainPart(
                        preferred = preferred,
                        spec = partSpec,
                        partIndex = index + 1,
                        partCount = spec.parts.size,
                        completedBefore = completedBytes,
                        totalAll = totalDownloadBytes,
                        callback = callback,
                    )
                    callback(ImageInstallEvent.Verifying)
                    require(sha256(partFile).equals(partSpec.sha256, ignoreCase = true)) {
                        "Package part SHA-256 mismatch: ${partSpec.name}"
                    }
                    downloader.cleanupState(partFile)
                    downloadedParts += partFile
                    completedBytes += if (partSpec.bytes > 0L) partSpec.bytes else partFile.length()
                    callback(
                        ImageInstallEvent.Downloading(
                            percent = if (totalDownloadBytes > 0L) ((completedBytes * 100L) / totalDownloadBytes).toInt().coerceIn(0, 100) else 100,
                            done = completedBytes,
                            total = totalDownloadBytes,
                            partIndex = index + 1,
                            partCount = spec.parts.size,
                            partDone = partFile.length(),
                            partTotal = if (partSpec.bytes > 0L) partSpec.bytes else partFile.length(),
                            bytesPerSecond = 0L,
                            etaSeconds = 0L,
                            threads = RuntimePartDownloader.DEFAULT_THREADS,
                        ),
                    )
                }

                val sources = downloadedParts.map { file ->
                    StreamPart(
                        name = file.name,
                        bytes = file.length(),
                        openStream = { FileInputStream(file) },
                        cleanup = {
                            downloader.cleanupState(file)
                            file.delete()
                        },
                    )
                }
                installFromSources(spec, sources, base, callback, cleanupSources = true)
            } catch (t: Throwable) {
                if (!cancelled.get()) callback(ImageInstallEvent.Failure(t.message ?: t.javaClass.simpleName))
            } finally {
                worker = null
            }
        }
        return true
    }

    fun importFromPublicDownloads(callback: (ImageInstallEvent) -> Unit): Boolean {
        if (isBusy()) return false
        cancelled.set(false)
        worker = thread(name = "rin-browser-runtime-import") {
            try {
                callback(ImageInstallEvent.Checking)
                if (!runtime.hasStorageAccess()) error("Storage permission is required")
                val base = runtime.defaultBaseDir.apply { mkdirs() }
                val spec = loadCompatibleSpec()
                val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                require(downloads.isDirectory) { "System Download folder is unavailable" }
                val children = downloads.walkTopDown().maxDepth(2).filter { it.isFile }.toList()
                val expectedTotal = spec.parts.sumOf { it.bytes.coerceAtLeast(0L) }
                var observedBytes = 0L
                val sources = mutableListOf<StreamPart>()
                spec.parts.forEach { part ->
                    val exact = children.filter { it.name == part.name }.maxByOrNull { it.lastModified() }
                    val progressCandidate = exact ?: children
                        .filter { it.name.startsWith(part.name) }
                        .maxByOrNull { it.length() }
                    observedBytes += minOf(progressCandidate?.length() ?: 0L, part.bytes.coerceAtLeast(0L))
                    if (exact != null && (part.bytes <= 0L || exact.length() == part.bytes)) {
                        sources += StreamPart(
                            name = part.name,
                            bytes = exact.length(),
                            openStream = { FileInputStream(exact) },
                        )
                    }
                }
                callback(ImageInstallEvent.BrowserWaiting(sources.size, spec.parts.size, observedBytes, expectedTotal))
                if (sources.size != spec.parts.size) return@thread
                ensureInstallSpace(base, 0L, spec.archiveBytes)
                spec.parts.zip(sources).forEach { (expected, source) ->
                    checkCancelled()
                    callback(ImageInstallEvent.Verifying)
                    require(sha256(source).equals(expected.sha256, ignoreCase = true)) {
                        "Browser-downloaded part SHA-256 mismatch: ${expected.name}"
                    }
                }
                installFromSources(spec, sources, base, callback, cleanupSources = false)
            } catch (t: Throwable) {
                if (!cancelled.get()) callback(ImageInstallEvent.Failure(t.message ?: t.javaClass.simpleName))
            } finally {
                worker = null
            }
        }
        return true
    }

    fun importFromDocumentTree(treeUri: Uri, callback: (ImageInstallEvent) -> Unit): Boolean {
        if (isBusy()) return false
        cancelled.set(false)
        worker = thread(name = "rin-browser-runtime-import") {
            try {
                callback(ImageInstallEvent.Checking)
                if (!runtime.hasStorageAccess()) error("Storage permission is required")
                val base = runtime.defaultBaseDir.apply { mkdirs() }
                val spec = loadCompatibleSpec()
                val root = DocumentFile.fromTreeUri(context, treeUri) ?: error("Could not open selected download folder")
                require(root.exists() && root.isDirectory) { "Selected download folder is unavailable" }
                val children = root.listFiles().toList()
                val expectedTotal = spec.parts.sumOf { it.bytes.coerceAtLeast(0L) }
                var observedBytes = 0L
                val sources = mutableListOf<StreamPart>()

                spec.parts.forEach { part ->
                    val exact = children.firstOrNull { it.name == part.name }
                    val progressCandidate = exact ?: children
                        .filter { candidate -> candidate.name?.startsWith(part.name) == true }
                        .maxByOrNull { it.length() }
                    observedBytes += minOf(progressCandidate?.length() ?: 0L, part.bytes.coerceAtLeast(0L))
                    if (exact != null && exact.isFile && (part.bytes <= 0L || exact.length() == part.bytes)) {
                        val uri = exact.uri
                        sources += StreamPart(
                            name = part.name,
                            bytes = exact.length(),
                            openStream = {
                                context.contentResolver.openInputStream(uri)
                                    ?: error("Could not open browser-downloaded part ${part.name}")
                            },
                        )
                    }
                }

                callback(
                    ImageInstallEvent.BrowserWaiting(
                        foundParts = sources.size,
                        totalParts = spec.parts.size,
                        done = observedBytes,
                        total = expectedTotal,
                    ),
                )
                if (sources.size != spec.parts.size) return@thread

                ensureInstallSpace(base, 0L, spec.archiveBytes)
                spec.parts.zip(sources).forEach { (expected, source) ->
                    checkCancelled()
                    callback(ImageInstallEvent.Verifying)
                    require(sha256(source).equals(expected.sha256, ignoreCase = true)) {
                        "Browser-downloaded part SHA-256 mismatch: ${expected.name}"
                    }
                }
                installFromSources(spec, sources, base, callback, cleanupSources = false)
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
        downloader.cancel()
        worker?.interrupt()
        worker = null
    }

    private fun loadCompatibleSpec(): PackageSpec {
        val manifest = fetchManifest()
        val packageJson = selectPublishedPackage(manifest)
        if (packageJson == null) error(unavailableMessage(manifest))
        return parsePackage(packageJson)
    }

    private fun fetchManifest(): JSONObject {
        val conn = open(URL(MANIFEST_URL), 0L)
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
        val patchesJson = item.optJSONArray("context_patches")
        val contextPatches = if (patchesJson == null) emptyList() else (0 until patchesJson.length()).map { patchIndex ->
            val patch = patchesJson.getJSONObject(patchIndex)
            val id = patch.optString("id", "context-patch-${patchIndex + 1}").trim()
            val target = patch.getString("target").replace('\\', '/').trimStart('/')
            require(target.isNotBlank() && target.split('/').none { it == ".." || it.isBlank() }) { "Invalid context patch target" }
            val sha = patch.getString("sha256").lowercase()
            require(sha.matches(Regex("^[0-9a-f]{64}$"))) { "Invalid context patch SHA-256" }
            val applyIf = jsonStrings(patch.optJSONArray("apply_if_sha256")).map { it.lowercase() }.toSet()
            require(applyIf.all { it.matches(Regex("^[0-9a-f]{64}$")) }) { "Invalid context patch legacy SHA-256" }
            val patchPartsJson = patch.getJSONArray("parts")
            require(patchPartsJson.length() > 0) { "Context patch has no downloadable parts" }
            val patchParts = (0 until patchPartsJson.length()).map { partIndex ->
                val part = patchPartsJson.getJSONObject(partIndex)
                val url = part.getString("url")
                val partSha = part.getString("sha256").lowercase()
                require(url.startsWith("https://")) { "Only HTTPS context patch parts are allowed" }
                require(partSha.matches(Regex("^[0-9a-f]{64}$"))) { "Invalid context patch part SHA-256" }
                PackagePart(
                    name = part.optString("name", "${id}.part${partIndex + 1}"),
                    url = url,
                    sha256 = partSha,
                    bytes = part.optLong("bytes", 0L),
                )
            }
            ContextPatch(
                id = id,
                target = target,
                bytes = patch.optLong("bytes", patchParts.sumOf { it.bytes }),
                sha256 = sha,
                applyIfSha256 = applyIf,
                parts = patchParts,
            )
        }
        return PackageSpec(
            id = item.optString("id", version),
            version = version,
            archiveSha256 = archiveSha,
            archiveBytes = archive.optLong("bytes", parts.sumOf { it.bytes }),
            parts = parts,
            contextPatches = contextPatches,
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
        partIndex: Int,
        partCount: Int,
        completedBefore: Long,
        totalAll: Long,
        callback: (ImageInstallEvent) -> Unit,
    ): File {
        if (preferred.isFile && !downloader.hasSegmentState(preferred) && spec.bytes > 0L &&
            preferred.length() == spec.bytes && sha256(preferred).equals(spec.sha256, ignoreCase = true)
        ) {
            return preferred
        }
        return downloader.download(
            target = preferred,
            url = spec.url,
            expectedBytes = spec.bytes,
            requestedThreads = RuntimePartDownloader.DEFAULT_THREADS,
        ) { partDone, partTotal, bytesPerSecond, _, threads ->
            val overallDone = completedBefore + partDone
            val overallTotal = if (totalAll > 0L) totalAll else completedBefore + partTotal
            val percent = if (overallTotal > 0L) ((overallDone * 100L) / overallTotal).toInt().coerceIn(0, 100) else 0
            val overallEta = if (bytesPerSecond > 0L && overallTotal > 0L) {
                ((overallTotal - overallDone).coerceAtLeast(0L) / bytesPerSecond)
            } else {
                -1L
            }
            callback(
                ImageInstallEvent.Downloading(
                    percent = percent,
                    done = overallDone,
                    total = overallTotal,
                    partIndex = partIndex,
                    partCount = partCount,
                    partDone = partDone,
                    partTotal = partTotal,
                    bytesPerSecond = bytesPerSecond,
                    etaSeconds = overallEta,
                    threads = threads,
                ),
            )
        }
    }

    private fun installFromSources(
        spec: PackageSpec,
        sources: List<StreamPart>,
        base: File,
        callback: (ImageInstallEvent) -> Unit,
        cleanupSources: Boolean,
    ) {
        checkCancelled()
        val combinedBytes = sources.sumOf { it.bytes }
        if (spec.archiveBytes > 0L) require(combinedBytes == spec.archiveBytes) { "Runtime archive size mismatch" }
        callback(ImageInstallEvent.Verifying)
        require(sha256(sources).equals(spec.archiveSha256, ignoreCase = true)) { "Runtime archive SHA-256 mismatch" }

        callback(ImageInstallEvent.Extracting)
        extractSafelyCreateNew(sources, base)
        repairContextPatches(spec, base, callback)
        restoreRuntimePermissions(base)
        installPrivatePythonRuntime(base)
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
        if (cleanupSources) sources.forEach { runCatching { it.cleanup() } }
        callback(ImageInstallEvent.Complete(spec.version))
    }

    private fun repairContextPatches(
        spec: PackageSpec,
        base: File,
        callback: (ImageInstallEvent) -> Unit,
    ): Boolean {
        if (spec.contextPatches.isEmpty()) return false
        var changed = false
        val root = base.canonicalFile
        for (patch in spec.contextPatches) {
            checkCancelled()
            val target = File(base, patch.target).canonicalFile
            require(target.path == root.path || target.path.startsWith(root.path + File.separator)) { "Unsafe context patch target" }
            if (!target.isFile) continue
            if (patch.bytes > 0L && target.length() == patch.bytes) {
                val current = sha256(target)
                if (current.equals(patch.sha256, ignoreCase = true)) continue
                if (patch.applyIfSha256.isNotEmpty() && current.lowercase() !in patch.applyIfSha256) {
                    error("Context patch ${patch.id} found an unexpected target SHA-256: $current")
                }
            } else {
                val current = sha256(target)
                if (current.equals(patch.sha256, ignoreCase = true)) continue
                if (patch.applyIfSha256.isNotEmpty() && current.lowercase() !in patch.applyIfSha256) {
                    error("Context patch ${patch.id} found an unexpected target SHA-256: $current")
                }
            }

            val patchDir = File(base, ".rin_install/context-patches/${sanitizeToken(patch.id)}").apply { mkdirs() }
            val temp = File(target.parentFile, target.name + ".rin-repair.tmp")
            val state = File(patchDir, "repair-state.txt")
            var completedParts = state.takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull()?.coerceIn(0, patch.parts.size) ?: 0
            val expectedPrefix = patch.parts.take(completedParts).sumOf { it.bytes.coerceAtLeast(0L) }
            if (completedParts == 0) {
                temp.delete()
            } else if (!temp.isFile || temp.length() != expectedPrefix) {
                completedParts = 0
                temp.delete()
                state.delete()
            }
            ensurePatchSpace(base, patch, temp.length())
            var completedDownloadBytes = patch.parts.take(completedParts).sumOf { it.bytes.coerceAtLeast(0L) }
            val totalDownloadBytes = patch.parts.sumOf { it.bytes.coerceAtLeast(0L) }

            for (index in completedParts until patch.parts.size) {
                checkCancelled()
                val part = patch.parts[index]
                val safeName = part.name.ifBlank { "${patch.id}.part${index + 1}" }
                require(!safeName.contains('/') && !safeName.contains('\\')) { "Invalid context patch part name" }
                val partFile = File(patchDir, safeName + ".part")
                val downloaded = obtainPart(
                    preferred = partFile,
                    spec = part,
                    partIndex = index + 1,
                    partCount = patch.parts.size,
                    completedBefore = completedDownloadBytes,
                    totalAll = totalDownloadBytes,
                    callback = callback,
                )
                callback(ImageInstallEvent.Verifying)
                require(sha256(downloaded).equals(part.sha256, ignoreCase = true)) {
                    "Context patch part SHA-256 mismatch: ${part.name}"
                }
                FileOutputStream(temp, true).use { output ->
                    FileInputStream(downloaded).use { input -> input.copyTo(output, BUFFER_SIZE) }
                    output.fd.sync()
                }
                completedDownloadBytes += if (part.bytes > 0L) part.bytes else downloaded.length()
                state.writeText((index + 1).toString(), Charsets.UTF_8)
                downloader.cleanupState(downloaded)
                downloaded.delete()
            }

            callback(ImageInstallEvent.Verifying)
            require(temp.isFile) { "Context patch temporary output is missing" }
            if (patch.bytes > 0L) require(temp.length() == patch.bytes) { "Context patch output size mismatch" }
            val finalSha = sha256(temp)
            require(finalSha.equals(patch.sha256, ignoreCase = true)) { "Context patch output SHA-256 mismatch" }
            target.parentFile?.mkdirs()
            runCatching {
                Files.move(
                    temp.toPath(),
                    target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE,
                )
            }.getOrElse {
                val backup = File(target.parentFile, target.name + ".rin-bad-context.bak")
                backup.delete()
                if (target.exists()) require(target.renameTo(backup)) { "Could not rotate old context before repair" }
                if (!temp.renameTo(target)) {
                    if (backup.exists()) backup.renameTo(target)
                    error("Could not install repaired context")
                }
                backup.delete()
            }
            state.delete()
            patchDir.listFiles()?.forEach { file ->
                downloader.cleanupState(file)
                if (file.isFile) file.delete()
            }
            patchDir.delete()
            changed = true
        }
        return changed
    }

    private fun ensurePatchSpace(base: File, patch: ContextPatch, alreadyWritten: Long) {
        val reserve = 512L * 1024L * 1024L
        val remaining = (patch.bytes - alreadyWritten).coerceAtLeast(0L)
        val largestPart = patch.parts.maxOfOrNull { it.bytes.coerceAtLeast(0L) } ?: 0L
        val needed = remaining + largestPart + reserve
        if (needed <= 0L) return
        val free = StatFs(base.absolutePath).availableBytes
        require(free > needed) { "Not enough free storage for the context repair" }
    }

    private fun modelPayloadComplete(status: ImageRuntimeStatus): Boolean =
        status.storagePermission && status.missingCommonFiles.isEmpty() && status.availableResolutions.isNotEmpty()

    private fun cleanupCachedParts(spec: PackageSpec, base: File) {
        val installDir = File(base, ".rin_install")
        spec.parts.forEach { part ->
            val file = File(installDir, part.name + ".part")
            downloader.cleanupState(file)
            if (file.isFile) file.delete()
        }
        installDir.listFiles()?.filter { it.isFile && (it.name.endsWith(".segments.json") || it.name.endsWith(".range-state.json")) }
            ?.forEach { it.delete() }
    }

    private fun ensureInstallSpace(base: File, downloadBytes: Long, archiveBytes: Long) {
        val reserve = 768L * 1024L * 1024L
        val needed = downloadBytes.coerceAtLeast(0L) + archiveBytes.coerceAtLeast(0L) + reserve
        if (needed <= 0L) return
        val free = StatFs(base.absolutePath).availableBytes
        require(free > needed) { "Not enough free storage for the image runtime package" }
    }

    private fun extractSafelyCreateNew(parts: List<StreamPart>, destination: File) {
        val root = destination.canonicalFile
        ZipInputStream(BufferedInputStream(ConcatenatedInputStream(parts), BUFFER_SIZE)).use { zis ->
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
                    ).use { output -> zis.copyTo(output, BUFFER_SIZE) }
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
        setRequestProperty("User-Agent", "Rin-NPU-Agent/1.5.10")
        setRequestProperty("Cache-Control", "no-cache")
        setRequestProperty("Accept-Encoding", "identity")
        if (rangeStart > 0L) setRequestProperty("Range", "bytes=$rangeStart-")
        connect()
    }

    private fun sha256(file: File): String = sha256(
        StreamPart(file.name, file.length(), { FileInputStream(file) }),
    )

    private fun sha256(source: StreamPart): String {
        val digest = MessageDigest.getInstance("SHA-256")
        source.openStream().use { input -> updateDigest(digest, input) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sha256(sources: List<StreamPart>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        sources.forEach { source -> source.openStream().use { input -> updateDigest(digest, input) } }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun updateDigest(digest: MessageDigest, input: InputStream) {
        val buffer = ByteArray(BUFFER_SIZE)
        while (true) {
            checkCancelled()
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
    }

    private fun sanitizeToken(value: String): String = value.replace(Regex("[^0-9A-Za-z_.-]+"), "_")

    private fun checkCancelled() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Install cancelled")
    }

    private class ConcatenatedInputStream(private val parts: List<StreamPart>) : InputStream() {
        private var index = 0
        private var current: InputStream? = null

        private fun ensureCurrent(): Boolean {
            while (current == null && index < parts.size) {
                current = parts[index++].openStream()
            }
            return current != null
        }

        override fun read(): Int {
            while (ensureCurrent()) {
                val value = current!!.read()
                if (value >= 0) return value
                current!!.close()
                current = null
            }
            return -1
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            if (length == 0) return 0
            while (ensureCurrent()) {
                val n = current!!.read(buffer, offset, length)
                if (n >= 0) return n
                current!!.close()
                current = null
            }
            return -1
        }

        override fun close() {
            runCatching { current?.close() }
            current = null
        }
    }

    companion object {
        const val MANIFEST_URL = "https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json"
        const val RELEASE_PAGE_URL = "https://github.com/reimurin/Rin-NPU-Agent/releases/tag/wai-sm8750-r1-test"
        private const val BUFFER_SIZE = 1024 * 1024
    }
}
