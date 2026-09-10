package com.geniex.demo.image

import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.StatFs
import com.geniex.demo.BuildConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

enum class ModelSourcePreference { AUTO, GITHUB, CHINA_MIRROR }

data class DeviceNpuTarget(
    val socModel: String,
    val qnnSocId: Int?,
    val dspArch: Int?,
    val tokens: List<String>,
) {
    val label: String get() = buildString {
        append(socModel.ifBlank { "Unknown SoC" })
        qnnSocId?.let { append(" · QNN $it") }
        dspArch?.let { append(" · V$it") }
    }
}

data class InstalledModelPack(
    val id: String,
    val version: String,
    val directory: File,
    val resolutions: List<ImageResolution>,
    val loraAbi: String,
)

data class ModelPackRelease(
    val id: String,
    val version: String,
    val runtimeAbi: Int,
    val manifestSchema: Int,
    val minAppRuntime: Int,
    val minAppVersionCode: Int,
    val socModels: List<String>,
    val qnnSocId: Int?,
    val dspArch: Int?,
    val resolutions: List<ImageResolution>,
    val loraAbi: String,
    val loraRank: Int,
    val totalDownloadBytes: Long,
    val manifestPath: String,
    val manifestSha256: String,
    val manifestSignaturePath: String,
    val githubBase: String,
    val mirrorBase: String,
    val notesZh: String,
    val notesEn: String,
)

sealed class ModelUpdateCheck {
    abstract val device: DeviceNpuTarget
    data class UpToDate(override val device: DeviceNpuTarget, val local: InstalledModelPack, val remote: ModelPackRelease, val source: ModelSourcePreference) : ModelUpdateCheck()
    data class UpdateAvailable(override val device: DeviceNpuTarget, val local: InstalledModelPack?, val remote: ModelPackRelease, val source: ModelSourcePreference) : ModelUpdateCheck()
    data class AppUpdateRequired(override val device: DeviceNpuTarget, val remote: ModelPackRelease, val source: ModelSourcePreference) : ModelUpdateCheck()
    data class Ignored(override val device: DeviceNpuTarget, val local: InstalledModelPack?, val remote: ModelPackRelease, val source: ModelSourcePreference) : ModelUpdateCheck()
    data class Cancelled(override val device: DeviceNpuTarget) : ModelUpdateCheck()
    data class Unsupported(override val device: DeviceNpuTarget, val message: String) : ModelUpdateCheck()
    data class Error(override val device: DeviceNpuTarget, val message: String) : ModelUpdateCheck()
}

sealed class ModelInstallEvent {
    data class Starting(val release: ModelPackRelease) : ModelInstallEvent()
    data class Downloading(val index: Int, val count: Int, val name: String, val done: Long, val total: Long, val speed: Long, val source: ModelSourcePreference) : ModelInstallEvent()
    data class Verifying(val name: String) : ModelInstallEvent()
    data class Switching(val version: String) : ModelInstallEvent()
    data class Complete(val pack: InstalledModelPack) : ModelInstallEvent()
    data class Failure(val message: String) : ModelInstallEvent()
}

class ModelPackUpdateManager(private val context: Context) {
    private data class FileSpec(val path: String, val bytes: Long, val sha256: String)
    private data class RemoteManifest(val rawBytes: ByteArray, val signatureRaw: String, val root: JSONObject, val files: List<FileSpec>)

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val cancelled = AtomicBoolean(false)
    private val downloader = RuntimePartDownloader(cancelled)
    @Volatile private var worker: Thread? = null

    val baseDir: File get() = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "sdxl_qnn")

    fun isBusy() = worker?.isAlive == true

    fun deviceTarget(): DeviceNpuTarget {
        val tokens = listOf(Build.SOC_MODEL, Build.HARDWARE, Build.BOARD).map { it.orEmpty().trim() }.filter { it.isNotBlank() }.distinct()
        val joined = tokens.joinToString(" ").uppercase(Locale.ROOT)
        val sm8750 = joined.contains("SM8750")
        return DeviceNpuTarget(Build.SOC_MODEL.orEmpty().ifBlank { tokens.firstOrNull().orEmpty() }, if (sm8750) 69 else null, if (sm8750) 79 else null, tokens)
    }

    fun sourcePreference(): ModelSourcePreference = runCatching {
        ModelSourcePreference.valueOf(prefs.getString(KEY_SOURCE, ModelSourcePreference.AUTO.name) ?: ModelSourcePreference.AUTO.name)
    }.getOrDefault(ModelSourcePreference.AUTO)

    fun setSourcePreference(value: ModelSourcePreference) { prefs.edit().putString(KEY_SOURCE, value.name).apply() }
    fun lastCheckAt(): Long = prefs.getLong(KEY_LAST_CHECK_AT, 0L)
    fun ignoreVersion(release: ModelPackRelease) { prefs.edit().putString(KEY_IGNORED_ID, release.id).putString(KEY_IGNORED_VERSION, release.version).apply() }
    fun clearIgnoredVersion() { prefs.edit().remove(KEY_IGNORED_ID).remove(KEY_IGNORED_VERSION).apply() }
    private fun isIgnored(release: ModelPackRelease): Boolean = prefs.getString(KEY_IGNORED_ID, null) == release.id && prefs.getString(KEY_IGNORED_VERSION, null) == release.version
    fun currentPack(): InstalledModelPack? = loadCurrentPack()

    fun checkAsync(callback: (ModelUpdateCheck) -> Unit): Boolean {
        if (isBusy()) return false
        cancelled.set(false)
        worker = thread(name = "rin-model-pack-check") {
            val result = try { checkNow() } catch (x: Throwable) {
                if (cancelled.get() || x is InterruptedException) ModelUpdateCheck.Cancelled(deviceTarget())
                else ModelUpdateCheck.Error(deviceTarget(), x.message ?: x.javaClass.simpleName)
            }
            if (result !is ModelUpdateCheck.Cancelled) prefs.edit().putLong(KEY_LAST_CHECK_AT, System.currentTimeMillis()).apply()
            worker = null
            callback(result)
        }
        return true
    }

    fun installAsync(release: ModelPackRelease, callback: (ModelInstallEvent) -> Unit): Boolean {
        if (isBusy()) return false
        cancelled.set(false)
        worker = thread(name = "rin-model-pack-install") {
            try {
                callback(ModelInstallEvent.Starting(release))
                require(StorageAccess.granted()) { "Storage permission is required" }
                installRelease(release, callback)
            } catch (t: Throwable) {
                if (!cancelled.get()) callback(ModelInstallEvent.Failure(t.message ?: t.javaClass.simpleName))
            } finally { worker = null }
        }
        return true
    }

    fun cancel() { cancelled.set(true); downloader.cancel(); worker?.interrupt() }

    fun checkNow(): ModelUpdateCheck {
        val device = deviceTarget()
        val (index, source) = fetchIndex()
        val releases = mutableListOf<ModelPackRelease>()
        val array = index.optJSONArray("packages") ?: JSONArray()
        for (i in 0 until array.length()) {
            val item = array.optJSONObject(i) ?: continue
            if (!item.optString("status", "published").equals("published", true)) continue
            parseRelease(item)?.takeIf { hardwareMatches(device, it) }?.let(releases::add)
        }
        val remote = releases.maxWithOrNull(Comparator { a, b -> compareVersions(a.version, b.version) })
            ?: return ModelUpdateCheck.Unsupported(device, "No published NPU image model pack matches this SoC")
        val local = loadCurrentPack()
        return when {
            local != null && local.id == remote.id && compareVersions(remote.version, local.version) <= 0 -> ModelUpdateCheck.UpToDate(device, local, remote, source)
            isIgnored(remote) -> ModelUpdateCheck.Ignored(device, local, remote, source)
            remote.runtimeAbi != RUNTIME_ABI || remote.manifestSchema > MANIFEST_SCHEMA || remote.minAppRuntime > APP_RUNTIME_VERSION || remote.minAppVersionCode > BuildConfig.VERSION_CODE || remote.loraRank > MAX_LORA_RANK -> ModelUpdateCheck.AppUpdateRequired(device, remote, source)
            else -> ModelUpdateCheck.UpdateAvailable(device, local, remote, source)
        }
    }

    private fun fetchIndex(): Pair<JSONObject, ModelSourcePreference> {
        var last: Throwable? = null
        for (source in sourceOrder()) {
            val url = if (source == ModelSourcePreference.CHINA_MIRROR) INDEX_MODELSCOPE_URL else INDEX_GITHUB_URL
            val signatureUrl = if (source == ModelSourcePreference.CHINA_MIRROR) INDEX_MODELSCOPE_SIG_URL else INDEX_GITHUB_SIG_URL
            try {
                val raw = fetchText(url)
                val detached = fetchText(signatureUrl).trim()
                require(verifyIndexSignature(raw, detached)) { "Model index signature verification failed" }
                return JSONObject(raw) to source
            } catch (t: Throwable) { last = t }
        }
        throw last ?: IllegalStateException("No update source available")
    }

    private fun sourceOrder(): List<ModelSourcePreference> = when (sourcePreference()) {
        ModelSourcePreference.GITHUB -> listOf(ModelSourcePreference.GITHUB, ModelSourcePreference.CHINA_MIRROR)
        ModelSourcePreference.CHINA_MIRROR -> listOf(ModelSourcePreference.CHINA_MIRROR, ModelSourcePreference.GITHUB)
        ModelSourcePreference.AUTO -> if (isChinaMainland()) listOf(ModelSourcePreference.CHINA_MIRROR, ModelSourcePreference.GITHUB) else listOf(ModelSourcePreference.GITHUB, ModelSourcePreference.CHINA_MIRROR)
    }

    private fun isChinaMainland(): Boolean {
        val now = System.currentTimeMillis()
        val at = prefs.getLong(KEY_REGION_AT, 0L)
        prefs.getString(KEY_REGION, null)?.takeIf { now - at in 0 until REGION_CACHE_MS }?.let { return it == "CN" }
        val country = runCatching {
            fetchText(CLOUDFLARE_TRACE_URL, 5_000, 8_000).lineSequence().firstOrNull { it.startsWith("loc=") }?.substringAfter('=')?.trim()?.uppercase(Locale.ROOT)
        }.getOrNull()
        if (!country.isNullOrBlank()) prefs.edit().putString(KEY_REGION, country).putLong(KEY_REGION_AT, now).apply()
        return country == "CN"
    }

    private fun parseRelease(item: JSONObject): ModelPackRelease? = runCatching {
        val target = item.getJSONObject("target")
        val lora = item.getJSONObject("lora")
        val sources = item.getJSONObject("sources")
        val manifest = item.getJSONObject("manifest")
        val signature = manifest.getJSONObject("signature")
        require(signature.optString("algorithm") == ModelManifestSignatureVerifier.ALGORITHM)
        val resolutionArray = item.optJSONArray("supported_resolutions") ?: item.optJSONArray("resolutions") ?: error("Missing supported_resolutions")
        val sha = manifest.getString("sha256").lowercase(Locale.ROOT)
        require(sha.matches(SHA_RE))
        val loraAbi = lora.optString("abi_signature").lowercase(Locale.ROOT)
        val loraRank = lora.optInt("rank_capacity", 0)
        require(loraAbi.matches(SHA_RE) && loraRank in 1..MAX_LORA_RANK)
        ModelPackRelease(
            item.getString("id"), item.getString("version"), item.optInt("runtime_abi", RUNTIME_ABI),
            manifest.optInt("schema", item.optInt("manifest_schema", MANIFEST_SCHEMA)), item.optInt("min_app_runtime", 1), item.optInt("min_app_version_code", 1),
            jsonStrings(target.optJSONArray("soc_models")), target.optInt("qnn_soc_id", -1).takeIf { it >= 0 }, target.optInt("dsp_arch", -1).takeIf { it >= 0 },
            parseResolutions(resolutionArray), loraAbi, loraRank,
            item.optLong("total_download_bytes", item.optLong("total_model_bytes", 0L)).coerceAtLeast(0L),
            manifest.getString("path").trimStart('/'), sha, signature.getString("path").trimStart('/'),
            sources.getString("github_base").trimEnd('/'), sources.getString("modelscope_base").trimEnd('/'),
            item.optString("notes_zh"), item.optString("notes_en"),
        )
    }.getOrNull()

    private fun hardwareMatches(device: DeviceNpuTarget, release: ModelPackRelease): Boolean {
        if (release.socModels.isNotEmpty() && release.socModels.none { req -> req == "*" || device.tokens.any { actual -> actual.contains(req, true) || req.contains(actual, true) } }) return false
        if (device.qnnSocId != null && release.qnnSocId != null && device.qnnSocId != release.qnnSocId) return false
        if (device.dspArch != null && release.dspArch != null && device.dspArch != release.dspArch) return false
        return true
    }

    private fun installRelease(release: ModelPackRelease, callback: (ModelInstallEvent) -> Unit) {
        val manifest = fetchManifest(release)
        validateManifest(release, manifest.root)
        val root = packRoot().apply { mkdirs() }
        val finalName = "${safe(release.id)}_${safe(release.version)}"
        val finalDir = File(root, finalName)
        if (finalDir.isDirectory && readInstalledPack(finalDir) != null) {
            switchCurrent(root, finalName, release)
            callback(ModelInstallEvent.Complete(readInstalledPack(finalDir)!!)); return
        }
        val staging = File(root, ".staging_$finalName").apply { mkdirs() }
        val reusableDir = loadCurrentPack()?.takeIf { it.id == release.id }?.directory
        val total = manifest.files.sumOf { it.bytes }
        require(StatFs(root.absolutePath).availableBytes > total + RESERVE_BYTES) { "Not enough free storage for the model update" }
        var completed = manifest.files.sumOf { s -> File(staging, s.path).takeIf { it.isFile && it.length() == s.bytes }?.length() ?: 0L }
        manifest.files.forEachIndexed { index, spec ->
            checkCancelled()
            val target = inside(staging, spec.path); target.parentFile?.mkdirs()
            if (target.isFile && target.length() == spec.bytes && sha256(target).equals(spec.sha256, true)) return@forEachIndexed
            if (reusableDir != null && !target.exists()) {
                val source = runCatching { inside(reusableDir, spec.path) }.getOrNull()
                if (source != null && source.isFile && source.length() == spec.bytes) {
                    callback(ModelInstallEvent.Verifying("本机复用 · ${spec.path}"))
                    if (VerifiedFileReuse.copy(source, target, spec.bytes, spec.sha256) { cancelled.get() || Thread.currentThread().isInterrupted }) {
                        completed += spec.bytes
                        callback(ModelInstallEvent.Downloading(index + 1, manifest.files.size, "本机复用 · ${spec.path}", completed, total, 0L, ModelSourcePreference.AUTO))
                        return@forEachIndexed
                    }
                }
            }
            val part = File(target.parentFile, target.name + ".part")
            var got: File? = null; var last: Throwable? = null; var used = ModelSourcePreference.GITHUB
            for ((source, base) in sourceBases(release)) {
                try {
                    used = source
                    got = downloader.download(part, "$base/${spec.path}", spec.bytes, RuntimePartDownloader.DEFAULT_THREADS) { done, _, speed, _, _ ->
                        callback(ModelInstallEvent.Downloading(index + 1, manifest.files.size, spec.path, completed + done, total, speed, source))
                    }
                    break
                } catch (t: Throwable) { last = t }
            }
            val file = got ?: throw last ?: IllegalStateException("Download failed: ${spec.path}")
            callback(ModelInstallEvent.Verifying(spec.path))
            require(file.length() == spec.bytes && sha256(file).equals(spec.sha256, true)) { "Integrity check failed: ${spec.path}" }
            downloader.cleanupState(file); target.delete(); Files.move(file.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            completed += spec.bytes
            callback(ModelInstallEvent.Downloading(index + 1, manifest.files.size, spec.path, completed, total, 0L, used))
        }
        File(staging, "model_manifest.json.pending").writeBytes(manifest.rawBytes)
        File(staging, LOCAL_MANIFEST_SIGNATURE_FILE).writeText(manifest.signatureRaw, Charsets.UTF_8)
        val pending = File(staging, "model_manifest.json.pending"); val committed = File(staging, "model_manifest.json"); committed.delete(); require(pending.renameTo(committed))
        require(readInstalledPack(staging) != null) { "Downloaded model pack failed validation" }
        callback(ModelInstallEvent.Switching(release.version))
        if (finalDir.exists()) finalDir.deleteRecursively()
        require(staging.renameTo(finalDir)) { "Could not activate model pack" }
        switchCurrent(root, finalName, release)
        callback(ModelInstallEvent.Complete(readInstalledPack(finalDir) ?: error("Activated pack is invalid")))
    }

    private fun fetchManifest(release: ModelPackRelease): RemoteManifest {
        var last: Throwable? = null
        for ((_, base) in sourceBases(release)) {
            try {
                val rawBytes = fetchBytes("$base/${release.manifestPath}", MAX_MANIFEST_BYTES)
                val actualSha = sha256(rawBytes)
                require(actualSha.equals(release.manifestSha256, true)) { "Manifest SHA-256 mismatch" }
                val signatureRaw = fetchBytes("$base/${release.manifestSignaturePath}", MAX_SIGNATURE_BYTES).toString(Charsets.UTF_8)
                ModelManifestSignatureVerifier.verify(rawBytes, JSONObject(signatureRaw), actualSha)
                val root = JSONObject(rawBytes.toString(Charsets.UTF_8))
                return RemoteManifest(rawBytes, signatureRaw, root, parseFiles(root.getJSONArray("files")))
            } catch (x: Throwable) { last = x }
        }
        throw last ?: IllegalStateException("Could not download and verify model manifest")
    }

    private fun validateManifest(release: ModelPackRelease, root: JSONObject) {
        require(root.optInt("schema") == release.manifestSchema && release.manifestSchema <= MANIFEST_SCHEMA)
        require(root.optBoolean("complete", false) && root.getString("id") == release.id && root.getString("version") == release.version)
        require(root.optInt("runtime_abi") == RUNTIME_ABI)
        require(root.optInt("min_app_runtime", 1) == release.minAppRuntime && release.minAppRuntime <= APP_RUNTIME_VERSION)
        val target = root.getJSONObject("target"); release.qnnSocId?.let { require(target.optInt("qnn_soc_id") == it) }; release.dspArch?.let { require(target.optInt("dsp_arch") == it) }
        val lora = root.getJSONObject("lora"); require(lora.optBoolean("external_dynamic_ab", false) && lora.optInt("rank_capacity") == release.loraRank && lora.optString("abi_signature").equals(release.loraAbi, true))
        require(parseResolutions(resolutionArray(root)) == release.resolutions)
    }

    private fun switchCurrent(root: File, directory: String, release: ModelPackRelease) {
        val tmp = File(root, "current.json.tmp"); val target = File(root, "current.json")
        tmp.writeText(JSONObject().put("schema", 1).put("directory", directory).put("id", release.id).put("version", release.version).toString(2), Charsets.UTF_8)
        try { Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE) }
        catch (_: Throwable) { target.delete(); require(tmp.renameTo(target)) }
    }

    private fun loadCurrentPack(): InstalledModelPack? {
        val root = packRoot(); val pointer = File(root, "current.json")
        if (!pointer.isFile || pointer.length() !in 2..65_536) return null
        val pointed = runCatching { inside(root, JSONObject(pointer.readText(Charsets.UTF_8)).getString("directory")) }.getOrNull() ?: return null
        return readInstalledPack(pointed)
    }

    private fun readInstalledPack(dir: File): InstalledModelPack? = runCatching {
        val file = File(dir, "model_manifest.json")
        val signatureFile = File(dir, LOCAL_MANIFEST_SIGNATURE_FILE)
        if (!file.isFile || file.length() !in 2..MAX_MANIFEST_BYTES.toLong() || !signatureFile.isFile || signatureFile.length() !in 2..MAX_SIGNATURE_BYTES.toLong()) return@runCatching null
        val manifestBytes = file.readBytes()
        val actualSha = sha256(manifestBytes)
        ModelManifestSignatureVerifier.verify(manifestBytes, JSONObject(signatureFile.readText(Charsets.UTF_8)), actualSha)
        val root = JSONObject(manifestBytes.toString(Charsets.UTF_8))
        if (root.optInt("schema") !in 1..MANIFEST_SCHEMA || !root.optBoolean("complete", false) || root.optInt("runtime_abi") != RUNTIME_ABI || root.optInt("min_app_runtime", 1) > APP_RUNTIME_VERSION) return@runCatching null
        val target = root.getJSONObject("target"); val device = deviceTarget(); if (device.qnnSocId != null && target.optInt("qnn_soc_id") != device.qnnSocId) return@runCatching null; if (device.dspArch != null && target.optInt("dsp_arch") != device.dspArch) return@runCatching null
        val lora = root.getJSONObject("lora"); if (!lora.optBoolean("external_dynamic_ab", false) || lora.optInt("rank_capacity") !in 1..MAX_LORA_RANK || !lora.optString("abi_signature").lowercase(Locale.ROOT).matches(SHA_RE)) return@runCatching null
        if (!parseFiles(root.getJSONArray("files")).all { s -> inside(dir, s.path).let { it.isFile && it.length() == s.bytes } }) return@runCatching null
        InstalledModelPack(root.getString("id"), root.getString("version"), dir, parseResolutions(resolutionArray(root)), lora.optString("abi_signature").lowercase(Locale.ROOT))
    }.getOrNull()

    private fun parseFiles(array: JSONArray): List<FileSpec> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i); val path = o.getString("path").replace('\\', '/').trimStart('/'); val sha = o.getString("sha256").lowercase(Locale.ROOT); val bytes = o.getLong("bytes")
        require(path.matches(PATH_RE) && path.split('/').none { it.isBlank() || it == "." || it == ".." } && bytes > 0 && sha.matches(SHA_RE)); FileSpec(path, bytes, sha)
    }

    private fun resolutionArray(root: JSONObject): JSONArray = root.optJSONArray("supported_resolutions") ?: root.getJSONArray("resolutions")
    private fun parseResolutions(array: JSONArray): List<ImageResolution> = (0 until array.length()).map { i -> array.getJSONObject(i).let { ImageResolution(it.getInt("width"), it.getInt("height")) }.also { r -> require(r.width in 64..8192 && r.height in 64..8192 && r.width % 8 == 0 && r.height % 8 == 0) } }
    private fun jsonStrings(array: JSONArray?): List<String> = if (array == null) emptyList() else (0 until array.length()).mapNotNull { array.optString(it).trim().takeIf { s -> s.isNotBlank() } }
    private fun sourceBases(r: ModelPackRelease): List<Pair<ModelSourcePreference, String>> =
        sourceOrder().mapNotNull { source ->
            when (source) {
                ModelSourcePreference.GITHUB ->
                    r.githubBase.takeIf { it.startsWith("https://") }?.let { source to it }
                ModelSourcePreference.CHINA_MIRROR ->
                    r.mirrorBase.takeIf { it.startsWith("https://") }?.let { source to it }
                ModelSourcePreference.AUTO -> null
            }
        }.distinctBy { it.second }
    private fun packRoot() = File(baseDir, "context/model_packs")
    private fun inside(root: File, relative: String): File { val base = root.canonicalFile; val f = File(root, relative).canonicalFile; require(f.path == base.path || f.path.startsWith(base.path + File.separator)); return f }
    private fun safe(v: String) = v.replace(Regex("[^0-9A-Za-z_.-]+"), "_").take(96)

    private fun fetchBytes(raw: String, maxBytes: Int, connect: Int = 12_000, read: Int = 25_000): ByteArray {
        var current = URL(raw)
        repeat(8) {
            checkCancelled()
            val connection = current.openConnection() as HttpURLConnection
            connection.connectTimeout = connect
            connection.readTimeout = read
            connection.instanceFollowRedirects = false
            connection.setRequestProperty("User-Agent", "Rin-NPU-Agent/${BuildConfig.VERSION_NAME}")
            connection.setRequestProperty("Accept-Encoding", "identity")
            try {
                val code = connection.responseCode
                if (code in REDIRECTS) {
                    current = URL(current, connection.getHeaderField("Location") ?: error("Redirect without Location"))
                } else {
                    require(code in 200..299) { "HTTP $code from ${current.host}" }
                    val declared = connection.contentLengthLong
                    require(declared < 0L || declared <= maxBytes.toLong()) { "Remote metadata exceeds size limit" }
                    val output = ByteArrayOutputStream(if (declared in 1..maxBytes.toLong()) declared.toInt() else minOf(maxBytes, 64 * 1024))
                    connection.inputStream.use { input ->
                        val buffer = ByteArray(32 * 1024)
                        var total = 0
                        while (true) {
                            checkCancelled()
                            val count = input.read(buffer)
                            if (count < 0) break
                            total += count
                            require(total <= maxBytes) { "Remote metadata exceeds size limit" }
                            output.write(buffer, 0, count)
                        }
                    }
                    return output.toByteArray()
                }
            } finally { connection.disconnect() }
        }
        error("Too many redirects")
    }

    private fun fetchText(raw: String, connect: Int = 12_000, read: Int = 25_000): String =
        fetchBytes(raw, MAX_TEXT_BYTES, connect, read).toString(Charsets.UTF_8)

    private fun sha256(file: File): String { val d = MessageDigest.getInstance("SHA-256"); file.inputStream().buffered(1024 * 1024).use { input -> val b = ByteArray(1024 * 1024); while (true) { checkCancelled(); val n = input.read(b); if (n < 0) break; d.update(b, 0, n) } }; return d.digest().joinToString("") { "%02x".format(it) } }
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun checkCancelled() { if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Operation cancelled") }

    fun compareVersions(a: String, b: String): Int {
        fun p(v: String) = Regex("\\d+").findAll(v).map { it.value.toIntOrNull() ?: 0 }.toList()
        val aa = p(a)
        val bb = p(b)
        for (i in 0 until maxOf(aa.size, bb.size)) { val x = aa.getOrElse(i) { 0 }; val y = bb.getOrElse(i) { 0 }; if (x != y) return x.compareTo(y) }
        return a.compareTo(b, ignoreCase = true)
    }

    companion object {
        internal fun verifyIndexSignature(raw: String, detachedBase64: String): Boolean = runCatching {
            ModelManifestSignatureVerifier.verifyDetached(INDEX_SIGNING_PUBLIC_RAW_B64, raw.toByteArray(Charsets.UTF_8), detachedBase64)
            true
        }.getOrDefault(false)

        const val RUNTIME_ABI = 1
        const val APP_RUNTIME_VERSION = 1
        const val INDEX_GITHUB_URL = "https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/model-pack-index.json"
        const val INDEX_GITHUB_SIG_URL = "https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/model-pack-index.json.sig"
        const val INDEX_MODELSCOPE_URL = "https://modelscope.cn/models/reimurin/Rin-NPU-Agent/resolve/master/models/model-pack-index.json"
        const val INDEX_MODELSCOPE_SIG_URL = "https://modelscope.cn/models/reimurin/Rin-NPU-Agent/resolve/master/models/model-pack-index.json.sig"
        private const val INDEX_SIGNING_PUBLIC_RAW_B64 = "zE7NARTmhoS2rvE59NYCn1ZbmF4KTv5R/ciIzADc7nk="
        private const val CLOUDFLARE_TRACE_URL = "https://www.cloudflare.com/cdn-cgi/trace"
        const val MANIFEST_SCHEMA = 2
        const val MAX_LORA_RANK = 64
        private const val PREFS = "rin_model_pack_updates"; private const val KEY_SOURCE = "source"; private const val KEY_REGION = "region"; private const val KEY_REGION_AT = "region_at"
        private const val KEY_LAST_CHECK_AT = "last_check_at"; private const val KEY_IGNORED_ID = "ignored_id"; private const val KEY_IGNORED_VERSION = "ignored_version"
        private const val REGION_CACHE_MS = 86_400_000L; private const val RESERVE_BYTES = 768L * 1024L * 1024L
        private const val MAX_TEXT_BYTES = 4 * 1024 * 1024
        private const val MAX_MANIFEST_BYTES = 4 * 1024 * 1024
        private const val MAX_SIGNATURE_BYTES = 16 * 1024
        private const val LOCAL_MANIFEST_SIGNATURE_FILE = "model_manifest.sig.json"
        private val REDIRECTS = setOf(301, 302, 303, 307, 308); private val SHA_RE = Regex("^[0-9a-f]{64}$"); private val PATH_RE = Regex("^[0-9A-Za-z._/-]+$")
    }
}
