package com.geniex.demo.image

import android.content.Context
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

@Serializable
data class PositivePromptPreset(
    val id: String, val name: String, val note: String = "", val prompt: String,
    val tags: List<String> = emptyList(), val createdAt: Long, val updatedAt: Long,
    val lastUsedAt: Long = 0L, val pinnedAt: Long = 0L,
)
@Serializable
data class NegativePromptPreset(
    val id: String, val name: String, val note: String = "", val prompt: String,
    val isDefault: Boolean = false, val createdAt: Long, val updatedAt: Long,
    val lastUsedAt: Long = 0L, val pinnedAt: Long = 0L,
)
@Serializable
private data class PositivePresetFile(val version: Int = 1, val items: List<PositivePromptPreset> = emptyList())
@Serializable
private data class NegativePresetFile(val version: Int = 1, val items: List<NegativePromptPreset> = emptyList())

class PromptPresetStore internal constructor(private val root: File, private val clock: () -> Long = System::currentTimeMillis) {
    constructor(context: Context) : this(File(context.filesDir, "image_presets"))
    private val lock = synchronized(locks) { locks.getOrPut(root.canonicalPath) { Any() } }
    private val positiveFile = File(root, "positive.json")
    private val negativeFile = File(root, "negative.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = true }

    fun listPositive(): List<PositivePromptPreset> = synchronized(lock) {
        readPositive().items.sortedWith(compareByDescending<PositivePromptPreset> { it.pinnedAt > 0 }
            .thenByDescending { it.pinnedAt }.thenByDescending { it.lastUsedAt }.thenByDescending { it.updatedAt })
    }
    fun listNegative(): List<NegativePromptPreset> = synchronized(lock) {
        readNegative().items.sortedWith(compareByDescending<NegativePromptPreset> { it.pinnedAt > 0 }
            .thenByDescending { it.pinnedAt }.thenByDescending { it.isDefault }
            .thenByDescending { it.lastUsedAt }.thenByDescending { it.updatedAt })
    }
    fun savePositive(id: String? = null, name: String, note: String, prompt: String, tags: List<String>? = null): PositivePromptPreset = synchronized(lock) {
        val file = readPositive()
        val old = id?.let { key -> file.items.firstOrNull { it.id == key } }
        require(id == null || old != null) { "该预设已被删除，请刷新列表" }
        val now = clock()
        val item = PositivePromptPreset(old?.id ?: UUID.randomUUID().toString(), cleanName(name),
            note.trim().take(MAX_NOTE_CHARS), cleanPrompt(prompt),
            tags?.map { it.trim() }?.filter { it.isNotBlank() }?.distinct()?.take(32) ?: old?.tags.orEmpty(),
            old?.createdAt ?: now, now, old?.lastUsedAt ?: 0L, old?.pinnedAt ?: 0L)
        writePositive(file.copy(items = file.items.filterNot { it.id == item.id } + item)); item
    }
    fun saveNegative(id: String? = null, name: String, note: String, prompt: String, makeDefault: Boolean? = null): NegativePromptPreset = synchronized(lock) {
        val file = readNegative()
        val old = id?.let { key -> file.items.firstOrNull { it.id == key } }
        require(id == null || old != null) { "该预设已被删除，请刷新列表" }
        val now = clock(); val selectedDefault = makeDefault ?: old?.isDefault ?: false
        val item = NegativePromptPreset(old?.id ?: UUID.randomUUID().toString(), cleanName(name),
            note.trim().take(MAX_NOTE_CHARS), cleanPrompt(prompt), selectedDefault,
            old?.createdAt ?: now, now, old?.lastUsedAt ?: 0L, old?.pinnedAt ?: 0L)
        val others = file.items.filterNot { it.id == item.id }.map { if (selectedDefault) it.copy(isDefault = false) else it }
        writeNegative(file.copy(items = others + item)); item
    }
    fun touchPositive(id: String): PositivePromptPreset? = synchronized(lock) {
        val file = readPositive(); val found = file.items.firstOrNull { it.id == id } ?: return@synchronized null
        val updated = found.copy(lastUsedAt = clock())
        writePositive(file.copy(items = file.items.map { if (it.id == id) updated else it })); updated
    }
    fun touchNegative(id: String): NegativePromptPreset? = synchronized(lock) {
        val file = readNegative(); val found = file.items.firstOrNull { it.id == id } ?: return@synchronized null
        val updated = found.copy(lastUsedAt = clock())
        writeNegative(file.copy(items = file.items.map { if (it.id == id) updated else it })); updated
    }
    fun setPositivePinned(id: String, pinned: Boolean): Boolean = synchronized(lock) {
        val file = readPositive(); if (file.items.none { it.id == id }) return@synchronized false
        val order = if (pinned) maxOf(clock(), (file.items.maxOfOrNull { it.pinnedAt } ?: 0L) + 1L, 1L) else 0L
        writePositive(file.copy(items = file.items.map { if (it.id == id) it.copy(pinnedAt = order) else it })); true
    }
    fun setNegativePinned(id: String, pinned: Boolean): Boolean = synchronized(lock) {
        val file = readNegative(); if (file.items.none { it.id == id }) return@synchronized false
        val order = if (pinned) maxOf(clock(), (file.items.maxOfOrNull { it.pinnedAt } ?: 0L) + 1L, 1L) else 0L
        writeNegative(file.copy(items = file.items.map { if (it.id == id) it.copy(pinnedAt = order) else it })); true
    }
    fun setDefaultNegative(id: String): NegativePromptPreset? = synchronized(lock) {
        val file = readNegative(); val item = file.items.firstOrNull { it.id == id } ?: return@synchronized null
        val updated = item.copy(isDefault = true, updatedAt = clock())
        writeNegative(file.copy(items = file.items.map { if (it.id == id) updated else it.copy(isDefault = false) })); updated
    }
    fun defaultNegative(): NegativePromptPreset? = synchronized(lock) { readNegative().items.firstOrNull { it.isDefault } }
    fun deletePositive(id: String): Boolean = synchronized(lock) {
        val file = readPositive(); val items = file.items.filterNot { it.id == id }
        if (items.size == file.items.size) return@synchronized false
        writePositive(file.copy(items = items)); true
    }
    fun deleteNegative(id: String): Boolean = synchronized(lock) {
        val file = readNegative(); val items = file.items.filterNot { it.id == id }
        if (items.size == file.items.size) return@synchronized false
        writeNegative(file.copy(items = items)); true
    }
    private fun readText(file: File): String? {
        val source = if (file.isFile) file else File(file.path + ".bak").takeIf { it.isFile } ?: return null
        require(source.length() <= 32L * 1024 * 1024) { "预设文件过大，已保留原文件" }
        return source.readText(Charsets.UTF_8)
    }
    private fun readPositive(): PositivePresetFile = try {
        val text = readText(positiveFile)
        if (text == null) PositivePresetFile() else json.decodeFromString<PositivePresetFile>(text).also {
            require(it.version in 1..2 && it.items.map { p -> p.id }.distinct().size == it.items.size)
        }
    } catch (e: Exception) { throw IllegalStateException("正向预设读取失败，原文件已保留，未覆盖：${e.message}", e) }
    private fun readNegative(): NegativePresetFile = try {
        val text = readText(negativeFile)
        if (text == null) NegativePresetFile() else json.decodeFromString<NegativePresetFile>(text).also {
            require(it.version in 1..2 && it.items.map { p -> p.id }.distinct().size == it.items.size)
        }
    } catch (e: Exception) { throw IllegalStateException("负向预设读取失败，原文件已保留，未覆盖：${e.message}", e) }
    private fun writePositive(value: PositivePresetFile) = writeAtomic(positiveFile, json.encodeToString(value.copy(version = 2)))
    private fun writeNegative(value: NegativePresetFile) = writeAtomic(negativeFile, json.encodeToString(value.copy(version = 2)))
    private fun writeAtomic(target: File, content: String) {
        check(root.isDirectory || root.mkdirs()) { "无法访问预设目录" }
        // Keep the first pre-management snapshot without overwriting it on later edits.
        val backup = File(root, target.name + ".before_1_6.bak")
        if (target.isFile && !backup.exists()) target.copyTo(backup)
        val pending = File(root, target.name + ".pending")
        FileOutputStream(pending).use { stream ->
            stream.write(content.toByteArray(Charsets.UTF_8)); stream.fd.sync()
        }
        // Same-directory rename is atomic. Never delete the old destination first.
        Files.move(pending.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        check(target.readText(Charsets.UTF_8) == content) { "预设写入验证失败，已保留升级前备份" }
    }
    private fun cleanName(value: String): String = value.trim().replace(Regex("\\s+"), " ").also {
        require(it.isNotBlank() && it.length <= 80) { "名称不能为空，且最多 80 字符" }
    }
    private fun cleanPrompt(value: String): String = value.trim().also {
        require(it.isNotBlank() && it.length <= 100_000) { "提示词不能为空，且最多 100000 字符" }
    }
    companion object {
        private val locks = mutableMapOf<String, Any>()
        private const val MAX_NOTE_CHARS = 4_000
    }
}
