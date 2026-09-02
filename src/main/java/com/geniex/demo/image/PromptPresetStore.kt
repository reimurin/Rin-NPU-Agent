package com.geniex.demo.image

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

@Serializable
data class PositivePromptPreset(
    val id: String,
    val name: String,
    val note: String = "",
    val prompt: String,
    val tags: List<String> = emptyList(),
    val createdAt: Long,
    val updatedAt: Long,
    val lastUsedAt: Long = 0L,
)

@Serializable
data class NegativePromptPreset(
    val id: String,
    val name: String,
    val note: String = "",
    val prompt: String,
    val isDefault: Boolean = false,
    val createdAt: Long,
    val updatedAt: Long,
    val lastUsedAt: Long = 0L,
)

@Serializable
private data class PositivePresetFile(val version: Int = 1, val items: List<PositivePromptPreset> = emptyList())

@Serializable
private data class NegativePresetFile(val version: Int = 1, val items: List<NegativePromptPreset> = emptyList())

class PromptPresetStore(context: Context) {
    private val root = File(context.filesDir, "image_presets").apply { mkdirs() }
    private val positiveFile = File(root, "positive.json")
    private val negativeFile = File(root, "negative.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = true }

    @Synchronized
    fun listPositive(): List<PositivePromptPreset> = readPositive().items
        .sortedWith(compareByDescending<PositivePromptPreset> { it.lastUsedAt }.thenByDescending { it.updatedAt })

    @Synchronized
    fun listNegative(): List<NegativePromptPreset> = readNegative().items
        .sortedWith(compareByDescending<NegativePromptPreset> { it.isDefault }.thenByDescending { it.lastUsedAt }.thenByDescending { it.updatedAt })

    @Synchronized
    fun savePositive(
        id: String? = null,
        name: String,
        note: String,
        prompt: String,
        tags: List<String> = emptyList(),
    ): PositivePromptPreset {
        val cleanName = normalizeName(name)
        val cleanPrompt = normalizePrompt(prompt)
        val now = System.currentTimeMillis()
        val file = readPositive()
        val old = id?.let { target -> file.items.firstOrNull { it.id == target } }
        val preset = PositivePromptPreset(
            id = old?.id ?: UUID.randomUUID().toString(),
            name = cleanName,
            note = note.trim().take(MAX_NOTE_CHARS),
            prompt = cleanPrompt,
            tags = tags.map { it.trim() }.filter { it.isNotBlank() }.distinct().take(MAX_TAGS),
            createdAt = old?.createdAt ?: now,
            updatedAt = now,
            lastUsedAt = old?.lastUsedAt ?: 0L,
        )
        writePositive(file.copy(items = file.items.filterNot { it.id == preset.id } + preset))
        return preset
    }

    @Synchronized
    fun saveNegative(
        id: String? = null,
        name: String,
        note: String,
        prompt: String,
        makeDefault: Boolean = false,
    ): NegativePromptPreset {
        val cleanName = normalizeName(name)
        val cleanPrompt = normalizePrompt(prompt)
        val now = System.currentTimeMillis()
        val file = readNegative()
        val old = id?.let { target -> file.items.firstOrNull { it.id == target } }
        val targetId = old?.id ?: UUID.randomUUID().toString()
        val itemsWithoutTarget = file.items.filterNot { it.id == targetId }
            .map { if (makeDefault) it.copy(isDefault = false) else it }
        val preset = NegativePromptPreset(
            id = targetId,
            name = cleanName,
            note = note.trim().take(MAX_NOTE_CHARS),
            prompt = cleanPrompt,
            isDefault = makeDefault || old?.isDefault == true,
            createdAt = old?.createdAt ?: now,
            updatedAt = now,
            lastUsedAt = old?.lastUsedAt ?: 0L,
        )
        val normalized = if (preset.isDefault) {
            itemsWithoutTarget.map { it.copy(isDefault = false) } + preset
        } else {
            itemsWithoutTarget + preset
        }
        writeNegative(file.copy(items = normalized))
        return preset
    }

    @Synchronized
    fun touchPositive(id: String): PositivePromptPreset? {
        val file = readPositive()
        val now = System.currentTimeMillis()
        var found: PositivePromptPreset? = null
        val items = file.items.map {
            if (it.id == id) it.copy(lastUsedAt = now).also { updated -> found = updated } else it
        }
        if (found != null) writePositive(file.copy(items = items))
        return found
    }

    @Synchronized
    fun touchNegative(id: String): NegativePromptPreset? {
        val file = readNegative()
        val now = System.currentTimeMillis()
        var found: NegativePromptPreset? = null
        val items = file.items.map {
            if (it.id == id) it.copy(lastUsedAt = now).also { updated -> found = updated } else it
        }
        if (found != null) writeNegative(file.copy(items = items))
        return found
    }

    @Synchronized
    fun setDefaultNegative(id: String): NegativePromptPreset? {
        val file = readNegative()
        val now = System.currentTimeMillis()
        var found: NegativePromptPreset? = null
        val items = file.items.map {
            if (it.id == id) it.copy(isDefault = true, updatedAt = now).also { updated -> found = updated }
            else it.copy(isDefault = false)
        }
        if (found != null) writeNegative(file.copy(items = items))
        return found
    }

    @Synchronized
    fun defaultNegative(): NegativePromptPreset? = readNegative().items.firstOrNull { it.isDefault }

    @Synchronized
    fun deletePositive(id: String): Boolean {
        val file = readPositive()
        val items = file.items.filterNot { it.id == id }
        if (items.size == file.items.size) return false
        writePositive(file.copy(items = items))
        return true
    }

    @Synchronized
    fun deleteNegative(id: String): Boolean {
        val file = readNegative()
        val items = file.items.filterNot { it.id == id }
        if (items.size == file.items.size) return false
        writeNegative(file.copy(items = items))
        return true
    }

    private fun readPositive(): PositivePresetFile {
        if (!positiveFile.isFile) return PositivePresetFile()
        return runCatching { json.decodeFromString<PositivePresetFile>(positiveFile.readText(Charsets.UTF_8)) }
            .getOrDefault(PositivePresetFile())
    }

    private fun readNegative(): NegativePresetFile {
        if (!negativeFile.isFile) return NegativePresetFile()
        return runCatching { json.decodeFromString<NegativePresetFile>(negativeFile.readText(Charsets.UTF_8)) }
            .getOrDefault(NegativePresetFile())
    }

    private fun writePositive(value: PositivePresetFile) = writeAtomic(positiveFile, json.encodeToString(value))
    private fun writeNegative(value: NegativePresetFile) = writeAtomic(negativeFile, json.encodeToString(value))

    private fun writeAtomic(target: File, content: String) {
        root.mkdirs()
        val temp = File(root, target.name + ".tmp")
        temp.writeText(content, Charsets.UTF_8)
        if (target.exists() && !target.delete()) error("Could not replace preset file")
        if (!temp.renameTo(target)) {
            target.writeText(temp.readText(Charsets.UTF_8), Charsets.UTF_8)
            temp.delete()
        }
    }

    private fun normalizeName(value: String): String {
        val clean = value.trim().replace(Regex("\\s+"), " ").take(MAX_NAME_CHARS)
        require(clean.isNotBlank()) { "Preset name is empty" }
        return clean
    }

    private fun normalizePrompt(value: String): String {
        val clean = value.trim().take(MAX_PROMPT_CHARS)
        require(clean.isNotBlank()) { "Prompt is empty" }
        return clean
    }

    companion object {
        private const val MAX_NAME_CHARS = 80
        private const val MAX_NOTE_CHARS = 4_000
        private const val MAX_PROMPT_CHARS = 100_000
        private const val MAX_TAGS = 32
    }
}
