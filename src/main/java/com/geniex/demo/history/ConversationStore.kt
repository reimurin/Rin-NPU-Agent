package com.geniex.demo.history

import android.content.Context
import com.geniex.demo.Message
import com.geniex.demo.MessageType
import com.geniex.demo.R
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

@Serializable
data class StoredMessage(
    val content: String,
    val type: Int,
    val images: List<String> = emptyList(),
    val audio: List<String> = emptyList(),
)

@Serializable
data class StoredConversation(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val customTitle: Boolean = false,
    val messages: List<StoredMessage> = emptyList(),
)

data class ConversationSummary(
    val id: String,
    val title: String,
    val updatedAt: Long,
    val messageCount: Int,
)

class ConversationStore(private val context: Context) {
    private val root = File(context.filesDir, "conversations").apply { mkdirs() }
    private val prefs = context.getSharedPreferences("rin_conversation_state", Context.MODE_PRIVATE)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    var currentId: String?
        get() = prefs.getString(KEY_CURRENT_ID, null)
        private set(value) {
            prefs.edit().putString(KEY_CURRENT_ID, value).apply()
        }

    @Synchronized
    fun create(): StoredConversation {
        val now = System.currentTimeMillis()
        val conversation = StoredConversation(
            id = UUID.randomUUID().toString(),
            title = context.getString(R.string.new_conversation_title),
            createdAt = now,
            updatedAt = now,
        )
        writeAtomic(conversation)
        currentId = conversation.id
        return conversation
    }

    @Synchronized
    fun loadCurrentOrCreate(): StoredConversation {
        currentId?.let(::load)?.let { return it }
        list().firstOrNull()?.let { summary ->
            load(summary.id)?.let {
                currentId = it.id
                return it
            }
        }
        return create()
    }

    @Synchronized
    fun select(id: String): StoredConversation? {
        val conversation = load(id) ?: return null
        currentId = id
        return conversation
    }

    @Synchronized
    fun load(id: String): StoredConversation? {
        val file = fileFor(id)
        if (!file.isFile) return null
        return runCatching { json.decodeFromString<StoredConversation>(file.readText(Charsets.UTF_8)) }.getOrNull()
    }

    @Synchronized
    fun list(limit: Int = 40): List<ConversationSummary> =
        root.listFiles { file -> file.isFile && file.extension.equals("json", true) }
            .orEmpty()
            .mapNotNull { file ->
                runCatching { json.decodeFromString<StoredConversation>(file.readText(Charsets.UTF_8)) }
                    .getOrNull()
                    ?.let { ConversationSummary(it.id, it.title, it.updatedAt, it.messages.size) }
            }
            .sortedByDescending { it.updatedAt }
            .take(limit)

    @Synchronized
    fun save(id: String, messages: List<Message>) {
        val old = load(id) ?: StoredConversation(
            id = id,
            title = context.getString(R.string.new_conversation_title),
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
        )
        val storedMessages = messages
            .filter { it.type != MessageType.LOADING }
            .map { message ->
                StoredMessage(
                    content = message.content.take(MAX_MESSAGE_CHARS),
                    type = message.type.value,
                    images = message.images.map { it.absolutePath },
                    audio = message.audio.map { it.absolutePath },
                )
            }
        val title = if (old.customTitle) old.title else deriveTitle(messages)
        writeAtomic(old.copy(title = title, updatedAt = System.currentTimeMillis(), messages = storedMessages))
        currentId = id
    }

    @Synchronized
    fun rename(id: String, title: String): Boolean {
        val clean = title.trim().take(MAX_TITLE_CHARS)
        if (clean.isEmpty()) return false
        val old = load(id) ?: return false
        writeAtomic(old.copy(title = clean, customTitle = true, updatedAt = System.currentTimeMillis()))
        return true
    }

    @Synchronized
    fun delete(id: String): Boolean {
        val deleted = !fileFor(id).exists() || fileFor(id).delete()
        if (deleted && currentId == id) currentId = null
        return deleted
    }

    fun toMessages(conversation: StoredConversation): List<Message> = conversation.messages.map { stored ->
        Message(
            content = stored.content,
            type = MessageType.from(stored.type),
            images = stored.images.map(::File),
            audio = stored.audio.map(::File),
        )
    }

    private fun deriveTitle(messages: List<Message>): String {
        val first = messages.firstOrNull { it.type == MessageType.USER && it.content.isNotBlank() }
            ?.content
            ?.replace(Regex("\\s+"), " ")
            ?.trim()
            .orEmpty()
        return first.take(MAX_TITLE_CHARS).ifBlank { context.getString(R.string.new_conversation_title) }
    }

    private fun writeAtomic(conversation: StoredConversation) {
        root.mkdirs()
        val target = fileFor(conversation.id)
        val tmp = File(root, conversation.id + ".json.tmp")
        tmp.writeText(json.encodeToString(conversation), Charsets.UTF_8)
        if (target.exists() && !target.delete()) error("Could not replace conversation file")
        if (!tmp.renameTo(target)) {
            target.writeText(tmp.readText(Charsets.UTF_8), Charsets.UTF_8)
            tmp.delete()
        }
    }

    private fun fileFor(id: String) = File(root, "$id.json")

    companion object {
        private const val KEY_CURRENT_ID = "current_id"
        private const val MAX_TITLE_CHARS = 48
        private const val MAX_MESSAGE_CHARS = 250_000
    }
}
