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
data class StoredProject(
    val id: String,
    val name: String,
    val createdAt: Long,
    val updatedAt: Long,
    val currentConversationId: String? = null,
    val workspaceUri: String? = null,
    val workspaceFlags: Int = 0,
    val workspaceProbeOk: Boolean? = null,
)

data class ProjectSummary(
    val id: String,
    val name: String,
    val updatedAt: Long,
    val conversationCount: Int,
)

class ProjectSessionStore(private val context: Context) {
    private val root = File(context.filesDir, "projects").apply { mkdirs() }
    private val prefs = context.getSharedPreferences("rin_project_session_state", Context.MODE_PRIVATE)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    init {
        migrateLegacyIfNeeded()
    }

    var currentProjectId: String?
        get() = prefs.getString(KEY_CURRENT_PROJECT_ID, null)
        private set(value) {
            prefs.edit().putString(KEY_CURRENT_PROJECT_ID, value).apply()
        }

    @Synchronized
    fun loadCurrentProjectOrCreate(): StoredProject {
        currentProjectId?.let(::loadProject)?.let { return it }
        listProjects().firstOrNull()?.let { summary ->
            loadProject(summary.id)?.let {
                currentProjectId = it.id
                return it
            }
        }
        return createProject(context.getString(R.string.default_project))
    }

    @Synchronized
    fun currentProject(): StoredProject = loadCurrentProjectOrCreate()

    @Synchronized
    fun listProjects(limit: Int = 50): List<ProjectSummary> =
        root.listFiles { file -> file.isDirectory && File(file, PROJECT_FILE).isFile }
            .orEmpty()
            .mapNotNull { dir -> loadProject(dir.name) }
            .map { project ->
                ProjectSummary(
                    id = project.id,
                    name = project.name,
                    updatedAt = project.updatedAt,
                    conversationCount = conversationDir(project.id).listFiles { file -> file.isFile && file.extension.equals("json", true) }.orEmpty().size,
                )
            }
            .sortedByDescending { it.updatedAt }
            .take(limit)

    @Synchronized
    fun createProject(name: String): StoredProject {
        val clean = name.trim().take(MAX_PROJECT_NAME_CHARS).ifBlank { context.getString(R.string.default_project) }
        val now = System.currentTimeMillis()
        val project = StoredProject(
            id = UUID.randomUUID().toString(),
            name = clean,
            createdAt = now,
            updatedAt = now,
        )
        projectDir(project.id).mkdirs()
        conversationDir(project.id).mkdirs()
        writeProjectAtomic(project)
        currentProjectId = project.id
        createConversation(project.id)
        return loadProject(project.id) ?: project
    }

    @Synchronized
    fun selectProject(id: String): StoredProject? {
        val project = loadProject(id) ?: return null
        currentProjectId = id
        return project
    }

    @Synchronized
    fun loadProject(id: String): StoredProject? {
        val file = projectFile(id)
        if (!file.isFile) return null
        return runCatching { json.decodeFromString<StoredProject>(file.readText(Charsets.UTF_8)) }.getOrNull()
    }

    @Synchronized
    fun renameProject(id: String, name: String): Boolean {
        val clean = name.trim().take(MAX_PROJECT_NAME_CHARS)
        if (clean.isEmpty()) return false
        val old = loadProject(id) ?: return false
        writeProjectAtomic(old.copy(name = clean, updatedAt = System.currentTimeMillis()))
        return true
    }

    @Synchronized
    fun deleteProject(id: String): StoredProject {
        val dir = projectDir(id)
        if (dir.exists() && !dir.deleteRecursively()) error("Could not delete project")
        if (currentProjectId == id) currentProjectId = null
        return loadCurrentProjectOrCreate()
    }

    @Synchronized
    fun updateCurrentWorkspace(uri: String?, flags: Int, probeOk: Boolean?) {
        val project = currentProject()
        writeProjectAtomic(
            project.copy(
                workspaceUri = uri,
                workspaceFlags = flags,
                workspaceProbeOk = probeOk,
                updatedAt = System.currentTimeMillis(),
            ),
        )
    }

    @Synchronized
    fun updateCurrentWorkspaceProbe(probeOk: Boolean?) {
        val project = currentProject()
        writeProjectAtomic(project.copy(workspaceProbeOk = probeOk, updatedAt = System.currentTimeMillis()))
    }

    @Synchronized
    fun createConversation(projectId: String = currentProject().id): StoredConversation {
        require(loadProject(projectId) != null) { "Project not found" }
        val now = System.currentTimeMillis()
        val conversation = StoredConversation(
            id = UUID.randomUUID().toString(),
            title = context.getString(R.string.new_conversation_title),
            createdAt = now,
            updatedAt = now,
        )
        writeConversationAtomic(projectId, conversation)
        updateProjectCurrentConversation(projectId, conversation.id)
        return conversation
    }

    @Synchronized
    fun loadCurrentConversationOrCreate(projectId: String = currentProject().id): StoredConversation {
        val project = loadProject(projectId) ?: error("Project not found")
        project.currentConversationId?.let { loadConversation(projectId, it) }?.let { return it }
        listConversations(projectId).firstOrNull()?.let { summary ->
            loadConversation(projectId, summary.id)?.let {
                updateProjectCurrentConversation(projectId, it.id)
                return it
            }
        }
        return createConversation(projectId)
    }

    @Synchronized
    fun selectConversation(projectId: String, conversationId: String): StoredConversation? {
        val conversation = loadConversation(projectId, conversationId) ?: return null
        updateProjectCurrentConversation(projectId, conversationId)
        return conversation
    }

    @Synchronized
    fun loadConversation(projectId: String, conversationId: String): StoredConversation? {
        val file = conversationFile(projectId, conversationId)
        if (!file.isFile) return null
        return runCatching { json.decodeFromString<StoredConversation>(file.readText(Charsets.UTF_8)) }.getOrNull()
    }

    @Synchronized
    fun listConversations(projectId: String, limit: Int = 80): List<ConversationSummary> =
        conversationDir(projectId).listFiles { file -> file.isFile && file.extension.equals("json", true) }
            .orEmpty()
            .mapNotNull { file ->
                runCatching { json.decodeFromString<StoredConversation>(file.readText(Charsets.UTF_8)) }
                    .getOrNull()
                    ?.let { ConversationSummary(it.id, it.title, it.updatedAt, it.messages.size) }
            }
            .sortedByDescending { it.updatedAt }
            .take(limit)

    @Synchronized
    fun saveConversation(projectId: String, conversationId: String, messages: List<Message>) {
        val old = loadConversation(projectId, conversationId) ?: StoredConversation(
            id = conversationId,
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
        writeConversationAtomic(
            projectId,
            old.copy(title = title, updatedAt = System.currentTimeMillis(), messages = storedMessages),
        )
        updateProjectCurrentConversation(projectId, conversationId, touch = true)
    }

    @Synchronized
    fun renameConversation(projectId: String, conversationId: String, title: String): Boolean {
        val clean = title.trim().take(MAX_TITLE_CHARS)
        if (clean.isEmpty()) return false
        val old = loadConversation(projectId, conversationId) ?: return false
        writeConversationAtomic(projectId, old.copy(title = clean, customTitle = true, updatedAt = System.currentTimeMillis()))
        touchProject(projectId)
        return true
    }

    @Synchronized
    fun deleteConversation(projectId: String, conversationId: String): StoredConversation {
        val file = conversationFile(projectId, conversationId)
        if (file.exists() && !file.delete()) error("Could not delete conversation")
        val project = loadProject(projectId) ?: error("Project not found")
        if (project.currentConversationId == conversationId) updateProjectCurrentConversation(projectId, null)
        return loadCurrentConversationOrCreate(projectId)
    }

    fun toMessages(conversation: StoredConversation): List<Message> = conversation.messages.map { stored ->
        Message(
            content = stored.content,
            type = MessageType.from(stored.type),
            images = stored.images.map(::File),
            audio = stored.audio.map(::File),
        )
    }

    @Synchronized
    fun resolveProject(ref: String?): StoredProject? {
        if (ref.isNullOrBlank()) return loadCurrentProjectOrCreate()
        loadProject(ref)?.let { return it }
        return listProjects(200).firstOrNull { it.name.equals(ref.trim(), ignoreCase = true) }?.let { loadProject(it.id) }
    }

    @Synchronized
    fun resolveConversation(projectId: String, ref: String): StoredConversation? {
        loadConversation(projectId, ref)?.let { return it }
        return listConversations(projectId, 500)
            .firstOrNull { it.title.equals(ref.trim(), ignoreCase = true) }
            ?.let { loadConversation(projectId, it.id) }
    }

    @Synchronized
    fun readConversationContext(projectRef: String?, conversationRef: String, maxMessages: Int = 80, maxChars: Int = 160_000): String {
        val project = resolveProject(projectRef) ?: error("Project not found: ${projectRef.orEmpty()}")
        val conversation = resolveConversation(project.id, conversationRef) ?: error("Conversation not found: $conversationRef")
        val relevant = conversation.messages
            .filter { it.type == MessageType.USER.value || it.type == MessageType.ASSISTANT.value }
            .takeLast(maxMessages.coerceIn(1, 200))
        val sb = StringBuilder()
        sb.append("Project: ").append(project.name).append('\n')
        sb.append("Conversation: ").append(conversation.title).append('\n')
        sb.append("Conversation ID: ").append(conversation.id).append("\n\n")
        for (message in relevant) {
            val role = if (message.type == MessageType.USER.value) "user" else "assistant"
            val line = buildString {
                append('[').append(role).append("] ").append(message.content)
                if (message.images.isNotEmpty()) append("\n[image attachments: ").append(message.images.joinToString { File(it).name }).append(']')
                append("\n\n")
            }
            if (sb.length + line.length > maxChars) {
                val remaining = (maxChars - sb.length).coerceAtLeast(0)
                if (remaining > 0) sb.append(line.take(remaining))
                break
            }
            sb.append(line)
        }
        return sb.toString()
    }

    private fun deriveTitle(messages: List<Message>): String {
        val first = messages.firstOrNull { it.type == MessageType.USER && it.content.isNotBlank() }
            ?.content
            ?.replace(Regex("\\s+"), " ")
            ?.trim()
            .orEmpty()
        return first.take(MAX_TITLE_CHARS).ifBlank { context.getString(R.string.new_conversation_title) }
    }

    private fun updateProjectCurrentConversation(projectId: String, conversationId: String?, touch: Boolean = false) {
        val project = loadProject(projectId) ?: return
        writeProjectAtomic(
            project.copy(
                currentConversationId = conversationId,
                updatedAt = if (touch) System.currentTimeMillis() else project.updatedAt,
            ),
        )
    }

    private fun touchProject(projectId: String) {
        val project = loadProject(projectId) ?: return
        writeProjectAtomic(project.copy(updatedAt = System.currentTimeMillis()))
    }

    private fun writeProjectAtomic(project: StoredProject) {
        val dir = projectDir(project.id).apply { mkdirs() }
        conversationDir(project.id).mkdirs()
        val target = File(dir, PROJECT_FILE)
        writeAtomicText(target, json.encodeToString(project))
    }

    private fun writeConversationAtomic(projectId: String, conversation: StoredConversation) {
        val dir = conversationDir(projectId).apply { mkdirs() }
        val target = File(dir, "${conversation.id}.json")
        writeAtomicText(target, json.encodeToString(conversation))
    }

    private fun writeAtomicText(target: File, content: String) {
        target.parentFile?.mkdirs()
        val tmp = File(target.parentFile, target.name + ".tmp")
        tmp.writeText(content, Charsets.UTF_8)
        if (target.exists() && !target.delete()) error("Could not replace ${target.name}")
        if (!tmp.renameTo(target)) {
            target.writeText(tmp.readText(Charsets.UTF_8), Charsets.UTF_8)
            tmp.delete()
        }
    }

    private fun projectDir(id: String) = File(root, id)
    private fun projectFile(id: String) = File(projectDir(id), PROJECT_FILE)
    private fun conversationDir(projectId: String) = File(projectDir(projectId), "conversations")
    private fun conversationFile(projectId: String, conversationId: String) = File(conversationDir(projectId), "$conversationId.json")

    private fun migrateLegacyIfNeeded() {
        if (prefs.getBoolean(KEY_MIGRATION_DONE, false)) return
        val existing = root.listFiles { file -> file.isDirectory && File(file, PROJECT_FILE).isFile }.orEmpty()
        if (existing.isNotEmpty()) {
            if (currentProjectId == null) currentProjectId = existing.first().name
            prefs.edit().putBoolean(KEY_MIGRATION_DONE, true).apply()
            return
        }

        val legacyRoot = File(context.filesDir, "conversations")
        val legacyFiles = legacyRoot.listFiles { file -> file.isFile && file.extension.equals("json", true) }.orEmpty()
        val legacyConversationPrefs = context.getSharedPreferences("rin_conversation_state", Context.MODE_PRIVATE)
        val legacyWorkspacePrefs = context.getSharedPreferences("rin_agent_workspace", Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        val projectId = UUID.randomUUID().toString()
        val convDir = conversationDir(projectId).apply { mkdirs() }
        val migrated = mutableListOf<StoredConversation>()
        legacyFiles.forEach { source ->
            runCatching { json.decodeFromString<StoredConversation>(source.readText(Charsets.UTF_8)) }
                .getOrNull()
                ?.let { conversation ->
                    writeConversationAtomic(projectId, conversation)
                    migrated += conversation
                }
        }
        val legacyCurrent = legacyConversationPrefs.getString("current_id", null)
        val currentConversationId = when {
            legacyCurrent != null && migrated.any { it.id == legacyCurrent } -> legacyCurrent
            migrated.isNotEmpty() -> migrated.maxByOrNull { it.updatedAt }?.id
            else -> null
        }
        val workspaceUri = legacyWorkspacePrefs.getString("root_uri", null)
        val workspaceFlags = legacyWorkspacePrefs.getInt("root_flags", 0)
        val workspaceProbe = if (legacyWorkspacePrefs.contains("probe_ok")) legacyWorkspacePrefs.getBoolean("probe_ok", false) else null
        val project = StoredProject(
            id = projectId,
            name = context.getString(R.string.default_project),
            createdAt = migrated.minOfOrNull { it.createdAt } ?: now,
            updatedAt = migrated.maxOfOrNull { it.updatedAt } ?: now,
            currentConversationId = currentConversationId,
            workspaceUri = workspaceUri,
            workspaceFlags = workspaceFlags,
            workspaceProbeOk = workspaceProbe,
        )
        writeProjectAtomic(project)
        currentProjectId = projectId
        if (migrated.isEmpty()) createConversation(projectId)
        prefs.edit().putBoolean(KEY_MIGRATION_DONE, true).apply()
    }

    companion object {
        private const val PROJECT_FILE = "project.json"
        private const val KEY_CURRENT_PROJECT_ID = "current_project_id"
        private const val KEY_MIGRATION_DONE = "project_migration_v1_done"
        private const val MAX_PROJECT_NAME_CHARS = 64
        private const val MAX_TITLE_CHARS = 48
        private const val MAX_MESSAGE_CHARS = 250_000
    }
}
