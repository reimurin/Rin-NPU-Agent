package com.geniex.demo.agent

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import com.geniex.demo.history.ProjectSessionStore
import java.io.FileNotFoundException

data class WorkspaceAccessStatus(
    val selected: Boolean,
    val name: String? = null,
    val persistedRead: Boolean = false,
    val persistedWrite: Boolean = false,
    val exists: Boolean = false,
    val readable: Boolean = false,
    val writable: Boolean = false,
    val probeOk: Boolean? = null,
    val detail: String? = null,
) {
    val ready: Boolean
        get() = selected && persistedRead && persistedWrite && exists && readable && writable && probeOk == true
}

class ProjectWorkspace(
    private val context: Context,
    private val sessions: ProjectSessionStore,
) {
    val rootUri: Uri?
        get() = sessions.currentProject().workspaceUri?.let(Uri::parse)

    val rootName: String?
        get() = inspect().name

    fun setRoot(uri: Uri, grantedFlags: Int) {
        val persistFlags = grantedFlags and (Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        require((persistFlags and Intent.FLAG_GRANT_READ_URI_PERMISSION) != 0) { "Selected folder did not grant read access" }
        context.contentResolver.takePersistableUriPermission(uri, persistFlags)
        sessions.updateCurrentWorkspace(uri.toString(), persistFlags, false)
        val status = inspect()
        require(status.exists && status.readable) { "Selected folder is not readable" }
    }

    fun clearRoot() {
        val project = sessions.currentProject()
        val uri = project.workspaceUri?.let(Uri::parse)
        if (uri != null && project.workspaceFlags != 0) {
            runCatching { context.contentResolver.releasePersistableUriPermission(uri, project.workspaceFlags) }
        }
        sessions.updateCurrentWorkspace(null, 0, null)
    }

    fun inspect(): WorkspaceAccessStatus {
        val project = sessions.currentProject()
        val uri = project.workspaceUri?.let(Uri::parse) ?: return WorkspaceAccessStatus(selected = false)
        val permission = context.contentResolver.persistedUriPermissions.firstOrNull { it.uri == uri }
        val doc = runCatching { DocumentFile.fromTreeUri(context, uri) }.getOrNull()
        return WorkspaceAccessStatus(
            selected = true,
            name = doc?.name,
            persistedRead = permission?.isReadPermission == true,
            persistedWrite = permission?.isWritePermission == true,
            exists = runCatching { doc?.exists() == true }.getOrDefault(false),
            readable = runCatching { doc?.canRead() == true }.getOrDefault(false),
            writable = runCatching { doc?.canWrite() == true }.getOrDefault(false),
            probeOk = project.workspaceProbeOk,
        )
    }

    fun probeAccess(): WorkspaceAccessStatus {
        val base = inspect()
        if (!base.selected || !base.persistedRead || !base.persistedWrite || !base.exists || !base.readable || !base.writable) {
            sessions.updateCurrentWorkspaceProbe(false)
            return base.copy(probeOk = false, detail = "Persisted read/write permission is incomplete")
        }
        val root = runCatching { rootDocument() }.getOrElse {
            sessions.updateCurrentWorkspaceProbe(false)
            return base.copy(probeOk = false, detail = it.message)
        }
        var probe: DocumentFile? = null
        return try {
            val name = ".rin_agent_probe_${System.currentTimeMillis()}.txt"
            probe = root.createFile("text/plain", name) ?: error("Provider refused to create a test file")
            val marker = "rin-agent-write-probe"
            context.contentResolver.openOutputStream(probe.uri, "wt")?.bufferedWriter()?.use { it.write(marker) }
                ?: error("Could not open test file for writing")
            val roundTrip = context.contentResolver.openInputStream(probe.uri)?.bufferedReader()?.use { it.readText() }
                ?: error("Could not read test file back")
            check(roundTrip == marker) { "Read/write verification did not round-trip" }
            sessions.updateCurrentWorkspaceProbe(true)
            base.copy(probeOk = true, detail = null)
        } catch (e: Exception) {
            sessions.updateCurrentWorkspaceProbe(false)
            base.copy(probeOk = false, detail = e.message ?: e.javaClass.simpleName)
        } finally {
            runCatching { probe?.delete() }
        }
    }

    fun list(relativePath: String = ""): String {
        requireReadable()
        val dir = resolve(relativePath, createDirs = false) ?: throw FileNotFoundException(relativePath)
        require(dir.isDirectory) { "Not a directory: $relativePath" }
        return dir.listFiles().sortedWith(compareByDescending<DocumentFile> { it.isDirectory }.thenBy { it.name ?: "" })
            .joinToString("\n") { if (it.isDirectory) "[DIR] ${it.name}" else "[FILE] ${it.name} (${it.length()} bytes)" }
    }

    fun readText(relativePath: String, maxChars: Int = 120_000): String {
        requireReadable()
        val file = resolve(relativePath, createDirs = false) ?: throw FileNotFoundException(relativePath)
        require(file.isFile) { "Not a file: $relativePath" }
        return context.contentResolver.openInputStream(file.uri)?.bufferedReader()?.use { it.readText().take(maxChars) }
            ?: throw FileNotFoundException(relativePath)
    }

    fun createDirectory(relativePath: String): String {
        requireWritable()
        val dir = resolve(relativePath, createDirs = true) ?: error("Could not create directory: $relativePath")
        require(dir.isDirectory)
        return "Created directory: $relativePath"
    }

    fun writeText(relativePath: String, content: String): String {
        requireWritable()
        val clean = cleanPath(relativePath)
        require(clean.isNotEmpty()) { "File path is empty" }
        val extension = clean.substringAfterLast('.', "").lowercase()
        require(extension !in BINARY_EXTENSIONS) { "Binary file type .$extension must use a dedicated artifact tool" }
        require(content.length <= MAX_TEXT_WRITE_CHARS) { "Text write exceeds $MAX_TEXT_WRITE_CHARS characters" }
        val (parent, fileName) = resolveParent(clean)
        val existing = parent.findFile(fileName)
        val file = existing ?: parent.createFile(mimeFor(fileName), fileName) ?: error("Could not create $fileName")
        require(file.isFile) { "Path is not a file: $relativePath" }
        context.contentResolver.openOutputStream(file.uri, "wt")?.bufferedWriter()?.use { it.write(content) }
            ?: error("Could not open $relativePath for writing")
        return "Wrote ${content.length} characters to $relativePath"
    }

    fun writeBytes(relativePath: String, bytes: ByteArray, mimeType: String): String {
        requireWritable()
        require(bytes.size <= MAX_BINARY_WRITE_BYTES) { "Binary artifact exceeds $MAX_BINARY_WRITE_BYTES bytes" }
        val clean = cleanPath(relativePath)
        require(clean.isNotEmpty()) { "File path is empty" }
        val (parent, fileName) = resolveParent(clean)
        val existing = parent.findFile(fileName)
        val file = existing ?: parent.createFile(mimeType, fileName) ?: error("Could not create $fileName")
        require(file.isFile) { "Path is not a file: $relativePath" }
        context.contentResolver.openOutputStream(file.uri, "w")?.use { it.write(bytes) }
            ?: error("Could not open $relativePath for binary writing")
        return "Wrote ${bytes.size} bytes to $relativePath"
    }

    private fun resolveParent(clean: String): Pair<DocumentFile, String> {
        val parts = clean.split('/')
        val fileName = parts.last()
        val parentPath = parts.dropLast(1).joinToString("/")
        val parent = resolve(parentPath, createDirs = true) ?: error("Could not resolve parent directory")
        require(parent.isDirectory) { "Parent path is not a directory" }
        return parent to fileName
    }

    private fun requireReadable() {
        val status = inspect()
        require(status.persistedRead && status.exists && status.readable) { "Workspace read permission is unavailable; re-select the project folder" }
    }

    private fun requireWritable() {
        val status = inspect()
        require(status.ready) { "Workspace write permission has not been verified for the current project" }
    }

    private fun resolve(relativePath: String, createDirs: Boolean): DocumentFile? {
        var current: DocumentFile? = rootDocument()
        val clean = cleanPath(relativePath)
        if (clean.isEmpty()) return current
        for (segment in clean.split('/')) {
            val next = current?.findFile(segment)
            current = when {
                next != null -> next
                createDirs -> current?.createDirectory(segment)
                else -> null
            }
            if (current == null) return null
        }
        return current
    }

    private fun rootDocument(): DocumentFile {
        val uri = rootUri ?: error("Project folder has not been selected for the current logical project")
        val doc = DocumentFile.fromTreeUri(context, uri) ?: error("Project folder is no longer accessible")
        require(doc.exists()) { "Project folder no longer exists" }
        return doc
    }

    private fun cleanPath(path: String): String {
        val raw = path.replace('\\', '/').trim()
        require(!raw.startsWith("/")) { "Absolute paths are not allowed" }
        require(!Regex("^[A-Za-z]:/").containsMatchIn(raw)) { "Drive-qualified paths are not allowed" }
        val normalized = raw.trim('/')
        val parts = normalized.split('/').filter { it.isNotBlank() }
        require(parts.none { it == "." || it == ".." }) { "Path traversal is not allowed" }
        return parts.joinToString("/")
    }

    private fun mimeFor(name: String): String = when (name.substringAfterLast('.', "").lowercase()) {
        "kt", "kts", "java", "py", "js", "ts", "tsx", "jsx", "c", "cc", "cpp", "h", "hpp", "rs", "go", "sh", "ps1", "md", "txt", "json", "xml", "yaml", "yml", "toml", "gradle", "properties", "html", "css" -> "text/plain"
        else -> "application/octet-stream"
    }

    companion object {
        private val BINARY_EXTENSIONS = setOf("pptx", "docx", "xlsx", "pdf", "zip", "jar", "apk")
        private const val MAX_TEXT_WRITE_CHARS = 1_000_000
        private const val MAX_BINARY_WRITE_BYTES = 64 * 1024 * 1024
    }
}
