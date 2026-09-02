package com.geniex.demo.agent

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import androidx.core.content.IntentCompat
import java.io.File
import java.io.FileOutputStream

data class SharedAttachment(
    val sourceUri: Uri?,
    val localFile: File,
    val displayName: String,
    val mimeType: String,
    val kind: Kind,
) {
    enum class Kind { IMAGE, PDF, TEXT, OTHER }
}

class SharedInputManager(private val context: Context) {
    fun fromIntent(intent: Intent?): List<SharedAttachment> {
        if (intent == null) return emptyList()
        val out = mutableListOf<SharedAttachment>()
        val uris = when (intent.action) {
            Intent.ACTION_SEND -> listOfNotNull(IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java))
            Intent.ACTION_SEND_MULTIPLE -> IntentCompat.getParcelableArrayListExtra(intent, Intent.EXTRA_STREAM, Uri::class.java).orEmpty()
            else -> emptyList()
        }
        uris.forEach { uri -> runCatching { copyUri(uri) }.getOrNull()?.let(out::add) }
        intent.getStringExtra(Intent.EXTRA_TEXT)?.takeIf { it.isNotBlank() }?.let { text ->
            val file = File(shareCache(), "shared_${System.currentTimeMillis()}.txt").apply { writeText(text) }
            out += SharedAttachment(null, file, file.name, "text/plain", SharedAttachment.Kind.TEXT)
        }
        return out
    }

    fun textContext(attachments: List<SharedAttachment>, maxCharsPerFile: Int = 40_000): String =
        attachments.filter { it.kind == SharedAttachment.Kind.TEXT }.joinToString("\n\n") { a ->
            val text = runCatching { a.localFile.readText().take(maxCharsPerFile) }.getOrElse { "[unreadable]" }
            "--- ${a.displayName} ---\n$text"
        }

    fun visionFiles(attachments: List<SharedAttachment>, maxPdfPages: Int = 4): List<File> {
        val images = attachments.filter { it.kind == SharedAttachment.Kind.IMAGE }.map { it.localFile }.toMutableList()
        attachments.filter { it.kind == SharedAttachment.Kind.PDF }.forEach { images += renderPdf(it.localFile, maxPdfPages) }
        return images
    }

    private fun copyUri(uri: Uri): SharedAttachment {
        val type = context.contentResolver.getType(uri) ?: "application/octet-stream"
        val ext = when {
            type.startsWith("image/") -> ".jpg"
            type == "application/pdf" -> ".pdf"
            type.startsWith("text/") -> ".txt"
            else -> ".bin"
        }
        val file = File(shareCache(), "share_${System.currentTimeMillis()}_${uri.hashCode().toUInt()}$ext")
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open shared URI" }
            file.outputStream().use { input.copyTo(it) }
        }
        val kind = when {
            type.startsWith("image/") -> SharedAttachment.Kind.IMAGE
            type == "application/pdf" -> SharedAttachment.Kind.PDF
            type.startsWith("text/") || looksTextual(uri) -> SharedAttachment.Kind.TEXT
            else -> SharedAttachment.Kind.OTHER
        }
        return SharedAttachment(uri, file, file.name, type, kind)
    }

    private fun looksTextual(uri: Uri): Boolean {
        val name = uri.lastPathSegment.orEmpty().lowercase()
        return listOf(".kt", ".kts", ".java", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".md", ".json", ".xml", ".yaml", ".yml", ".toml", ".gradle", ".properties", ".html", ".css", ".sh", ".ps1").any(name::endsWith)
    }

    private fun renderPdf(file: File, maxPages: Int): List<File> {
        val output = mutableListOf<File>()
        val pfd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
        PdfRenderer(pfd).use { renderer ->
            val count = minOf(renderer.pageCount, maxPages)
            for (i in 0 until count) {
                renderer.openPage(i).use { page ->
                    val scale = minOf(1.8f, 1440f / page.width.toFloat())
                    val width = (page.width * scale).toInt().coerceAtLeast(1)
                    val height = (page.height * scale).toInt().coerceAtLeast(1)
                    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    bitmap.eraseColor(android.graphics.Color.WHITE)
                    page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    val out = File(shareCache(), "${file.nameWithoutExtension}_page_${i + 1}.jpg")
                    FileOutputStream(out).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                    bitmap.recycle()
                    output += out
                }
            }
        }
        return output
    }

    private fun shareCache(): File = File(context.cacheDir, "shared_inputs").apply { mkdirs() }
}
