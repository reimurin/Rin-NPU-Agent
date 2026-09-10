package com.geniex.demo.image

import java.io.File
import java.security.MessageDigest

internal object VerifiedFileReuse {
    fun copy(
        source: File,
        target: File,
        expectedBytes: Long,
        expectedSha256: String,
        shouldCancel: () -> Boolean = { false },
    ): Boolean {
        if (!source.isFile || source.length() != expectedBytes || target.exists()) return false
        target.parentFile?.mkdirs()
        val digest = MessageDigest.getInstance("SHA-256")
        var copied = 0L
        source.inputStream().buffered(8 * 1024 * 1024).use { input ->
            target.outputStream().buffered(8 * 1024 * 1024).use { output ->
                val buffer = ByteArray(8 * 1024 * 1024)
                while (true) {
                    if (shouldCancel()) throw InterruptedException("Operation cancelled")
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    digest.update(buffer, 0, count)
                    copied += count
                }
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        return copied == expectedBytes && target.length() == expectedBytes && actual.equals(expectedSha256, ignoreCase = true)
    }
}
