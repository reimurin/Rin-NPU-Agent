package com.geniex.demo.image

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.security.MessageDigest

class VerifiedFileReuseTest {
    @get:Rule val temp = TemporaryFolder()
    private fun sha(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test fun matchingFileIsCopiedAndVerified() {
        val data = "rin-local-reuse".toByteArray()
        val source = temp.newFile("source.bin").apply { writeBytes(data) }
        val target = temp.root.resolve("nested/target.bin")
        assertTrue(VerifiedFileReuse.copy(source, target, data.size.toLong(), sha(data)))
        assertArrayEquals(data, target.readBytes())
    }

    @Test fun wrongDigestIsNeverAccepted() {
        val data = "rin-local-reuse".toByteArray()
        val source = temp.newFile("source-bad.bin").apply { writeBytes(data) }
        val target = temp.root.resolve("target-bad.bin")
        assertFalse(VerifiedFileReuse.copy(source, target, data.size.toLong(), "0".repeat(64)))
    }
}
