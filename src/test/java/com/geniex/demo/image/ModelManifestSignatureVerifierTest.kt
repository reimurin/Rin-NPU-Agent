package com.geniex.demo.image

import android.app.Application
import org.json.JSONObject
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Base64

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [31], application = Application::class)
class ModelManifestSignatureVerifierTest {
    private val manifest = Base64.getDecoder().decode("eyJzY2hlbWEiOjEsImNvbXBsZXRlIjp0cnVlLCJpZCI6InJpbi1zaWduYXR1cmUtc2VsZnRlc3QiLCJ2ZXJzaW9uIjoiMS4wLjAiLCJydW50aW1lX2FiaSI6MSwibWluX2FwcF9ydW50aW1lIjoxfQ==")
    private val signatureDocument = JSONObject("""{"schema":1,"algorithm":"Ed25519","key_id":"rin-model-pack-ed25519-2026-09-01","manifest_sha256":"f4a3e922379346119b71cb4e3f92037dc8c06315a3aa59c48f6f10afa5f514db","signature":"0RWbXe4Uthg90O7eblkr4/y9sWAuaMjw4Xkeh6fL5gVmkxloS9NQZJagytezqBq8CzxW+nLQ+ZR9dmMcXQcWAw=="}""")
    private val manifestSha = "f4a3e922379346119b71cb4e3f92037dc8c06315a3aa59c48f6f10afa5f514db"

    @Test fun validReleaseSignaturePassesOnApi31() {
        ModelManifestSignatureVerifier.verify(manifest, signatureDocument, manifestSha)
    }

    @Test fun modelIndexDetachedSignatureUsesSameReleaseKey() {
        val raw = manifest.toString(Charsets.UTF_8)
        val detached = signatureDocument.getString("signature")
        org.junit.Assert.assertTrue(ModelPackUpdateManager.verifyIndexSignature(raw, detached))
    }

    @Test fun tamperedManifestFails() {
        val changed = manifest.copyOf()
        changed[changed.lastIndex] = (changed.last().toInt() xor 1).toByte()
        assertThrows(IllegalArgumentException::class.java) {
            ModelManifestSignatureVerifier.verify(changed, signatureDocument, manifestSha)
        }
    }

    @Test fun wrongKeyIdFails() {
        val changed = JSONObject(signatureDocument.toString()).put("key_id", "untrusted-test-key")
        assertThrows(IllegalArgumentException::class.java) {
            ModelManifestSignatureVerifier.verify(manifest, changed, manifestSha)
        }
    }

    @Test fun signatureMetadataShaMismatchFails() {
        val changed = JSONObject(signatureDocument.toString()).put("manifest_sha256", "0".repeat(64))
        assertThrows(IllegalArgumentException::class.java) {
            ModelManifestSignatureVerifier.verify(manifest, changed, manifestSha)
        }
    }
}
