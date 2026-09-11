package com.geniex.demo.image

import com.google.crypto.tink.subtle.Ed25519Verify
import org.json.JSONObject
import java.util.Base64
import java.util.Locale

object ModelManifestSignatureVerifier {
    const val ALGORITHM = "Ed25519"
    const val CURRENT_KEY_ID = "rin-model-pack-ed25519-2026-09-01"

    private val trustedPublicKeys = mapOf(
        CURRENT_KEY_ID to "zE7NARTmhoS2rvE59NYCn1ZbmF4KTv5R/ciIzADc7nk=",
    )
    private val sha256Pattern = Regex("^[0-9a-f]{64}$")

    fun verify(manifestBytes: ByteArray, signatureDocument: JSONObject, expectedManifestSha256: String) {
        require(signatureDocument.optInt("schema", -1) == 1) { "Unsupported model manifest signature schema" }
        require(signatureDocument.optString("algorithm") == ALGORITHM) { "Unsupported model manifest signature algorithm" }

        val keyId = signatureDocument.optString("key_id")
        val publicKeyBase64 = trustedPublicKeys[keyId]
            ?: throw IllegalArgumentException("Untrusted model manifest signing key: $keyId")
        val publicKey = Base64.getDecoder().decode(publicKeyBase64)
        require(publicKey.size == 32) { "Invalid trusted Ed25519 public key" }

        val expectedSha = expectedManifestSha256.lowercase(Locale.ROOT)
        require(expectedSha.matches(sha256Pattern)) { "Invalid expected manifest SHA-256" }
        val signedSha = signatureDocument.optString("manifest_sha256").lowercase(Locale.ROOT)
        require(signedSha == expectedSha) { "Manifest signature metadata SHA-256 mismatch" }

        verifyDetached(publicKeyBase64, manifestBytes, signatureDocument.getString("signature"))
    }

    internal fun verifyDetached(publicKeyBase64: String, message: ByteArray, signatureBase64: String) {
        val publicKey = runCatching { Base64.getDecoder().decode(publicKeyBase64) }
            .getOrElse { throw IllegalArgumentException("Malformed Ed25519 public key", it) }
        require(publicKey.size == 32) { "Invalid Ed25519 public key length" }
        val signature = runCatching { Base64.getDecoder().decode(signatureBase64.trim()) }
            .getOrElse { throw IllegalArgumentException("Malformed Ed25519 signature", it) }
        require(signature.size == 64) { "Invalid Ed25519 signature length" }
        runCatching { Ed25519Verify(publicKey).verify(signature, message) }
            .getOrElse { throw IllegalArgumentException("Invalid Ed25519 signature", it) }
    }
}
