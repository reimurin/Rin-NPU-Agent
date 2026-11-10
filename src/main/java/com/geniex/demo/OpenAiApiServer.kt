package com.geniex.demo

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.net.InetAddress
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

data class ApiMessage(val role: String, val content: String)
data class ApiCompletionProfile(val promptTokens: Int, val completionTokens: Int, val ttftMs: Double, val prefillTokensPerSecond: Double, val decodeTokensPerSecond: Double)
data class ApiCompletionResult(val text: String, val profile: ApiCompletionProfile)

/** OpenAI-compatible HTTP server backed by the loaded model. */
class OpenAiApiServer(
    private val modelId: String,
    private val log: (String) -> Unit,
    private val complete: suspend (List<ApiMessage>, Int, (String) -> Unit) -> ApiCompletionResult,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val running = AtomicBoolean(false)
    private var server: ServerSocket? = null

    fun start(): String {
        val endpoint = preferredEndpoint()
        if (!running.compareAndSet(false, true)) return endpoint
        server = ServerSocket(PORT, 8, InetAddress.getByName(BIND_ADDRESS))
        scope.launch {
            log("HTTP server listening on $BIND_ADDRESS:$PORT; client endpoint $endpoint")
            while (running.get()) {
                val socket = runCatching { server?.accept() }.getOrNull() ?: break
                launch { handle(socket) }
            }
        }
        return endpoint
    }

    private fun preferredEndpoint(): String {
        val addresses = runCatching {
            NetworkInterface.getNetworkInterfaces().toList()
                .flatMap { it.inetAddresses.toList() }
                .filterIsInstance<Inet4Address>()
                .filterNot { it.isLoopbackAddress }
                .mapNotNull { it.hostAddress }
        }.getOrDefault(emptyList())
        val tailnet = addresses.firstOrNull { address ->
            val octets = address.split('.').mapNotNull(String::toIntOrNull)
            octets.size == 4 && octets[0] == 100 && octets[1] in 64..127
        }
        return "http://${tailnet ?: addresses.firstOrNull() ?: LOOPBACK_ADDRESS}:$PORT/v1"
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        runCatching { server?.close() }
        scope.cancel()
        log("HTTP server stopped")
    }

    private suspend fun handle(socket: Socket) {
        socket.use {
            it.soTimeout = 30_000
            val input = BufferedInputStream(it.getInputStream())
            val output = BufferedOutputStream(it.getOutputStream())
            val request = readLine(input)?.split(' ') ?: return
            if (request.size < 2) return
            val method = request[0]
            val path = request[1].substringBefore('?')
            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = readLine(input) ?: break
                if (line.isEmpty()) break
                val at = line.indexOf(':')
                if (at > 0) headers[line.substring(0, at).lowercase()] = line.substring(at + 1).trim()
            }
            val length = headers["content-length"]?.toIntOrNull() ?: 0
            require(length in 0..MAX_BODY_BYTES) { "Request body too large" }
            val bytes = ByteArray(length)
            var offset = 0
            while (offset < length) {
                val count = input.read(bytes, offset, length - offset)
                if (count < 0) break
                offset += count
            }
            val id = "chatcmpl-${UUID.randomUUID()}"
            log("[$id] $method $path $offset bytes")
            try {
                when {
                    method == "GET" && path == "/v1/models" -> sendJson(output, 200, models())
                    method == "GET" && path == "/health" -> sendJson(output, 200, JSONObject().put("status", "ok").put("model", modelId))
                    method == "POST" && path == "/v1/chat/completions" -> chat(output, String(bytes, 0, offset, StandardCharsets.UTF_8), id)
                    else -> sendError(output, 404, "Not found", "invalid_request_error")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "request failed", t)
                log("[$id] ERROR ${t.stackTraceToString()}")
                runCatching { sendError(output, 500, t.message ?: "Inference failed", "server_error") }
            }
            runCatching { output.flush() }
        }
    }

    private suspend fun chat(output: BufferedOutputStream, body: String, id: String) {
        val request = JSONObject(body)
        val input = request.optJSONArray("messages") ?: throw IllegalArgumentException("messages is required")
        val messages = buildList {
            for (i in 0 until input.length()) {
                val item = input.getJSONObject(i)
                add(ApiMessage(item.getString("role"), textContent(item.opt("content"))))
            }
        }
        val maxTokens = request.optInt("max_completion_tokens", request.optInt("max_tokens", 2048)).coerceIn(1, 8192)
        val stream = request.optBoolean("stream", false)
        log("[$id] model=${request.optString("model", modelId)} stream=$stream max_tokens=$maxTokens messages=${messages.size}")
        messages.forEachIndexed { i, message -> log("[$id] prompt[$i] ${message.role}: ${message.content}") }
        val started = System.nanoTime()
        if (stream) {
            writeHeaders(output, 200, "text/event-stream", null)
            output.flush()
            var first = true
            val result = complete(messages, maxTokens) { token ->
                val delta = JSONObject().put("content", token)
                if (first) delta.put("role", "assistant")
                first = false
                sendSse(output, base(id, "chat.completion.chunk").put("choices", JSONArray().put(JSONObject().put("index", 0).put("delta", delta).put("finish_reason", JSONObject.NULL))).toString())
                log("[$id] token: $token")
            }
            sendSse(output, base(id, "chat.completion.chunk").put("choices", JSONArray().put(JSONObject().put("index", 0).put("delta", JSONObject()).put("finish_reason", "stop"))).put("usage", usage(result.profile)).toString())
            sendSse(output, "[DONE]")
            logResult(id, result, started)
        } else {
            val result = complete(messages, maxTokens) { token -> log("[$id] token: $token") }
            sendJson(output, 200, base(id, "chat.completion")
                .put("choices", JSONArray().put(JSONObject().put("index", 0).put("message", JSONObject().put("role", "assistant").put("content", result.text)).put("finish_reason", "stop")))
                .put("usage", usage(result.profile)))
            logResult(id, result, started)
        }
    }

    private fun logResult(id: String, result: ApiCompletionResult, started: Long) {
        val p = result.profile
        log("[$id] completed elapsed_ms=%.2f ttft_ms=%.2f prompt_tokens=%d completion_tokens=%d prefill_tok_s=%.2f decode_tok_s=%.2f".format((System.nanoTime() - started) / 1_000_000.0, p.ttftMs, p.promptTokens, p.completionTokens, p.prefillTokensPerSecond, p.decodeTokensPerSecond))
        log("[$id] response: ${result.text}")
    }

    private fun models() = JSONObject().put("object", "list").put("data", JSONArray().put(JSONObject().put("id", modelId).put("object", "model").put("created", System.currentTimeMillis() / 1000).put("owned_by", "local-npu")))
    private fun base(id: String, kind: String) = JSONObject().put("id", id).put("object", kind).put("created", System.currentTimeMillis() / 1000).put("model", modelId)
    private fun usage(p: ApiCompletionProfile) = JSONObject().put("prompt_tokens", p.promptTokens).put("completion_tokens", p.completionTokens).put("total_tokens", p.promptTokens + p.completionTokens)
    private fun textContent(value: Any?): String = when (value) {
        is String -> value
        is JSONArray -> buildList { for (i in 0 until value.length()) value.optJSONObject(i)?.takeIf { it.optString("type") == "text" }?.let { add(it.optString("text")) } }.joinToString("\n")
        else -> value?.toString().orEmpty()
    }
    private fun sendJson(out: BufferedOutputStream, status: Int, value: JSONObject) { val bytes = value.toString().toByteArray(StandardCharsets.UTF_8); writeHeaders(out, status, "application/json", bytes.size); out.write(bytes) }
    private fun sendError(out: BufferedOutputStream, status: Int, message: String, type: String) = sendJson(out, status, JSONObject().put("error", JSONObject().put("message", message).put("type", type).put("code", JSONObject.NULL)))
    private fun sendSse(out: BufferedOutputStream, value: String) { out.write("data: $value\n\n".toByteArray(StandardCharsets.UTF_8)); out.flush() }
    private fun writeHeaders(out: BufferedOutputStream, status: Int, type: String, length: Int?) { out.write(buildString { append("HTTP/1.1 $status ${if (status in 200..299) "OK" else "Error"}\r\nContent-Type: $type\r\nCache-Control: no-cache\r\n"); if (length != null) append("Content-Length: $length\r\n"); append("Connection: close\r\n\r\n") }.toByteArray(StandardCharsets.US_ASCII)) }
    private fun readLine(input: BufferedInputStream): String? { val bytes = ArrayList<Byte>(); while (true) { val value = input.read(); if (value < 0) return if (bytes.isEmpty()) null else bytes.toByteArray().toString(StandardCharsets.UTF_8); if (value == '\n'.code) break; if (value != '\r'.code) bytes.add(value.toByte()); require(bytes.size <= 16_384) { "HTTP header too long" } }; return bytes.toByteArray().toString(StandardCharsets.UTF_8) }

    companion object {
        const val PORT = 8080
        const val BIND_ADDRESS = "0.0.0.0"
        const val LOOPBACK_ADDRESS = "127.0.0.1"
        private const val MAX_BODY_BYTES = 2 * 1024 * 1024
        private const val TAG = "OpenAiApiServer"
    }
}
