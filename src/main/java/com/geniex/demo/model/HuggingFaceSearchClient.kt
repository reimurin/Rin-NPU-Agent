package com.geniex.demo.model

import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

class HuggingFaceSearchClient {
    fun search(query: String, limit: Int = 30): List<CatalogModel> {
        val encoded = URLEncoder.encode(query.trim(), Charsets.UTF_8.name())
        val url = URL("https://huggingface.co/api/models?search=$encoded&limit=$limit&sort=downloads&direction=-1&full=true")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = 12_000
            readTimeout = 20_000
            requestMethod = "GET"
            setRequestProperty("Accept", "application/json")
            setRequestProperty("User-Agent", "Rin-NPU-Agent/0.1")
        }
        try {
            if (conn.responseCode !in 200..299) error("HTTP ${conn.responseCode}")
            val raw = conn.inputStream.bufferedReader().use { it.readText() }
            return Json.parseToJsonElement(raw).jsonArray.mapNotNull { node ->
                val obj = node.jsonObject
                val id = obj["id"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val tags = obj["tags"]?.jsonArray?.mapNotNull { it.jsonPrimitive.contentOrNull }.orEmpty()
                val pipeline = obj["pipeline_tag"]?.jsonPrimitive?.contentOrNull.orEmpty()
                val lower = id.lowercase()
                val gguf = lower.contains("gguf") || tags.any { it.equals("gguf", true) }
                val multimodal = pipeline.contains("image", true) || tags.any { it.contains("image-text-to-text", true) } || lower.contains("vl") || lower.contains("qwen3.5")
                CatalogModel(id, id, "Q4_0", multimodal, gguf, obj["downloads"]?.jsonPrimitive?.longOrNull ?: 0L)
            }
        } finally {
            conn.disconnect()
        }
    }
}
