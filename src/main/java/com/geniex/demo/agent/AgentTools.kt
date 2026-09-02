package com.geniex.demo.agent

import com.geniex.demo.artifact.PptxBuilder
import com.geniex.demo.history.ProjectSessionStore
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class AgentToolExecutor(
    private val workspace: ProjectWorkspace,
    private val sessions: ProjectSessionStore,
    private val crossConversationReadAuthorized: () -> Boolean,
) {
    fun definitions(includeFiles: Boolean, includeHistory: Boolean): String {
        val chunks = mutableListOf<String>()
        if (includeFiles) chunks += FILE_TOOL_DEFINITIONS
        if (includeHistory) chunks += HISTORY_TOOL_DEFINITIONS
        return "[${chunks.joinToString(",")}]"
    }

    fun execute(name: String, argumentsJson: String): String = runCatching {
        require(name in ALLOWED_TOOLS) { "Unknown tool: $name" }
        require(argumentsJson.length <= MAX_TOOL_ARGUMENT_CHARS) { "Tool arguments are too large" }
        val args = Json.parseToJsonElement(argumentsJson).jsonObject
        when (name) {
            "list_directory" -> workspace.list(args.string("path", ""))
            "read_file" -> workspace.readText(args.string("path"))
            "create_directory" -> workspace.createDirectory(args.string("path"))
            "write_file" -> workspace.writeText(args.string("path"), args.string("content"))
            "create_presentation" -> createPresentation(args)
            "list_projects" -> {
                requireHistoryAuthorization()
                sessions.listProjects(100).joinToString("\n") { project ->
                    "${project.id} | ${project.name} | conversations=${project.conversationCount}"
                }
            }
            "list_conversations" -> {
                requireHistoryAuthorization()
                val project = sessions.resolveProject(args.optionalString("project")) ?: error("Project not found")
                sessions.listConversations(project.id, 200).joinToString("\n") { conversation ->
                    "${conversation.id} | ${conversation.title} | messages=${conversation.messageCount}"
                }
            }
            "read_conversation_context" -> {
                requireHistoryAuthorization()
                sessions.readConversationContext(
                    projectRef = args.optionalString("project"),
                    conversationRef = args.string("conversation"),
                    maxMessages = args.int("max_messages", 80).coerceIn(1, 200),
                    maxChars = args.int("max_chars", 160_000).coerceIn(4_000, 240_000),
                )
            }
            else -> error("Unknown tool: $name")
        }
    }.fold(onSuccess = { it }, onFailure = { "Tool error: ${it.message}" })

    private fun requireHistoryAuthorization() {
        require(crossConversationReadAuthorized()) {
            "Cross-conversation context access is not authorized for this turn. The user must explicitly ask to read/reference another project or conversation."
        }
    }

    private fun createPresentation(args: JsonObject): String {
        val pathRaw = args.string("path")
        val path = if (pathRaw.lowercase().endsWith(".pptx")) pathRaw else "$pathRaw.pptx"
        val title = args.string("title", "Presentation").take(200)
        val slidesArray = args["slides"]?.jsonArray ?: error("Missing argument: slides")
        require(slidesArray.isNotEmpty()) { "Presentation needs at least one slide" }
        require(slidesArray.size <= 60) { "Presentation is limited to 60 slides" }
        val slides = slidesArray.mapIndexed { index, element ->
            val slide = element.jsonObject
            val slideTitle = slide.string("title", "Slide ${index + 1}").take(300)
            val bullets = slide["bullets"]?.jsonArray?.map { it.jsonPrimitive.content.take(2_000) }.orEmpty().take(16)
            PptxBuilder.Slide(slideTitle, bullets)
        }
        val bytes = PptxBuilder.build(title, slides)
        return workspace.writeBytes(
            path,
            bytes,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    }

    private fun JsonObject.string(key: String, default: String? = null): String =
        this[key]?.jsonPrimitive?.content ?: default ?: error("Missing argument: $key")

    private fun JsonObject.optionalString(key: String): String? =
        this[key]?.jsonPrimitive?.content?.trim()?.takeIf { it.isNotEmpty() }

    private fun JsonObject.int(key: String, default: Int): Int = this[key]?.jsonPrimitive?.intOrNull ?: default

    companion object {
        val ALLOWED_TOOLS = setOf(
            "list_directory",
            "read_file",
            "create_directory",
            "write_file",
            "create_presentation",
            "list_projects",
            "list_conversations",
            "read_conversation_context",
        )
        private const val MAX_TOOL_ARGUMENT_CHARS = 2_000_000

        private const val FILE_TOOL_DEFINITIONS = """
          {"type":"function","function":{"name":"list_directory","description":"List files/folders in the current logical project's authorized Android workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":[]}}},
          {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text/code file in the current logical project's authorized workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
          {"type":"function","function":{"name":"create_directory","description":"Create a directory inside the current logical project's authorized workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
          {"type":"function","function":{"name":"write_file","description":"Create or replace a text/code file. Never use for PPTX/DOCX/XLSX/PDF/ZIP.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
          {"type":"function","function":{"name":"create_presentation","description":"Create a real PPTX in the current authorized project workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"},"title":{"type":"string"},"slides":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},"bullets":{"type":"array","items":{"type":"string"}}},"required":["title","bullets"]}}},"required":["path","title","slides"]}}}
        """

        private const val HISTORY_TOOL_DEFINITIONS = """
          {"type":"function","function":{"name":"list_projects","description":"List persistent logical projects. Use ONLY when the user explicitly asks to read/reference another project or conversation.","parameters":{"type":"object","properties":{},"required":[]}}},
          {"type":"function","function":{"name":"list_conversations","description":"List conversations in a persistent project. Use ONLY on an explicit user request to read/reference history.","parameters":{"type":"object","properties":{"project":{"type":"string","description":"Project name or ID; omit for current project"}},"required":[]}}},
          {"type":"function","function":{"name":"read_conversation_context","description":"Read user/assistant context from one persistent conversation. This is read-only and available ONLY for the current turn when the user explicitly asks to read/reference another conversation/project.","parameters":{"type":"object","properties":{"project":{"type":"string","description":"Project name or ID; omit for current project"},"conversation":{"type":"string","description":"Conversation title or ID"},"max_messages":{"type":"integer"},"max_chars":{"type":"integer"}},"required":["conversation"]}}}
        """
    }
}

data class ParsedToolCall(val name: String, val arguments: String)

object AgentToolParser {
    private val qwenTag = Regex("<tool_call>\\s*(.*?)\\s*</tool_call>", setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE))
    private val jsonFence = Regex("```(?:json)?\\s*(\\{.*?})\\s*```", setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE))

    fun parse(text: String): List<ParsedToolCall> {
        if (text.length > MAX_PARSE_CHARS) return emptyList()
        val payloads = mutableListOf<String>()
        payloads += qwenTag.findAll(text).map { it.groupValues[1] }.toList()
        if (payloads.isEmpty()) payloads += jsonFence.findAll(text).map { it.groupValues[1] }.toList()
        if (payloads.isEmpty()) {
            val trimmed = text.trim()
            if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) payloads += trimmed
        }
        return payloads.flatMap { parsePayload(it) }.distinctBy { it.name + "\u0000" + it.arguments }
    }

    private fun parsePayload(payload: String): List<ParsedToolCall> = runCatching {
        parseElement(Json.parseToJsonElement(payload))
    }.getOrDefault(emptyList())

    private fun parseElement(element: JsonElement): List<ParsedToolCall> = when (element) {
        is JsonArray -> element.flatMap(::parseElement)
        is JsonObject -> {
            val nested = element["tool_calls"]
            if (nested != null) {
                parseElement(nested)
            } else {
                val function = element["function"] as? JsonObject
                val name = (function?.get("name") ?: element["name"])?.jsonPrimitive?.content
                if (name == null || name !in AgentToolExecutor.ALLOWED_TOOLS) {
                    emptyList()
                } else {
                    val argsElement = function?.get("arguments") ?: element["arguments"]
                    val args = when (argsElement) {
                        null -> "{}"
                        is JsonObject -> argsElement.toString()
                        else -> argsElement.jsonPrimitive.content
                    }
                    listOf(ParsedToolCall(name, args))
                }
            }
        }
        else -> emptyList()
    }

    private const val MAX_PARSE_CHARS = 2_000_000
}
