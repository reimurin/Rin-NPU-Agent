package com.geniex.demo.image

import java.text.Normalizer

internal data class LoraSelection(val name: String, val weight: Double)
internal data class LoraPrompt(val text: String, val selections: List<LoraSelection>)
internal object LoraTags {
    private val tag = Regex("<lora:([^<>]+)>", RegexOption.IGNORE_CASE)
    private val number = Regex("[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)")

    private fun normalizeName(raw: String): String {
        var name = Normalizer.normalize(raw.trim(), Normalizer.Form.NFC)
        if (name.endsWith(".safetensors", true)) name = name.dropLast(12)
        require(name.isNotBlank() && name != "." && name != ".." &&
            name.none { it in "/\\<>:" || it.code < 32 }) { "LoRA 名称无效" }
        return name
    }

    fun parse(prompt: String): LoraPrompt {
        require(prompt.length <= 100_000) { "提示词过长" }
        val selections = linkedMapOf<String, LoraSelection>()
        val text = tag.replace(prompt) { match ->
            val body = match.groupValues[1]
            val colon = body.lastIndexOf(':')
            val rawName = (if (colon >= 0) body.substring(0, colon) else body).trim()
            val value = if (colon >= 0) body.substring(colon + 1).trim() else "1.0"
            require(number.matches(value)) { "LoRA 权重格式错误：$value" }
            val weight = value.toDoubleOrNull() ?: error("LoRA 权重无效")
            require(weight.isFinite() && weight in -2.0..2.0) { "当前测试权重范围为 -2 到 2" }
            val name = normalizeName(rawName)
            val old = selections[name]
            require(old == null || old.weight == weight) { "同一个 LoRA 出现冲突权重：$name" }
            selections[name] = LoraSelection(name, weight)
            ""
        }
        require(!Regex("<\\s*lora\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)) { "LoRA 标签未闭合或格式错误" }
        return LoraPrompt(text, selections.values.toList())
    }

    fun format(name: String, weight: Double = 0.8): String {
        val label = "<lora:${normalizeName(name)}:$weight>"
        parse(label)
        return label
    }

    fun contains(prompt: String, name: String): Boolean {
        val target = normalizeName(name)
        return parse(prompt).selections.any { it.name == target }
    }

    fun remove(prompt: String, name: String): String {
        val target = normalizeName(name)
        val withoutTarget = tag.replace(prompt) { match ->
            val parsed = parse(match.value).selections.singleOrNull()
            if (parsed?.name == target) "" else match.value
        }
        return withoutTarget
            .replace(Regex("\\s*,\\s*,+\\s*"), ", ")
            .replace(Regex("^\\s*,\\s*"), "")
            .replace(Regex("\\s*,\\s*$"), "")
            .replace(Regex("[ \\t]{2,}"), " ")
            .trim()
    }
}
