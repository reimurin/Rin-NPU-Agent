package com.geniex.demo.model

data class CatalogModel(
    val repoId: String,
    val displayName: String = repoId,
    val quant: String? = null,
    val multimodal: Boolean = false,
    val compatible: Boolean = true,
    val downloads: Long = 0,
    val hub: String = "HUGGINGFACE",
    val chipset: String? = null,
    val runtime: String = "llama_cpp",
)

object RecommendedModels {
    val primary = CatalogModel(
        repoId = "ai-hub-models/Qwen3-4B-Instruct-2507",
        displayName = "Qwen3-4B-Instruct-2507 (NPU)",
        hub = "AUTO",
        chipset = "SM8750",
        runtime = "qairt",
    )

    val vision = CatalogModel(
        repoId = "ai-hub-models/Qwen2.5-VL-7B-Instruct",
        displayName = "Qwen2.5-VL-7B-Instruct (NPU)",
        multimodal = true,
        hub = "AUTO",
        chipset = "SM8750",
        runtime = "qairt",
    )

    val ggufFallback = CatalogModel(
        repoId = "unsloth/Qwen3.5-2B-GGUF",
        displayName = "Qwen3.5-2B (GGUF fallback)",
        quant = "Q4_0",
        multimodal = true,
    )

    val all = listOf(primary, vision, ggufFallback)
}
