package com.geniex.demo.utils

import android.content.Context
import com.geniex.demo.R

object UiErrorLocalizer {
    fun message(context: Context, raw: String?): String {
        val text = raw.orEmpty().lowercase()
        val resId = when {
            text.contains("out of memory") || text.contains("memory allocation") || text.contains("oom") -> R.string.sdk_error_memory
            (text.contains("npu") || text.contains("htp") || text.contains("qnn")) &&
                (text.contains("fail") || text.contains("error") || text.contains("unavailable")) -> R.string.sdk_error_npu
            text.contains("timeout") || text.contains("timed out") || text.contains("connection") || text.contains("network") -> R.string.sdk_error_network
            text.contains("tokenizer") -> R.string.sdk_error_tokenizer
            text.contains("unsupported") || text.contains("not supported") -> R.string.sdk_error_unsupported
            text.contains("model") && (text.contains("not found") || text.contains("path") || text.contains("missing")) -> R.string.sdk_error_model_files
            text.contains("load") && (text.contains("fail") || text.contains("error")) -> R.string.sdk_error_load
            else -> R.string.sdk_error_generic
        }
        return context.getString(resId)
    }
}
