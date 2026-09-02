// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.app.Application
import android.system.Os
import android.util.Log
import java.io.File

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        runCatching {
            Os.setenv("GENIEX_DL_FILE_CONCURRENCY", "4", true)
            Os.setenv("GENIEX_DL_CHUNK_CONCURRENCY", "4", true)
            Os.setenv("GENIEX_DL_CHUNK_SIZE", "8388608", true)
        }
        clearLegacyModelsDir()
    }

    /**
     * Old builds downloaded models into `filesDir/models/{id}/...` with a
     * hand-rolled manifest. The Rust model manager owns its own layout
     * under `filesDir/geniex/models/{org}/{repo}/...` and cannot read the
     * old files. Wipe the legacy dir on first launch so users don't keep
     * paying for stranded gigabytes.
     */
    private fun clearLegacyModelsDir() {
        val legacy = File(filesDir, "models")
        if (!legacy.exists()) return
        val ok = runCatching { legacy.deleteRecursively() }.getOrElse { false }
        Log.i(TAG, "legacy models dir cleanup: ok=$ok path=${legacy.absolutePath}")
    }

    companion object {
        private const val TAG = "GenieXDemo"
    }
}
