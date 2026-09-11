// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.app.Application
import com.geniex.demo.image.StartupDiagnostics
import android.system.Os

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        runCatching {
            Os.setenv("GENIEX_DL_FILE_CONCURRENCY", "4", true)
            Os.setenv("GENIEX_DL_CHUNK_CONCURRENCY", "4", true)
            Os.setenv("GENIEX_DL_CHUNK_SIZE", "8388608", true)
        }
        StartupDiagnostics.install(this)
        // Upgrade must not delete user model or preset data automatically.
    }

    companion object {
        private const val TAG = "GenieXDemo"
    }
}
