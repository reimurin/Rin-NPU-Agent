package com.geniex.demo.image
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import com.geniex.demo.BuildConfig
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

internal object StartupDiagnostics {
    @Volatile private var installed=false
    private const val NAME="startup_last_error.txt"
    @Synchronized fun install(context:Context) {
        if(installed)return
        installed=true
        val app=context.applicationContext ?: context
        val previous=Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread,error ->
            record(app,"uncaught thread=${thread.name}",error)
            previous?.uncaughtException(thread,error)
        }
    }
    fun record(context:Context,phase:String,error:Throwable) {
        runCatching {
            val trace=StringWriter().also { error.printStackTrace(PrintWriter(it)) }.toString()
            File(context.filesDir,NAME).writeText("Rin startup diagnostic\nversion=${BuildConfig.VERSION_NAME}\npackage=${context.packageName}\napi=${Build.VERSION.SDK_INT}\nphase=$phase\n\n"+trace.take(48_000))
        }
    }
    fun share(activity:Activity) {
        val file=File(activity.filesDir,NAME)
        val text=if(file.isFile)file.readText().take(48_000) else "尚未记录到启动异常。"
        activity.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type="text/plain";putExtra(Intent.EXTRA_TEXT,text)
        },"分享启动诊断"))
    }
}
