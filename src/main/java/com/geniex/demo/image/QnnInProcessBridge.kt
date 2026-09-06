package com.geniex.demo.image

import android.content.Context
import android.system.Os
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

internal object QnnInProcessNative {
    init {
        System.loadLibrary("rinqnnbridge")
    }

    external fun runContext(
        backendPath: String,
        systemLibraryPath: String,
        contextPath: String,
        inputListPath: String,
        outputDir: String,
        nativeInput: Boolean,
        nativeOutput: Boolean,
    ): String
}

internal class QnnInProcessBridgeServer(
    private val context: Context,
    private val baseDir: File,
    private val diagDir: File,
) : AutoCloseable {
    private val running = AtomicBoolean(false)
    private var server: ServerSocket? = null
    private var worker: Thread? = null
    private val logFile = File(diagDir, "qnn_inprocess_bridge.log")

    @Volatile
    var preloadSummary: String = "not-started"
        private set

    val port: Int
        get() = server?.localPort ?: 0

    fun start(): QnnInProcessBridgeServer {
        if (running.get()) return this
        diagDir.mkdirs()
        preloadSummary = configureAndPreload()
        val socket = ServerSocket(0, 2, InetAddress.getByName("127.0.0.1"))
        server = socket
        running.set(true)
        appendLog("START port=${socket.localPort} preload=$preloadSummary")
        worker = thread(name = "rin-qnn-inprocess-bridge", isDaemon = true) {
            while (running.get()) {
                try {
                    val client = socket.accept()
                    client.use { connection ->
                        connection.soTimeout = 190_000
                        val line = connection.getInputStream().bufferedReader().readLine()
                        val response = if (line.isNullOrBlank()) {
                            JSONObject().put("ok", false).put("stage", "protocol").put("detail", "empty request").toString()
                        } else {
                            handleRequest(line)
                        }
                        connection.getOutputStream().bufferedWriter().use { writer ->
                            writer.write(response)
                            writer.newLine()
                            writer.flush()
                        }
                    }
                } catch (_: SocketException) {
                    if (running.get()) appendLog("SOCKET_ERROR while running")
                } catch (t: Throwable) {
                    appendLog("SERVER_ERROR ${t.javaClass.simpleName}: ${t.message ?: "unknown"}")
                }
            }
        }
        return this
    }

    private fun configureAndPreload(): String {
        val nativeDir = File(context.applicationInfo.nativeLibraryDir).absolutePath
        val modelLib = File(baseDir, "lib").absolutePath
        val adsp = listOf(
            nativeDir,
            "/odm/lib/rfsa/adsp/aiboost/signed",
            "/odm/lib/rfsa/adsp",
            "/vendor/lib64/rfs/dsp",
            "/vendor/lib/rfsa/adsp",
            "/vendor/dsp",
            modelLib,
        ).distinct().joinToString(";")
        val oldLd = System.getenv("LD_LIBRARY_PATH").orEmpty()
        val ld = listOf(nativeDir, modelLib, oldLd).filter { it.isNotBlank() }.distinct().joinToString(":")
        runCatching { Os.setenv("ADSP_LIBRARY_PATH", adsp, true) }
        runCatching { Os.setenv("LD_LIBRARY_PATH", ld, true) }

        val results = mutableListOf<String>()
        for (name in listOf(
            "cdsprpc",
            "adsprpc",
            "cdsprpc_system",
            "adsprpc_system",
            "QnnHtpV79Stub",
            "QnnSystem",
            "QnnHtp",
        )) {
            val result = runCatching {
                System.loadLibrary(name)
                "$name=ok"
            }.getOrElse { "$name=failed:${it.javaClass.simpleName}:${it.message ?: "unknown"}" }
            results += result
        }
        results += "ADSP=$adsp"
        results += "LD=$ld"
        return results.joinToString(" | ")
    }

    private fun handleRequest(line: String): String {
        val req = JSONObject(line)
        val op = req.optString("op", "run")
        if (op != "run") {
            return JSONObject().put("ok", false).put("stage", "protocol").put("detail", "unsupported op=$op").toString()
        }
        val ctx = req.getString("context")
        val inputList = req.getString("input_list")
        val outputDir = req.getString("output_dir")
        val nativeInput = req.optBoolean("native_input", false)
        val nativeOutput = req.optBoolean("native_output", false)
        val stage = req.optString("stage", "qnn")
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val backend = File(nativeDir, "libQnnHtp.so")
        val system = File(nativeDir, "libQnnSystem.so")
        if (!backend.isFile || !system.isFile) {
            return JSONObject()
                .put("ok", false)
                .put("stage", "bridge_runtime")
                .put("detail", "QNN backend/system library missing from nativeLibraryDir")
                .toString()
        }
        File(outputDir).mkdirs()
        appendLog("REQUEST stage=$stage ctx=$ctx input=$inputList out=$outputDir nativeIn=$nativeInput nativeOut=$nativeOutput")
        val started = System.nanoTime()
        return try {
            val raw = QnnInProcessNative.runContext(
                backend.absolutePath,
                system.absolutePath,
                ctx,
                inputList,
                outputDir,
                nativeInput,
                nativeOutput,
            )
            val elapsed = (System.nanoTime() - started) / 1_000_000.0
            appendLog("RESULT stage=$stage elapsedMs=${"%.1f".format(elapsed)} raw=$raw")
            raw
        } catch (t: Throwable) {
            val elapsed = (System.nanoTime() - started) / 1_000_000.0
            val detail = "${t.javaClass.simpleName}: ${t.message ?: "native bridge failure"}"
            appendLog("NATIVE_EXCEPTION stage=$stage elapsedMs=${"%.1f".format(elapsed)} detail=$detail")
            JSONObject()
                .put("ok", false)
                .put("stage", "jni_exception")
                .put("detail", detail)
                .put("elapsed_ms", elapsed)
                .toString()
        }
    }

    private fun appendLog(line: String) {
        Log.i(TAG, line.take(4_000))
        runCatching {
            diagDir.mkdirs()
            logFile.appendText("${System.currentTimeMillis()} $line\n")
        }
    }

    override fun close() {
        if (!running.getAndSet(false)) return
        runCatching { server?.close() }
        runCatching { worker?.join(1_500) }
        appendLog("STOP")
        server = null
        worker = null
    }

    companion object {
        private const val TAG = "RinQnnBridge"
    }
}
