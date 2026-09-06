package com.geniex.demo.image

import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

internal class RuntimePartDownloader(
    private val cancelled: AtomicBoolean,
) {
    private data class Segment(val index: Int, val start: Long, val end: Long) {
        val length: Long get() = end - start + 1L
    }

    @Volatile private var activePool: ExecutorService? = null

    fun cancel() {
        activePool?.shutdownNow()
    }

    fun hasSegmentState(target: File): Boolean = stateFile(target).isFile

    fun cleanupState(target: File) {
        stateFile(target).delete()
    }

    fun download(
        target: File,
        url: String,
        expectedBytes: Long,
        requestedThreads: Int = DEFAULT_THREADS,
        onProgress: (done: Long, total: Long, bytesPerSecond: Long, etaSeconds: Long, threads: Int) -> Unit,
    ): File {
        target.parentFile?.mkdirs()
        if (expectedBytes <= 0L) return singleDownload(target, url, expectedBytes, onProgress)

        val threadCount = requestedThreads.coerceIn(1, MAX_THREADS)
        val segments = buildSegments(expectedBytes, threadCount)
        var completed = loadState(target, expectedBytes, segments.size)
        if (!target.isFile || target.length() != expectedBytes) {
            completed = BooleanArray(segments.size)
            stateFile(target).delete()
        }
        if (target.isFile && completed.indices.all { completed[it] }) {
            onProgress(expectedBytes, expectedBytes, 0L, 0L, segments.size)
            return target
        }

        if (!supportsRanges(url, expectedBytes)) {
            stateFile(target).delete()
            target.delete()
            return singleDownload(target, url, expectedBytes, onProgress)
        }

        RandomAccessFile(target, "rw").use { raf ->
            if (raf.length() != expectedBytes) raf.setLength(expectedBytes)
        }
        persistState(target, expectedBytes, completed)

        val initialDone = segments.filter { completed[it.index] }.sumOf { it.length }
        val totalDone = AtomicLong(initialDone)
        val sessionStart = SystemClock.elapsedRealtime()
        val lastEmit = AtomicLong(0L)
        val failure = AtomicReference<Throwable?>(null)
        val stateLock = Any()
        val incomplete = segments.filterNot { completed[it.index] }
        if (incomplete.isEmpty()) return target

        val pool = Executors.newFixedThreadPool(incomplete.size.coerceAtMost(threadCount))
        activePool = pool
        val latch = CountDownLatch(incomplete.size)

        fun emit(force: Boolean = false) {
            val now = SystemClock.elapsedRealtime()
            val previous = lastEmit.get()
            if (!force && now - previous < PROGRESS_INTERVAL_MS) return
            if (!force && !lastEmit.compareAndSet(previous, now)) return
            if (force) lastEmit.set(now)
            val done = totalDone.get().coerceAtMost(expectedBytes)
            val elapsedMs = (now - sessionStart).coerceAtLeast(1L)
            val sessionBytes = (done - initialDone).coerceAtLeast(0L)
            val speed = (sessionBytes * 1000L / elapsedMs).coerceAtLeast(0L)
            val eta = if (speed > 0L) ((expectedBytes - done).coerceAtLeast(0L) / speed) else -1L
            onProgress(done, expectedBytes, speed, eta, incomplete.size.coerceAtMost(threadCount))
        }

        emit(true)
        incomplete.forEach { segment ->
            pool.execute {
                try {
                    checkCancelled()
                    val conn = openConnection(url, "bytes=${segment.start}-${segment.end}")
                    try {
                        require(conn.responseCode == HttpURLConnection.HTTP_PARTIAL) {
                            "Range request HTTP ${conn.responseCode}"
                        }
                        val contentRange = conn.getHeaderField("Content-Range").orEmpty()
                        require(contentRange.startsWith("bytes ${segment.start}-${segment.end}/")) {
                            "Unexpected Content-Range: $contentRange"
                        }
                        RandomAccessFile(target, "rw").use { raf ->
                            raf.seek(segment.start)
                            BufferedInputStream(conn.inputStream, BUFFER_SIZE).use { input ->
                                val buffer = ByteArray(BUFFER_SIZE)
                                var remaining = segment.length
                                while (remaining > 0L) {
                                    checkCancelled()
                                    val n = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
                                    if (n < 0) error("Range ended early for segment ${segment.index + 1}")
                                    raf.write(buffer, 0, n)
                                    remaining -= n.toLong()
                                    totalDone.addAndGet(n.toLong())
                                    emit(false)
                                }
                            }
                        }
                    } finally {
                        conn.disconnect()
                    }
                    synchronized(stateLock) {
                        completed[segment.index] = true
                        persistState(target, expectedBytes, completed)
                    }
                    emit(true)
                } catch (t: Throwable) {
                    failure.compareAndSet(null, t)
                } finally {
                    latch.countDown()
                }
            }
        }

        try {
            latch.await()
        } finally {
            pool.shutdownNow()
            if (activePool === pool) activePool = null
        }
        failure.get()?.let { throw it }
        checkCancelled()
        require(completed.all { it }) { "Parallel download did not complete every segment" }
        require(target.length() == expectedBytes) { "Downloaded package part size mismatch" }
        emit(true)
        return target
    }

    private fun singleDownload(
        target: File,
        url: String,
        expectedBytes: Long,
        onProgress: (done: Long, total: Long, bytesPerSecond: Long, etaSeconds: Long, threads: Int) -> Unit,
    ): File {
        var existing = if (target.isFile) target.length() else 0L
        if (expectedBytes > 0L && existing > expectedBytes) {
            target.delete()
            existing = 0L
        }
        if (expectedBytes > 0L && existing == expectedBytes) {
            onProgress(existing, expectedBytes, 0L, 0L, 1)
            return target
        }
        var conn = openConnection(url, if (existing > 0L) "bytes=$existing-" else null)
        if (existing > 0L && conn.responseCode != HttpURLConnection.HTTP_PARTIAL) {
            conn.disconnect()
            target.delete()
            existing = 0L
            conn = openConnection(url, null)
        }
        require(conn.responseCode == HttpURLConnection.HTTP_OK || conn.responseCode == HttpURLConnection.HTTP_PARTIAL) {
            "Runtime package HTTP ${conn.responseCode}"
        }
        val contentLength = conn.contentLengthLong.coerceAtLeast(0L)
        val total = if (expectedBytes > 0L) expectedBytes else existing + contentLength
        val startDone = existing
        val startedAt = SystemClock.elapsedRealtime()
        var lastEmit = 0L
        RandomAccessFile(target, "rw").use { raf ->
            raf.seek(existing)
            BufferedInputStream(conn.inputStream, BUFFER_SIZE).use { input ->
                val buffer = ByteArray(BUFFER_SIZE)
                var done = existing
                while (true) {
                    checkCancelled()
                    val n = input.read(buffer)
                    if (n < 0) break
                    raf.write(buffer, 0, n)
                    done += n.toLong()
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastEmit >= PROGRESS_INTERVAL_MS) {
                        lastEmit = now
                        val elapsed = (now - startedAt).coerceAtLeast(1L)
                        val speed = ((done - startDone).coerceAtLeast(0L) * 1000L / elapsed).coerceAtLeast(0L)
                        val eta = if (speed > 0L && total > 0L) ((total - done).coerceAtLeast(0L) / speed) else -1L
                        onProgress(done, total, speed, eta, 1)
                    }
                }
            }
        }
        conn.disconnect()
        if (expectedBytes > 0L) require(target.length() == expectedBytes) { "Downloaded package part size mismatch" }
        onProgress(target.length(), total, 0L, 0L, 1)
        return target
    }

    private fun supportsRanges(url: String, expectedBytes: Long): Boolean = runCatching {
        val conn = openConnection(url, "bytes=0-0")
        try {
            if (conn.responseCode != HttpURLConnection.HTTP_PARTIAL) return@runCatching false
            val range = conn.getHeaderField("Content-Range").orEmpty()
            if (!range.endsWith("/$expectedBytes")) return@runCatching false
            conn.inputStream.use { it.read() }
            true
        } finally {
            conn.disconnect()
        }
    }.getOrDefault(false)

    private fun openConnection(rawUrl: String, range: String?): HttpURLConnection {
        var current = URL(rawUrl)
        repeat(MAX_REDIRECTS) {
            val conn = (current.openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", "Rin-NPU-Agent/1.5.4")
                setRequestProperty("Accept-Encoding", "identity")
                setRequestProperty("Cache-Control", "no-cache")
                if (range != null) setRequestProperty("Range", range)
            }
            val code = conn.responseCode
            if (code in REDIRECT_CODES) {
                val location = conn.getHeaderField("Location") ?: error("Redirect without Location")
                val next = URL(current, location)
                conn.disconnect()
                current = next
            } else {
                return conn
            }
        }
        error("Too many HTTP redirects")
    }

    private fun buildSegments(total: Long, count: Int): List<Segment> {
        val actual = count.coerceAtMost(total.coerceAtLeast(1L).toInt().coerceAtLeast(1))
        val base = total / actual
        val remainder = total % actual
        var cursor = 0L
        return (0 until actual).map { index ->
            val length = base + if (index < remainder) 1L else 0L
            val segment = Segment(index, cursor, cursor + length - 1L)
            cursor += length
            segment
        }
    }

    private fun loadState(target: File, expectedBytes: Long, segmentCount: Int): BooleanArray {
        val result = BooleanArray(segmentCount)
        val file = stateFile(target)
        if (!file.isFile) return result
        runCatching {
            val json = JSONObject(file.readText())
            if (json.optLong("expectedBytes") != expectedBytes || json.optInt("segments") != segmentCount) return result
            val array = json.optJSONArray("completed") ?: return result
            for (i in 0 until minOf(segmentCount, array.length())) result[i] = array.optBoolean(i, false)
        }
        return result
    }

    private fun persistState(target: File, expectedBytes: Long, completed: BooleanArray) {
        val file = stateFile(target)
        val tmp = File(file.parentFile, file.name + ".tmp")
        val array = JSONArray()
        completed.forEach { array.put(it) }
        val json = JSONObject()
            .put("expectedBytes", expectedBytes)
            .put("segments", completed.size)
            .put("completed", array)
        tmp.writeText(json.toString())
        if (file.exists()) file.delete()
        check(tmp.renameTo(file)) { "Could not persist download state" }
    }

    private fun stateFile(target: File): File = File(target.parentFile, target.name + ".segments.json")

    private fun checkCancelled() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) throw InterruptedException("Install cancelled")
    }

    companion object {
        const val DEFAULT_THREADS = 8
        private const val MAX_THREADS = 12
        private const val MAX_REDIRECTS = 8
        private const val BUFFER_SIZE = 1024 * 1024
        private const val PROGRESS_INTERVAL_MS = 250L
        private val REDIRECT_CODES = setOf(301, 302, 303, 307, 308)
    }
}
