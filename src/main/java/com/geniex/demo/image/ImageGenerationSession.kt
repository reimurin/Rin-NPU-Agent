package com.geniex.demo.image

import android.content.Context
import java.util.concurrent.CopyOnWriteArraySet
import kotlin.concurrent.thread

data class ImageGenerationSessionSnapshot(
    val running: Boolean = false,
    val request: ImageGenerationRequest? = null,
    val progress: ImageGenerationEvent.Progress? = null,
    val latestPreview: ImageGenerationEvent.Preview? = null,
    val warning: ImageGenerationEvent.Warning? = null,
    val terminal: ImageGenerationEvent? = null,
    val startedAt: Long = 0L,
)

internal fun reduceImageGenerationSessionSnapshot(
    current: ImageGenerationSessionSnapshot,
    event: ImageGenerationEvent,
): ImageGenerationSessionSnapshot = when (event) {
    is ImageGenerationEvent.Progress -> current.copy(progress = event)
    is ImageGenerationEvent.Preview -> current.copy(latestPreview = event)
    is ImageGenerationEvent.Warning -> current.copy(warning = event)
    is ImageGenerationEvent.Complete,
    is ImageGenerationEvent.Failure,
    ImageGenerationEvent.Cancelled -> current.copy(running = false, terminal = event)
}

internal fun replayImageGenerationEvents(snapshot: ImageGenerationSessionSnapshot): List<ImageGenerationEvent> {
    val events = mutableListOf<ImageGenerationEvent>()
    snapshot.latestPreview?.let(events::add)
    snapshot.progress?.let(events::add)
    snapshot.warning?.let(events::add)
    snapshot.terminal?.let(events::add)
    return events
}

object ImageGenerationSession {
    private val lock = Any()
    private val listeners = CopyOnWriteArraySet<(ImageGenerationEvent) -> Unit>()
    @Volatile private var runtime: ImageGenerationRuntime? = null
    @Volatile private var state = ImageGenerationSessionSnapshot()
    @Volatile private var userStopRequested = false

    fun isRunning(): Boolean = state.running
    fun currentRequest(): ImageGenerationRequest? = state.request
    fun snapshot(): ImageGenerationSessionSnapshot = state

    fun attach(listener: (ImageGenerationEvent) -> Unit, replay: Boolean = true) { listeners += listener; if (replay) replay(listener) }
    fun detach(listener: (ImageGenerationEvent) -> Unit) { listeners -= listener }
    fun replay(listener: (ImageGenerationEvent) -> Unit) {
        replayImageGenerationEvents(state).forEach { event -> runCatching { listener(event) } }
    }

    fun start(context: Context, request: ImageGenerationRequest): Boolean {
        val app = context.applicationContext
        val preparing = ImageGenerationEvent.Progress(2, ImageGenerationEvent.Stage.PREPARING, totalSteps = request.steps)
        val rt: ImageGenerationRuntime
        synchronized(lock) {
            if (state.running) return false
            userStopRequested = false
            rt = ImageGenerationRuntime(app)
            runtime = rt
            state = ImageGenerationSessionSnapshot(true, request, preparing, startedAt = System.currentTimeMillis())
        }
        dispatch(preparing)
        try { ImageGenerationForegroundService.start(context) } catch (t: Throwable) {
            receive(app, ImageGenerationEvent.Failure("无法启动后台生图服务：${t.message ?: t.javaClass.simpleName}")); return false
        }
        val started = rt.generate(request, callback = { receive(app, it) })
        if (!started) receive(app, ImageGenerationEvent.Failure("已有生图任务正在运行"))
        return started
    }

    fun stop(context: Context) {
        val rt: ImageGenerationRuntime?
        synchronized(lock) { if (!state.running) return; userStopRequested = true; rt = runtime }
        rt?.stop()
        val event = ImageGenerationEvent.Cancelled
        synchronized(lock) { state = state.copy(running = false, terminal = event); runtime = null }
        dispatch(event)
        ImageGenerationForegroundService.stop(context.applicationContext)
    }

    private fun receive(app: Context, event: ImageGenerationEvent) {
        var terminal = false
        var completedSeed: Long? = null
        synchronized(lock) {
            if (userStopRequested && event !is ImageGenerationEvent.Cancelled) return
            if (!state.running && state.terminal != null) return
            if (event is ImageGenerationEvent.Complete) completedSeed = state.request?.seed
            terminal = event is ImageGenerationEvent.Complete || event is ImageGenerationEvent.Failure || event === ImageGenerationEvent.Cancelled
            state = reduceImageGenerationSessionSnapshot(state, event)
            if (terminal) runtime = null
        }
        dispatch(event)
        if (event is ImageGenerationEvent.Complete && completedSeed != null) {
            val seed = completedSeed!!
            thread(name = "rin-image-history") { runCatching { LoraHistoryStore.recordFromDiagnostics(app, ImageGenerationRuntime(app).defaultBaseDir, event.file, seed) } }
        }
        if (terminal) ImageGenerationForegroundService.stop(app)
    }

    private fun dispatch(event: ImageGenerationEvent) { listeners.forEach { listener -> runCatching { listener(event) } } }
}
