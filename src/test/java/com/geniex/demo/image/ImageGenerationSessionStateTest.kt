package com.geniex.demo.image

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ImageGenerationSessionStateTest {
    private val request = ImageGenerationRequest(
        prompt = "alpha8",
        negativePrompt = "",
        resolution = ImageResolution(832, 1216),
        steps = 8,
        seed = 42L,
    )

    @Test fun replayRetainsLatestPreviewProgressAndWarningAcrossUiReattach() {
        val progress = ImageGenerationEvent.Progress(47, ImageGenerationEvent.Stage.DENOISING, 4, 8)
        val preview = ImageGenerationEvent.Preview(File("preview_current.png"), 4, 8)
        val warning = ImageGenerationEvent.Warning("TAESD unavailable; latent preview active")
        var state = ImageGenerationSessionSnapshot(running = true, request = request, startedAt = 1L)
        state = reduceImageGenerationSessionSnapshot(state, progress)
        state = reduceImageGenerationSessionSnapshot(state, preview)
        state = reduceImageGenerationSessionSnapshot(state, warning)

        assertTrue(state.running)
        assertEquals(listOf(preview, progress, warning), replayImageGenerationEvents(state))
    }

    @Test fun terminalEventStopsSessionWithoutDroppingLatestPreview() {
        val preview = ImageGenerationEvent.Preview(File("preview_current.png"), 8, 8)
        val complete = ImageGenerationEvent.Complete(File("final.png"), 12.5)
        var state = ImageGenerationSessionSnapshot(running = true, request = request)
        state = reduceImageGenerationSessionSnapshot(state, preview)
        state = reduceImageGenerationSessionSnapshot(state, complete)

        assertFalse(state.running)
        assertEquals(preview, state.latestPreview)
        assertEquals(complete, state.terminal)
        assertEquals(listOf(preview, complete), replayImageGenerationEvents(state))
    }
}
