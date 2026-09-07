package com.geniex.demo.image

import android.app.Application
import android.content.Intent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class ImageSeedOverrideTest {
    @Test
    fun explicitAdbSeedWinsOnlyForAutomation() {
        val intent = Intent().putExtra(ImageModeController.EXTRA_TEST_SEED, 123456789L)
        assertEquals(123456789L, resolveImageGenerationSeed(intent, allowTestOverride = true) { 77L })
    }

    @Test
    fun staleAdbSeedIsIgnoredForManualGeneration() {
        val intent = Intent().putExtra(ImageModeController.EXTRA_TEST_SEED, 123456789L)
        assertEquals(77L, resolveImageGenerationSeed(intent) { 77L })
    }

    @Test
    fun absentOverrideUsesExistingTimeMaskBehavior() {
        val now = 0x8000002aL
        assertEquals(42L, resolveImageGenerationSeed(Intent()) { now })
    }

    @Test
    fun outOfRangeAutomationOverrideFallsBackToClock() {
        val intent = Intent().putExtra(ImageModeController.EXTRA_TEST_SEED, 0x80000000L)
        assertEquals(99L, resolveImageGenerationSeed(intent, allowTestOverride = true) { 99L })
    }

    @Test
    fun automationIsDisabledByDefault() {
        val test = resolveImageTestAutomation(Intent())
        assertNull(test.prompt)
        assertFalse(test.autorun)
    }

    @Test
    fun automationRequiresExplicitPromptAndAutorunFlag() {
        val enabled = resolveImageTestAutomation(
            Intent()
                .putExtra(ImageModeController.EXTRA_TEST_PROMPT, "fixed_prompt")
                .putExtra(ImageModeController.EXTRA_TEST_AUTORUN, true),
        )
        assertEquals("fixed_prompt", enabled.prompt)
        assertTrue(enabled.autorun)

        val promptOnly = resolveImageTestAutomation(
            Intent().putExtra(ImageModeController.EXTRA_TEST_PROMPT, "fixed_prompt"),
        )
        assertEquals("fixed_prompt", promptOnly.prompt)
        assertFalse(promptOnly.autorun)
    }
}
