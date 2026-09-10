package com.geniex.demo.image

import org.junit.Assert.*
import org.junit.Test

class ImagePreviewProgressTest {
    @Test fun parsesCurrentPythonPreviewFormat() {
        assertEquals(ImagePreviewStep(2, 8), parseImagePreviewStep("  [PREVIEW step 2/8] LATENT_RGB 14ms", 8))
    }

    @Test fun keepsLegacyPreviewFormatCompatible() {
        assertEquals(ImagePreviewStep(6, 8), parseImagePreviewStep("[PREVIEW 6/8] CPU 22ms", 8))
    }

    @Test fun rejectsInvalidOrUnrelatedLines() {
        assertNull(parseImagePreviewStep("[UNet 2/8] 999ms", 8))
        assertNull(parseImagePreviewStep("[PREVIEW step 9/8] LATENT_RGB", 8))
    }
}
