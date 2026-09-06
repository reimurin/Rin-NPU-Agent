package com.geniex.demo.image
import org.junit.Test
import org.junit.Assert.*
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import java.io.File
class ResolutionCatalogTest {
    @get:Rule val temp=TemporaryFolder()
    private fun complete(dir:File){dir.mkdirs();ResolutionCatalog.contextNames.forEach{File(dir,it).writeBytes(byteArrayOf(1))}}
    @Test fun legacyIsOnly1024(){val r=temp.newFolder();complete(File(r,"context"));assertEquals(listOf(ImageResolution(1024,1024)),ResolutionCatalog.discover(r))}
    @Test fun scopedDimensionsDiscovered(){val r=temp.newFolder();complete(File(r,"context/832x1216"));assertEquals(listOf(ImageResolution(832,1216)),ResolutionCatalog.discover(r))}
    @Test fun incompleteBucketNotSupported(){val r=temp.newFolder();val d=File(r,"context/832x1216");complete(d);File(d,ResolutionCatalog.contextNames.first()).writeBytes(byteArrayOf());assertTrue(ResolutionCatalog.discover(r).isEmpty())}
    @Test fun duplicate1024Deduped(){val r=temp.newFolder();complete(File(r,"context"));complete(File(r,"context/1024x1024"));assertEquals(1,ResolutionCatalog.discover(r).size)}
    @Test fun malformedAndOversizedDimensionsRejected(){for(s in listOf("0x1024","1025x1024","999999999999999x1024","-1x1024","8193x8193","abc"))assertNull(ResolutionCatalog.parse(s))}
    @Test fun missingModelFolderIsEmpty(){assertTrue(ResolutionCatalog.discover(temp.newFolder()).isEmpty())}
    @Test fun noImplicitRectangleFromSquare(){val r=temp.newFolder();complete(File(r,"context"));assertFalse(ImageResolution(1216,832) in ResolutionCatalog.discover(r))}
}
