package com.geniex.demo.image
import android.app.Application
import org.junit.Test
import org.junit.Assert.*
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34],application=Application::class)
class PresetStoreTest {
    @get:Rule val temp=TemporaryFolder()
    private var clock=100L
    private fun store(root:File)=PromptPresetStore(root){++clock}
    @Test fun legacyJsonSurvivesManagement() {
        val root=temp.newFolder();val file=File(root,"positive.json")
        val original="""{"version":1,"items":[{"id":"old","name":"旧预设","prompt":"cat","note":"备注","tags":["x"],"createdAt":1,"updatedAt":2,"lastUsedAt":3}]}"""
        file.writeText(original);val s=store(root);assertEquals(0,s.listPositive().single().pinnedAt)
        s.setPositivePinned("old",true);s.savePositive("old","修改后","新备注","dog")
        val saved=store(root).listPositive().single();assertEquals("old",saved.id);assertEquals("dog",saved.prompt);assertEquals(1,saved.createdAt);assertEquals(listOf("x"),saved.tags);assertTrue(saved.pinnedAt>0)
        assertEquals(original,File(root,"positive.json.before_1_6.bak").readText())
    }
    @Test fun pinOverridesRecentUseAndPersists() {
        val root=temp.newFolder();val s=store(root);val a=s.savePositive(name="a",note="",prompt="a");val b=s.savePositive(name="b",note="",prompt="b")
        s.setPositivePinned(a.id,true);s.touchPositive(b.id);assertEquals(a.id,store(root).listPositive().first().id)
        s.setPositivePinned(a.id,false);assertEquals(b.id,s.listPositive().first().id)
    }
    @Test fun pinOrderStableAfterEditing() {
        val s=store(temp.newFolder());val a=s.savePositive(name="a",note="",prompt="a");val b=s.savePositive(name="b",note="",prompt="b")
        s.setPositivePinned(a.id,true);s.setPositivePinned(b.id,true);s.savePositive(a.id,"edited","","x")
        assertEquals(listOf(b.id,a.id),s.listPositive().map{it.id})
    }
    @Test fun positiveDeletionPersists() {
        val root=temp.newFolder();val s=store(root);val a=s.savePositive(name="a",note="",prompt="a")
        assertTrue(s.deletePositive(a.id));assertFalse(s.deletePositive(a.id));assertTrue(store(root).listPositive().isEmpty())
    }
    @Test fun negativeDefaultsAndPinsAreIndependent() {
        val s=store(temp.newFolder());val a=s.saveNegative(name="a",note="",prompt="a",makeDefault=true);val b=s.saveNegative(name="b",note="",prompt="b")
        s.setNegativePinned(b.id,true);assertEquals(b.id,s.listNegative().first().id);assertEquals(a.id,s.defaultNegative()?.id)
        s.saveNegative(a.id,"a2","","a2",false);assertNull(s.defaultNegative());assertEquals(b.id,s.listNegative().first().id)
    }
    @Test fun deletingDefaultDoesNotPromoteAnother() {
        val s=store(temp.newFolder());val a=s.saveNegative(name="a",note="",prompt="a",makeDefault=true);s.saveNegative(name="b",note="",prompt="b")
        s.deleteNegative(a.id);assertNull(s.defaultNegative());assertEquals(1,s.listNegative().size)
    }
    @Test fun oneDefaultOnly() {
        val s=store(temp.newFolder());s.saveNegative(name="a",note="",prompt="a",makeDefault=true);val b=s.saveNegative(name="b",note="",prompt="b",makeDefault=true)
        assertEquals(1,s.listNegative().count{it.isDefault});assertEquals(b.id,s.defaultNegative()?.id)
    }
    @Test fun corruptedFileNeverOverwritten() {
        val root=temp.newFolder();val file=File(root,"positive.json");file.writeText("{broken")
        val s=store(root)
        try{s.savePositive(name="x",note="",prompt="x");fail()}catch(_:IllegalStateException){}
        assertEquals("{broken",file.readText())
    }
    @Test fun deletedIdCannotBeResurrectedByEditor() {
        val s=store(temp.newFolder());val a=s.savePositive(name="a",note="",prompt="a");s.deletePositive(a.id)
        try{s.savePositive(a.id,"x","","x");fail()}catch(_:IllegalArgumentException){}
        assertTrue(s.listPositive().isEmpty())
    }
    @Test fun sameNameDifferentIdsCanBeManagedSeparately() {
        val s=store(temp.newFolder());val a=s.savePositive(name="same",note="",prompt="a");val b=s.savePositive(name="same",note="",prompt="b")
        s.deletePositive(a.id);assertEquals(b.id,s.listPositive().single().id)
    }
    @Test fun invalidEditPreservesPreviousValue() {
        val s=store(temp.newFolder());val a=s.savePositive(name="a",note="",prompt="a")
        try{s.savePositive(a.id,"","","b");fail()}catch(_:IllegalArgumentException){}
        assertEquals(a,s.listPositive().single())
    }
    @Test fun twoStoreInstancesDoNotLoseUpdates() {
        val root=temp.newFolder();val a=store(root);val b=store(root)
        val first=a.savePositive(name="a",note="",prompt="a");b.savePositive(name="b",note="",prompt="b");a.setPositivePinned(first.id,true)
        assertEquals(2,b.listPositive().size)
    }
}
