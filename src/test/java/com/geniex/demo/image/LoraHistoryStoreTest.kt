package com.geniex.demo.image

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LoraHistoryStoreTest {
    private val shaA="a".repeat(64)
    private val shaB="b".repeat(64)

    @Test fun inactiveDiagnosticsDoNotCreateHistory() {
        val root=JSONObject().put("active",false).put("files",JSONArray().put(JSONObject().put("name","a").put("sha256",shaA)))
        assertTrue(LoraHistoryStore.parseEvidence(root.toString()).isEmpty())
    }

    @Test fun activeDiagnosticsKeepAllValidAdapters() {
        val files=JSONArray()
            .put(JSONObject().put("name","alpha").put("sha256",shaA).put("weight",0.6))
            .put(JSONObject().put("name","beta").put("sha256",shaB).put("weight",1.1))
        val result=LoraHistoryStore.parseEvidence(JSONObject().put("active",true).put("files",files).toString())
        assertEquals(listOf("alpha","beta"),result.map{it.getString("name")})
        assertEquals(listOf(0.6,1.1),result.map{it.getDouble("weight")})
    }

    @Test fun invalidShaEntryIsIgnored() {
        val files=JSONArray()
            .put(JSONObject().put("name","bad").put("sha256","123"))
            .put(JSONObject().put("name","good").put("sha256",shaA))
        val result=LoraHistoryStore.parseEvidence(JSONObject().put("active",true).put("files",files).toString())
        assertEquals(listOf("good"),result.map{it.getString("name")})
    }
}
