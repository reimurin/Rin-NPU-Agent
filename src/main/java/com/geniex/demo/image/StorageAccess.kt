package com.geniex.demo.image
import android.os.Environment
internal object StorageAccess {
    @Volatile var lastError:String?=null
        private set
    fun granted(query:()->Boolean={Environment.isExternalStorageManager()}):Boolean = try {
        query().also { lastError=null }
    } catch(e:Exception) {
        lastError="${e.javaClass.simpleName}: ${e.message}";false
    }
}
