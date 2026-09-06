package com.geniex.demo.image

import android.content.Context
import android.os.Build
import android.os.StatFs
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

internal object LoraModelComponent {
    const val ID="wai-v170-sm8750-lora-r64-1024-partitioned-v1"
    const val RELATIVE="context/lora/$ID"
    const val INDEX_URL="https://github.com/reimurin/Rin-NPU-Agent/releases/download/wai-lora-sm8750-r64-partitioned-v1/component.json"
    val busy=AtomicBoolean(false)
    private val cancelled=AtomicBoolean(false)
    @Volatile var status="LoRA 生图组件尚未检查"
    @Volatile private var downloader:RuntimePartDownloader?=null
    private val checksum=Regex("[0-9a-f]{64}")
    private val names=((0..2).map{"encoder_p$it.bin"}+(0..3).map{"decoder_p$it.bin"}+"lora_template.json").toSet()
    internal data class Part(val name:String,val url:String,val bytes:Long,val sha:String)
    internal data class Asset(val name:String,val bytes:Long,val sha:String,val parts:List<Part>)
    internal fun parseIndex(data:JSONObject):List<Asset> {
        require(data.getInt("schema")==1&&data.getString("model_id")==ID) { "LoRA 组件版本不匹配" }
        require(data.getInt("soc")==69&&data.getInt("dsp")==79) { "LoRA 组件芯片不匹配" }
        val array=data.getJSONArray("files");require(array.length()==names.size)
        val used=mutableSetOf<String>();val partNames=mutableSetOf<String>()
        return (0 until array.length()).map { i ->
            val item=array.getJSONObject(i);val name=item.getString("name");val bytes=item.getLong("bytes");val sha=item.getString("sha256")
            require(name in names&&used.add(name)&&bytes in 1..4L*1024*1024*1024&&checksum.matches(sha)) { "LoRA 文件清单无效" }
            if(name=="lora_template.json")require(bytes<=16L*1024*1024)
            val parts=item.getJSONArray("parts");require(parts.length() in 1..16)
            val parsed=(0 until parts.length()).map { j->
                val p=parts.getJSONObject(j);val pn=p.getString("name");val url=p.getString("url");val length=p.getLong("bytes");val digest=p.getString("sha256")
                require(Regex("[A-Za-z0-9._-]+").matches(pn)&&pn!="."&&pn!=".."&&partNames.add(pn)) { "LoRA 分卷名称无效" }
                require(url=="https://github.com/reimurin/Rin-NPU-Agent/releases/download/wai-lora-sm8750-r64-partitioned-v1/$pn"&&length in 1..1024L*1024*1024&&checksum.matches(digest)) { "LoRA 下载来源或分卷信息无效" }
                Part(pn,url,length,digest)
            }
            require(parsed.sumOf{it.bytes}==bytes) { "LoRA 文件总长度不匹配" }
            Asset(name,bytes,sha,parsed)
        }
    }
    private fun directory(base:File):File {
        val root=base.canonicalFile;val dir=File(root,RELATIVE).canonicalFile
        require(dir.path.startsWith(root.path+File.separator)) { "LoRA 目录超出运行时" }
        return dir
    }
    internal fun contextSpecs(manifest:JSONObject):List<JSONObject> {
        require(manifest.getInt("schema")==1&&manifest.getBoolean("complete")&&manifest.getString("model_id")==ID)
        require(manifest.getInt("soc")==69&&manifest.getInt("dsp")==79&&manifest.getInt("rank_capacity")==64)
        val dimensions=manifest.getJSONArray("resolution");require(dimensions.length()==2&&dimensions.getInt(0)==1024&&dimensions.getInt(1)==1024)
        val graphs=manifest.getJSONObject("graphs");require(graphs.length()==2)
        return listOf("encoder","decoder").flatMap { graph->
            val parts=graphs.getJSONObject(graph).getJSONArray("parts");require(parts.length()==if(graph=="encoder")3 else 4)
            (0 until parts.length()).map { index->
                parts.getJSONObject(index).also { part->
                    require(part.getString("id")=="${graph}_p$index"&&part.getString("context_file")=="${graph}_p$index.bin")
                    require(part.getLong("context_bytes") in 1..4L*1024*1024*1024&&checksum.matches(part.getString("context_sha256")))
                }
            }
        }
    }
    fun installed(base:File):Boolean=runCatching {
        val dir=directory(base);val file=File(dir,"lora_template.json")
        if(!file.isFile||file.length()>16L*1024*1024)return@runCatching false
        contextSpecs(JSONObject(file.readText())).all { part->
            File(dir,part.getString("context_file")).let{it.isFile&&it.length()==part.getLong("context_bytes")}
        }
    }.getOrDefault(false)
    fun cancel(){cancelled.set(true);status="正在暂停下载；已完成分卷将保留。"}
    fun install(context:Context,base:File,onUpdate:(String)->Unit):Boolean {
        if(!busy.compareAndSet(false,true))return false
        cancelled.set(false);val app=context.applicationContext
        thread(name="rin-lora-component-install") {
            var notifiedAt=0L
            fun update(message:String,percent:Int=0) {
                status=message
                val now=android.os.SystemClock.elapsedRealtime()
                if(message.startsWith("LoRA 组件 ")&&percent<100&&now-notifiedAt<750)return
                notifiedAt=now;onUpdate(message)
                runCatching { LoraComponentForegroundService.update(app,percent,message) }
            }
            try {
                require(StorageAccess.granted()) { "请先授予文件访问权限" }
                require(Build.SOC_MODEL.contains("SM8750",true)) { "当前 LoRA 组件仅适配 SM8750" }
                if(installed(base)){update("LoRA 生图组件已安装",100);return@thread}
                LoraComponentForegroundService.start(app,"正在检查 LoRA 生图组件")
                update("正在获取 LoRA 组件清单…")
                val assets=parseIndex(fetchIndex());val target=directory(base)
                require(!target.exists()) { "发现未完整安装的 LoRA 目录，请先检查该组件；原底模未受影响" }
                val parent=target.parentFile!!;check(parent.isDirectory||parent.mkdirs())
                val stage=File(parent,".$ID.installing");check(stage.isDirectory||stage.mkdirs())
                val partsDir=File(base,".rin_lora/downloads/$ID");check(partsDir.isDirectory||partsDir.mkdirs())
                val total=assets.sumOf{it.bytes};val current=stage.listFiles().orEmpty().filter{it.isFile}.sumOf{it.length()}
                val required=(total-current).coerceAtLeast(0)+(assets.maxOfOrNull{it.bytes} ?: 0)+1024L*1024*1024
                require(StatFs(base.absolutePath).availableBytes>=required) { "LoRA 组件安装空间不足，请额外预留 ${(required/1024/1024/1024)+1} GiB" }
                val worker=RuntimePartDownloader(cancelled);downloader=worker
                var completed=0L
                for(asset in assets.sortedBy{if(it.name=="lora_template.json")1 else 0}) {
                    ensureActive();val destination=File(stage,asset.name)
                    if(destination.isFile&&destination.length()==asset.bytes&&sha(destination)==asset.sha){completed+=asset.bytes;continue}
                    val pending=File(stage,asset.name+".assembling")
                    FileOutputStream(pending).use { output->
                        for(part in asset.parts) {
                            ensureActive();val file=File(partsDir,part.name+".part")
                            if(!(file.isFile&&file.length()==part.bytes&&!worker.hasSegmentState(file)&&sha(file)==part.sha)) {
                                worker.download(file,part.url,part.bytes,4) { done,_,speed,_,_->
                                    val percent=((completed+done)*100/total).toInt().coerceIn(0,99)
                                    update("LoRA 组件 $percent% · ${speed/1024/1024} MiB/s",percent)
                                }
                            }
                            ensureActive();require(file.length()==part.bytes&&sha(file)==part.sha) { "LoRA 分卷校验失败：${part.name}" }
                            file.inputStream().use{it.copyTo(output,1024*1024)}
                            worker.cleanupState(file)
                            completed+=part.bytes
                        }
                        output.fd.sync()
                    }
                    require(pending.length()==asset.bytes&&sha(pending)==asset.sha) { "LoRA 模型文件校验失败：${asset.name}" }
                    Files.move(pending.toPath(),destination.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
                    for(part in asset.parts)File(partsDir,part.name+".part").takeIf{it.isFile}?.delete()
                }
                ensureActive();val template=JSONObject(File(stage,"lora_template.json").readText())
                require(template.getString("model_id")==ID&&template.getBoolean("complete")) { "LoRA 模板不完整" }
                for(spec in contextSpecs(template)) {
                    val asset=assets.single{it.name==spec.getString("context_file")}
                    require(spec.getLong("context_bytes")==asset.bytes&&spec.getString("context_sha256")==asset.sha)
                }
                Files.move(stage.toPath(),target.toPath(),StandardCopyOption.ATOMIC_MOVE)
                require(installed(base));update("LoRA 生图组件安装完成；可返回生图页测试兼容 LoRA。",100)
            } catch(e:Exception) {
                status=if(cancelled.get())"LoRA 下载已暂停；再次点击安装可继续。" else "LoRA 组件未安装：${e.message}"
                onUpdate(status)
            } finally {
                downloader=null;busy.set(false);runCatching{LoraComponentForegroundService.stop(app)}
            }
        }
        return true
    }
    private fun ensureActive(){if(cancelled.get())throw InterruptedException("LoRA 下载已暂停")}
    private fun fetchIndex():JSONObject {
        var address=URL(INDEX_URL)
        repeat(6) {
            val connection=address.openConnection() as HttpURLConnection
            connection.connectTimeout=15000;connection.readTimeout=30000;connection.instanceFollowRedirects=false
            connection.setRequestProperty("User-Agent","Rin-LoRA-Component");connection.setRequestProperty("Cache-Control","no-cache")
            try {
                val code=connection.responseCode
                if(code in listOf(301,302,303,307,308)) {
                    address=URL(address,connection.getHeaderField("Location") ?: error("缺少重定向地址"));require(address.protocol=="https")
                } else {
                    require(code==200) { if(code==404)"LoRA 模型组件尚未发布，原生图功能仍可使用" else "组件清单 HTTP $code" }
                    val data=connection.inputStream.use{it.readBytesLimited(1024*1024)}
                    return JSONObject(String(data,Charsets.UTF_8))
                }
            } finally {connection.disconnect()}
        }
        error("组件清单重定向过多")
    }
    private fun java.io.InputStream.readBytesLimited(limit:Int):ByteArray {
        val output=java.io.ByteArrayOutputStream();val buffer=ByteArray(8192)
        while(true){val n=read(buffer);if(n<0)break;require(output.size()+n<=limit);output.write(buffer,0,n)}
        return output.toByteArray()
    }
    private fun sha(file:File):String {
        val hash=MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream->val buffer=ByteArray(1024*1024);while(true){val n=stream.read(buffer);if(n<0)break;hash.update(buffer,0,n)} }
        return hash.digest().joinToString(""){"%02x".format(it)}
    }
}
