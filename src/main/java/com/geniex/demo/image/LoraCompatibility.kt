package com.geniex.demo.image
import android.content.Context
import org.json.JSONObject
internal class LoraCompatibility private constructor(val capacity:Int,private val aliases:Map<String,Triple<String,Int,Int>>) {
    fun check(prefix:String,down:List<Long>,up:List<Long>):String {
        val item=aliases[prefix] ?: error("当前组件未覆盖该层，不会忽略：$prefix")
        require(down[0]<=capacity) { "LoRA 秩超过组件上限 $capacity" }
        require(down[1]==item.second.toLong()&&up[0]==item.third.toLong()) { "LoRA 与 WAI 层形状不匹配：$prefix" }
        return item.first
    }
    companion object {
        fun load(context:Context):LoraCompatibility {
            val data=context.assets.open("lora_compatibility.json").bufferedReader().use{JSONObject(it.readText())}
            require(data.getInt("schema")==1&&data.getString("model_id")==LoraModelComponent.ID)
            val map=linkedMapOf<String,Triple<String,Int,Int>>();val layers=data.getJSONArray("layers")
            for(i in 0 until layers.length()) {
                val layer=layers.getJSONObject(i);val item=Triple(layer.getString("module"),layer.getInt("in_features"),layer.getInt("out_features"));val aliases=layer.getJSONArray("aliases")
                for(j in 0 until aliases.length()){val key=aliases.getString(j);require(map[key]==null||map[key]==item);map[key]=item}
            }
            return LoraCompatibility(data.getInt("rank_capacity"),map)
        }
    }
}
