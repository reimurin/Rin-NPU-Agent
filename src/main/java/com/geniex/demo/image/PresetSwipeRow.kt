package com.geniex.demo.image
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.MotionEvent
import android.view.ViewConfiguration
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.widget.AppCompatButton
import androidx.core.content.ContextCompat
import com.geniex.demo.R
import kotlin.math.abs
import kotlin.math.min

internal class PresetSwipeRow(context: Context) : FrameLayout(context) {
    private val actions=LinearLayout(context).apply { orientation=LinearLayout.HORIZONTAL;gravity=Gravity.CENTER_VERTICAL;setPadding(dp(4),dp(8),dp(4),dp(8)) }
    private val front=LinearLayout(context).apply { orientation=LinearLayout.HORIZONTAL;gravity=Gravity.CENTER_VERTICAL }
    private val texts=LinearLayout(context).apply { orientation=LinearLayout.VERTICAL }
    private val title=TextView(context).apply { textSize=16f;maxLines=2 }
    private val note=TextView(context).apply { textSize=13f;maxLines=2 }
    private val more=RinControls.button(context,RinControls.Tone.GHOST).apply { text="⋮";textSize=22f;minWidth=0;minimumWidth=0;contentDescription="展开预设操作" }
    private var actionWidth=dp(216).toFloat()
    private var startX=0f;private var startY=0f;private var origin=0f
    private var dragging=false;private var hitsActions=false
    private val slop=ViewConfiguration.get(context).scaledTouchSlop
    private var opened=false
    var onOpened:((PresetSwipeRow)->Unit)?=null
    val isOpen:Boolean get()=opened
    init {
        minimumHeight=dp(100);clipChildren=true
        front.background=GradientDrawable().apply { cornerRadius=dp(20).toFloat();setColor(ContextCompat.getColor(context,R.color.rin_surface));setStroke(dp(1),ContextCompat.getColor(context,R.color.rin_outline)) }
        front.setPadding(dp(14),dp(12),dp(6),dp(12))
        title.setTextColor(ContextCompat.getColor(context,R.color.rin_text_primary));note.setTextColor(ContextCompat.getColor(context,R.color.rin_text_secondary))
        texts.addView(title);texts.addView(note)
        front.addView(texts,LinearLayout.LayoutParams(0,-2,1f));front.addView(more,LinearLayout.LayoutParams(dp(44),dp(48)))
        addView(actions,LayoutParams(actionWidth.toInt(),-1,Gravity.RIGHT));addView(front,LayoutParams(-1,-1))
        more.setOnClickListener { revealActions(!isOpen) };front.setOnLongClickListener { revealActions(true);true }
    }
    fun bind(item:PresetListItem,select:()->Unit,pin:()->Unit,edit:()->Unit,delete:()->Unit) {
        revealActions(false,false);tag=item.id
        title.text=(if(item.pinned)"置顶 · " else "")+(if(item.isDefault)"默认 · " else "")+item.name
        note.text=item.note.ifBlank { item.prompt.replace('\n',' ') }
        front.setOnClickListener { if(isOpen)revealActions(false) else select() }
        actions.removeAllViews()
        listOf(Triple(if(item.pinned)"取消置顶" else "置顶","pin",pin),Triple("编辑","edit",edit),Triple("删除","delete",delete)).forEach { (label,id,handler)->
            val button=RinControls.button(context,if(id=="delete") RinControls.Tone.DANGER else RinControls.Tone.SECONDARY).apply {
                text=label;textSize=13f;isAllCaps=false;minWidth=0;minimumWidth=0;setPadding(dp(2),dp(4),dp(2),dp(4))
                tag="action_$id";contentDescription="$label：${item.name}"
                setTextColor(ContextCompat.getColor(context,if(id=="delete")R.color.rin_danger else R.color.rin_text_primary))
                setOnClickListener { if(isOpen)handler() }
            }
            actions.addView(button,LinearLayout.LayoutParams(0,-2,1f).apply { marginStart=dp(3);marginEnd=dp(3) })
        }
    }
    override fun onSizeChanged(w:Int,h:Int,oldw:Int,oldh:Int) {
        super.onSizeChanged(w,h,oldw,oldh);actionWidth=min(dp(216).toFloat(),w*0.82f)
        actions.layoutParams=LayoutParams(actionWidth.toInt(),-1,Gravity.RIGHT)
        front.translationX=if(opened)-actionWidth else 0f
    }
    fun revealActions(open:Boolean,animate:Boolean=true) {
        opened=open;front.animate().cancel();if(open)onOpened?.invoke(this)
        val target=if(open)-actionWidth else 0f
        if(animate)front.animate().translationX(target).setDuration(150).start() else front.translationX=target
        actions.importantForAccessibility=if(open)IMPORTANT_FOR_ACCESSIBILITY_AUTO else IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
    }
    override fun onInterceptTouchEvent(e:MotionEvent):Boolean {
        when(e.actionMasked) {
            MotionEvent.ACTION_DOWN->{startX=e.x;startY=e.y;origin=front.translationX;dragging=false;hitsActions=opened&&e.x>=width+front.translationX}
            MotionEvent.ACTION_MOVE->{val dx=e.x-startX;val dy=e.y-startY
                if(!hitsActions&&abs(dx)>slop&&abs(dx)>abs(dy)*1.3f&&(dx<0||origin<0)) {
                    dragging=true;parent?.requestDisallowInterceptTouchEvent(true);return true
                }
            }
        }
        return super.onInterceptTouchEvent(e)
    }
    override fun onTouchEvent(e:MotionEvent):Boolean {
        if(!dragging)return super.onTouchEvent(e)
        when(e.actionMasked) {
            MotionEvent.ACTION_MOVE->front.translationX=(origin+e.x-startX).coerceIn(-actionWidth,0f)
            MotionEvent.ACTION_UP->{revealActions(front.translationX < -actionWidth*0.35f);dragging=false;parent?.requestDisallowInterceptTouchEvent(false)}
            MotionEvent.ACTION_CANCEL->{revealActions(origin < -actionWidth*0.5f);dragging=false;parent?.requestDisallowInterceptTouchEvent(false)}
        }
        return true
    }
    private fun dp(v:Int)=(v*resources.displayMetrics.density).toInt()
}
