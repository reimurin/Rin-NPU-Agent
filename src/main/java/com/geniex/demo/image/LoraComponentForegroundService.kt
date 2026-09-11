package com.geniex.demo.image

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.geniex.demo.MainActivity
import com.geniex.demo.R

class LoraComponentForegroundService : Service() {
    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        val percent = intent?.getIntExtra(EXTRA_PERCENT, 0)?.coerceIn(0, 100) ?: 0
        val text = intent?.getStringExtra(EXTRA_TEXT).orEmpty().ifBlank { getString(R.string.image_runtime_install_checking) }
        startForeground(NOTIFICATION_ID, buildNotification(percent, text))
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.runtime_download_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = getString(R.string.runtime_download_notification_channel_desc)
                setShowBadge(false)
            },
        )
    }

    private fun buildNotification(percent: Int, text: String): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("统一 WAI 模型升级")
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(pending)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, percent, false)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "rin_lora_component_download"
        private const val NOTIFICATION_ID = 16620
        private const val ACTION_UPDATE = "com.geniex.demo.image.LORA_COMPONENT_DOWNLOAD_UPDATE"
        private const val ACTION_STOP = "com.geniex.demo.image.LORA_COMPONENT_DOWNLOAD_STOP"
        private const val EXTRA_PERCENT = "percent"
        private const val EXTRA_TEXT = "text"

        fun start(context: Context, text: String = context.getString(R.string.image_runtime_install_checking)) {
            val intent = Intent(context, LoraComponentForegroundService::class.java)
                .setAction(ACTION_UPDATE)
                .putExtra(EXTRA_PERCENT, 0)
                .putExtra(EXTRA_TEXT, text)
            ContextCompat.startForegroundService(context, intent)
        }

        fun update(context: Context, percent: Int, text: String) {
            val intent = Intent(context, LoraComponentForegroundService::class.java)
                .setAction(ACTION_UPDATE)
                .putExtra(EXTRA_PERCENT, percent.coerceIn(0, 100))
                .putExtra(EXTRA_TEXT, text)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, LoraComponentForegroundService::class.java))
        }
    }
}
