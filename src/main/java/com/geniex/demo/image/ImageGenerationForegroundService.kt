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
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.geniex.demo.MainActivity
import com.geniex.demo.R

class ImageGenerationForegroundService : Service() {
    private var wakeLock: PowerManager.WakeLock? = null
    private var attached = false
    private var foregroundStarted = false
    private var powerManager: PowerManager? = null
    private var thermalListenerRegistered = false
    private val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
        applyThermalPerformance(status)
    }
    private val listener: (ImageGenerationEvent) -> Unit = {
        if (foregroundStarted) {
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(it))
        }
    }

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        registerThermalListener()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            ImageGenerationSession.stop(this)
            return START_NOT_STICKY
        }
        acquireWakeLock()
        startForeground(NOTIFICATION_ID, buildNotification(ImageGenerationSession.snapshot().progress))
        foregroundStarted = true
        if (!attached) {
            attached = true
            ImageGenerationSession.attach(listener, replay = true)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            applyThermalPerformance(powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE)
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        unregisterThermalListener()
        if (attached) ImageGenerationSession.detach(listener)
        attached = false
        foregroundStarted = false
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun registerThermalListener() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || thermalListenerRegistered) return
        runCatching {
            powerManager?.addThermalStatusListener(thermalListener)
            thermalListenerRegistered = true
        }.onFailure { Log.w(TAG, "thermal listener unavailable", it) }
    }

    private fun unregisterThermalListener() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || !thermalListenerRegistered) return
        runCatching { powerManager?.removeThermalStatusListener(thermalListener) }
        thermalListenerRegistered = false
    }

    private fun applyThermalPerformance(status: Int) {
        if (!foregroundStarted || !ImageGenerationSession.isRunning()) return
        val mode = if (status >= PowerManager.THERMAL_STATUS_SEVERE) {
            QnnInProcessBridgeServer.HTP_MODE_BALANCED
        } else {
            QnnInProcessBridgeServer.HTP_MODE_BOOST
        }
        runCatching { QnnInProcessNative.setHtpPerformanceMode(mode) }
            .onSuccess { Log.i(TAG, "HTP thermal mode=$mode status=$status raw=$it") }
            .onFailure { Log.w(TAG, "HTP thermal vote unavailable mode=$mode status=$status", it) }
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = (powerManager ?: getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "rin:image_generation")
            .apply {
                setReferenceCounted(false)
                acquire(MAX_WAKE_MS)
            }
    }

    private fun releaseWakeLock() {
        runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
        wakeLock = null
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.image_generation_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = getString(R.string.image_generation_notification_channel_desc)
                setShowBadge(false)
            },
        )
    }

    private fun buildNotification(event: ImageGenerationEvent?): Notification {
        val progress = event as? ImageGenerationEvent.Progress ?: ImageGenerationSession.snapshot().progress
        val percent = progress?.percent?.coerceIn(0, 100) ?: 0
        val stage = when (progress?.stage) {
            ImageGenerationEvent.Stage.PREPARING -> "准备"
            ImageGenerationEvent.Stage.CLIP -> "CLIP"
            ImageGenerationEvent.Stage.DENOISING -> if (progress.totalSteps > 0) "去噪 ${progress.step}/${progress.totalSteps}" else "去噪"
            ImageGenerationEvent.Stage.VAE -> "VAE"
            ImageGenerationEvent.Stage.SAVING -> "保存"
            ImageGenerationEvent.Stage.COMPLETE -> "完成"
            null -> "准备"
        }
        val openPending = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopPending = PendingIntent.getService(
            this,
            1,
            Intent(this, ImageGenerationForegroundService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentTitle(getString(R.string.image_generation_notification_title))
            .setContentText(getString(R.string.image_generation_notification_progress, percent, stage))
            .setContentIntent(openPending)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, percent, false)
            .addAction(0, getString(R.string.image_generation_notification_stop), stopPending)
            .build()
    }

    companion object {
        private const val TAG = "RinImageGenService"
        private const val CHANNEL_ID = "rin_image_generation"
        private const val NOTIFICATION_ID = 15108
        private const val ACTION_START = "com.geniex.demo.image.IMAGE_GENERATION_START"
        private const val ACTION_STOP = "com.geniex.demo.image.IMAGE_GENERATION_STOP"
        private const val MAX_WAKE_MS = 30L * 60L * 1000L

        fun start(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ImageGenerationForegroundService::class.java).setAction(ACTION_START),
            )
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, ImageGenerationForegroundService::class.java))
        }
    }
}
