package com.grounded.engine

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Un-Killed Android Foreground Service.
 *
 * Lifecycle:
 *   onCreate()  → init Rust engine, acquire WakeLock, register Sensor listeners
 *   onStart()   → START_STICKY, start cognitive daemon thread
 *   onDestroy() → stop daemon, persist graph, release WakeLock
 *
 * The service runs with a persistent low-importance notification and a
 * PARTIAL_WAKE_LOCK to stay alive when the device sleeps.
 *
 * Rust engine runs on a native OS thread — NOT on the main thread and
 * NOT in the Kotlin coroutine scope. Communication is via the lock-free
 * UniFFI bridge functions.
 */
class CognitiveForegroundService : Service(), SensorEventListener {

    companion object {
        const val CHANNEL_ID = "semantic_engine_channel"
        const val NOTIFICATION_ID = 1
        const val TAG = "SemanticEngine"

        fun start(context: Context) {
            val intent = Intent(context, CognitiveForegroundService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, CognitiveForegroundService::class.java))
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var sensorManager: SensorManager? = null
    private var isRunning = AtomicBoolean(false)
    private var keepaliveTimer: java.util.Timer? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "onCreate — initializing Rust engine")

        // ── 1. Init Rust engine with data directory ──
        val dataDir = File(filesDir, "semantic_engine").also { it.mkdirs() }
        SemanticEngine.init(dataDir.absolutePath)

        // ── 2. Acquire partial wake lock for sustained background execution ──
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$TAG:DaemonLock"
        )
        wakeLock?.acquire(4 * 60 * 60 * 1000L) // 4 hour timeout — renewable

        // ── 3. Register sensor listeners ──
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        registerSensors()

        // ── 4. Create notification channel ──
        createNotificationChannel()

        // ── 5. Start foreground with persistent notification ──
        val notification = buildNotification()
        startForeground(NOTIFICATION_ID, notification)

        // ── 6. Start keepalive watchdog (5s interval) ──
        startKeepaliveWatchdog()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand — starting cognitive daemon")
        SemanticEngine.start()
        isRunning.set(true)
        return START_STICKY  // restart if killed by OS
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy — shutting down")
        isRunning.set(false)

        // ── 0. Cancel keepalive watchdog ──
        keepaliveTimer?.cancel()
        keepaliveTimer = null

        // ── 1. Stop Rust engine (persists graph to disk) ──
        SemanticEngine.stop()

        // ── 2. Unregister sensors ──
        sensorManager?.unregisterListener(this)

        // ── 3. Release wake lock ──
        wakeLock?.let {
            if (it.isHeld) it.release()
        }

        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ── Sensor ingestion ──

    private fun registerSensors() {
        sensorManager?.let { sm ->
            sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
                Log.i(TAG, "Registered accelerometer")
            }
            sm.getDefaultSensor(Sensor.TYPE_PROXIMITY)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
                Log.i(TAG, "Registered proximity")
            }
            sm.getDefaultSensor(Sensor.TYPE_LIGHT)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
                Log.i(TAG, "Registered light sensor")
            }
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isRunning.get()) return

        val sensorName = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> "accelerometer"
            Sensor.TYPE_PROXIMITY -> "proximity"
            Sensor.TYPE_LIGHT -> "light"
            else -> return
        }

        // Push each axis value as a separate channel
        for (i in event.values.indices) {
            SemanticEngine.feedSensor(sensorName, i.toByte(), event.values[i])
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op for cognitive purposes
    }

    // ── Notification ──

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Semantic Engine",
                NotificationManager.IMPORTANCE_LOW  // low = no sound, minimal visual
            ).apply {
                description = "Background cognitive engine for grounded semantic processing"
                setShowBadge(false)
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        val stopIntent = PendingIntent.getService(
            this,
            0,
            Intent(this, CognitiveForegroundService::class.java).apply {
                action = "STOP"
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            Notification.Builder(this)
        }

        return builder
            .setContentTitle("Semantic Engine")
            .setContentText("Cognitive processing active")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(Notification.PRIORITY_LOW)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopIntent)
            .build()
    }

    // ── Keepalive watchdog ──

    private fun startKeepaliveWatchdog() {
        keepaliveTimer = java.util.Timer("KeepaliveWatchdog", false).apply {
            schedule(object : java.util.TimerTask() {
                override fun run() {
                    if (!isRunning.get()) return

                    val alive = SemanticEngine.keepalive()
                    if (!alive) {
                        val missed = SemanticEngine.missedHeartbeats()
                        Log.w(TAG, "Engine heartbeat MISSED ($missed missed) — restarting")
                        SemanticEngine.stop()
                        SemanticEngine.start()
                        // Renew wakelock after restart
                        renewWakeLock()
                    } else {
                        // Renew wakelock on every heartbeat to maintain 4h window
                        renewWakeLock()
                    }

                    // Drain any pending output actions
                    drainAndDispatch()
                }
            }, 5_000L, 5_000L) // initial delay 5s, period 5s
        }
        Log.i(TAG, "Keepalive watchdog started (5s interval)")
    }

    private fun renewWakeLock() {
        wakeLock?.let {
            if (it.isHeld) it.release()
            try {
                it.acquire(4 * 60 * 60 * 1000L)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to renew wakelock", e)
            }
        }
    }

    /**
     * Drain output actions from the Rust engine and dispatch them as intents.
     * Called periodically from a timer or on UI tick.
     */
    fun drainAndDispatch() {
        val outputs = SemanticEngine.drainOutputs()
        for (json in outputs) {
            dispatchAction(json)
        }
    }

    private fun dispatchAction(json: String) {
        try {
            val parsed = org.json.JSONObject(json)
            val action = parsed.optString("action", "")

            when (action) {
                "lockScreen" -> {
                    val intent = Intent(Intent.ACTION_LOCK_SCREEN).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    // Requires device admin for ACTION_LOCK_SCREEN on modern Android
                    Log.i(TAG, "Action: lockScreen")
                }
                "toggleFlashlight" -> {
                    // Typically requires camera permission + flashlight toggle
                    Log.i(TAG, "Action: toggleFlashlight")
                }
                "system_action" -> {
                    val template = parsed.optString("template", "{}")
                    val nested = org.json.JSONObject(template)
                    val nestedAction = nested.optString("action", "")
                    Log.i(TAG, "Action from template: $nestedAction")
                }
                else -> {
                    Log.i(TAG, "Unknown action: $action — json: $json")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse action JSON: $json", e)
        }
    }
}
