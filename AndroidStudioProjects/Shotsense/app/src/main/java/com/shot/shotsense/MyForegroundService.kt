package com.shot.shotsense

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import ch.hsr.geohash.GeoHash
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.net.HttpURLConnection
import java.net.URL
import java.util.Timer
import kotlin.concurrent.scheduleAtFixedRate
import kotlin.math.absoluteValue

class MyForegroundService : Service() {
    private val CHANNEL_ID_FOREGROUND = "ForegroundServiceChannel"
    private val NOTIFICATION_ID_FOREGROUND = 1
    private val CHANNEL_ID_GUNSHOT = "GunshotNotificationChannel"
    private val NOTIFICATION_ID_GUNSHOT = 2
    private var modelPath = "gunshotmodel.tflite"
    private var probabilityThreshold: Float = 0.9f
    private val volumeThreshold: Double = 20.0
    private lateinit var locationManager: LocationManager
    private lateinit var locationListener: LocationListener
    private val flaskServerUrl = "http://www.pragmaticcreations.com/add_geohash"
    private val secondFlaskServerBaseUrl = "http://www.pragmaticcreations.com/"
    private lateinit var  notificationManager: NotificationManagerCompat
    private lateinit var notification: Notification

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate() {
        super.onCreate()
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannels()

        val notification: Notification = createForegroundNotification()

        startForeground(NOTIFICATION_ID_FOREGROUND, notification)

        startForeground(1, notification)
        showGunshotNotification()
        val classifier = AudioClassifier.createFromFile(this, modelPath)

        val tensor = classifier.createInputTensorAudio()

        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"

        val record = classifier.createAudioRecord()
        record.startRecording()

        locationListener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                val latitude = location.latitude
                val longitude = location.longitude
                val geohashValue = GeoHash.withCharacterPrecision(latitude, longitude, 9).toBase32()

                // Perform your operations with the geohash value here
            }

            override fun onProviderEnabled(provider: String) {}

            override fun onProviderDisabled(provider: String) {}

            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        }



        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager



        startLocationUpdates()

        val geohashValue = getGeohashValue()
        val secondFlaskServerUrl = "$secondFlaskServerBaseUrl$geohashValue"

        CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                val flag = checkFlagFromServer(secondFlaskServerUrl)
                if (flag) {
                    showGunshotNotification()
                }
                delay(5000)
            }
        }

        Timer().scheduleAtFixedRate(1, 1000) {
            val numberOfSamples = tensor.load(record)
            val output = classifier.classify(tensor)

            val volumeLevel =
                30 * Math.log10(numberOfSamples.absoluteValue.toDouble() / Short.MAX_VALUE.toDouble())

            var filteredModelOutput = output[1].categories.filter {
                it.label == "Gunshot" && it.score > probabilityThreshold
            }

            if (filteredModelOutput.isNotEmpty()) {
                if (checkPermissions()) {
                    val location = getLastKnownLocation()
                    location?.let { sendLocationToServer(location) }
                }
            }
        }

        return START_STICKY
    }
    @RequiresApi(Build.VERSION_CODES.O)
    private fun createForegroundNotification(): Notification {
        createNotificationChannels()

        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            notificationIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID_FOREGROUND)
            .setContentTitle("Shot Sense")
            .setContentText("Sensing for gunshots")
            .setSmallIcon(R.mipmap.ic_gunshot)
            .setContentIntent(pendingIntent)
            .build()
    }
    private fun checkPermissions(): Boolean {
        return ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED
    }

    private fun checkFlagFromServer(secondFlaskServerUrl: String): Boolean {
        val url = URL(secondFlaskServerUrl)
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"

        return try {
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val inputStream = connection.inputStream.bufferedReader()
                val response = inputStream.readText()
                inputStream.close()
                response.toBoolean()
            } else {
                false
            }
        } catch (e: Exception) {
            false
        } finally {
            connection.disconnect()
        }
    }

    private fun startLocationUpdates() {
        if (checkPermissions()) {
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return
            }
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                0L,
                0f,
                locationListener
            )
        }
    }

    private fun getLastKnownLocation(): Location? {
        if (checkPermissions()) {
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return null
            }
            return locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        }
        return null
    }

    private fun sendLocationToServer(location: Location) {
        val latitude = location.latitude
        val longitude = location.longitude
        val geohashValue = GeoHash.withCharacterPrecision(latitude, longitude, 9).toBase32()

        val url = URL("$flaskServerUrl?geohash=$geohashValue")
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"

        // Send the request to the server
        connection.connect()

        connection.disconnect()
    }

    private fun getGeohashValue(): String {
        if (checkPermissions()) {
            val location = getLastKnownLocation()
            location?.let {
                val latitude = location.latitude
                val longitude = location.longitude
                return GeoHash.withCharacterPrecision(latitude, longitude, 7).toBase32()
            }
        }
        return ""
    }

    private fun showGunshotNotification() {
        val notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID_GUNSHOT)
            .setSmallIcon(R.drawable.ic_gunshot)
            .setContentTitle("Gunshot Sensed")
            .setContentText("A gunshot was sensed near your location.")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)

        val notificationManager = NotificationManagerCompat.from(this)
        notificationManager.notify(NOTIFICATION_ID_GUNSHOT, notificationBuilder.build())
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createNotificationChannels() {
        val nameForeground = "Foreground Service"
        val descriptionForeground = "Foreground service channel"
        val importanceForeground = NotificationManager.IMPORTANCE_DEFAULT
        val channelForeground = NotificationChannel(CHANNEL_ID_FOREGROUND, nameForeground, importanceForeground).apply {
            description = descriptionForeground
        }

        val nameGunshot = "Gunshot Notifications"
        val descriptionGunshot = "Notification channel for gunshot events"
        val importanceGunshot = NotificationManager.IMPORTANCE_DEFAULT
        val channelGunshot = NotificationChannel(CHANNEL_ID_GUNSHOT, nameGunshot, importanceGunshot).apply {
            description = descriptionGunshot
        }

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channelForeground)
        notificationManager.createNotificationChannel(channelGunshot)
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }
}
