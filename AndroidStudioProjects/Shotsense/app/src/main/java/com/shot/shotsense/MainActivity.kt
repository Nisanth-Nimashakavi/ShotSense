package com.shot.shotsense

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Shader
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.SpannableString
import android.text.style.UnderlineSpan
import android.view.View
import android.widget.EditText
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import ch.hsr.geohash.GeoHash
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.*
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.net.HttpURLConnection
import java.net.URL
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate
import kotlin.math.absoluteValue
import java.util.logging.Logger;

class MainActivity : AppCompatActivity() {
    private var TAG = "MainActivity"
    private var modelPath = "gunshotmodel.tflite"
    private var probabilityThreshold: Float = 0.85f
    private val volumeThreshold: Double = 20.0
    private lateinit var locationManager: LocationManager
    private lateinit var locationListener: LocationListener
    private val flaskServerUrl = "http://www.pragmaticcreations.com/add_geohash"
    private val secondFlaskServerBaseUrl = "http://www.pragmaticcreations.com/"
    lateinit var textView: TextView
    lateinit var edittext: TextView
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        textView = findViewById<TextView>(R.id.textView)
        runOnUiThread {
            textView.text = "Gunshot\nnot\nDetected"
        }
        val floatingActionButton = findViewById<FloatingActionButton>(R.id.floatingActionButton)
        edittext = findViewById<TextView>(R.id.textView2)
        val myShader: Shader = LinearGradient(
            0f, 0f, 1200f, 0f,
            Color.rgb(166,101,204), Color.rgb(74,156,156),
            Shader.TileMode.CLAMP
        )
        edittext.getPaint().setShader(myShader)

        locationListener = object : LocationListener {

            override fun onLocationChanged(location: Location) {
                val latitude = location.latitude
                val longitude = location.longitude
                val geohashValue = GeoHash.withCharacterPrecision(latitude, longitude, 9).toBase32()

            }

            override fun onProviderEnabled(provider: String) {}

            override fun onProviderDisabled(provider: String) {}

            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        }

        requestPermissions()

        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager


        val locationPermission = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )



        val classifier = AudioClassifier.createFromFile(this, modelPath)

        val tensor = classifier.createInputTensorAudio()

        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"

        val record = classifier.createAudioRecord()
        record.startRecording()

        val CHANNEL_ID = "gunshot_notification_channel"
        val name = "Gunshot Notifications"
        val descriptionText = "Notification channel for gunshot events"
        val importance = NotificationManager.IMPORTANCE_DEFAULT
        val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
            description = descriptionText
        }
        // Register the channel with the systeme: /home/nimnim/gunshot/odml-pathways/audio_classification/codelab2/android/final/app/src/main/java/com/example/mysoundclassification/MainActivity.kt: (73, 38): Unresolved reference: output
        val notificationManager: NotificationManager =
            getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)




        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(locationPermission, PERMISSION_REQUEST_CODE)
        }
        val serviceIntent = Intent(this, MyForegroundService::class.java)
        startService(serviceIntent)





        val geohashValue = getGeohashValue()
        val secondFlaskServerUrl = "$secondFlaskServerBaseUrl$geohashValue"
        CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                val flag = checkFlagFromServer(secondFlaskServerUrl)
                if (flag) {

                    showNotification()
                    runOnUiThread {
                        textView.text ="Gunshot\nDetected"
                    }

                } else{
                    runOnUiThread {
                        textView.text ="Gunshot\nnot\nDetected"
                    }


                }
                delay(5000)

            }}
        Timer().scheduleAtFixedRate(1, 1000) {
            val numberOfSamples = tensor.load(record)
            val output = classifier.classify(tensor)

            val volumeLevel =
                30 * Math.log10(numberOfSamples.absoluteValue.toDouble() / Short.MAX_VALUE.toDouble())

            var filteredModelOutput = output[1].categories.filter {
                (it.label == "Other") && it.score > probabilityThreshold
            }
            println(filteredModelOutput);
            println(filteredModelOutput);
            println(filteredModelOutput);
            var filteredModelOutput2 = output[1].categories.filter {
                (it.label == "Gunshot" ) && it.score > probabilityThreshold
            }
            println(filteredModelOutput2);
            if (filteredModelOutput.isNotEmpty()) {
                if (checkPermissions()) {
                    val location = getLastKnownLocation()
                    location?.let { sendLocationToServer(location) }
                }
            } else {

            }
        }


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
    fun onFloatingActionButtonClick(view: View?) {
        val websiteUrl = "http://shotsense.000webhostapp.com/"
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(websiteUrl))
        startActivity(intent)
    }
    private fun checkFlagFromServer(secondFlaskServerUrl: String): Boolean {
        val url = URL(secondFlaskServerUrl)
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        println(secondFlaskServerUrl)
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

    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.RECORD_AUDIO
        )
        ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startLocationUpdates()
            } else {
                // Permission denied, handle it accordingly
            }
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
        connection.connect()        // Read the response if needed
        // val responseCode = connection.responseCode
        // val responseMessage = connection.responseMessage

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

    private fun showNotification() {
        val CHANNEL_ID = "gunshot_notification_channel"
        val notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID) // Add channelId here            .setSmallIcon(R.drawable.ic_gunshot)
            .setSmallIcon(R.drawable.ic_gunshot)
            .setContentTitle("Gunshot Sensed")
            .setContentText("A gunshot was sensed near your location.")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)

        val notificationManager = NotificationManagerCompat.from(this)
        notificationManager.notify(0, notificationBuilder.build())
    }

    companion object {
        const val PERMISSION_REQUEST_CODE = 123
    }
}
