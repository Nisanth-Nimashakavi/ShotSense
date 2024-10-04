import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.room.jarjarred.org.stringtemplate.v4.Interpreter
import androidx.room.jarjarred.org.stringtemplate.v4.STGroup
import com.google.firebase.crashlytics.buildtools.reloc.org.apache.commons.cli.Options
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class AudioClassifier(private val assetManager: AssetManager) {
    companion object {
        private const val MODEL_FILENAME = "yamnet.tflite"
        private const val NUM_FRAMES = 96
        private const val NUM_BANDS = 64
        private const val INPUT_SIZE = NUM_FRAMES * NUM_BANDS
        private const val OUTPUT_SIZE = 521
        private const val BUFFER_SIZE = 2 * INPUT_SIZE
    }

    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE)
    private val interpreter: Interpreter

    init {
        // Load the model
        val modelPath = getModelPath()
        val options = Interpreter.Options()
        interpreter = Interpreter(loadModelFile(modelPath), options)

        // Initialize the input buffer
        inputBuffer.order(ByteOrder.nativeOrder())
    }

    private fun getModelPath(): String {
        val fileDescriptor: AssetFileDescriptor = assetManager.openFd(MODEL_FILENAME)
        return fileDescriptor.fileDescriptor.toString()
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer? {
        val inputStream = FileInputStream(modelPath)
        val fileChannel = inputStream.channel
        val startOffset = fileChannel.position()
        val declaredLength = fileChannel.size()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classify(audioData: FloatArray): FloatArray {
        // Prepare the input buffer
        inputBuffer.rewind()
        for (audioSample in audioData) {
            inputBuffer.putFloat(audioSample)
        }

        // Run inference
        val output = Array(1) { FloatArray(OUTPUT_SIZE) }
        interpreter.run(inputBuffer, output)

        return output[0]
    }
}

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val SAMPLE_RATE = 16000
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT
    )

    private lateinit var audioClassifier: AudioClassifier
    private lateinit var audioRecord: AudioRecord
    private lateinit var recordingThread: Thread

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            audioClassifier = AudioClassifier(assets)
        } catch (e: IOException) {
            Log.e(TAG, "Failed to initialize AudioClassifier", e)
        }
    }

    override fun onStart() {
        super.onStart()
        startRecording()
    }

    override fun onStop() {
        super.onStop()
        stopRecording()
    }

    private fun startRecording() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        audioRecord = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(BUFFER_SIZE)
            .build()

        audioRecord.startRecording()

        recordingThread = Thread {
            val audioBuffer = FloatArray(BUFFER_SIZE / 4) // 4 bytes per float

            while (!Thread.currentThread().isInterrupted) {
                val numSamples = audioRecord.read(audioBuffer, 0, audioBuffer.size, AudioRecord.READ_BLOCKING)
                if (numSamples > 0) {
                    val audioData = audioBuffer.copyOfRange(0, numSamples)

                    // Classify audio data
                    val output = audioClassifier.classify(audioData)
                    // Process the output as needed

                    // Example: Print the top class and its score
                    var maxIndex = 0
                    var maxScore = 0.0f
                    for (i in output.indices) {
                        if (output[i] > maxScore) {
                            maxScore = output[i]
                            maxIndex = i
                        }
                    }
                    Log.d(TAG, "Top class: $maxIndex, Score: $maxScore")
                }
            }
        }
        recordingThread.start()
    }

    private fun stopRecording() {
        if (::audioRecord.isInitialized) {
            audioRecord.stop()
            audioRecord.release()
        }

        if (::recordingThread.isInitialized) {
            recordingThread.interrupt()
        }
    }
}
