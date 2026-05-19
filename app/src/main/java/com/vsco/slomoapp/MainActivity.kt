package com.vsco.slomoapp

import android.content.ContentValues
import android.content.Intent
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.OutputStream

class MainActivity : ComponentActivity() {

    private var selectedVideoUri by mutableStateOf<Uri?>(null)
    private var isProcessing by mutableStateOf(false)
    private var isCancelled = false
    private var isUpscaleEnabled by mutableStateOf(false)
    private var fpsMultiplier by mutableStateOf(1)
    private var processingProgress by mutableStateOf(0f)
    private var factor by mutableStateOf(1.0f)

    private val selectVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        selectedVideoUri = uri
    }

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier.fillMaxSize().padding(16.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("SlomoApp", style = MaterialTheme.typography.headlineMedium)
                        Spacer(modifier = Modifier.height(16.dp))

                        Button(onClick = { selectVideoLauncher.launch("video/*") }) {
                            Text("Select Video")
                        }

                        Spacer(modifier = Modifier.height(8.dp))
                        Text(selectedVideoUri?.let { "Video Selected" } ?: "No video selected")

                        Text("Slow-Mo Factor: ${if (factor == 1.0f) "Normal" else factor.toString() + "x"}")
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            listOf(1.0f, 2.0f, 4.0f, 8.0f, 16.0f).forEach { f ->
                                FilterChip(
                                    selected = factor == f,
                                    onClick = { factor = f },
                                    label = { Text(if (f == 1.0f) "None" else "${f.toInt()}x") }
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))
                        Text("FPS Multiplier: ${fpsMultiplier}x")
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            listOf(1, 2, 4, 6, 8, 16).forEach { m ->
                                FilterChip(
                                    selected = fpsMultiplier == m,
                                    onClick = { fpsMultiplier = m },
                                    label = { Text("${m}x") },
                                    enabled = !isProcessing
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        if (isProcessing) {
                            LinearProgressIndicator(
                                progress = processingProgress,
                                modifier = Modifier.fillMaxWidth()
                            )
                            Text("Processing: ${(processingProgress * 100).toInt()}%")
                            Spacer(modifier = Modifier.height(8.dp))
                            Button(onClick = { isCancelled = true }) {
                                Text("Cancel")
                            }
                            Spacer(modifier = Modifier.height(16.dp))
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(
                                checked = isUpscaleEnabled,
                                onCheckedChange = { isUpscaleEnabled = it },
                                enabled = !isProcessing
                            )
                            Text("Enable 4K AI Upscaling")
                        }

                        Spacer(modifier = Modifier.height(16.dp))
                        val coroutineScope = rememberCoroutineScope()
                        Button(
                            onClick = {
                                selectedVideoUri?.let { uri ->
                                    isProcessing = true
                                    isCancelled = false
                                    processingProgress = 0f
                                    coroutineScope.launch {
                                        processVideoFile(uri)
                                    }
                                }
                            },
                            enabled = selectedVideoUri != null && !isProcessing
                        ) {
                            Text(if (isProcessing) "Processing..." else "Apply Slow-Mo")
                        }
                    }
                }
            }
        }
    }

    private suspend fun processVideoFile(uri: Uri) {
        withContext(Dispatchers.IO) {
            try {
                // Copy to cache dir to get a real file path for OpenCV
                val inputStream = contentResolver.openInputStream(uri)
                val inputFile = File(cacheDir, "input.mp4")
                val outputStream = FileOutputStream(inputFile)
                inputStream?.copyTo(outputStream)
                inputStream?.close()
                outputStream.close()

                val outputFile = File(cacheDir, "output_slomo.mp4")

                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(inputFile.absolutePath)
                val totalFramesStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                val totalFrames = totalFramesStr?.toIntOrNull() ?: 0
                val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                val rotation = rotationStr?.toIntOrNull() ?: 0
                retriever.release()

                val success = processVideo(
                    inputFile.absolutePath,
                    outputFile.absolutePath,
                    1.0f / (factor * fpsMultiplier),
                    totalFrames,
                    rotation,
                    isUpscaleEnabled,
                    fpsMultiplier
                )

                withContext(Dispatchers.Main) {
                    if (success) {
                        if (!isCancelled) {
                            val galleryUri = saveVideoToGallery(outputFile)
                            if (galleryUri != null) {
                                Toast.makeText(this@MainActivity, "Saved to Gallery!", Toast.LENGTH_LONG).show()
                            } else {
                                Toast.makeText(this@MainActivity, "Saved to ${outputFile.absolutePath}", Toast.LENGTH_LONG).show()
                            }
                        } else {
                            Toast.makeText(this@MainActivity, "Processing cancelled", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        val message = if (isCancelled) "Processing cancelled" else "Processing failed"
                        Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
                    }
                    isProcessing = false
                    processingProgress = 0f
                    isCancelled = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    isProcessing = false
                    processingProgress = 0f
                }
            }
        }
    }

    /**
     * Called from native code to update progress
     */
    fun onProgressUpdate(percentage: Float) {
        runOnUiThread {
            processingProgress = percentage
        }
    }

    /**
     * Called from native code to check if processing should be cancelled
     */
    fun isCancelled(): Boolean {
        return isCancelled
    }

    private suspend fun saveVideoToGallery(file: File): Uri? {
        return withContext(Dispatchers.IO) {
            try {
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, "Slomo_${System.currentTimeMillis()}.mp4")
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/SlomoApp")
                        put(MediaStore.Video.Media.IS_PENDING, 1)
                    }
                }

                val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                } else {
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                }

                val uri = contentResolver.insert(collection, values) ?: return@withContext null

                contentResolver.openOutputStream(uri)?.use { outputStream ->
                    FileInputStream(file).use { inputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear()
                    values.put(MediaStore.Video.Media.IS_PENDING, 0)
                    contentResolver.update(uri, values, null, null)
                }
                uri
            } catch (e: Exception) {
                null
            }
        }
    }

    external fun processVideo(
        inputPath: String,
        outputPath: String,
        factor: Float,
        totalFrames: Int,
        rotation: Int,
        upscale: Boolean,
        fpsMultiplier: Int
    ): Boolean

    companion object {
        init {
            System.loadLibrary("slomoapp")
        }
    }
}