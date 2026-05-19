package com.vsco.slomoapp

import android.content.ContentValues
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
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

class MainActivity : ComponentActivity() {

    private var selectedVideoUri by mutableStateOf<Uri?>(null)
    private var isProcessing by mutableStateOf(false)
    private var isCancelled = false
    private var processingProgress by mutableStateOf(0f)
    private var factor by mutableStateOf(0.5f)

    private val selectVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        selectedVideoUri = uri
    }

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

                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Slow-Mo Factor: ${String.format("%.2f", factor)}")
                        Slider(
                            value = factor,
                            onValueChange = { factor = it },
                            valueRange = 0.1f..1.0f,
                            steps = 8
                        )

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
                val inputStream = contentResolver.openInputStream(uri)
                val inputFile = File(cacheDir, "input.mp4")
                val outputStream = FileOutputStream(inputFile)
                inputStream?.copyTo(outputStream)
                inputStream?.close()
                outputStream.close()

                val outputFile = File(cacheDir, "output_slomo.mp4")
                if (outputFile.exists()) outputFile.delete()

                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(inputFile.absolutePath)
                val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                val rotation = rotationStr?.toIntOrNull() ?: 0
                retriever.release()

                val processor = HardwareVideoProcessor()
                val success = processor.processVideo(
                    inputPath = inputFile.absolutePath,
                    outputPath = outputFile.absolutePath,
                    factor = factor,
                    rotationDegrees = rotation,
                    onProgress = { progress ->
                        runOnUiThread { processingProgress = progress }
                    },
                    isCancelled = { isCancelled }
                )

                withContext(Dispatchers.Main) {
                    if (success) {
                        val galleryUri = saveVideoToGallery(outputFile)
                        if (galleryUri != null) {
                            Toast.makeText(this@MainActivity, "Saved to Gallery!", Toast.LENGTH_LONG).show()
                        } else {
                            Toast.makeText(this@MainActivity, "Saved to ${outputFile.absolutePath}", Toast.LENGTH_LONG).show()
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
}