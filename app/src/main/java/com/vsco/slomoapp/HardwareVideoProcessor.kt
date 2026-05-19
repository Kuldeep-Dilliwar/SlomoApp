package com.vsco.slomoapp

import android.media.*
import android.opengl.EGL14
import android.opengl.EGLExt
import android.opengl.GLES30
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface

class HardwareVideoProcessor {

    fun processVideo(
        inputPath: String,
        outputPath: String,
        factor: Float,
        rotationDegrees: Int,
        onProgress: (Float) -> Unit,
        isCancelled: () -> Boolean
    ): Boolean {
        val extractor = MediaExtractor()
        var handlerThread: HandlerThread? = null

        try {
            extractor.setDataSource(inputPath)

            var videoTrackIndex = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    videoTrackIndex = i
                    format = f
                    break
                }
            }

            if (videoTrackIndex < 0 || format == null) return false
            extractor.selectTrack(videoTrackIndex)

            val width = format.getInteger(MediaFormat.KEY_WIDTH)
            val height = format.getInteger(MediaFormat.KEY_HEIGHT)
            val durationUs = format.getLong(MediaFormat.KEY_DURATION)
            val frameRate = try { format.getInteger(MediaFormat.KEY_FRAME_RATE) } catch (e: Exception) { 30 }

            // 1. Setup hardware encoder
            val outputFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, 15000000)
                setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }

            val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            encoder.configure(outputFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val inputSurface = encoder.createInputSurface()
            encoder.start()

            // 2. Setup EGL Context
            val eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val version = IntArray(2)
            EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
                EGLExt.EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)

            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE)
            val eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

            val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
            val eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], inputSurface, surfaceAttribs, 0)
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            // 3. Setup OpenGL Program & Direct Buffers
            val shaderProgram = SlomoShader.createProgram()
            GLES30.glUseProgram(shaderProgram)

            val vertices = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
            val texCoords = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)

            val vertexBuffer = java.nio.ByteBuffer.allocateDirect(vertices.size * 4).order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer().put(vertices).apply { position(0) }
            val texBuffer = java.nio.ByteBuffer.allocateDirect(texCoords.size * 4).order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer().put(texCoords).apply { position(0) }

            val posLoc = GLES30.glGetAttribLocation(shaderProgram, "aPosition")
            val texLoc = GLES30.glGetAttribLocation(shaderProgram, "aTexCoord")

            GLES30.glVertexAttribPointer(posLoc, 2, GLES30.GL_FLOAT, false, 0, vertexBuffer)
            GLES30.glEnableVertexAttribArray(posLoc)
            GLES30.glVertexAttribPointer(texLoc, 2, GLES30.GL_FLOAT, false, 0, texBuffer)
            GLES30.glEnableVertexAttribArray(texLoc)

            // 4. Setup hardware decoder & Texture Synchronization
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            val surfaceTexture = android.graphics.SurfaceTexture(textures[0])

            val frameSyncObject = Object()
            var frameAvailable = false
            handlerThread = HandlerThread("FrameHandlerThread").apply { start() }
            val handler = Handler(handlerThread.looper)

            surfaceTexture.setOnFrameAvailableListener({
                synchronized(frameSyncObject) {
                    frameAvailable = true
                    frameSyncObject.notifyAll()
                }
            }, handler)

            val decoderSurface = Surface(surfaceTexture)
            val decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            decoder.configure(format, decoderSurface, null, 0)
            decoder.start()

            // 5. Muxer Setup
            val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer.setOrientationHint(rotationDegrees)
            var muxerStarted = false
            var trackIndex = -1

            // 6. Loop State Variables
            val decoderInfo = MediaCodec.BufferInfo() // SEPARATE decoder info
            val encoderInfo = MediaCodec.BufferInfo() // SEPARATE encoder info

            var isExtractorEOS = false
            var isDecoderEOS = false
            var isEncoderEOS = false
            var outPtsUs = 0L
            val ptsIncrementUs = (1000000L / frameRate)

            // Helper function to safely drain the encoder without deadlocking
            val drainEncoder = { timeoutUs: Long ->
                var encoderStatus = encoder.dequeueOutputBuffer(encoderInfo, timeoutUs)
                while (encoderStatus >= 0 || encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        trackIndex = muxer.addTrack(encoder.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    } else if (encoderStatus >= 0) {
                        val encodedData = encoder.getOutputBuffer(encoderStatus)!!
                        if (encoderInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            encoderInfo.size = 0
                        }
                        if (encoderInfo.size != 0 && muxerStarted) {
                            encodedData.position(encoderInfo.offset)
                            encodedData.limit(encoderInfo.offset + encoderInfo.size)
                            muxer.writeSampleData(trackIndex, encodedData, encoderInfo)
                        }
                        encoder.releaseOutputBuffer(encoderStatus, false)

                        if ((encoderInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            isEncoderEOS = true
                        }
                    }
                    if (isEncoderEOS) break
                    encoderStatus = encoder.dequeueOutputBuffer(encoderInfo, 0)
                }
            }

            // 7. The Main Pipeline Loop (Runs until ENCODER finishes, not just decoder)
            while (!isEncoderEOS) {
                if (isCancelled()) break

                // Feed Decoder
                if (!isExtractorEOS) {
                    val inIndex = decoder.dequeueInputBuffer(10000)
                    if (inIndex >= 0) {
                        val buffer = decoder.getInputBuffer(inIndex)!!
                        val sampleSize = extractor.readSampleData(buffer, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            isExtractorEOS = true
                        } else {
                            decoder.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                            onProgress((extractor.sampleTime.toFloat() / durationUs.toFloat()).coerceIn(0f, 1f))
                        }
                    }
                }

                // Drain Decoder -> Render -> Feed Encoder
                if (!isDecoderEOS) {
                    val outIndex = decoder.dequeueOutputBuffer(decoderInfo, 10000)
                    if (outIndex >= 0) {
                        val render = decoderInfo.size != 0
                        decoder.releaseOutputBuffer(outIndex, render)

                        if (render) {
                            // Synchronously wait for the texture to latch
                            synchronized(frameSyncObject) {
                                var waitCount = 0
                                while (!frameAvailable && waitCount < 10) {
                                    try { frameSyncObject.wait(250) } catch (e: InterruptedException) {}
                                    waitCount++
                                }
                                frameAvailable = false
                            }

                            try { surfaceTexture.updateTexImage() } catch (e: Exception) {}

                            val numIntermediate = (1.0f / factor).toInt() - 1
                            for (i in 0..numIntermediate) {
                                val t = i.toFloat() / (numIntermediate + 1)

                                GLES30.glViewport(0, 0, width, height)
                                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
                                GLES30.glUniform1f(GLES30.glGetUniformLocation(shaderProgram, "uT"), t)
                                GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

                                EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, outPtsUs * 1000)
                                EGL14.eglSwapBuffers(eglDisplay, eglSurface)
                                outPtsUs += ptsIncrementUs

                                // PREVENT DEADLOCK: Empty the encoder queue instantly after pushing a frame
                                drainEncoder(0)
                            }
                        }

                        if ((decoderInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            encoder.signalEndOfInputStream()
                            isDecoderEOS = true
                        }
                    }
                }

                // Master Encoder Drain (catches anything left over)
                drainEncoder(10000)
            }

            // 8. Cleanup
            decoder.stop()
            decoder.release()
            encoder.stop()
            encoder.release()
            if (muxerStarted) muxer.stop()
            muxer.release()
            extractor.release()

            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)
            handlerThread?.quitSafely()

            return !isCancelled()

        } catch (e: Exception) {
            e.printStackTrace()
            handlerThread?.quitSafely()
            return false
        }
    }
}