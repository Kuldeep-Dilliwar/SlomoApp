#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <vector>

#define LOG_TAG "SlomoAppNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

const char* VERTEX_SHADER =
        "#version 300 es\n"
        "in vec4 aPosition;\n"
        "in vec2 aTexCoord;\n"
        "out vec2 vTexCoord;\n"
        "void main() {\n"
        "  gl_Position = aPosition;\n"
        "  vTexCoord = aTexCoord;\n"
        "}\n";

const char* FRAGMENT_SHADER =
        "#version 300 es\n"
        "precision mediump float;\n"
        "uniform sampler2D uTexPrev;\n"
        "uniform sampler2D uTexCurr;\n"
        "uniform sampler2D uTexFlow;\n"
        "uniform float uT;\n"
        "uniform vec2 uTexSize;\n"
        "in vec2 vTexCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "  vec2 flow = texture(uTexFlow, vTexCoord).rg;\n"
        "  \n"
        "  // Correct warping math for future/past pixel interpolation\n"
        "  vec2 coordPrev = vTexCoord - flow * uT / uTexSize;\n"
        "  vec2 coordCurr = vTexCoord + flow * (1.0 - uT) / uTexSize;\n"
        "  \n"
        "  vec4 colorPrev = texture(uTexPrev, coordPrev);\n"
        "  vec4 colorCurr = texture(uTexCurr, coordCurr);\n"
        "  \n"
        "  fragColor = mix(colorPrev, colorCurr, uT);\n"
        "}\n";

GLuint createShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char* infoLog = (char*)malloc(sizeof(char) * infoLen);
            glGetShaderInfoLog(shader, infoLen, nullptr, infoLog);
            LOGE("Error compiling shader: %s", infoLog);
            free(infoLog);
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vsco_slomoapp_MainActivity_processVideo(
        JNIEnv* env,
        jobject thiz,
        jstring inputPath,
        jstring outputPath,
        jfloat factor,
        jint totalFrames,
        jint rotation) {

    const char* inPath = env->GetStringUTFChars(inputPath, nullptr);
    const char* outPath = env->GetStringUTFChars(outputPath, nullptr);

    LOGI("Processing video: %s to %s with factor %f, rotation %d", inPath, outPath, factor, rotation);

    jboolean result = JNI_FALSE;
    cv::VideoCapture cap;
    cv::VideoWriter writer;

    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;

    try {
        cap.open(inPath);
        if (!cap.isOpened()) {
            throw cv::Exception(cv::Error::StsError, "Failed to open input video", "processVideo", __FILE__, __LINE__);
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int videoFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

        if (videoFrames <= 0) videoFrames = totalFrames;

        int targetWidth = width;
        int targetHeight = height;
        if (rotation == 90 || rotation == 270) {
            targetWidth = height;
            targetHeight = width;
        }

        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        writer.open(outPath, fourcc, fps, cv::Size(targetWidth, targetHeight));
        if (!writer.isOpened()) {
            fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            writer.open(outPath, fourcc, fps, cv::Size(targetWidth, targetHeight));
            if (!writer.isOpened()) {
                throw cv::Exception(cv::Error::StsError, "Failed to open writer", "processVideo", __FILE__, __LINE__);
            }
        }

        // Setup EGL Headless (GLES 3.0)
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);
        EGLint configAttribs[] = {
                EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                EGL_NONE
        };
        EGLConfig config;
        EGLint numConfigs;
        eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
        EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);

        EGLint pbufferAttribs[] = { EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        eglMakeCurrent(display, surface, surface, context);

        // OPTIMIZATION: Prevent memory alignment crashes on cropped/odd resolution videos
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        GLuint vs = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
        GLuint fs = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        GLuint program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glUseProgram(program);

        GLint posLoc = glGetAttribLocation(program, "aPosition");
        GLint texLoc = glGetAttribLocation(program, "aTexCoord");
        GLint texSizeLoc = glGetUniformLocation(program, "uTexSize");
        GLint tLoc = glGetUniformLocation(program, "uT");

        glUniform2f(texSizeLoc, (float)width, (float)height);

        float vertices[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
        float texCoords[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
        glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, vertices);
        glEnableVertexAttribArray(posLoc);
        glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
        glEnableVertexAttribArray(texLoc);

        GLuint textures[3];
        glGenTextures(3, textures);
        for(int i=0; i<3; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            // FIX: Float textures (optical flow) require GL_NEAREST to function in GLES3
            GLint filter = (i == 2) ? GL_NEAREST : GL_LINEAR;
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        jclass clazz = env->GetObjectClass(thiz);
        jmethodID progressMethodId = env->GetMethodID(clazz, "onProgressUpdate", "(F)V");
        jmethodID cancelMethodId = env->GetMethodID(clazz, "isCancelled", "()Z");

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
        cv::Mat prevFrame, currFrame, prevGray, currGray;

        cap >> prevFrame;
        if (prevFrame.empty()) {
            throw cv::Exception(cv::Error::StsError, "Video is empty", "processVideo", __FILE__, __LINE__);
        }

        // Centralized frame handler: manages rotation and strict RGB output for MediaCodec
        auto handleFrame = [&](cv::Mat& frame, bool write) {
            cv::Mat processed = frame.clone();
            if (rotation == 90) cv::rotate(processed, processed, cv::ROTATE_90_CLOCKWISE);
            else if (rotation == 180) cv::rotate(processed, processed, cv::ROTATE_180);
            else if (rotation == 270) cv::rotate(processed, processed, cv::ROTATE_90_COUNTERCLOCKWISE);

            if (write) {
                cv::Mat rgbFrame;
                // FIX: MediaCodec underlying VideoWriter expects RGB, converting from BGR fixes Smurf Vision
                cv::cvtColor(processed, rgbFrame, cv::COLOR_BGR2RGB);
                writer.write(rgbFrame);
            }
        };

        cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
        handleFrame(prevFrame, true);

        int frameCount = 1;
        while (true) {
            if (cancelMethodId != nullptr && env->CallBooleanMethod(thiz, cancelMethodId)) break;

            cap >> currFrame;
            if (currFrame.empty()) break;

            cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);
            cv::Mat flow;
            dis->calc(prevGray, currGray, flow);

            // OPTIMIZATION: Texture uploads are pulled OUTSIDE the interpolation loop.
            // This prevents redundant CPU-to-GPU memory copies for intermediate frames.
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, prevFrame.data);
            glUniform1i(glGetUniformLocation(program, "uTexPrev"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textures[1]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);
            glUniform1i(glGetUniformLocation(program, "uTexCurr"), 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, textures[2]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, flow.data);
            glUniform1i(glGetUniformLocation(program, "uTexFlow"), 2);

            int numIntermediate = (int)(1.0f / factor) - 1;
            for (int i = 1; i <= numIntermediate; ++i) {
                float t = (float)i / (numIntermediate + 1);

                glUniform1f(tLoc, t);
                glViewport(0, 0, width, height);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

                // FIX: Read as RGBA to strictly comply with GLES3 specifications and prevent crashes
                cv::Mat blendedRGBA(height, width, CV_8UC4);
                glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, blendedRGBA.data);

                // Strip the alpha channel to return to our baseline BGR format for handleFrame
                cv::Mat blendedBGR;
                cv::cvtColor(blendedRGBA, blendedBGR, cv::COLOR_BGRA2BGR);

                handleFrame(blendedBGR, true);
            }

            handleFrame(currFrame, true);
            prevGray = currGray.clone();
            prevFrame = currFrame.clone();
            frameCount++;

            if (videoFrames > 0 && progressMethodId != nullptr) {
                env->CallVoidMethod(thiz, progressMethodId, (float)frameCount / videoFrames);
            }
        }
        result = JNI_TRUE;

    } catch (const cv::Exception& e) { LOGE("OpenCV Exception: %s", e.what()); }
    catch (const std::exception& e) { LOGE("Standard Exception: %s", e.what()); }

    if (display != EGL_NO_DISPLAY) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        eglTerminate(display);
    }
    if (cap.isOpened()) cap.release();
    if (writer.isOpened()) writer.release();
    env->ReleaseStringUTFChars(inputPath, inPath);
    env->ReleaseStringUTFChars(outputPath, outPath);

    return result;
}