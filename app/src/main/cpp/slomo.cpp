#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <vector>
#include <cmath>

#define LOG_TAG "SlomoAppNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

const char* DIS_COMPUTE_SHADER =
    "#version 310 es\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "\n"
    "layout(binding = 0, r32f) readonly uniform highp image2D imgPrev;\n"
    "layout(binding = 1, r32f) readonly uniform highp image2D imgCurr;\n"
    "layout(binding = 2, rg32f) uniform highp image2D flowMap;\n"
    "\n"
    "const int PATCH_SIZE = 8;\n"
    "const int MAX_ITERATIONS = 5;\n"
    "\n"
    "void main() {\n"
    "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
    "    ivec2 imgSize = imageSize(imgPrev);\n"
    "\n"
    "    if (pos.x >= imgSize.x || pos.y >= imgSize.y) return;\n"
    "\n"
    "    vec2 currentFlow = imageLoad(flowMap, pos).rg;\n"
    "\n"
    "    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {\n"
    "        float H11 = 0.0, H12 = 0.0, H22 = 0.0;\n"
    "        float b1 = 0.0, b2 = 0.0;\n"
    "\n"
    "        for (int y = -PATCH_SIZE/2; y < PATCH_SIZE/2; ++y) {\n"
    "            for (int x = -PATCH_SIZE/2; x < PATCH_SIZE/2; ++x) {\n"
    "                ivec2 p = pos + ivec2(x, y);\n"
    "                if (p.x <= 0 || p.x >= imgSize.x - 1 || p.y <= 0 || p.y >= imgSize.y - 1) continue;\n"
    "\n"
    "                float I_x = (imageLoad(imgPrev, p + ivec2(1, 0)).r - imageLoad(imgPrev, p - ivec2(1, 0)).r) * 0.5;\n"
    "                float I_y = (imageLoad(imgPrev, p + ivec2(0, 1)).r - imageLoad(imgPrev, p - ivec2(0, 1)).r) * 0.5;\n"
    "\n"
    "                vec2 warpedP = vec2(p) + currentFlow;\n"
    "                if (warpedP.x < 0.0 || warpedP.x >= float(imgSize.x-1) || warpedP.y < 0.0 || warpedP.y >= float(imgSize.y-1)) continue;\n"
    "\n"
    "                // Bi-linear interpolation for current frame sampling\n"
    "                ivec2 p00 = ivec2(floor(warpedP));\n"
    "                vec2 f = warpedP - vec2(p00);\n"
    "                float val00 = imageLoad(imgCurr, p00).r;\n"
    "                float val10 = imageLoad(imgCurr, p00 + ivec2(1, 0)).r;\n"
    "                float val01 = imageLoad(imgCurr, p00 + ivec2(0, 1)).r;\n"
    "                float val11 = imageLoad(imgCurr, p00 + ivec2(1, 1)).r;\n"
    "                float I_curr = mix(mix(val00, val10, f.x), mix(val01, val11, f.x), f.y);\n"
    "\n"
    "                float I_t = I_curr - imageLoad(imgPrev, p).r;\n"
    "\n"
    "                H11 += I_x * I_x;\n"
    "                H12 += I_x * I_y;\n"
    "                H22 += I_y * I_y;\n"
    "                b1 += I_x * I_t;\n"
    "                b2 += I_y * I_t;\n"
    "            }\n"
    "        }\n"
    "\n"
    "        float det = H11 * H22 - H12 * H12;\n"
    "        if (abs(det) < 1e-5) break;\n"
    "\n"
    "        vec2 deltaFlow;\n"
    "        deltaFlow.x = (H22 * b1 - H12 * b2) / det;\n"
    "        deltaFlow.y = (H11 * b2 - H12 * b1) / det;\n"
    "\n"
    "        currentFlow -= deltaFlow;\n"
    "        if (length(deltaFlow) < 0.01) break;\n"
    "    }\n"
    "    imageStore(flowMap, pos, vec4(currentFlow, 0.0, 0.0));\n"
    "}\n";

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

    LOGI("Processing video: %s to %s with factor %f, rotation %d (GPU DIS Accelerated)", inPath, outPath, factor, rotation);

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

        // Setup EGL Headless (GLES 3.1 for Compute Shaders)
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

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        // Compile and Link Compute Shader
        GLuint cs = createShader(GL_COMPUTE_SHADER, DIS_COMPUTE_SHADER);
        GLuint computeProgram = glCreateProgram();
        glAttachShader(computeProgram, cs);
        glLinkProgram(computeProgram);

        // Compile and Link Interpolation Shaders
        GLuint vs = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
        GLuint fs = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        GLuint renderProgram = glCreateProgram();
        glAttachShader(renderProgram, vs);
        glAttachShader(renderProgram, fs);
        glLinkProgram(renderProgram);

        GLint posLoc = glGetAttribLocation(renderProgram, "aPosition");
        GLint texLoc = glGetAttribLocation(renderProgram, "aTexCoord");
        GLint texSizeLoc = glGetUniformLocation(renderProgram, "uTexSize");
        GLint tLoc = glGetUniformLocation(renderProgram, "uT");

        float vertices[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
        float texCoords[] = { 0, 0, 1, 0, 0, 1, 1, 1 };

        GLuint textures[5];
        glGenTextures(5, textures);

        // Textures for compute shader inputs (Gray)
        for(int i=0; i<2; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        // Texture for flow map (Output of compute, input of interpolation)
        glBindTexture(GL_TEXTURE_2D, textures[2]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // Clear flow map initially by uploading zero data
        {
            std::vector<float> zeroData(width * height * 2, 0.0f);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RG, GL_FLOAT, zeroData.data());
        }

        // Textures for interpolation (RGB)
        for(int i=3; i<5; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        jclass clazz = env->GetObjectClass(thiz);
        jmethodID progressMethodId = env->GetMethodID(clazz, "onProgressUpdate", "(F)V");
        jmethodID cancelMethodId = env->GetMethodID(clazz, "isCancelled", "()Z");

        cv::Mat prevFrame, currFrame, prevGray, currGray;

        cap >> prevFrame;
        if (prevFrame.empty()) {
            throw cv::Exception(cv::Error::StsError, "Video is empty", "processVideo", __FILE__, __LINE__);
        }

        auto handleFrame = [&](cv::Mat& frame, bool write) {
            cv::Mat processed = frame.clone();
            if (rotation == 90) cv::rotate(processed, processed, cv::ROTATE_90_CLOCKWISE);
            else if (rotation == 180) cv::rotate(processed, processed, cv::ROTATE_180);
            else if (rotation == 270) cv::rotate(processed, processed, cv::ROTATE_90_COUNTERCLOCKWISE);

            if (write) {
                cv::Mat rgbFrame;
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

            // 1. Upload Grayscale frames for Compute Shader
            cv::Mat prevGrayFloat, currGrayFloat;
            prevGray.convertTo(prevGrayFloat, CV_32F, 1.0/255.0);
            currGray.convertTo(currGrayFloat, CV_32F, 1.0/255.0);

            glBindTexture(GL_TEXTURE_2D, textures[0]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, prevGrayFloat.data);

            glBindTexture(GL_TEXTURE_2D, textures[1]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, currGrayFloat.data);

            // 2. Dispatch Compute Shader for Optical Flow
            glUseProgram(computeProgram);
            glBindImageTexture(0, textures[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
            glBindImageTexture(1, textures[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
            glBindImageTexture(2, textures[2], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);

            glDispatchCompute((width + 15) / 16, (height + 15) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

            // 3. Upload RGB frames for Interpolation
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textures[3]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, prevFrame.data);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textures[4]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);

            // 4. Interpolation Loop
            glUseProgram(renderProgram);
            glUniform2f(texSizeLoc, (float)width, (float)height);
            glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, vertices);
            glEnableVertexAttribArray(posLoc);
            glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
            glEnableVertexAttribArray(texLoc);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textures[3]);
            glUniform1i(glGetUniformLocation(renderProgram, "uTexPrev"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textures[4]);
            glUniform1i(glGetUniformLocation(renderProgram, "uTexCurr"), 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, textures[2]);
            glUniform1i(glGetUniformLocation(renderProgram, "uTexFlow"), 2);

            int numIntermediate = (int)(1.0f / factor) - 1;
            for (int i = 1; i <= numIntermediate; ++i) {
                float t = (float)i / (numIntermediate + 1);
                glUniform1f(tLoc, t);
                glViewport(0, 0, width, height);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

                cv::Mat blendedRGBA(height, width, CV_8UC4);
                glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, blendedRGBA.data);

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