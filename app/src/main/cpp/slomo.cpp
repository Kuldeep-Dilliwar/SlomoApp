#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <vector>

#define LOG_TAG "SlomoAppNative"
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

const char* SLOMO_FRAGMENT_SHADER =
        "#version 300 es\n"
        "precision highp float;\n"
        "uniform sampler2D uTexPrev;\n"
        "uniform sampler2D uTexCurr;\n"
        "uniform sampler2D uTexFlowForward;\n"
        "uniform sampler2D uTexFlowBackward;\n"
        "uniform float uT;\n"
        "uniform vec2 uTexSize;\n"
        "in vec2 vTexCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    vec2 flowF = texture(uTexFlowForward, vTexCoord).rg;\n"
        "    vec2 flowB = texture(uTexFlowBackward, vTexCoord).rg;\n"
        "    \n"
        "    // Bidirectional Consistency Check (Occlusion Detection)\n"
        "    vec2 nextPos = vTexCoord + (flowF / uTexSize);\n"
        "    vec2 flowB_at_nextPos = texture(uTexFlowBackward, nextPos).rg;\n"
        "    float errorA = length(flowF + flowB_at_nextPos);\n"
        "    \n"
        "    vec2 prevPos = vTexCoord + (flowB / uTexSize);\n"
        "    vec2 flowF_at_prevPos = texture(uTexFlowForward, prevPos).rg;\n"
        "    float errorB = length(flowB + flowF_at_prevPos);\n"
        "    \n"
        "    float maskA = (errorA < 1.5) ? 1.0 : 0.0;\n"
        "    float maskB = (errorB < 1.5) ? 1.0 : 0.0;\n"
        "    \n"
        "    vec2 coordPrev = vTexCoord - flowF * uT / uTexSize;\n"
        "    vec2 coordCurr = vTexCoord - flowB * (1.0 - uT) / uTexSize;\n"
        "    \n"
        "    vec4 colorPrev = texture(uTexPrev, coordPrev);\n"
        "    vec4 colorCurr = texture(uTexCurr, coordCurr);\n"
        "    \n"
        "    if (maskA > 0.5 && maskB > 0.5) {\n"
        "        fragColor = mix(colorPrev, colorCurr, uT);\n"
        "    } else if (maskA < 0.5 && maskB > 0.5) {\n"
        "        fragColor = colorCurr;\n"
        "    } else if (maskA > 0.5 && maskB < 0.5) {\n"
        "        fragColor = colorPrev;\n"
        "    } else {\n"
        "        fragColor = mix(colorPrev, colorCurr, uT);\n"
        "    }\n"
        "}\n";

const char* UPSCALE_FRAGMENT_SHADER =
        "#version 300 es\n"
        "precision highp float;\n"
        "uniform sampler2D uTexCurr;\n"
        "uniform sampler2D uTexHistory;\n"
        "uniform sampler2D uTexFlow;\n"
        "uniform vec2 uTexelSizeLow;\n"
        "in vec2 vTexCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    vec2 flow = texture(uTexFlow, vTexCoord).rg;\n"
        "    vec2 historyCoord = vTexCoord - (flow * uTexelSizeLow);\n"
        "    vec4 historyColor = texture(uTexHistory, historyCoord);\n"
        "    \n"
        "    vec2 dx = vec2(uTexelSizeLow.x, 0.0);\n"
        "    vec2 dy = vec2(0.0, uTexelSizeLow.y);\n"
        "    \n"
        "    vec4 c  = texture(uTexCurr, vTexCoord);\n"
        "    vec4 t  = texture(uTexCurr, vTexCoord - dy);\n"
        "    vec4 b  = texture(uTexCurr, vTexCoord + dy);\n"
        "    vec4 l  = texture(uTexCurr, vTexCoord - dx);\n"
        "    vec4 r  = texture(uTexCurr, vTexCoord + dx);\n"
        "    vec4 tl = texture(uTexCurr, vTexCoord - dx - dy);\n"
        "    vec4 tr = texture(uTexCurr, vTexCoord + dx - dy);\n"
        "    vec4 bl = texture(uTexCurr, vTexCoord - dx + dy);\n"
        "    vec4 br = texture(uTexCurr, vTexCoord + dx + dy);\n"
        "    \n"
        "    vec4 colorMin = min(min(min(c, t), min(b, l)), min(min(r, tl), min(tr, bl)));\n"
        "    colorMin = min(colorMin, br);\n"
        "    vec4 colorMax = max(max(max(c, t), max(b, l)), max(max(r, tl), max(tr, bl)));\n"
        "    colorMax = max(colorMax, br);\n"
        "    \n"
        "    historyColor = clamp(historyColor, colorMin, colorMax);\n"
        "    fragColor = mix(historyColor, c, 0.10);\n"
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
            LOGE("Shader compile error: %s", infoLog);
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
        jint rotation,
        jboolean upscale,
        jint fpsMultiplier) {

    const char* inPath = env->GetStringUTFChars(inputPath, nullptr);
    const char* outPath = env->GetStringUTFChars(outputPath, nullptr);

    jboolean result = JNI_FALSE;
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;

    try {
        cap.open(inPath);
        if (!cap.isOpened()) throw cv::Exception(9001, "Input fail", "open", __FILE__, __LINE__);

        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double inputFps = cap.get(cv::CAP_PROP_FPS);
        double outputFps = inputFps * fpsMultiplier;

        int outWidth = upscale ? width * 2 : width;
        int outHeight = upscale ? height * 2 : height;

        int targetWidth = outWidth;
        int targetHeight = outHeight;
        if (rotation == 90 || rotation == 270) {
            targetWidth = outHeight;
            targetHeight = outWidth;
        }

        writer.open(outPath, cv::VideoWriter::fourcc('H','2','6','4'), outputFps, cv::Size(targetWidth, targetHeight));
        if (!writer.isOpened()) writer.open(outPath, cv::VideoWriter::fourcc('a','v','c','1'), outputFps, cv::Size(targetWidth, targetHeight));

        // EGL Setup
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);
        EGLint configAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE };
        EGLConfig config; EGLint numConfigs;
        eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
        EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);

        EGLint pbufferAttribs[] = { EGL_WIDTH, outWidth, EGL_HEIGHT, outHeight, EGL_NONE };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        eglMakeCurrent(display, surface, surface, context);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        GLuint vs = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
        GLuint fsSlomo = createShader(GL_FRAGMENT_SHADER, SLOMO_FRAGMENT_SHADER);
        GLuint progSlomo = glCreateProgram();
        glAttachShader(progSlomo, vs); glAttachShader(progSlomo, fsSlomo);
        glLinkProgram(progSlomo);

        GLuint fsUpscale = createShader(GL_FRAGMENT_SHADER, UPSCALE_FRAGMENT_SHADER);
        GLuint progUpscale = glCreateProgram();
        glAttachShader(progUpscale, vs); glAttachShader(progUpscale, fsUpscale);
        glLinkProgram(progUpscale);

        GLuint texPrev, texCurr, texFlowF, texFlowB, texHistory[2], fbo;
        glGenTextures(1, &texPrev); glGenTextures(1, &texCurr);
        glGenTextures(1, &texFlowF); glGenTextures(1, &texFlowB);
        glGenTextures(2, texHistory); glGenFramebuffers(1, &fbo);

        auto setupTex = [&](GLuint tex, int w, int h, GLint fmt, GLint type) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, (fmt == GL_RG32F ? GL_RG : GL_RGB), type, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (fmt == GL_RG32F ? GL_NEAREST : GL_LINEAR));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (fmt == GL_RG32F ? GL_NEAREST : GL_LINEAR));
        };
        setupTex(texPrev, width, height, GL_RGB, GL_UNSIGNED_BYTE);
        setupTex(texCurr, width, height, GL_RGB, GL_UNSIGNED_BYTE);
        setupTex(texFlowF, width, height, GL_RG32F, GL_FLOAT);
        setupTex(texFlowB, width, height, GL_RG32F, GL_FLOAT);
        setupTex(texHistory[0], outWidth, outHeight, GL_RGB, GL_UNSIGNED_BYTE);
        setupTex(texHistory[1], outWidth, outHeight, GL_RGB, GL_UNSIGNED_BYTE);

        auto handleFrame = [&](cv::Mat& frame) {
            cv::Mat processed = frame.clone();
            if (rotation == 90) cv::rotate(processed, processed, cv::ROTATE_90_CLOCKWISE);
            else if (rotation == 180) cv::rotate(processed, processed, cv::ROTATE_180);
            else if (rotation == 270) cv::rotate(processed, processed, cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::Mat rgb; cv::cvtColor(processed, rgb, cv::COLOR_BGR2RGB);
            writer.write(rgb);
        };

        float quadVerts[] = { -1,-1, 1,-1, -1,1, 1,1 };
        float quadTex[] = { 0,0, 1,0, 0,1, 1,1 };

        auto drawQuad = [&](GLuint prog, GLint posLoc, GLint texLoc) {
            glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, quadVerts);
            glEnableVertexAttribArray(posLoc);
            glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 0, quadTex);
            glEnableVertexAttribArray(texLoc);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        };

        jclass clazz = env->GetObjectClass(thiz);
        jmethodID progressId = env->GetMethodID(clazz, "onProgressUpdate", "(F)V");
        jmethodID cancelId = env->GetMethodID(clazz, "isCancelled", "()Z");
        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

        cv::Mat prevFrame, currFrame, prevGray, currGray;
        cap >> prevFrame;
        cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

        cv::Mat initial4K; cv::resize(prevFrame, initial4K, cv::Size(outWidth, outHeight));
        glBindTexture(GL_TEXTURE_2D, texHistory[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, outWidth, outHeight, GL_RGB, GL_UNSIGNED_BYTE, initial4K.data);
        handleFrame(initial4K);

        int frameCount = 1; int historyIdx = 0;
        while (true) {
            if (cancelId && env->CallBooleanMethod(thiz, cancelId)) break;
            cap >> currFrame; if (currFrame.empty()) break;
            cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

            cv::Mat flowF, flowB;
            dis->calc(prevGray, currGray, flowF);
            dis->calc(currGray, prevGray, flowB);

            // 1. UPSCALE STEP
            if (upscale) {
                glUseProgram(progUpscale);
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texCurr);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);
                glUniform1i(glGetUniformLocation(progUpscale, "uTexCurr"), 0);

                glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texHistory[historyIdx]);
                glUniform1i(glGetUniformLocation(progUpscale, "uTexHistory"), 1);

                glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, texFlowF);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RG, GL_FLOAT, flowF.data);
                glUniform1i(glGetUniformLocation(progUpscale, "uTexFlow"), 2);

                glUniform2f(glGetUniformLocation(progUpscale, "uTexelSizeLow"), 1.0f/width, 1.0f/height);

                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texHistory[1-historyIdx], 0);
                glViewport(0, 0, outWidth, outHeight);
                drawQuad(progUpscale, glGetAttribLocation(progUpscale, "aPosition"), glGetAttribLocation(progUpscale, "aTexCoord"));

                historyIdx = 1 - historyIdx;
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }

            // 2. INTERPOLATION STEP
            int numIntermediate = (int)(1.0f / factor) - 1;
            if (numIntermediate > 0) {
                glUseProgram(progSlomo);

                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texPrev);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, prevFrame.data);
                glUniform1i(glGetUniformLocation(progSlomo, "uTexPrev"), 0);

                glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texCurr);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);
                glUniform1i(glGetUniformLocation(progSlomo, "uTexCurr"), 1);

                glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, texFlowF);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RG, GL_FLOAT, flowF.data);
                glUniform1i(glGetUniformLocation(progSlomo, "uTexFlowForward"), 2);

                glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, texFlowB);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RG, GL_FLOAT, flowB.data);
                glUniform1i(glGetUniformLocation(progSlomo, "uTexFlowBackward"), 3);

                glUniform2f(glGetUniformLocation(progSlomo, "uTexSize"), (float)width, (float)height);

                for (int i = 1; i <= numIntermediate; ++i) {
                    float t = (float)i / (numIntermediate + 1);
                    glUniform1f(glGetUniformLocation(progSlomo, "uT"), t);
                    glViewport(0, 0, width, height);

                    drawQuad(progSlomo, glGetAttribLocation(progSlomo, "aPosition"), glGetAttribLocation(progSlomo, "aTexCoord"));

                    cv::Mat res(height, width, CV_8UC4);
                    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, res.data);

                    // The explicit cv::flip has been REMOVED here.

                    cv::Mat bgr; cv::cvtColor(res, bgr, cv::COLOR_BGRA2BGR);

                    if (upscale) {
                        cv::Mat up; cv::resize(bgr, up, cv::Size(outWidth, outHeight));
                        handleFrame(up);
                    } else {
                        handleFrame(bgr);
                    }
                }
            }

            // Write the current anchor frame
            if (upscale) {
                cv::Mat res(outHeight, outWidth, CV_8UC4);
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texHistory[historyIdx], 0);
                glReadPixels(0, 0, outWidth, outHeight, GL_RGBA, GL_UNSIGNED_BYTE, res.data);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);

                // The explicit cv::flip has been REMOVED here as well.

                cv::Mat bgr; cv::cvtColor(res, bgr, cv::COLOR_BGRA2BGR);
                handleFrame(bgr);
            } else {
                handleFrame(currFrame);
            }

            prevFrame = currFrame.clone(); prevGray = currGray.clone(); frameCount++;
            if (progressId) env->CallVoidMethod(thiz, progressId, (float)frameCount/totalFrames);
        }
        result = JNI_TRUE;
    } catch (...) { LOGE("Processing error"); }

    if (display != EGL_NO_DISPLAY) { eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); eglTerminate(display); }
    if (cap.isOpened()) cap.release(); if (writer.isOpened()) writer.release();
    env->ReleaseStringUTFChars(inputPath, inPath); env->ReleaseStringUTFChars(outputPath, outPath);
    return result;
}