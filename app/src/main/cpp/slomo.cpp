#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <vector>

#define LOG_TAG "SlomoAppNative"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

const int PYRAMID_LEVELS = 4; // 0=Base, 1=Half, 2=Quarter, 3=Eighth

// --- THE NEW PYRAMIDAL COMPUTE SHADER ---
const char* PYRAMID_FLOW_COMPUTE_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, r32f) readonly uniform highp image2D imgFrom;\n"
        "layout(binding = 1, r32f) readonly uniform highp image2D imgTo;\n"
        "layout(binding = 2, rg32f) writeonly uniform highp image2D outFlowMap;\n"
        "layout(binding = 3, rg32f) readonly uniform highp image2D inFlowMap;\n"
        "\n"
        "uniform int uIsTopLevel;\n"
        "const int PATCH_SIZE = 8;\n"
        "const int MAX_ITERATIONS = 5;\n"
        "\n"
        "void main() {\n"
        "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    ivec2 imgSize = imageSize(imgFrom);\n"
        "    if (pos.x >= imgSize.x || pos.y >= imgSize.y) return;\n"
        "\n"
        "    vec2 currentFlow = vec2(0.0);\n"
        "    if (uIsTopLevel == 0) {\n"
        "        ivec2 prevPos = pos / 2; // Exact nearest-neighbor fetch from previous level\n"
        "        currentFlow = imageLoad(inFlowMap, prevPos).rg * 2.0;\n" // Multiply guess by 2
        "    }\n"
        "\n"
        "    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {\n"
        "        float H11 = 0.0, H12 = 0.0, H22 = 0.0, b1 = 0.0, b2 = 0.0;\n"
        "        for (int y = -PATCH_SIZE/2; y < PATCH_SIZE/2; ++y) {\n"
        "            for (int x = -PATCH_SIZE/2; x < PATCH_SIZE/2; ++x) {\n"
        "                ivec2 p = pos + ivec2(x, y);\n"
        "                if (p.x <= 0 || p.x >= imgSize.x - 1 || p.y <= 0 || p.y >= imgSize.y - 1) continue;\n"
        "\n"
        "                float I_x = (imageLoad(imgFrom, p + ivec2(1, 0)).r - imageLoad(imgFrom, p - ivec2(1, 0)).r) * 0.5;\n"
        "                float I_y = (imageLoad(imgFrom, p + ivec2(0, 1)).r - imageLoad(imgFrom, p - ivec2(0, 1)).r) * 0.5;\n"
        "\n"
        "                vec2 warpedP = vec2(p) + currentFlow;\n"
        "                if (warpedP.x < 0.0 || warpedP.x >= float(imgSize.x-1) || warpedP.y < 0.0 || warpedP.y >= float(imgSize.y-1)) continue;\n"
        "\n"
        "                ivec2 p00 = ivec2(floor(warpedP));\n"
        "                vec2 f = warpedP - vec2(p00);\n"
        "                float val00 = imageLoad(imgTo, p00).r;\n"
        "                float val10 = imageLoad(imgTo, p00 + ivec2(1, 0)).r;\n"
        "                float val01 = imageLoad(imgTo, p00 + ivec2(0, 1)).r;\n"
        "                float val11 = imageLoad(imgTo, p00 + ivec2(1, 1)).r;\n"
        "                float I_curr = mix(mix(val00, val10, f.x), mix(val01, val11, f.x), f.y);\n"
        "\n"
        "                float I_t = I_curr - imageLoad(imgFrom, p).r;\n"
        "                H11 += I_x * I_x; H12 += I_x * I_y; H22 += I_y * I_y;\n"
        "                b1 += I_x * I_t; b2 += I_y * I_t;\n"
        "            }\n"
        "        }\n"
        "        float det = H11 * H22 - H12 * H12;\n"
        "        if (abs(det) < 1e-5) break;\n"
        "        vec2 deltaFlow = vec2((H22 * b1 - H12 * b2) / det, (H11 * b2 - H12 * b1) / det);\n"
        "        currentFlow -= deltaFlow;\n"
        "        if (length(deltaFlow) < 0.01) break;\n"
        "    }\n"
        "    imageStore(outFlowMap, pos, vec4(currentFlow, 0.0, 0.0));\n"
        "}\n";

const char* VERTEX_SHADER =
        "#version 300 es\n"
        "in vec4 aPosition;\n"
        "in vec2 aTexCoord;\n"
        "out vec2 vTexCoord;\n"
        "void main() { gl_Position = aPosition; vTexCoord = aTexCoord; }\n";

const char* SLOMO_FRAGMENT_SHADER =
        "#version 300 es\n"
        "precision highp float;\n"
        "uniform sampler2D uTexPrev;\n"
        "uniform sampler2D uTexCurr;\n"
        "uniform sampler2D uTexFlowForward;\n"
        "uniform sampler2D uTexFlowBackward;\n"
        "uniform float uT;\n"
        "uniform vec2 uTexSize;\n"
        "uniform float uAntiGhosting;\n"
        "in vec2 vTexCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    vec2 flowF = texture(uTexFlowForward, vTexCoord).rg;\n"
        "    vec2 flowB = texture(uTexFlowBackward, vTexCoord).rg;\n"
        "    vec2 coordPrev = vTexCoord - flowF * uT / uTexSize;\n"
        "    vec2 coordCurr = vTexCoord - flowB * (1.0 - uT) / uTexSize;\n"
        "    vec4 colorPrev = texture(uTexPrev, coordPrev);\n"
        "    vec4 colorCurr = texture(uTexCurr, coordCurr);\n"
        "    \n"
        "    if (uAntiGhosting > 0.5) {\n"
        "        vec2 nextPos = vTexCoord + (flowF / uTexSize);\n"
        "        float errorA = length(flowF + texture(uTexFlowBackward, nextPos).rg);\n"
        "        vec2 prevPos = vTexCoord + (flowB / uTexSize);\n"
        "        float errorB = length(flowB + texture(uTexFlowForward, prevPos).rg);\n"
        "        float maskA = (errorA < 1.5) ? 1.0 : 0.0;\n"
        "        float maskB = (errorB < 1.5) ? 1.0 : 0.0;\n"
        "        if (maskA > 0.5 && maskB > 0.5) fragColor = mix(colorPrev, colorCurr, uT);\n"
        "        else if (maskA < 0.5 && maskB > 0.5) fragColor = colorCurr;\n"
        "        else if (maskA > 0.5 && maskB < 0.5) fragColor = colorPrev;\n"
        "        else fragColor = mix(colorPrev, colorCurr, uT);\n"
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
        "    vec2 dx = vec2(uTexelSizeLow.x, 0.0); vec2 dy = vec2(0.0, uTexelSizeLow.y);\n"
        "    vec4 c  = texture(uTexCurr, vTexCoord);\n"
        "    vec4 t  = texture(uTexCurr, vTexCoord - dy);\n"
        "    vec4 b  = texture(uTexCurr, vTexCoord + dy);\n"
        "    vec4 l  = texture(uTexCurr, vTexCoord - dx);\n"
        "    vec4 r  = texture(uTexCurr, vTexCoord + dx);\n"
        "    vec4 tl = texture(uTexCurr, vTexCoord - dx - dy);\n"
        "    vec4 tr = texture(uTexCurr, vTexCoord + dx - dy);\n"
        "    vec4 bl = texture(uTexCurr, vTexCoord - dx + dy);\n"
        "    vec4 br = texture(uTexCurr, vTexCoord + dx + dy);\n"
        "    vec4 colorMin = min(min(min(c, t), min(b, l)), min(min(r, tl), min(tr, bl)));\n"
        "    colorMin = min(colorMin, br);\n"
        "    vec4 colorMax = max(max(max(c, t), max(b, l)), max(max(r, tl), max(tr, bl)));\n"
        "    colorMax = max(colorMax, br);\n"
        "    fragColor = mix(clamp(historyColor, colorMin, colorMax), c, 0.10);\n"
        "}\n";

GLuint createShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled; glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLen = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            std::vector<char> infoLog(infoLen);
            glGetShaderInfoLog(shader, infoLen, nullptr, infoLog.data());
            LOGE("Shader error: %s", infoLog.data());
        }
        glDeleteShader(shader); return 0;
    }
    return shader;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vsco_slomoapp_MainActivity_processVideo(
        JNIEnv* env, jobject thiz, jstring inputPath, jstring outputPath,
        jfloat factor, jint totalFrames, jint rotation, jboolean upscale, jint fpsMultiplier, jboolean antiGhosting) {

    const char* inPath = env->GetStringUTFChars(inputPath, nullptr);
    const char* outPath = env->GetStringUTFChars(outputPath, nullptr);
    jboolean result = JNI_FALSE;
    cv::VideoCapture cap; cv::VideoWriter writer;
    EGLDisplay display = EGL_NO_DISPLAY; EGLContext context = EGL_NO_CONTEXT; EGLSurface surface = EGL_NO_SURFACE;

    try {
        cap.open(inPath);
        if (!cap.isOpened()) throw std::runtime_error("Input fail");

        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps = cap.get(cv::CAP_PROP_FPS) * fpsMultiplier;

        int outWidth = upscale ? width * 2 : width;
        int outHeight = upscale ? height * 2 : height;
        int targetWidth = (rotation == 90 || rotation == 270) ? outHeight : outWidth;
        int targetHeight = (rotation == 90 || rotation == 270) ? outWidth : outHeight;

        writer.open(outPath, cv::VideoWriter::fourcc('H','2','6','4'), outputFps, cv::Size(targetWidth, targetHeight));
        if (!writer.isOpened()) writer.open(outPath, cv::VideoWriter::fourcc('a','v','c','1'), outputFps, cv::Size(targetWidth, targetHeight));

        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);
        EGLint configAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_NONE };
        EGLConfig config; EGLint numConfigs;
        eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
        EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
        EGLint pbufferAttribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        eglMakeCurrent(display, surface, surface, context);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); glPixelStorei(GL_PACK_ALIGNMENT, 1);

        GLuint vs = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
        GLuint progCompute = glCreateProgram();
        glAttachShader(progCompute, createShader(GL_COMPUTE_SHADER, PYRAMID_FLOW_COMPUTE_SHADER));
        glLinkProgram(progCompute);

        GLuint progSlomo = glCreateProgram();
        glAttachShader(progSlomo, vs); glAttachShader(progSlomo, createShader(GL_FRAGMENT_SHADER, SLOMO_FRAGMENT_SHADER));
        glLinkProgram(progSlomo);

        GLuint progUpscale = glCreateProgram();
        glAttachShader(progUpscale, vs); glAttachShader(progUpscale, createShader(GL_FRAGMENT_SHADER, UPSCALE_FRAGMENT_SHADER));
        glLinkProgram(progUpscale);

        GLuint texGrayP, texGrayC, texRGBP, texRGBC, texTargetSlomo, texTargetUpscale, texHistory[2];
        GLuint texFlowF_A, texFlowF_B, texFlowB_A, texFlowB_B;

        glGenTextures(1, &texGrayP); glGenTextures(1, &texGrayC);
        glGenTextures(1, &texRGBP); glGenTextures(1, &texRGBC);
        glGenTextures(1, &texFlowF_A); glGenTextures(1, &texFlowF_B);
        glGenTextures(1, &texFlowB_A); glGenTextures(1, &texFlowB_B);
        glGenTextures(1, &texTargetSlomo); glGenTextures(1, &texTargetUpscale);
        glGenTextures(2, texHistory);

        // Standard texture setup
        auto setupTex = [&](GLuint tex, int w, int h, GLint fmt, GLint intFmt, GLint type, GLint filter) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexStorage2D(GL_TEXTURE_2D, 1, intFmt, w, h);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        };

        // PYRAMID texture setup (Allows glGenerateMipmap)
        auto setupPyramidTex = [&](GLuint tex, int w, int h, GLint intFmt) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexStorage2D(GL_TEXTURE_2D, PYRAMID_LEVELS, intFmt, w, h);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        };

        setupPyramidTex(texGrayP, width, height, GL_R32F);
        setupPyramidTex(texGrayC, width, height, GL_R32F);

        // Full resolution Flow Ping-Pong buffers
        setupTex(texFlowF_A, width, height, GL_RG, GL_RG32F, GL_FLOAT, GL_LINEAR);
        setupTex(texFlowF_B, width, height, GL_RG, GL_RG32F, GL_FLOAT, GL_LINEAR);
        setupTex(texFlowB_A, width, height, GL_RG, GL_RG32F, GL_FLOAT, GL_LINEAR);
        setupTex(texFlowB_B, width, height, GL_RG, GL_RG32F, GL_FLOAT, GL_LINEAR);

        setupTex(texRGBP, width, height, GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE, GL_LINEAR);
        setupTex(texRGBC, width, height, GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE, GL_LINEAR);
        setupTex(texTargetSlomo, width, height, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR);
        if (upscale) {
            setupTex(texTargetUpscale, outWidth, outHeight, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR);
            setupTex(texHistory[0], outWidth, outHeight, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR);
            setupTex(texHistory[1], outWidth, outHeight, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR);
        }

        GLuint fboSlomo, fboUpscale;
        glGenFramebuffers(1, &fboSlomo);
        glBindFramebuffer(GL_FRAMEBUFFER, fboSlomo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texTargetSlomo, 0);

        if (upscale) {
            glGenFramebuffers(1, &fboUpscale);
            glBindFramebuffer(GL_FRAMEBUFFER, fboUpscale);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texTargetUpscale, 0);
        }

        GLuint pbo[2];
        glGenBuffers(2, pbo);
        int pboSize = outWidth * outHeight * 4;
        for(int i=0; i<2; ++i) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
            glBufferData(GL_PIXEL_PACK_BUFFER, pboSize, nullptr, GL_DYNAMIC_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        int pboIdx = 0; bool pboPrimed = false;

        float quadVerts[] = { -1,-1, 1,-1, -1,1, 1,1 };
        float quadTex[] = { 0,0, 1,0, 0,1, 1,1 };
        auto drawQuad = [&](GLuint prog) {
            GLint pLoc = glGetAttribLocation(prog, "aPosition");
            GLint tLoc = glGetAttribLocation(prog, "aTexCoord");
            glVertexAttribPointer(pLoc, 2, GL_FLOAT, GL_FALSE, 0, quadVerts);
            glEnableVertexAttribArray(pLoc);
            glVertexAttribPointer(tLoc, 2, GL_FLOAT, GL_FALSE, 0, quadTex);
            glEnableVertexAttribArray(tLoc);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        };

        cv::Mat prevFrame, currFrame, prevGray, currGray, prevGrayFloat, currGrayFloat;
        cv::Mat bgrMat(outHeight, outWidth, CV_8UC3); cv::Mat rotMat;

        auto handleMappedBuffer = [&](GLubyte* ptr) {
            cv::Mat bgra(outHeight, outWidth, CV_8UC4, ptr);
            cv::cvtColor(bgra, bgrMat, cv::COLOR_BGRA2RGB);
            cv::Mat* finalMat = &bgrMat;
            if (rotation == 90) { cv::rotate(bgrMat, rotMat, cv::ROTATE_90_CLOCKWISE); finalMat = &rotMat; }
            else if (rotation == 180) { cv::rotate(bgrMat, rotMat, cv::ROTATE_180); finalMat = &rotMat; }
            else if (rotation == 270) { cv::rotate(bgrMat, rotMat, cv::ROTATE_90_COUNTERCLOCKWISE); finalMat = &rotMat; }
            writer.write(*finalMat);
        };

        // --- THE CASCADE ENGINE ---
        // Runs the compute shader from lowest resolution up to full resolution
        auto computePyramidFlow = [&](GLuint imgFrom, GLuint imgTo, GLuint texPing, GLuint texPong, GLuint& finalFlow) {
            glUseProgram(progCompute);
            for (int level = PYRAMID_LEVELS - 1; level >= 0; --level) {
                int curW = std::max(1, width >> level);
                int curH = std::max(1, height >> level);

                glBindImageTexture(0, imgFrom, level, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(1, imgTo, level, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(2, texPing, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

                glUniform1i(glGetUniformLocation(progCompute, "uIsTopLevel"), (level == PYRAMID_LEVELS - 1) ? 1 : 0);

                if (level < PYRAMID_LEVELS - 1) {
                    glBindImageTexture(3, texPong, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
                }

                glDispatchCompute((curW + 15) / 16, (curH + 15) / 16, 1);
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

                std::swap(texPing, texPong);
            }
            finalFlow = texPong; // After the last swap, pong holds the full resolution result
        };

        jclass clazz = env->GetObjectClass(thiz);
        jmethodID progressId = env->GetMethodID(clazz, "onProgressUpdate", "(F)V");
        jmethodID cancelId = env->GetMethodID(clazz, "isCancelled", "()Z");

        cap >> prevFrame;
        cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

        int frameCount = 1; int historyIdx = 0;
        int numIntermediate = (int)(1.0f / factor) - 1;

        while (true) {
            if (cancelId && env->CallBooleanMethod(thiz, cancelId)) break;
            cap >> currFrame; if (currFrame.empty()) break;
            cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

            prevGray.convertTo(prevGrayFloat, CV_32F, 1.0/255.0);
            currGray.convertTo(currGrayFloat, CV_32F, 1.0/255.0);

            // Upload to Level 0 (Base) and generate Hardware Mipmaps instantly
            glBindTexture(GL_TEXTURE_2D, texGrayP);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, prevGrayFloat.data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glBindTexture(GL_TEXTURE_2D, texGrayC);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, currGrayFloat.data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glBindTexture(GL_TEXTURE_2D, texRGBP); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, prevFrame.data);
            glBindTexture(GL_TEXTURE_2D, texRGBC); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);

            // Run Hardware Cascades
            GLuint finalFlowF, finalFlowB;
            computePyramidFlow(texGrayP, texGrayC, texFlowF_A, texFlowF_B, finalFlowF);
            computePyramidFlow(texGrayC, texGrayP, texFlowB_A, texFlowB_B, finalFlowB);

            // Interpolation loop
            for (int i = 0; i <= numIntermediate; ++i) {
                float t = (i == 0) ? 0.0f : (float)i / (numIntermediate + 1);

                glBindFramebuffer(GL_FRAMEBUFFER, fboSlomo);
                glViewport(0, 0, width, height);
                glUseProgram(progSlomo);

                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texRGBP); glUniform1i(glGetUniformLocation(progSlomo, "uTexPrev"), 0);
                glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texRGBC); glUniform1i(glGetUniformLocation(progSlomo, "uTexCurr"), 1);

                // Bind the completed, full-res maps
                glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, finalFlowF); glUniform1i(glGetUniformLocation(progSlomo, "uTexFlowForward"), 2);
                glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, finalFlowB); glUniform1i(glGetUniformLocation(progSlomo, "uTexFlowBackward"), 3);

                glUniform1f(glGetUniformLocation(progSlomo, "uT"), t);
                glUniform2f(glGetUniformLocation(progSlomo, "uTexSize"), (float)width, (float)height);
                glUniform1f(glGetUniformLocation(progSlomo, "uAntiGhosting"), antiGhosting ? 1.0f : 0.0f);
                drawQuad(progSlomo);

                if (upscale) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fboUpscale);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texHistory[1-historyIdx], 0);
                    glViewport(0, 0, outWidth, outHeight);
                    glUseProgram(progUpscale);

                    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texTargetSlomo); glUniform1i(glGetUniformLocation(progUpscale, "uTexCurr"), 0);
                    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texHistory[historyIdx]); glUniform1i(glGetUniformLocation(progUpscale, "uTexHistory"), 1);
                    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, finalFlowF); glUniform1i(glGetUniformLocation(progUpscale, "uTexFlow"), 2);
                    glUniform2f(glGetUniformLocation(progUpscale, "uTexelSizeLow"), 1.0f/width, 1.0f/height);
                    drawQuad(progUpscale);
                    historyIdx = 1 - historyIdx;
                } else {
                    glBindFramebuffer(GL_FRAMEBUFFER, fboSlomo);
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[pboIdx]);
                glReadPixels(0, 0, outWidth, outHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);

                if (pboPrimed) {
                    int readIdx = 1 - pboIdx;
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[readIdx]);
                    GLubyte* ptr = (GLubyte*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, pboSize, GL_MAP_READ_BIT);
                    if (ptr) { handleMappedBuffer(ptr); glUnmapBuffer(GL_PIXEL_PACK_BUFFER); }
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                pboIdx = 1 - pboIdx;
                pboPrimed = true;
            }

            prevFrame = currFrame.clone(); prevGray = currGray.clone(); frameCount++;
            if (progressId) env->CallVoidMethod(thiz, progressId, (float)frameCount/totalFrames);
        }

        if (pboPrimed) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1 - pboIdx]);
            GLubyte* ptr = (GLubyte*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, pboSize, GL_MAP_READ_BIT);
            if (ptr) { handleMappedBuffer(ptr); glUnmapBuffer(GL_PIXEL_PACK_BUFFER); }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }

        result = JNI_TRUE;
    } catch (const std::exception& e) { LOGE("Pipeline error: %s", e.what()); }

    if (display != EGL_NO_DISPLAY) { eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); eglTerminate(display); }
    if (cap.isOpened()) cap.release(); if (writer.isOpened()) writer.release();
    env->ReleaseStringUTFChars(inputPath, inPath); env->ReleaseStringUTFChars(outputPath, outPath);
    return result;
}