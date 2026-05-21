#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#define LOG_TAG "SlomoAppNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

const int PYRAMID_LEVELS = 4;

// ============================================================================
// 1. COMPUTE SHADERS (TRUE MULTI-SCALE DIS ENGINE)
// ============================================================================

const char* DIS_CLEAR_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, rg32f) writeonly uniform highp image2D flow;\n"
        "void main() {\n"
        "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    if (pos.x >= imageSize(flow).x || pos.y >= imageSize(flow).y) return;\n"
        "    imageStore(flow, pos, vec4(0.0, 0.0, 0.0, 0.0));\n"
        "}\n";

const char* DIS_COPY_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, rg32f) readonly uniform highp image2D inFlow;\n"
        "layout(binding = 1, rg32f) writeonly uniform highp image2D outFlow;\n"
        "void main() {\n"
        "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    ivec2 outSize = imageSize(outFlow);\n"
        "    if (pos.x >= outSize.x || pos.y >= outSize.y) return;\n"
        "    imageStore(outFlow, pos, imageLoad(inFlow, pos));\n"
        "}\n";

const char* DIS_DOWNSCALE_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, r8) uniform readonly highp image2D srcImage;\n"
        "layout(binding = 1, r8) uniform writeonly highp image2D dstImage;\n"
        "void main() {\n"
        "    ivec2 dstPos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    ivec2 dstSize = imageSize(dstImage);\n"
        "    if (dstPos.x >= dstSize.x || dstPos.y >= dstSize.y) return;\n"
        "    ivec2 srcPos = dstPos * 2;\n"
        "    ivec2 srcSize = imageSize(srcImage);\n"
        "    float p00 = imageLoad(srcImage, min(srcPos + ivec2(0, 0), srcSize - 1)).r;\n"
        "    float p10 = imageLoad(srcImage, min(srcPos + ivec2(1, 0), srcSize - 1)).r;\n"
        "    float p01 = imageLoad(srcImage, min(srcPos + ivec2(0, 1), srcSize - 1)).r;\n"
        "    float p11 = imageLoad(srcImage, min(srcPos + ivec2(1, 1), srcSize - 1)).r;\n"
        "    float avg = (p00 + p10 + p01 + p11) * 0.25;\n"
        "    imageStore(dstImage, dstPos, vec4(avg, 0.0, 0.0, 0.0));\n"
        "}\n";

const char* DIS_GRADIENT_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, r8) readonly uniform highp image2D I0;\n"
        "layout(binding = 1, r16i) writeonly uniform highp iimage2D I0x;\n"
        "layout(binding = 2, r16i) writeonly uniform highp iimage2D I0y;\n"
        "void main() {\n"
        "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    ivec2 size = imageSize(I0);\n"
        "    if (pos.x >= size.x || pos.y >= size.y) return;\n"
        "    float left = imageLoad(I0, max(pos - ivec2(1, 0), ivec2(0))).r;\n"
        "    float right = imageLoad(I0, min(pos + ivec2(1, 0), size - 1)).r;\n"
        "    float top = imageLoad(I0, max(pos - ivec2(0, 1), ivec2(0))).r;\n"
        "    float bottom = imageLoad(I0, min(pos + ivec2(0, 1), size - 1)).r;\n"
        "    imageStore(I0x, pos, ivec4(int((right - left) * 255.0), 0, 0, 0));\n"
        "    imageStore(I0y, pos, ivec4(int((bottom - top) * 255.0), 0, 0, 0));\n"
        "}\n";

const char* DIS_PRECOMPUTE_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, r16i) readonly uniform highp iimage2D I0x;\n"
        "layout(binding = 1, r16i) readonly uniform highp iimage2D I0y;\n"
        "layout(binding = 2, r32f) writeonly uniform highp image2D xx_buf;\n"
        "layout(binding = 3, r32f) writeonly uniform highp image2D yy_buf;\n"
        "layout(binding = 4, r32f) writeonly uniform highp image2D xy_buf;\n"
        "layout(binding = 5, r32f) writeonly uniform highp image2D x_buf;\n"
        "layout(binding = 6, r32f) writeonly uniform highp image2D y_buf;\n"
        "uniform int patch_size;\n"
        "uniform int patch_stride;\n"
        "uniform int w;\n"
        "uniform int h;\n"
        "void main() {\n"
        "    ivec2 patch_pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    int max_ws = (w + patch_stride - 1) / patch_stride;\n"
        "    int max_hs = (h + patch_stride - 1) / patch_stride;\n"
        "    if (patch_pos.x >= max_ws || patch_pos.y >= max_hs) return;\n"
        "    int i0 = patch_pos.y * patch_stride;\n"
        "    int j0 = patch_pos.x * patch_stride;\n"
        "    float xx = 0.0, yy = 0.0, xy = 0.0, x_sum = 0.0, y_sum = 0.0;\n"
        "    int psz2 = patch_size / 2;\n"
        "    for (int y = -psz2; y < psz2; ++y) {\n"
        "        for (int x = -psz2; x < psz2; ++x) {\n"
        "            ivec2 p = clamp(ivec2(j0 + x, i0 + y), ivec2(0), ivec2(w - 1, h - 1));\n"
        "            float ix = float(imageLoad(I0x, p).r);\n"
        "            float iy = float(imageLoad(I0y, p).r);\n"
        "            xx += ix * ix; yy += iy * iy; xy += ix * iy;\n"
        "            x_sum += ix; y_sum += iy;\n"
        "        }\n"
        "    }\n"
        "    imageStore(xx_buf, patch_pos, vec4(xx, 0, 0, 0));\n"
        "    imageStore(yy_buf, patch_pos, vec4(yy, 0, 0, 0));\n"
        "    imageStore(xy_buf, patch_pos, vec4(xy, 0, 0, 0));\n"
        "    imageStore(x_buf, patch_pos, vec4(x_sum, 0, 0, 0));\n"
        "    imageStore(y_buf, patch_pos, vec4(y_sum, 0, 0, 0));\n"
        "}\n";

const char* DIS_PATCH_SEARCH_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, r8) uniform readonly highp image2D I0;\n"
        "layout(binding = 1, r8) uniform readonly highp image2D I1;\n"
        "layout(binding = 2, r16i) uniform readonly highp iimage2D I0x;\n"
        "layout(binding = 3, r16i) uniform readonly highp iimage2D I0y;\n"
        "layout(binding = 4, r32f) uniform readonly highp image2D xx_buf;\n"
        "layout(binding = 5, r32f) uniform readonly highp image2D yy_buf;\n"
        "layout(binding = 6, r32f) uniform readonly highp image2D xy_buf;\n"
        "layout(binding = 7, r32f) uniform readonly highp image2D x_buf;\n"
        "layout(binding = 8, r32f) uniform readonly highp image2D y_buf;\n"
        "layout(binding = 9, rg32f) uniform readonly highp image2D U_flow_read;\n"
        "layout(binding = 10, rg32f) uniform writeonly highp image2D U_flow_write;\n"
        "\n"
        "uniform int patch_size;\n"
        "uniform int patch_stride;\n"
        "uniform int num_inner_iter;\n"
        "uniform int w;\n"
        "uniform int h;\n"
        "\n"
        "float sampleBilinear(vec2 pos) {\n"
        "    vec2 p = clamp(pos, vec2(0.0), vec2(float(w - 1), float(h - 1)));\n"
        "    ivec2 base = ivec2(p);\n"
        "    vec2 f = fract(p);\n"
        "    float w11 = f.x * f.y; float w10 = f.x * (1.0 - f.y);\n"
        "    float w01 = (1.0 - f.x) * f.y; float w00 = (1.0 - f.x) * (1.0 - f.y);\n"
        "    float v00 = imageLoad(I1, base).r;\n"
        "    float v10 = imageLoad(I1, min(base + ivec2(1, 0), ivec2(w-1, h-1))).r;\n"
        "    float v01 = imageLoad(I1, min(base + ivec2(0, 1), ivec2(w-1, h-1))).r;\n"
        "    float v11 = imageLoad(I1, min(base + ivec2(1, 1), ivec2(w-1, h-1))).r;\n"
        "    return (v00*w00 + v10*w10 + v01*w01 + v11*w11) * 255.0;\n"
        "}\n"
        "\n"
        "float computeSSD(vec2 U, ivec2 pos) {\n"
        "    float ssd = 0.0; float sum_diff = 0.0;\n"
        "    int psz2 = patch_size / 2;\n"
        "    for (int y = -psz2; y < psz2; ++y) {\n"
        "        for (int x = -psz2; x < psz2; ++x) {\n"
        "            ivec2 p = pos + ivec2(x, y);\n"
        "            float I0_val = imageLoad(I0, clamp(p, ivec2(0), ivec2(w-1,h-1))).r * 255.0;\n"
        "            float I1_val = sampleBilinear(vec2(p) + U);\n"
        "            float diff = I1_val - I0_val;\n"
        "            ssd += diff * diff; sum_diff += diff;\n"
        "        }\n"
        "    }\n"
        "    float N = float(patch_size * patch_size);\n"
        "    return ssd - (sum_diff * sum_diff) / N;\n"
        "}\n"
        "\n"
        "float processPatch(float dUx, float dUy, ivec2 pos, float x_grad_sum, float y_grad_sum, out float dx, out float dy) {\n"
        "    float SSD = 0.0; float sum_diff = 0.0; float b1 = 0.0; float b2 = 0.0;\n"
        "    int psz2 = patch_size / 2;\n"
        "    for (int y = -psz2; y < psz2; ++y) {\n"
        "        for (int x = -psz2; x < psz2; ++x) {\n"
        "            ivec2 p = pos + ivec2(x, y);\n"
        "            ivec2 p_clamp = clamp(p, ivec2(0), ivec2(w-1, h-1));\n"
        "            float I0_val = imageLoad(I0, p_clamp).r * 255.0;\n"
        "            float I1_val = sampleBilinear(vec2(p) + vec2(dUx, dUy));\n"
        "            float diff = I1_val - I0_val;\n"
        "            SSD += diff * diff; sum_diff += diff;\n"
        "            float ix = float(imageLoad(I0x, p_clamp).r);\n"
        "            float iy = float(imageLoad(I0y, p_clamp).r);\n"
        "            b1 += diff * ix; b2 += diff * iy;\n"
        "        }\n"
        "    }\n"
        "    float N = float(patch_size * patch_size);\n"
        "    float mean_diff = sum_diff / N;\n"
        "    dx = b1 - mean_diff * x_grad_sum;\n"
        "    dy = b2 - mean_diff * y_grad_sum;\n"
        "    SSD -= (sum_diff * sum_diff) / N;\n"
        "    return SSD;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    ivec2 patch_pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    int max_ws = (w + patch_stride - 1) / patch_stride;\n"
        "    int max_hs = (h + patch_stride - 1) / patch_stride;\n"
        "    if (patch_pos.x >= max_ws || patch_pos.y >= max_hs) return;\n"
        "\n"
        "    ivec2 pos = ivec2(patch_pos.x * patch_stride, patch_pos.y * patch_stride);\n"
        "    vec2 cur_U = imageLoad(U_flow_read, patch_pos).rg;\n"
        "    float min_SSD = computeSSD(cur_U, pos);\n"
        "\n"
        "    ivec2 neighbors[4] = ivec2[](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));\n"
        "    for(int n = 0; n < 4; n++) {\n"
        "        ivec2 n_pos = patch_pos + neighbors[n];\n"
        "        if(n_pos.x >= 0 && n_pos.y >= 0 && n_pos.x < max_ws && n_pos.y < max_hs) {\n"
        "            vec2 spatial_U = imageLoad(U_flow_read, n_pos).rg;\n"
        "            float spatial_SSD = computeSSD(spatial_U, pos);\n"
        "            if (spatial_SSD < min_SSD) {\n"
        "                min_SSD = spatial_SSD;\n"
        "                cur_U = spatial_U;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "\n"
        "    vec2 initial_U = cur_U;\n"
        "    float xx = imageLoad(xx_buf, patch_pos).r;\n"
        "    float yy = imageLoad(yy_buf, patch_pos).r;\n"
        "    float xy = imageLoad(xy_buf, patch_pos).r;\n"
        "    float detH = max(xx * yy - xy * xy, 0.001);\n"
        "    float invH11 = yy / detH; float invH12 = -xy / detH; float invH22 = xx / detH;\n"
        "    float x_grad_sum = imageLoad(x_buf, patch_pos).r;\n"
        "    float y_grad_sum = imageLoad(y_buf, patch_pos).r;\n"
        "\n"
        "    float dx, dy;\n"
        "    float current_SSD = processPatch(cur_U.x, cur_U.y, pos, x_grad_sum, y_grad_sum, dx, dy);\n"
        "    \n"
        "    // PROPER GRADIENT DESCENT FIX: Strictly reject bad diverging vectors!\n"
        "    for (int t = 0; t < num_inner_iter; t++) {\n"
        "        float dUx = invH11 * dx + invH12 * dy;\n"
        "        float dUy = invH12 * dx + invH22 * dy;\n"
        "        vec2 step_U = cur_U - vec2(dUx, dUy);\n"
        "        \n"
        "        float next_dx, next_dy;\n"
        "        float next_SSD = processPatch(step_U.x, step_U.y, pos, x_grad_sum, y_grad_sum, next_dx, next_dy);\n"
        "        \n"
        "        if (next_SSD >= current_SSD) break;\n"
        "        \n"
        "        cur_U = step_U;\n"
        "        current_SSD = next_SSD;\n"
        "        dx = next_dx; dy = next_dy;\n"
        "    }\n"
        "\n"
        "    if (distance(cur_U, initial_U) <= float(patch_size)) {\n"
        "        imageStore(U_flow_write, patch_pos, vec4(cur_U, 0.0, 0.0));\n"
        "    } else {\n"
        "        imageStore(U_flow_write, patch_pos, vec4(initial_U, 0.0, 0.0));\n"
        "    }\n"
        "}\n";

const char* DIS_UPSCALE_FLOW_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, rg32f) readonly uniform highp image2D inFlow;\n"
        "layout(binding = 1, rg32f) writeonly uniform highp image2D outFlow;\n"
        "void main() {\n"
        "    ivec2 outPos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    ivec2 outSize = imageSize(outFlow);\n"
        "    if (outPos.x >= outSize.x || outPos.y >= outSize.y) return;\n"
        "    ivec2 inPos = outPos / 2;\n"
        "    vec2 flow = imageLoad(inFlow, inPos).rg * 2.0;\n"
        "    imageStore(outFlow, outPos, vec4(flow, 0.0, 0.0));\n"
        "}\n";

const char* DIS_DENSIFICATION_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, rg32f) uniform readonly highp image2D sparse_flow;\n"
        "layout(binding = 1, r8) uniform readonly highp image2D I0;\n"
        "layout(binding = 2, r8) uniform readonly highp image2D I1;\n"
        "layout(binding = 3, rg32f) uniform writeonly highp image2D dense_flow;\n"
        "\n"
        "uniform int patch_size;\n"
        "uniform int patch_stride;\n"
        "uniform int w;\n"
        "uniform int h;\n"
        "\n"
        "float sampleBilinear(vec2 pos) {\n"
        "    vec2 p = clamp(pos, vec2(0.0), vec2(float(w - 1), float(h - 1)));\n"
        "    ivec2 base = ivec2(p);\n"
        "    vec2 f = fract(p);\n"
        "    float w11 = f.x * f.y; float w10 = f.x * (1.0 - f.y);\n"
        "    float w01 = (1.0 - f.x) * f.y; float w00 = (1.0 - f.x) * (1.0 - f.y);\n"
        "    float v00 = imageLoad(I1, base).r;\n"
        "    float v10 = imageLoad(I1, min(base + ivec2(1, 0), ivec2(w-1, h-1))).r;\n"
        "    float v01 = imageLoad(I1, min(base + ivec2(0, 1), ivec2(w-1, h-1))).r;\n"
        "    float v11 = imageLoad(I1, min(base + ivec2(1, 1), ivec2(w-1, h-1))).r;\n"
        "    return (v00*w00 + v10*w10 + v01*w01 + v11*w11) * 255.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    if (pixel_pos.x >= w || pixel_pos.y >= h) return;\n"
        "\n"
        "    float sum_Ux = 0.0; float sum_Uy = 0.0; float sum_weight = 0.0;\n"
        "    float I0_val = float(imageLoad(I0, pixel_pos).r) * 255.0;\n"
        "\n"
        "    int start_js = max(0, (pixel_pos.x - patch_size + patch_stride) / patch_stride);\n"
        "    int end_js = min((w + patch_stride - 1) / patch_stride - 1, pixel_pos.x / patch_stride);\n"
        "    int start_is = max(0, (pixel_pos.y - patch_size + patch_stride) / patch_stride);\n"
        "    int end_is = min((h + patch_stride - 1) / patch_stride - 1, pixel_pos.y / patch_stride);\n"
        "\n"
        "    for (int is = start_is; is <= end_is; is++) {\n"
        "        for (int js = start_js; js <= end_js; js++) {\n"
        "            vec2 U = imageLoad(sparse_flow, ivec2(js, is)).rg;\n"
        "            vec2 mapped_pos = vec2(pixel_pos) + U;\n"
        "            float I1_val = sampleBilinear(mapped_pos);\n"
        "            float diff = abs(I1_val - I0_val);\n"
        "            float weight = 1.0 / max(1.0, diff);\n"
        "            sum_Ux += U.x * weight;\n"
        "            sum_Uy += U.y * weight;\n"
        "            sum_weight += weight;\n"
        "        }\n"
        "    }\n"
        "    vec2 final_U = vec2(sum_Ux / sum_weight, sum_Uy / sum_weight);\n"
        "    imageStore(dense_flow, pixel_pos, vec4(final_U, 0.0, 0.0));\n"
        "}\n";

const char* DIS_VARIATIONAL_SHADER =
        "#version 310 es\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "layout(binding = 0, rg32f) uniform readonly highp image2D flow_in;\n"
        "layout(binding = 1, r8) uniform readonly highp image2D I0;\n"
        "layout(binding = 2, r8) uniform readonly highp image2D I1;\n"
        "layout(binding = 3, rg32f) uniform writeonly highp image2D flow_out;\n"
        "\n"
        "layout(std140, binding = 1) uniform VarParams {\n"
        "    int w;\n"
        "    int h;\n"
        "    int color_pass;\n"
        "    float alpha;\n"
        "} pc;\n"
        "\n"
        "float sampleBilinear(vec2 pos) {\n"
        "    vec2 p = clamp(pos, vec2(0.0), vec2(float(pc.w - 1), float(pc.h - 1)));\n"
        "    ivec2 base = ivec2(p);\n"
        "    vec2 f = fract(p);\n"
        "    float w11 = f.x * f.y; float w10 = f.x * (1.0 - f.y);\n"
        "    float w01 = (1.0 - f.x) * f.y; float w00 = (1.0 - f.x) * (1.0 - f.y);\n"
        "    float v00 = imageLoad(I1, base).r;\n"
        "    float v10 = imageLoad(I1, min(base + ivec2(1, 0), ivec2(pc.w-1, pc.h-1))).r;\n"
        "    float v01 = imageLoad(I1, min(base + ivec2(0, 1), ivec2(pc.w-1, pc.h-1))).r;\n"
        "    float v11 = imageLoad(I1, min(base + ivec2(1, 1), ivec2(pc.w-1, pc.h-1))).r;\n"
        "    return (v00*w00 + v10*w10 + v01*w01 + v11*w11) * 255.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
        "    if (pos.x >= pc.w || pos.y >= pc.h) return;\n"
        "\n"
        "    if ((pos.x + pos.y) % 2 != pc.color_pass) {\n"
        "        imageStore(flow_out, pos, vec4(imageLoad(flow_in, pos).rg, 0.0, 0.0));\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    vec2 u_c = imageLoad(flow_in, pos).rg;\n"
        "    vec2 u_t = pos.y > 0 ? imageLoad(flow_in, pos - ivec2(0, 1)).rg : u_c;\n"
        "    vec2 u_b = pos.y < pc.h - 1 ? imageLoad(flow_in, pos + ivec2(0, 1)).rg : u_c;\n"
        "    vec2 u_l = pos.x > 0 ? imageLoad(flow_in, pos - ivec2(1, 0)).rg : u_c;\n"
        "    vec2 u_r = pos.x < pc.w - 1 ? imageLoad(flow_in, pos + ivec2(1, 0)).rg : u_c;\n"
        "    vec2 u_avg = (u_t + u_b + u_l + u_r) * 0.25;\n"
        "\n"
        "    // CENTRAL DIFFERENCES FIX: Massively improves edge detection over forward differences\n"
        "    float i0_c = imageLoad(I0, pos).r * 255.0;\n"
        "    float i0_l = pos.x > 0 ? imageLoad(I0, pos - ivec2(1, 0)).r * 255.0 : i0_c;\n"
        "    float i0_r = pos.x < pc.w - 1 ? imageLoad(I0, pos + ivec2(1, 0)).r * 255.0 : i0_c;\n"
        "    float i0_t = pos.y > 0 ? imageLoad(I0, pos - ivec2(0, 1)).r * 255.0 : i0_c;\n"
        "    float i0_b = pos.y < pc.h - 1 ? imageLoad(I0, pos + ivec2(0, 1)).r * 255.0 : i0_c;\n"
        "    float ix = (i0_r - i0_l) * 0.5;\n"
        "    float iy = (i0_b - i0_t) * 0.5;\n"
        "\n"
        "    float i1_warp = sampleBilinear(vec2(pos) + u_c);\n"
        "    float it = i1_warp - i0_c;\n"
        "\n"
        "    float grad_mag = ix*ix + iy*iy;\n"
        "    float edge_weight = exp(-grad_mag / 50.0);\n"
        "    float local_alpha = pc.alpha * edge_weight + 5.0;\n"
        "\n"
        "    float num = ix * u_avg.x + iy * u_avg.y + it;\n"
        "    float den = local_alpha + ix * ix + iy * iy;\n"
        "    float du = ix * (num / den);\n"
        "    float dv = iy * (num / den);\n"
        "\n"
        "    vec2 u_new = u_avg - vec2(du, dv);\n"
        "    u_new = mix(u_c, u_new, 1.5); // Accelerated SOR convergence\n"
        "    imageStore(flow_out, pos, vec4(u_new, 0.0, 0.0));\n"
        "}\n";

// ============================================================================
// 2. RENDER SHADERS (THE "TWIXTOR MAGIC" BLEND)
// ============================================================================

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
        "uniform sampler2D uTexFlowForward;\n"
        "uniform sampler2D uTexFlowBackward;\n"
        "uniform float uT;\n"
        "uniform vec2 uTexSize;\n"
        "in vec2 vTexCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "  vec2 flowF = texture(uTexFlowForward, vTexCoord).rg;\n"
        "  vec2 flowB = texture(uTexFlowBackward, vTexCoord).rg;\n"
        "  \n"
        "  vec2 coordPrev = vTexCoord - flowF * uT / uTexSize;\n"
        "  vec2 coordCurr = vTexCoord - flowB * (1.0 - uT) / uTexSize;\n"
        "  \n"
        "  vec4 colorPrev = texture(uTexPrev, coordPrev);\n"
        "  vec4 colorCurr = texture(uTexCurr, coordCurr);\n"
        "  \n"
        "  // SOFT BI-DIRECTIONAL BLENDING (The magic fix! No more ghosting crossfades)\n"
        "  vec2 nextPosF = vTexCoord + (flowF / uTexSize);\n"
        "  float errF = length(flowF + texture(uTexFlowBackward, nextPosF).rg);\n"
        "  \n"
        "  vec2 prevPosB = vTexCoord + (flowB / uTexSize);\n"
        "  float errB = length(flowB + texture(uTexFlowForward, prevPosB).rg);\n"
        "  \n"
        "  // Convert error to a weight (lower error = exponentially higher weight)\n"
        "  float weightF = 1.0 / (1.0 + errF * errF);\n"
        "  float weightB = 1.0 / (1.0 + errB * errB);\n"
        "  \n"
        "  float W_Prev = (1.0 - uT) * weightF;\n"
        "  float W_Curr = uT * weightB;\n"
        "  float sumW = W_Prev + W_Curr + 0.0001; \n"
        "  \n"
        "  fragColor = (colorPrev * W_Prev + colorCurr * W_Curr) / sumW;\n"
        "}\n";

// ============================================================================
// 3. NATIVE PIPELINE SETUP
// ============================================================================

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
            std::vector<char> infoLog(infoLen);
            glGetShaderInfoLog(shader, infoLen, nullptr, infoLog.data());
            LOGE("Error compiling shader: %s", infoLog.data());
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

void setupStorageTex(GLuint tex, int w, int h, GLint internalFmt, GLint filter) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, internalFmt, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vsco_slomoapp_MainActivity_processVideo(
        JNIEnv* env, jobject thiz, jstring inputPath, jstring outputPath,
        jfloat factor, jint totalFrames, jint rotation) {

    const char* inPath = env->GetStringUTFChars(inputPath, nullptr);
    const char* outPath = env->GetStringUTFChars(outputPath, nullptr);

    jboolean result = JNI_FALSE;
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;

    // FIX: Variables moved to the top of the function to prevent out-of-scope errors during cleanup
    GLuint uboVar = 0;
    GLuint uboDIS = 0;

    try {
        cap.open(inPath);
        if (!cap.isOpened()) throw std::runtime_error("Failed to open input video");

        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int videoFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        if (videoFrames <= 0) videoFrames = totalFrames;

        int targetWidth = (rotation == 90 || rotation == 270) ? height : width;
        int targetHeight = (rotation == 90 || rotation == 270) ? width : height;

        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        writer.open(outPath, fourcc, fps, cv::Size(targetWidth, targetHeight));
        if (!writer.isOpened()) {
            fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            writer.open(outPath, fourcc, fps, cv::Size(targetWidth, targetHeight));
        }

        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);
        EGLint configAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_NONE };
        EGLConfig config; EGLint numConfigs;
        eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
        EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
        EGLint pbufferAttribs[] = { EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE };
        surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        eglMakeCurrent(display, surface, surface, context);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        // Compile Compute Programs
        GLuint progClear = glCreateProgram(); glAttachShader(progClear, createShader(GL_COMPUTE_SHADER, DIS_CLEAR_SHADER)); glLinkProgram(progClear);
        GLuint progCopyFlow = glCreateProgram(); glAttachShader(progCopyFlow, createShader(GL_COMPUTE_SHADER, DIS_COPY_SHADER)); glLinkProgram(progCopyFlow);
        GLuint progDownscale = glCreateProgram(); glAttachShader(progDownscale, createShader(GL_COMPUTE_SHADER, DIS_DOWNSCALE_SHADER)); glLinkProgram(progDownscale);
        GLuint progGrad = glCreateProgram(); glAttachShader(progGrad, createShader(GL_COMPUTE_SHADER, DIS_GRADIENT_SHADER)); glLinkProgram(progGrad);
        GLuint progPrecomp = glCreateProgram(); glAttachShader(progPrecomp, createShader(GL_COMPUTE_SHADER, DIS_PRECOMPUTE_SHADER)); glLinkProgram(progPrecomp);
        GLuint progSearch = glCreateProgram(); glAttachShader(progSearch, createShader(GL_COMPUTE_SHADER, DIS_PATCH_SEARCH_SHADER)); glLinkProgram(progSearch);
        GLuint progUpscaleFlow = glCreateProgram(); glAttachShader(progUpscaleFlow, createShader(GL_COMPUTE_SHADER, DIS_UPSCALE_FLOW_SHADER)); glLinkProgram(progUpscaleFlow);
        GLuint progDense = glCreateProgram(); glAttachShader(progDense, createShader(GL_COMPUTE_SHADER, DIS_DENSIFICATION_SHADER)); glLinkProgram(progDense);
        GLuint progVariational = glCreateProgram(); glAttachShader(progVariational, createShader(GL_COMPUTE_SHADER, DIS_VARIATIONAL_SHADER)); glLinkProgram(progVariational);

        // Compile Render Program
        GLuint progRender = glCreateProgram();
        glAttachShader(progRender, createShader(GL_VERTEX_SHADER, VERTEX_SHADER));
        glAttachShader(progRender, createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER));
        glLinkProgram(progRender);

        // ==========================================
        // CUSTOM ULTRA PRESET (Pushing beyond OpenCV)
        // ==========================================
        int patch_size = 8;
        int patch_stride = 3;
        int finest_scale = 0;
        int grad_descent_iter = 32;

        // GPU SPATIAL PROPAGATION FIX: Use 8 ping pongs to sweep the full image, preventing large motion tearing
        int ping_pong_passes = 8;
        int num_inner_iter = std::max(1, grad_descent_iter / ping_pong_passes);
        int variational_iterations = 20; // Massive GPU edge-snapping passes

        int max_possible_scale = std::min(
                (int)(std::log(std::max(width, height) / (4.0 * patch_size)) / std::log(2.0) + 0.5),
                (int)(std::log(std::min(width, height) / patch_size) / std::log(2.0))
        );
        int coarsest_scale = max_possible_scale;
        int total_levels = coarsest_scale + 1;

        std::vector<GLuint> texI0_pyr(total_levels);
        std::vector<GLuint> texI1_pyr(total_levels);
        std::vector<GLuint> texFlowA_pyr(total_levels);
        std::vector<GLuint> texFlowB_pyr(total_levels);

        glGenTextures(total_levels, texI0_pyr.data());
        glGenTextures(total_levels, texI1_pyr.data());
        glGenTextures(total_levels, texFlowA_pyr.data());
        glGenTextures(total_levels, texFlowB_pyr.data());

        for (int i = 0; i <= coarsest_scale; i++) {
            int curW = std::max(1, width >> i);
            int curH = std::max(1, height >> i);
            int gridW = (curW + patch_stride - 1) / patch_stride;
            int gridH = (curH + patch_stride - 1) / patch_stride;

            setupStorageTex(texI0_pyr[i], curW, curH, GL_R8, GL_LINEAR);
            setupStorageTex(texI1_pyr[i], curW, curH, GL_R8, GL_LINEAR);
            setupStorageTex(texFlowA_pyr[i], gridW, gridH, GL_RG32F, GL_NEAREST);
            setupStorageTex(texFlowB_pyr[i], gridW, gridH, GL_RG32F, GL_NEAREST);
        }

        int fullGridW = (width + patch_stride - 1) / patch_stride;
        int fullGridH = (height + patch_stride - 1) / patch_stride;
        GLuint texI0x, texI0y, texXX, texYY, texXY, texXSum, texYSum, texFlowDenseF, texFlowDenseB, texFlowDenseTemp, texRGB_Prev, texRGB_Curr;

        glGenTextures(1, &texI0x); setupStorageTex(texI0x, width, height, GL_R16I, GL_NEAREST);
        glGenTextures(1, &texI0y); setupStorageTex(texI0y, width, height, GL_R16I, GL_NEAREST);
        glGenTextures(1, &texXX); setupStorageTex(texXX, fullGridW, fullGridH, GL_R32F, GL_NEAREST);
        glGenTextures(1, &texYY); setupStorageTex(texYY, fullGridW, fullGridH, GL_R32F, GL_NEAREST);
        glGenTextures(1, &texXY); setupStorageTex(texXY, fullGridW, fullGridH, GL_R32F, GL_NEAREST);
        glGenTextures(1, &texXSum); setupStorageTex(texXSum, fullGridW, fullGridH, GL_R32F, GL_NEAREST);
        glGenTextures(1, &texYSum); setupStorageTex(texYSum, fullGridW, fullGridH, GL_R32F, GL_NEAREST);

        int finestW = std::max(1, width >> finest_scale);
        int finestH = std::max(1, height >> finest_scale);
        glGenTextures(1, &texFlowDenseF); setupStorageTex(texFlowDenseF, finestW, finestH, GL_RG32F, GL_LINEAR);
        glGenTextures(1, &texFlowDenseB); setupStorageTex(texFlowDenseB, finestW, finestH, GL_RG32F, GL_LINEAR);
        glGenTextures(1, &texFlowDenseTemp); setupStorageTex(texFlowDenseTemp, finestW, finestH, GL_RG32F, GL_LINEAR);

        glGenTextures(1, &texRGB_Prev); setupStorageTex(texRGB_Prev, width, height, GL_RGB8, GL_LINEAR);
        glGenTextures(1, &texRGB_Curr); setupStorageTex(texRGB_Curr, width, height, GL_RGB8, GL_LINEAR);

        struct VarParams { int w; int h; int color_pass; float alpha; };
        glGenBuffers(1, &uboVar);
        glBindBuffer(GL_UNIFORM_BUFFER, uboVar);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(VarParams), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboVar);

        float vertices[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
        float texCoords[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
        GLint posLoc = glGetAttribLocation(progRender, "aPosition");
        GLint texLoc = glGetAttribLocation(progRender, "aTexCoord");

        jclass clazz = env->GetObjectClass(thiz);
        jmethodID progressMethodId = env->GetMethodID(clazz, "onProgressUpdate", "(F)V");
        jmethodID cancelMethodId = env->GetMethodID(clazz, "isCancelled", "()Z");

        cv::Mat prevFrame, currFrame, prevGray, currGray;
        cap >> prevFrame;

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

        // Subroutine to run optical flow engine for a specific direction
        auto computeFlowDirection = [&](bool isForward, GLuint targetDenseFlow) {
            glUseProgram(progDownscale);
            for (int level = 0; level < coarsest_scale; ++level) {
                int dstW = std::max(1, width >> (level + 1));
                int dstH = std::max(1, height >> (level + 1));

                // Swap I0 and I1 source textures based on direction
                GLuint imgA = isForward ? texI0_pyr[level] : texI1_pyr[level];
                GLuint imgA_next = isForward ? texI0_pyr[level + 1] : texI1_pyr[level + 1];
                GLuint imgB = isForward ? texI1_pyr[level] : texI0_pyr[level];
                GLuint imgB_next = isForward ? texI1_pyr[level + 1] : texI0_pyr[level + 1];

                glBindImageTexture(0, imgA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                glBindImageTexture(1, imgA_next, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);
                glDispatchCompute((dstW + 15) / 16, (dstH + 15) / 16, 1);

                glBindImageTexture(0, imgB, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                glBindImageTexture(1, imgB_next, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);
                glDispatchCompute((dstW + 15) / 16, (dstH + 15) / 16, 1);
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            }

            int cGridW = ((width >> coarsest_scale) + patch_stride - 1) / patch_stride;
            int cGridH = ((height >> coarsest_scale) + patch_stride - 1) / patch_stride;
            glUseProgram(progClear);
            glBindImageTexture(0, texFlowA_pyr[coarsest_scale], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
            glDispatchCompute((cGridW + 15) / 16, (cGridH + 15) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            for (int level = coarsest_scale; level >= finest_scale; level--) {
                int curW = std::max(1, width >> level);
                int curH = std::max(1, height >> level);
                int gridW = (curW + patch_stride - 1) / patch_stride;
                int gridH = (curH + patch_stride - 1) / patch_stride;

                GLuint currentI0 = isForward ? texI0_pyr[level] : texI1_pyr[level];
                GLuint currentI1 = isForward ? texI1_pyr[level] : texI0_pyr[level];

                glUseProgram(progGrad);
                glBindImageTexture(0, currentI0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                glBindImageTexture(1, texI0x, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16I);
                glBindImageTexture(2, texI0y, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16I);
                glDispatchCompute((curW + 15) / 16, (curH + 15) / 16, 1);
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

                glUseProgram(progPrecomp);
                glBindImageTexture(0, texI0x, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16I);
                glBindImageTexture(1, texI0y, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16I);
                glBindImageTexture(2, texXX, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
                glBindImageTexture(3, texYY, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
                glBindImageTexture(4, texXY, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
                glBindImageTexture(5, texXSum, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
                glBindImageTexture(6, texYSum, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
                glUniform1i(glGetUniformLocation(progPrecomp, "patch_size"), patch_size);
                glUniform1i(glGetUniformLocation(progPrecomp, "patch_stride"), patch_stride);
                glUniform1i(glGetUniformLocation(progPrecomp, "w"), curW);
                glUniform1i(glGetUniformLocation(progPrecomp, "h"), curH);
                glDispatchCompute((gridW + 15) / 16, (gridH + 15) / 16, 1);
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

                glUseProgram(progSearch);
                glBindImageTexture(0, currentI0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                glBindImageTexture(1, currentI1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                glBindImageTexture(2, texI0x, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16I);
                glBindImageTexture(3, texI0y, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16I);
                glBindImageTexture(4, texXX, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(5, texYY, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(6, texXY, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(7, texXSum, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glBindImageTexture(8, texYSum, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
                glUniform1i(glGetUniformLocation(progSearch, "patch_size"), patch_size);
                glUniform1i(glGetUniformLocation(progSearch, "patch_stride"), patch_stride);
                glUniform1i(glGetUniformLocation(progSearch, "num_inner_iter"), num_inner_iter);
                glUniform1i(glGetUniformLocation(progSearch, "w"), curW);
                glUniform1i(glGetUniformLocation(progSearch, "h"), curH);

                for(int pass = 0; pass < ping_pong_passes; pass++) {
                    GLuint readTex = (pass % 2 == 0) ? texFlowA_pyr[level] : texFlowB_pyr[level];
                    GLuint writeTex = (pass % 2 == 0) ? texFlowB_pyr[level] : texFlowA_pyr[level];
                    glBindImageTexture(9, readTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
                    glBindImageTexture(10, writeTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
                    glDispatchCompute((gridW + 15) / 16, (gridH + 15) / 16, 1);
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                }

                if (level > finest_scale) {
                    int nextLevel = level - 1;
                    int nextGridW = ((width >> nextLevel) + patch_stride - 1) / patch_stride;
                    int nextGridH = ((height >> nextLevel) + patch_stride - 1) / patch_stride;
                    glUseProgram(progUpscaleFlow);
                    GLuint finalSparseTex = (ping_pong_passes % 2 == 0) ? texFlowA_pyr[level] : texFlowB_pyr[level];
                    glBindImageTexture(0, finalSparseTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
                    glBindImageTexture(1, texFlowA_pyr[nextLevel], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
                    glDispatchCompute((nextGridW + 15) / 16, (nextGridH + 15) / 16, 1);
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                }
            }

            glUseProgram(progDense);
            GLuint finalSparseTexFinest = (ping_pong_passes % 2 == 0) ? texFlowA_pyr[finest_scale] : texFlowB_pyr[finest_scale];
            GLuint currentI0 = isForward ? texI0_pyr[finest_scale] : texI1_pyr[finest_scale];
            GLuint currentI1 = isForward ? texI1_pyr[finest_scale] : texI0_pyr[finest_scale];

            glBindImageTexture(0, finalSparseTexFinest, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
            glBindImageTexture(1, currentI0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
            glBindImageTexture(2, currentI1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
            glBindImageTexture(3, targetDenseFlow, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
            glUniform1i(glGetUniformLocation(progDense, "patch_size"), patch_size);
            glUniform1i(glGetUniformLocation(progDense, "patch_stride"), patch_stride);
            glUniform1i(glGetUniformLocation(progDense, "w"), finestW);
            glUniform1i(glGetUniformLocation(progDense, "h"), finestH);
            glDispatchCompute((finestW + 15) / 16, (finestH + 15) / 16, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

            glUseProgram(progVariational);
            GLuint current_dense = targetDenseFlow;
            GLuint next_dense = texFlowDenseTemp;

            for (int iter = 0; iter < variational_iterations; iter++) {
                for (int color = 0; color < 2; color++) {
                    VarParams vParams = { finestW, finestH, color, 20.0f };
                    glBindBuffer(GL_UNIFORM_BUFFER, uboVar);
                    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(VarParams), &vParams);

                    glBindImageTexture(0, current_dense, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
                    glBindImageTexture(1, currentI0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                    glBindImageTexture(2, currentI1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
                    glBindImageTexture(3, next_dense, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

                    glDispatchCompute((finestW + 15) / 16, (finestH + 15) / 16, 1);
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

                    std::swap(current_dense, next_dense);
                }
            }

            if (current_dense != targetDenseFlow) {
                glUseProgram(progCopyFlow);
                glBindImageTexture(0, current_dense, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
                glBindImageTexture(1, targetDenseFlow, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
                glDispatchCompute((finestW + 15) / 16, (finestH + 15) / 16, 1);
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            }
        };

        int frameCount = 1;
        while (true) {
            if (cancelMethodId != nullptr && env->CallBooleanMethod(thiz, cancelMethodId)) break;
            cap >> currFrame; if (currFrame.empty()) break;
            cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

            glBindTexture(GL_TEXTURE_2D, texI0_pyr[0]); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, prevGray.data);
            glBindTexture(GL_TEXTURE_2D, texI1_pyr[0]); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, currGray.data);

            // Compute both directions required for bi-directional masking
            computeFlowDirection(true, texFlowDenseF);
            computeFlowDirection(false, texFlowDenseB);

            // RENDER & INTERPOLATE
            glUseProgram(progRender);
            glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, vertices);
            glEnableVertexAttribArray(posLoc);
            glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
            glEnableVertexAttribArray(texLoc);
            glUniform2f(glGetUniformLocation(progRender, "uTexSize"), (float)width, (float)height);

            glBindTexture(GL_TEXTURE_2D, texRGB_Prev); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, prevFrame.data);
            glBindTexture(GL_TEXTURE_2D, texRGB_Curr); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, currFrame.data);

            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texRGB_Prev); glUniform1i(glGetUniformLocation(progRender, "uTexPrev"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texRGB_Curr); glUniform1i(glGetUniformLocation(progRender, "uTexCurr"), 1);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, texFlowDenseF); glUniform1i(glGetUniformLocation(progRender, "uTexFlowForward"), 2);
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, texFlowDenseB); glUniform1i(glGetUniformLocation(progRender, "uTexFlowBackward"), 3);

            int numIntermediate = (int)(1.0f / factor) - 1;
            for (int i = 1; i <= numIntermediate; ++i) {
                float t = (float)i / (numIntermediate + 1);
                glUniform1f(glGetUniformLocation(progRender, "uT"), t);
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

    } catch (const std::exception& e) { LOGE("Pipeline Error: %s", e.what()); }

    if (display != EGL_NO_DISPLAY) { eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); eglTerminate(display); }
    if (uboVar) glDeleteBuffers(1, &uboVar);
    if (uboDIS) glDeleteBuffers(1, &uboDIS);

    if (cap.isOpened()) cap.release();
    if (writer.isOpened()) writer.release();
    env->ReleaseStringUTFChars(inputPath, inPath); env->ReleaseStringUTFChars(outputPath, outPath);

    return result;
}