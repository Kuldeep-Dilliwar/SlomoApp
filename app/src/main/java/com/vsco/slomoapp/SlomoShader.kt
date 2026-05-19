package com.vsco.slomoapp

import android.opengl.GLES30

object SlomoShader {
    private const val VERTEX_SHADER = """#version 300 es
        in vec4 aPosition;
        in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord;
        }
    """

    private const val FRAGMENT_SHADER = """#version 300 es
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump float;
        
        uniform samplerExternalOES uTexPrev;
        uniform samplerExternalOES uTexCurr;
        uniform float uT;
        
        in vec2 vTexCoord;
        out vec4 fragColor;
        
        void main() {
            vec4 colorPrev = texture(uTexPrev, vTexCoord);
            vec4 colorCurr = texture(uTexCurr, vTexCoord);
            
            // Standard GPU frame blending
            fragColor = mix(colorPrev, colorCurr, uT);
        }
    """

    fun createProgram(): Int {
        val vertexShader = loadShader(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fragmentShader = loadShader(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        return program
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, shaderCode)
        GLES30.glCompileShader(shader)
        return shader
    }
}