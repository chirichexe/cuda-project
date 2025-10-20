#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaEGL.h>
#include <iostream>

const int width = 512;
const int height = 512;

GLuint pbo;
cudaGraphicsResource* cudaPBO;

// Kernel CUDA minimale: colore dinamico
__global__ void fillColor(uchar4* ptr, int width, int height, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    unsigned char r = static_cast<unsigned char>((sinf(t) * 0.5f + 0.5f) * 255);
    unsigned char g = static_cast<unsigned char>((cosf(t) * 0.5f + 0.5f) * 255);
    ptr[idx] = make_uchar4(r, g, 0, 255);
}

// Controllo errori CUDA
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Controllo errori EGL
void checkEGL(EGLBoolean ok, const char* msg) {
    if (ok != EGL_TRUE) {
        std::cerr << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // --- Inizializza EGL ---
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDpy == EGL_NO_DISPLAY) { std::cerr << "Errore EGL get display\n"; return -1; }
    checkEGL(eglInitialize(eglDpy, nullptr, nullptr), "Errore EGL initialize");

    // --- Config EGL ---
    EGLConfig config;
    EGLint numConfig;
    EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE,8,
        EGL_GREEN_SIZE,8,
        EGL_BLUE_SIZE,8,
        EGL_ALPHA_SIZE,8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    checkEGL(eglChooseConfig(eglDpy, attribs, &config, 1, &numConfig), "Errore EGL choose config");

    // --- Crea superficie PBuffer ---
    EGLint pbufAttribs[] = { EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE };
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, config, pbufAttribs);
    if (eglSurf == EGL_NO_SURFACE) { std::cerr << "Errore creazione superficie EGL\n"; return -1; }

    // --- Crea contesto EGL ES2 ---
    EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext eglCtx = eglCreateContext(eglDpy, config, EGL_NO_CONTEXT, ctxAttribs);
    if (eglCtx == EGL_NO_CONTEXT) { std::cerr << "Errore creazione contesto EGL\n"; return -1; }

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

    // --- Creazione PBO OpenGL ES ---
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // --- Registrazione PBO con CUDA EGL ---
    checkCuda(cudaGraphicsEGLRegisterImage(&cudaPBO, (void*)pbo, cudaGraphicsRegisterFlagsWriteDiscard),
              "Errore registrazione PBO CUDA EGL");

    float t = 0.0f;
    while (true) { // loop infinito, Ctrl+C per uscire
        // --- Mappa PBO ---
        checkCuda(cudaGraphicsMapResources(1, &cudaPBO, 0), "Errore map PBO CUDA");
        uchar4* d_ptr;
        size_t num_bytes;
        checkCuda(cudaGraphicsSubResourceGetMappedArray((cudaArray**)&d_ptr, cudaPBO, 0, 0),
                  "Errore get mapped pointer");

        // --- Kernel CUDA ---
        dim3 block(16,16);
        dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
        fillColor<<<grid, block>>>(d_ptr, width, height, t);
        checkCuda(cudaGetLastError(), "Errore kernel CUDA");
        checkCuda(cudaDeviceSynchronize(), "Errore sincronizzazione CUDA");

        // --- Unmap PBO ---
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPBO, 0), "Errore unmap PBO CUDA");

        // --- Rendering minimale con GLES ---
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, pbo);
        glDrawArrays(GL_POINTS, 0, width*height);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        eglSwapBuffers(eglDpy, eglSurf);
        t += 0.05f;
    }

    // --- Cleanup ---
    checkCuda(cudaGraphicsUnregisterResource(cudaPBO), "Errore unregister PBO CUDA");
    glDeleteBuffers(1, &pbo);
    eglDestroyContext(eglDpy, eglCtx);
    eglDestroySurface(eglDpy, eglSurf);
    eglTerminate(eglDpy);
}

