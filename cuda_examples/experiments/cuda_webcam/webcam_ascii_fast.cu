#include <iostream>
#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>

#define WIDTH 640
#define HEIGHT 480
#define SCALE 8    // riduzione per ASCII
#define ASCII_LEN 10
const char* ASCII_CHARS = " .:-=+*#%@";

// --- Kernel CUDA: YUYV -> grigio -> indice ASCII ---
__global__ void yuyv_to_ascii_kernel(unsigned char* yuyv, unsigned char* ascii_idx, int width, int height, int scale)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_width = width / scale;
    int out_height = height / scale;

    if(col >= out_width || row >= out_height) return;

    int x = col * scale;
    int y = row * scale;

    int idx = (y * width + x) * 2;
    if(idx >= width*height*2) return;

    unsigned char Y = yuyv[idx];
    unsigned char U = (x & 1) ? yuyv[idx-1] : yuyv[idx+1];
    unsigned char V = (x & 1) ? yuyv[idx+1] : yuyv[idx+3];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int r = (298*C + 409*E + 128) >> 8;
    int g = (298*C - 100*D - 208*E + 128) >> 8;
    int b = (298*C + 516*D + 128) >> 8;

    r = r<0 ? 0 : (r>255 ? 255 : r);
    g = g<0 ? 0 : (g>255 ? 255 : g);
    b = b<0 ? 0 : (b>255 ? 255 : b);

    int gray = (r+g+b)/3;
    int ascii_index = gray * ASCII_LEN / 256;
    ascii_idx[row*out_width + col] = ascii_index;
}

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if(fd<0){ perror("Errore apertura webcam"); return 1; }

    // Formato YUYV 640x480
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if(ioctl(fd, VIDIOC_S_FMT, &fmt)<0){ perror("Errore impostazione formato"); return 1; }

    // Buffer mmap
    struct v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if(ioctl(fd, VIDIOC_REQBUFS, &req)<0){ perror("Errore richiesta buffer"); return 1; }

    unsigned char* buffers[req.count];
    struct v4l2_buffer buf;
    for(int i=0;i<req.count;i++){
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if(ioctl(fd, VIDIOC_QUERYBUF, &buf)<0){ perror("Errore query buffer"); return 1; }
        buffers[i] = (unsigned char*)mmap(NULL, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if(buffers[i]==MAP_FAILED){ perror("Errore mmap"); return 1; }
        if(ioctl(fd, VIDIOC_QBUF, &buf)<0){ perror("Errore queue buffer"); return 1; }
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMON, &type)<0){ perror("Errore stream on"); return 1; }

    size_t ascii_size = (WIDTH/SCALE)*(HEIGHT/SCALE);
    unsigned char* d_yuyv;
    unsigned char* d_ascii;
    cudaMalloc(&d_yuyv, WIDTH*HEIGHT*2);
    cudaMalloc(&d_ascii, ascii_size);

    unsigned char* ascii_host = new unsigned char[ascii_size];

    dim3 block(16,16);
    dim3 grid((WIDTH/SCALE+block.x-1)/block.x, (HEIGHT/SCALE+block.y-1)/block.y);

    while(true){
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(ioctl(fd, VIDIOC_DQBUF, &buf)<0){ perror("Errore lettura frame"); break; }

        cudaMemcpy(d_yuyv, buffers[buf.index], WIDTH*HEIGHT*2, cudaMemcpyHostToDevice);
        yuyv_to_ascii_kernel<<<grid, block>>>(d_yuyv, d_ascii, WIDTH, HEIGHT, SCALE);
        cudaMemcpy(ascii_host, d_ascii, ascii_size, cudaMemcpyDeviceToHost);

        int out_width = WIDTH/SCALE;
        int out_height = HEIGHT/SCALE;

        std::cout << "\033[H"; // reset cursore
        for(int i=0;i<out_height;i++){
            for(int j=0;j<out_width;j++){
                std::cout << ASCII_CHARS[ascii_host[i*out_width + j]];
            }
            std::cout << "\n";
        }
        std::cout << std::flush;

        if(ioctl(fd, VIDIOC_QBUF, &buf)<0){ perror("Errore riqueue buffer"); break; }
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for(int i=0;i<req.count;i++) munmap(buffers[i], buf.length);
    close(fd);
    cudaFree(d_yuyv);
    cudaFree(d_ascii);
    delete[] ascii_host;

    return 0;
}

