#include <iostream>
#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>

#define WIDTH 640
#define HEIGHT 480
#define SCALE 8
#define ASCII_LEN 10
const char* ASCII_CHARS = " .:-=+*#%@";

__global__ void yuyv_to_ascii_kernel(unsigned char* yuyv, unsigned char* ascii_idx, int width, int height, int scale)
{
    __shared__ unsigned char tile[32*32*2]; // tile di YUYV in shared memory
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_width = width / scale;
    int out_height = height / scale;

    if(out_col >= out_width || out_row >= out_height) return;

    int x = out_col * scale;
    int y = out_row * scale;
    int idx = (y*width + x)*2;
    if(idx >= width*height*2) return;

    unsigned char Y = yuyv[idx];
    unsigned char U = (x & 1) ? yuyv[idx-1] : yuyv[idx+1];
    unsigned char V = (x & 1) ? yuyv[idx+1] : yuyv[idx+3];

    int C = Y-16;
    int D = U-128;
    int E = V-128;

    int r = (298*C + 409*E + 128)>>8;
    int g = (298*C - 100*D -208*E +128)>>8;
    int b = (298*C + 516*D +128)>>8;

    r = r<0?0:(r>255?255:r);
    g = g<0?0:(g>255?255:g);
    b = b<0?0:(b>255?255:b);

    int gray = (r+g+b)/3;
    int ascii_index = gray*ASCII_LEN/256;

    ascii_idx[out_row*out_width + out_col] = ascii_index;
}

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if(fd<0){ perror("Errore apertura webcam"); return 1; }

    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if(ioctl(fd, VIDIOC_S_FMT, &fmt)<0){ perror("Errore impostazione formato"); return 1; }

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

    int out_width = WIDTH/SCALE;
    int out_height = HEIGHT/SCALE;
    size_t ascii_size = out_width*out_height;

    unsigned char* d_yuyv[2];
    unsigned char* d_ascii[2];
    cudaMalloc(&d_yuyv[0], WIDTH*HEIGHT*2);
    cudaMalloc(&d_yuyv[1], WIDTH*HEIGHT*2);
    cudaMalloc(&d_ascii[0], ascii_size);
    cudaMalloc(&d_ascii[1], ascii_size);
    unsigned char* ascii_host = new unsigned char[ascii_size];

    dim3 block(32,32);
    dim3 grid((out_width+block.x-1)/block.x,(out_height+block.y-1)/block.y);

    int toggle = 0;
    int frames = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while(frames<500){ // 500 frame benchmark
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(ioctl(fd, VIDIOC_DQBUF, &buf)<0){ perror("Errore lettura frame"); break; }

        // copia asincrona con doppio buffer
        cudaMemcpyAsync(d_yuyv[toggle], buffers[buf.index], WIDTH*HEIGHT*2, cudaMemcpyHostToDevice);

        yuyv_to_ascii_kernel<<<grid,block>>>(d_yuyv[toggle], d_ascii[toggle], WIDTH, HEIGHT, SCALE);

        // copia del buffer precedente mentre GPU calcola il nuovo
        if(frames>0) cudaMemcpyAsync(ascii_host,d_ascii[1-toggle],ascii_size,cudaMemcpyDeviceToHost);
        
        // print

        // Stampa a terminale
        /*
        std::cout << "\033[H\033[J"; // pulisce terminale
        for(int i=0;i<HEIGHT/SCALE;i++){
            for(int j=0;j<WIDTH/SCALE;j++){
                std::cout << ASCII_CHARS[ascii_host[i*out_width + j]];
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
        */
        // end print

        toggle = 1-toggle;
        frames++;
        if(ioctl(fd, VIDIOC_QBUF, &buf)<0){ perror("Errore riqueue buffer"); break; }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double,std::milli>(end-start).count();
    std::cout << "GPU ottimizzata: tempo medio per frame = " << elapsed/frames << " ms\n";
    std::cout << "FPS teorico massimo = " << 1000.0/(elapsed/frames) << std::endl;

    ioctl(fd, VIDIOC_STREAMOFF,&type);
    for(int i=0;i<req.count;i++) munmap(buffers[i],buf.length);
    close(fd);
    cudaFree(d_yuyv[0]);
    cudaFree(d_yuyv[1]);
    cudaFree(d_ascii[0]);
    cudaFree(d_ascii[1]);
    delete[] ascii_host;
    return 0;
}

