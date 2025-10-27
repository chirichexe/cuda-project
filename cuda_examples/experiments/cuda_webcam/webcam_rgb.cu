#include <iostream>
#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>

#define WIDTH 640
#define HEIGHT 480

// --- Kernel CUDA: converte YUYV -> RGB ---
__global__ void yuyv_to_rgb_kernel(unsigned char* yuyv, unsigned char* rgb, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 2;
    int frame_size = width * height * 2;
    if (idx + 3 >= frame_size) return;

    unsigned char Y = yuyv[idx];
    unsigned char U = (x & 1) ? yuyv[idx - 1] : yuyv[idx + 1];
    unsigned char V = (x & 1) ? yuyv[idx + 1] : yuyv[idx + 3];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int r = (298 * C + 409 * E + 128) >> 8;
    int g = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int b = (298 * C + 516 * D + 128) >> 8;

    unsigned char R = r < 0 ? 0 : (r > 255 ? 255 : r);
    unsigned char G = g < 0 ? 0 : (g > 255 ? 255 : g);
    unsigned char B = b < 0 ? 0 : (b > 255 ? 255 : b);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx + 0] = R;
    rgb[rgb_idx + 1] = G;
    rgb[rgb_idx + 2] = B;
}

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if (fd < 0) { perror("Errore apertura webcam"); return 1; }

    // --- Imposta formato ---
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("Errore impostazione formato"); return 1; }

    // --- Richiesta buffer mmap ---
    struct v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("Errore richiesta buffer"); return 1; }

    unsigned char* buffers[req.count];
    struct v4l2_buffer buf;
    for(int i = 0; i < req.count; i++){
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if(ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0){ perror("Errore query buffer"); return 1; }
        buffers[i] = (unsigned char*)mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if(buffers[i] == MAP_FAILED){ perror("Errore mmap"); return 1; }
        if(ioctl(fd, VIDIOC_QBUF, &buf) < 0){ perror("Errore queue buffer"); return 1; }
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMON, &type) < 0){ perror("Errore stream on"); return 1; }

    size_t rgb_size = WIDTH * HEIGHT * 3;
    unsigned char* d_yuyv;
    unsigned char* d_rgb;
    cudaMalloc(&d_yuyv, WIDTH * HEIGHT * 2);
    cudaMalloc(&d_rgb, rgb_size);

    cv::namedWindow("Webcam RGB", cv::WINDOW_AUTOSIZE);

    while(true){
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(ioctl(fd, VIDIOC_DQBUF, &buf) < 0){ perror("Errore lettura frame"); break; }

        cudaMemcpy(d_yuyv, buffers[buf.index], WIDTH * HEIGHT * 2, cudaMemcpyHostToDevice);

        dim3 block(16,16);
        dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);
        yuyv_to_rgb_kernel<<<grid, block>>>(d_yuyv, d_rgb, WIDTH, HEIGHT);

        unsigned char* rgb_host = new unsigned char[rgb_size];
        cudaMemcpy(rgb_host, d_rgb, rgb_size, cudaMemcpyDeviceToHost);

        cv::Mat img(HEIGHT, WIDTH, CV_8UC3, rgb_host);
        cv::imshow("Webcam RGB", img);

        if(cv::waitKey(1) == 27) { delete[] rgb_host; break; }
        delete[] rgb_host;

        if(ioctl(fd, VIDIOC_QBUF, &buf) < 0){ perror("Errore riqueue buffer"); break; }
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for(int i=0;i<req.count;i++) munmap(buffers[i], buf.length);
    close(fd);
    cudaFree(d_yuyv);
    cudaFree(d_rgb);

    return 0;
}

