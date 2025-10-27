#include <iostream>
#include <chrono>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>

#define WIDTH 640
#define HEIGHT 480
#define SCALE 8
#define ASCII_LEN 10
const char* ASCII_CHARS = " .:-=+*#%@";

// --- Convert YUYV -> ASCII su CPU ---
unsigned char yuyv_to_ascii(unsigned char* yuyv, int x, int y) {
    int idx = (y*WIDTH + x)*2;
    unsigned char Y = yuyv[idx];
    unsigned char U = (x & 1) ? yuyv[idx-1] : yuyv[idx+1];
    unsigned char V = (x & 1) ? yuyv[idx+1] : yuyv[idx+3];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int r = (298*C + 409*E + 128) >> 8;
    int g = (298*C - 100*D - 208*E + 128) >> 8;
    int b = (298*C + 516*D + 128) >> 8;

    r = r<0?0:(r>255?255:r);
    g = g<0?0:(g>255?255:g);
    b = b<0?0:(b>255?255:b);

    int gray = (r+g+b)/3;
    int idx_ascii = gray*ASCII_LEN/256;
    return ASCII_CHARS[idx_ascii];
}

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if(fd < 0){ perror("Errore apertura webcam"); return 1; }

    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if(ioctl(fd, VIDIOC_S_FMT, &fmt) < 0){ perror("Errore impostazione formato"); return 1; }

    struct v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if(ioctl(fd, VIDIOC_REQBUFS, &req) < 0){ perror("Errore richiesta buffer"); return 1; }

    unsigned char* buffers[req.count];
    struct v4l2_buffer buf;
    for(int i=0;i<req.count;i++){
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if(ioctl(fd, VIDIOC_QUERYBUF, &buf)<0){ perror("Errore query buffer"); return 1; }
        buffers[i] = (unsigned char*)mmap(NULL, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if(buffers[i] == MAP_FAILED){ perror("Errore mmap"); return 1; }
        if(ioctl(fd, VIDIOC_QBUF, &buf) < 0){ perror("Errore queue buffer"); return 1; }
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMON, &type) < 0){ perror("Errore stream on"); return 1; }

    int out_width = WIDTH/SCALE;
    int out_height = HEIGHT/SCALE;

    int frames = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while(frames < 1000){ // loop continuo, premi CTRL+C per fermare
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(ioctl(fd, VIDIOC_DQBUF, &buf) < 0){ perror("Errore lettura frame"); break; }

        // pulisce il terminale
        /*
        std::cout << "\033[H\033[J";

        for(int i=0;i<out_height;i++){
            for(int j=0;j<out_width;j++){
                std::cout << yuyv_to_ascii(buffers[buf.index], j*SCALE, i*SCALE);
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
        */
        frames++;
        if(ioctl(fd, VIDIOC_QBUF, &buf) < 0){ perror("Errore riqueue buffer"); break; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double,std::milli>(end-start).count();
    std::cout << "CPU: tempo medio per frame = " << elapsed/frames << " ms\n";
    std::cout << "FPS teorico massimo = " << 1000.0/(elapsed/frames) << std::endl;

    ioctl(fd, VIDIOC_STREAMOFF,&type);
    for(int i=0;i<req.count;i++) munmap(buffers[i],buf.length);
    close(fd);

    return 0;
}

