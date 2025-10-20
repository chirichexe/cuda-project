## Prerequisites
Ensure all dependencies listed in `DEPENDENCIES.md` are installed.

## Build
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
