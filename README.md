Optimized CPU Gaussian blur filter
==================================

Features :
* 24bpp uncompressed BMP reader/writer
* Cache efficient algorithm
* SSE 4.1 intrinsics
* Multithreading using OpenMP
* Cross-platform (Linux, Windows, OS X)

Build :
```shell
mkdir build; cd build/
cmake ..
cmake --build . --config Release
```

Usage :
```shell
./gbfilter input.bmp output.bmp blur_radius tile_width tile_height
```
