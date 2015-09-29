Optimized CPU Gaussian blur filter
==================================

Features :
* Cache efficient data access
* Multithreading using OpenMP
* Vectorization using SSE 4.1 intrinsics
* Cross-platform (Linux, Windows, OS X)
* 24bpp uncompressed BMP reader/writer

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
