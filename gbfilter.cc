// -----------------------------------------------------------------------------
// 2014-2015
// Thibault Coppex <tcoppex@gmail.com>
// For conditions of distribution and use, see copyright notice in LICENSE.md
// -----------------------------------------------------------------------------

#include <cstdlib>
#include <cstdio>
#include <cmath>

#ifdef GBF_OMP_STATS
# include "omp.h"
# define GET_TIME() omp_get_wtime()
#else
# define GET_TIME() 0.0
#endif

#if __SSE4_1__
# include "smmintrin.h"
#endif


// =============================================================================

/**
 * @name BMPFile
 * @brief Simple BMP file reader/writer
 * @note Only accept 24b uncompressed image
 */
class BMPFile {
public:
  // ---------------------------------------------------------------------------
  /// @name Public methods
  // ---------------------------------------------------------------------------

  BMPFile() :
    data_(NULL)
  {}

  ~BMPFile() {
    clear_data();
  }

  /// Clear pixels data
  void clear_data() {
    if (data_) {
      delete [] data_;
      data_ = NULL;
    }
  }

  /// Load a BMP from a file
  /// @param filename : bmp file to load
  /// @return true if the file was loaded successfully
  bool load(const char* filename);

  /// Save a BMP to a file
  /// @param filename : bmp file to save
  /// @return true if the file was saved successfully
  bool save(const char* filename) const;


  // ---------------------------------------------------------------------------
  /// @name Getters
  // ---------------------------------------------------------------------------

  unsigned int width() const {
    return bih_.biWidth;
  }

  unsigned int height() const {
    return bih_.biHeight;
  }

  unsigned int resolution() const {
    return width() * height();
  }

  unsigned char* data() {
    return data_;
  }

private:
  // ---------------------------------------------------------------------------
  /// @name Format headers
  // ---------------------------------------------------------------------------

  // Tighly pack the structure to avoid unwanted padding
#pragma pack(push, 1)
  struct BITMAPFILEHEADER_t { 
    unsigned short  bfType;             //< filetype, must be "BM" (== 19778)
    unsigned int    bfSize;             //< filesize in bytes
    unsigned short  bfReserved1;        //< reserved, must be 0
    unsigned short  bfReserved2;        //< reserved, must be 0
    unsigned int    bfOffBits;          //< offset in bytes, represent the header size
  };
#pragma pack(pop)

  struct BITMAPINFOHEADER_t {
    unsigned int    biSize;             //< structure size, must be 40
    int             biWidth;            //< image width
    int             biHeight;           //< image height. If negative, start TopLeft, BottomLeft otherwise
    unsigned short  biPlanes;           //< planes count, must be 1
    unsigned short  biBitCount;         //< bits per pixel
    unsigned int    biCompression;      //< compression method when height > 0
    unsigned int    biSizeImage;        //< Image size in bytes.
    int             biXPelsPerMeter;    //< horizontal resolution in pixel per meter
    int             biYPelsPerMeter;    //< vertical resolution in pixel per meter
    unsigned int    biClrUsed;          //< number of color used from the color palette
    unsigned int    biClrImportant;     //< number of important index color. All if set to 0
  };

  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  BITMAPFILEHEADER_t bfh_;              //< bitmap file header
  BITMAPINFOHEADER_t bih_;              //< bitmap info header
  unsigned char *data_;                 //< pixels data
};

// -----------------------------------------------------------------------------

bool BMPFile::load(const char* filename) {
  clear_data();

  // Open the bmp file
  FILE *fd = NULL;
  if (NULL == (fd = fopen(filename, "rb"))) {
    fprintf(stderr, "Error : Unable to open \"%s\".\n", filename);
    return false;
  }

  // Read the file headers
  const size_t bfh_res = fread(&bfh_, sizeof(bfh_), 1u, fd);
  const size_t bih_res = fread(&bih_, sizeof(bih_), 1u, fd);
  if (   1u != bfh_res
      || 1u != bih_res) {
    fprintf(stderr, "Error : the file header was not read correctly.\n");
    fclose(fd);
    return false;
  }

  // Check file format
#define GBFILTER_BMP_MAGICNUMBER  19778
  if (bfh_.bfType != GBFILTER_BMP_MAGICNUMBER) {
    fprintf(stderr, "Error : invalid file format.\n");
    fclose(fd);
    return false;
  }
#undef GBFILTER_BMP_MAGICNUMBER

  // Only 24bit uncompressed images are accepted
  if (bih_.biCompression != 0u) {
    fprintf(stderr, "Compressed BMP files are not handled yet.\n");
    fclose(fd);
    return false;
  }
  if (bih_.biBitCount != 24u) {
    fprintf(stderr, "Non 24bits BMP files are not handled yet.\n");
    fclose(fd);
    return false;
  }

  // Create internal buffer
  const unsigned int image_size = bih_.biSizeImage;
  data_ = new unsigned char[image_size];

  // Be sure to start at pixels data
  fseek(fd, bfh_.bfOffBits, SEEK_SET);

  // Load 24 bit uncompressed data [note: BGR order]  
  if (image_size != fread(data_, 1u, image_size, fd)) {
    fprintf(stderr, "Error : the file was not read correctly.\n");
    fclose(fd);
    return false;
  }

  fclose(fd);
  return true;
}

// -----------------------------------------------------------------------------

bool BMPFile::save(const char* filename) const {
  FILE *fd = NULL;

  if (NULL == (fd = fopen(filename, "wb"))) {
    return false;
  }

  // headers
  fwrite(&bfh_, sizeof(bfh_), 1u, fd);
  fwrite(&bih_, sizeof(bih_), 1u, fd);
  
  // datas
  fseek(fd, bfh_.bfOffBits, SEEK_SET);
  fwrite(data_, 1u, bih_.biSizeImage, fd);
  
  fclose(fd);

  return true;
}

// =============================================================================

#if __SSE4_1__
/**
 * @struct Vec4
 * @brief Simple Vector4 structure using SSE4.1 intrinsics
 */
struct __attribute__((aligned (16))) Vec4 {
  Vec4() : mmvalue(_mm_setzero_ps()) {}
  Vec4(float x, float y, float z, float w) : mmvalue(_mm_set_ps(w, z, y, x)) {}
  Vec4(float value) : mmvalue(_mm_set1_ps(value)) {}
  Vec4(__m128 mm) : mmvalue(mm) {}

  static
  unsigned int GetAlignedSize(unsigned int size) {
    return (size + 3u) / 4u;
  }

  static
  float Dot4(const Vec4 &u, const Vec4 &v) {
    return _mm_cvtss_f32(_mm_dp_ps(u.mmvalue, v.mmvalue, 0xF1));
  }

  union {
    struct { float x, y, z, w; };
    __m128 mmvalue;
  };
};
#endif

// =============================================================================

/**
 * @class GBFilter
 * @brief Apply a Gaussian filter on a bitmap file
 *
 * Taking advantages of the symmetric property of the 2D Gaussian filter, the
 * algorithm is splitted in two passes, one horizontale & one verticale, using
 * a simple 1D filter.
 *
 * @note :
 * The blur radius is set in sub-pixel. The fractional part is used to lerp the
 * filter's extremities. The kernel size is hence set as
 * 2 * ceil(blur_radius) + 1
 * As for discrete gaussian, the kernel size is always an odd number.
 */
class GBFilter {
public:
  GBFilter(float blur_radius);
  ~GBFilter();

  /// Apply the gaussian filter on a bmp file
  /// @param bmp : bitmap to filter
  /// @param tile_w : tile width
  /// @param tile_h : tile height
  void apply(BMPFile &bmp, unsigned int tile_w, unsigned int tile_h);

private:
  // ---------------------------------------------------------------------------
  /// @name Structures
  // ---------------------------------------------------------------------------

  /// Buffer storing separated RGB channels in floating point format
  struct RGBBuffer_t {
    RGBBuffer_t() : red(NULL), green(NULL), blue(NULL) {}
    float *red, *green, *blue;
  };

  /// Simplify passing constant layout attributes
  struct LayoutParam_t {
    unsigned int image_w;
    unsigned int image_h;
    unsigned int tile_w;
    unsigned int tile_h;
    unsigned int grid_w;
    unsigned int grid_h;
  };

  // ---------------------------------------------------------------------------
  /// @name Methods
  // ---------------------------------------------------------------------------
  
  /// Transpose a buffer into another [row to column]
  static
  void TransposeBuffer(const RGBBuffer_t &in,
                             RGBBuffer_t &out,
                             LayoutParam_t &layout);

  /// Initialize the 1D gaussian filter
  void init_filter1D();

  /// Run the blur passes (horizontal & vertical)
  /// @param layout : layout parameters
  void blur(LayoutParam_t &layout);

  /// Apply a horizontal blur on each tiles
  /// @param layout : layout parameters
  void blur_x(const LayoutParam_t &layout);

  /// Apply a vertical blur on each tiles
  /// @param layout : layout parameters
  void blur_y(const LayoutParam_t &layout);

  /// Generic blur filter pass
  /// @param layout : layout parameters
  /// @param tx : tile x-coordinate
  /// @param ty : tile y-coordinate
  /// @param blurX : Performs horizontal blur if true, vertical otherwise
  /// @param in : input buffer
  /// @param out : output buffer
  void blur_pass(const LayoutParam_t &layout,
                 const unsigned int tx,
                 const unsigned int ty,
                 const bool blurX,
                 const RGBBuffer_t &in,
                       RGBBuffer_t &out);

  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------
  // Kernel radius threshold after which the transpose buffer layout optimization is used
  static const float kTransposeRadiusThreshold;

  // number of RGB buffer used
  static const unsigned int kNumRGBBuffer = 2u;

  float *filter1D_;                     //< 1D Gaussian filter
  float blur_radius_;                   //< radius of the blur in sub-pixels
  unsigned int kernel_size_;            //< size of the filter, must be odd
  RGBBuffer_t buffer_[kNumRGBBuffer];   //< F32 RGB channels buffers

#if __SSE4_1__
  Vec4 *sse_filter_;                    //< 1D Gaussian filter using SSE4.1
#endif
};

const float GBFilter::kTransposeRadiusThreshold = 28.0f;

// -----------------------------------------------------------------------------

GBFilter::GBFilter(float blur_radius) :
  filter1D_(NULL),
  blur_radius_(blur_radius) 
{
  kernel_size_ = 2u * ceilf(blur_radius_) + 1u;
  init_filter1D();
}

// -----------------------------------------------------------------------------

GBFilter::~GBFilter() {
  if (filter1D_) {
    delete [] filter1D_;
#if __SSE4_1__
    delete [] sse_filter_;
#endif
  }

  for (unsigned int i=0u; i<kNumRGBBuffer; ++i) {
    delete [] buffer_[i].red;
    delete [] buffer_[i].green;
    delete [] buffer_[i].blue;
  } 
}

// -----------------------------------------------------------------------------

void GBFilter::apply(BMPFile &bmp, unsigned int tile_w, unsigned int tile_h) {
  const unsigned int kResolution = bmp.resolution();

  // Initialize RGB float buffers
  for (unsigned int i=0u; i<kNumRGBBuffer; ++i) {
    buffer_[i].red   = new float[kResolution];
    buffer_[i].green = new float[kResolution];
    buffer_[i].blue  = new float[kResolution];
  }

  unsigned char *pixels = bmp.data();

  // setup first blur buffer [uchar to float]
  const float scale = 1.0f / 255.0f;
  for (unsigned int i=0u; i<kResolution; ++i) {
    buffer_[0u].blue[i]  = scale * pixels[3u*i + 0u];
    buffer_[0u].green[i] = scale * pixels[3u*i + 1u];
    buffer_[0u].red[i]   = scale * pixels[3u*i + 2u];

    // [debug color]
    buffer_[1u].blue[i]  = 1.0f;
    buffer_[1u].green[i] = 0.0f;
    buffer_[1u].red[i]   = 1.0f;
  }

  // Constants layout parameters
  LayoutParam_t layout;
  layout.image_w = bmp.width();
  layout.image_h = bmp.height();
  layout.tile_w  = tile_w;
  layout.tile_h  = tile_h;
  layout.grid_w  = (layout.image_w + layout.tile_w - 1u) / layout.tile_w;
  layout.grid_h  = (layout.image_h + layout.tile_h - 1u) / layout.tile_h;

  // Apply blur
  blur(layout);

  // Update BMP buffer [float to uchar]
  for (unsigned int i=0u; i<kResolution; ++i) {
    pixels[3u*i + 0u] = (unsigned char)(255 * buffer_[0u].blue[i]);
    pixels[3u*i + 1u] = (unsigned char)(255 * buffer_[0u].green[i]);
    pixels[3u*i + 2u] = (unsigned char)(255 * buffer_[0u].red[i]);
  }
}

// -----------------------------------------------------------------------------

void GBFilter::init_filter1D() {
  filter1D_ = new float[kernel_size_];

  // Base parameters
  const int c = static_cast<int>(kernel_size_ / 2u);
  const float sigma = kernel_size_ / 3.0f; // heuristic
  const float s = 2.0f * sigma * sigma;
  const float inv_s = 1.0f / s;
  const float inv_s_pi = 1.0f / (3.14159265359f * s);

  // Calculate the Gaussian coefficients
  float sum = 0.0f;
  for (int x=-c; x<=c; ++x) {
    const float r = x*x;
    const float coeff = exp(-r * inv_s) * inv_s_pi;
    filter1D_[x+c] = coeff;
    sum += coeff;
  }

  // Normalize the filter
  const float inv_sum = 1.0f / sum;
  for (unsigned int i=0u; i<kernel_size_; ++i) {
    filter1D_[i] *= inv_sum;
  }

  // Lerp kernel boundaries with radius fractional part
  //filter1D_[0] = filter1D_[kernel_size_-1] *= (kernel_radius_ - int(kernel_radius_));

#if __SSE4_1__
  /// Create the Vec4 gaussian filter
  const unsigned int nvec = Vec4::GetAlignedSize(kernel_size_);
  sse_filter_ = new Vec4[nvec];

  unsigned int i=0u, j=0u;  
  for (; i+1u < nvec; ++i, j+=4u) {
    sse_filter_[i] = Vec4(filter1D_[j], filter1D_[j+1u], filter1D_[j+2u], filter1D_[j+3u]);
  }

  sse_filter_[i].x = filter1D_[j];
  sse_filter_[i].y = (j+1u < kernel_size_) ? filter1D_[j+1u] : 0.0f;
  sse_filter_[i].z = (j+2u < kernel_size_) ? filter1D_[j+2u] : 0.0f;
  sse_filter_[i].w = (j+3u < kernel_size_) ? filter1D_[j+3u] : 0.0f;
#endif
}

// -----------------------------------------------------------------------------

void GBFilter::TransposeBuffer(const RGBBuffer_t &in,
                                     RGBBuffer_t &out,
                                     LayoutParam_t &layout) {
# pragma omp parallel for collapse(2) num_threads(4)
  for (unsigned int x = 0u; x < layout.image_w; ++x) {
    for (unsigned int y = 0u; y < layout.image_h; ++y) {    
      unsigned int  in_idx = y * layout.image_w + x;
      unsigned int out_idx = x * layout.image_h + y;

        out.red[out_idx] = in.red[in_idx];
      out.green[out_idx] = in.green[in_idx];
       out.blue[out_idx] = in.blue[in_idx];
    }
  }

  // Transpose the layout
  LayoutParam_t tlayout;
  tlayout.image_w = layout.image_h;
  tlayout.image_h = layout.image_w;
  tlayout.tile_w  = layout.tile_h;
  tlayout.tile_h  = layout.tile_w;
  tlayout.grid_w  = layout.grid_h;
  tlayout.grid_h  = layout.grid_w;
  layout = tlayout;
}

// -----------------------------------------------------------------------------

void GBFilter::blur(LayoutParam_t &layout) {
  // @note
  // Vertical blur is not cache efficient.
  // This is especially noticeable for large blur radius.
  // A work around is to use a buffer to temporary transpose rows as
  // columns and applying a horizontal blur to it before transposing
  // columns back as rows.

  double t1 = GET_TIME();

  // Horizontal blur
  blur_x(layout);

  double t2 = GET_TIME();

  // Vertical blur
  if (blur_radius_ < kTransposeRadiusThreshold) {
    blur_y(layout);
  } else {
    TransposeBuffer(buffer_[1u], buffer_[0u], layout);
    blur_x(layout);
    TransposeBuffer(buffer_[1u], buffer_[0u], layout);
  }

  double t3 = GET_TIME();

#ifdef GBF_OMP_STATS
  fprintf(stderr, "x-blur : %.3f ms\ny-blur : %.3f ms\ntotal : %.3f ms\n",
                  t2-t1, t3-t2, t3-t1);
#endif
}

// -----------------------------------------------------------------------------

void GBFilter::blur_x(const LayoutParam_t &layout) {
# pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (unsigned int ty = 0u; ty < layout.grid_h; ++ty) {
    for (unsigned int tx = 0u; tx < layout.grid_w; ++tx) {
      blur_pass(layout, tx, ty, true, buffer_[0u], buffer_[1u]);
    }
  }  
}

// -----------------------------------------------------------------------------

void GBFilter::blur_y(const LayoutParam_t &layout) {
# pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (unsigned int ty = 0u; ty < layout.grid_h; ++ty) {
    for (unsigned int tx = 0u; tx < layout.grid_w; ++tx) {
      blur_pass(layout, tx, ty, false, buffer_[1u], buffer_[0u]);
    }
  }
}

// -----------------------------------------------------------------------------

namespace {

/// @return the minimum value between two
inline
unsigned int Min(const unsigned int a, const unsigned int b) {
  return (a < b) ? a : b;
}

/// Wrap an index mirrored if outside range [0, width]
/// @param x : index to wrap
/// @param width : range upper boundary
/// @return wrapped index
inline
unsigned int WrappedIndex(const int x, const int width) {
  //const int index = (x < 0) ? -x-1 : (x >= width) ? width-1 : x;
  const int index = (x < 0) ? -x : (x < width) ? x : width-2 + width-x;
  return static_cast<unsigned int>(index);
}

/// Get wrapped image index depending on blur offset & directions
inline
unsigned int GetBlurIndex(unsigned int x, unsigned int y, unsigned int w, 
                          unsigned int h, int dx, bool blurX) {
  return (blurX) ? y*w + WrappedIndex(int(x) + dx, w)
                 : WrappedIndex(int(y) + dx, h) * w + x;
}

} // namespace

// -----------------------------------------------------------------------------

void GBFilter::blur_pass(const LayoutParam_t &layout,
                         const unsigned int tx,
                         const unsigned int ty,
                         const bool blurX,
                         const RGBBuffer_t &in,
                               RGBBuffer_t &out) {
  // Discretized kernel radius
  const int c = static_cast<int>(kernel_size_ / 2u);

  // Tile start & end index
  const unsigned int start_x = tx * layout.tile_w;
  const unsigned int start_y = ty * layout.tile_h;
  const unsigned int w = layout.image_w;
  const unsigned int h = layout.image_h;
  const unsigned int end_x = Min(start_x + layout.tile_w, w);
  const unsigned int end_y = Min(start_y + layout.tile_h, h);

  for (unsigned int y = start_y; y < end_y; ++y) {
    for (unsigned int x = start_x; x < end_x; ++x) {
      unsigned int index = y*layout.image_w + x;

      // Pixel blur evaluation
      float rgb[3u] = {0.0f};
#if __SSE4_1__
      for (int dx=-c, cid=0; dx<=c; dx+=4, ++cid) {
        // Gaussian filter coefficients
        const Vec4 &GFC = sse_filter_[cid];

        unsigned int i1 = GetBlurIndex(x, y, w, h, dx+0, blurX);
        unsigned int i2 = GetBlurIndex(x, y, w, h, dx+1, blurX);
        unsigned int i3 = GetBlurIndex(x, y, w, h, dx+2, blurX);
        unsigned int i4 = GetBlurIndex(x, y, w, h, dx+3, blurX);
        Vec4 RED(in.red[i1], in.red[i2], in.red[i3], in.red[i4]);
        Vec4 GREEN(in.green[i1], in.green[i2], in.green[i3], in.green[i4]);
        Vec4 BLUE(in.blue[i1], in.blue[i2], in.blue[i3], in.blue[i4]);

        rgb[0u] += Vec4::Dot4(GFC, RED);
        rgb[1u] += Vec4::Dot4(GFC, GREEN);
        rgb[2u] += Vec4::Dot4(GFC, BLUE);
      }
#else
      for (int dx=-c; dx<=c; ++dx) {
        float gfc = filter1D_[dx+c];
        unsigned int w_index = GetBlurIndex(x, y, w, h, dx, blurX);

        rgb[0u] += gfc * in.red[w_index];
        rgb[1u] += gfc * in.green[w_index];
        rgb[2u] += gfc * in.blue[w_index];
      }
#endif  // __SSE4_1__

      // Final pixel, should not need to be clamped
      out.red[index]   = rgb[0u];
      out.green[index] = rgb[1u];
      out.blue[index]  = rgb[2u];
    }
  }
}

// =============================================================================

int main(int argc, char **argv) {
  // Retrieve command line arguments
  if (argc < 6) {
    fprintf(stderr, "usage :\n%s input_file output_file blur_radius tile_width " \
                    "tile_height\n", argv[0u]);
    exit(EXIT_FAILURE);
  }

  char *p_filename_in = argv[1u];
  char *p_filename_out = argv[2u];

  float blur_radius(0.0f);
  sscanf(argv[3u], "%f", &blur_radius);

  unsigned int tile_w(0u), tile_h(0u);
  sscanf(argv[4u], "%u", &tile_w);
  sscanf(argv[5u], "%u", &tile_h);
 
  // ---------------------------

  BMPFile bmp;

  // Load the image
  if (!bmp.load(p_filename_in)) {
    exit(EXIT_FAILURE);
  }
  
  // Apply a Gaussian filter to the input image
  GBFilter(blur_radius).apply(bmp, tile_w, tile_h);

  // Save the result
  bmp.save(p_filename_out);

  return EXIT_SUCCESS;
}

// =============================================================================
