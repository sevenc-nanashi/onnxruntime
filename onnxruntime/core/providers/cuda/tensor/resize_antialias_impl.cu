// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/resize_impl.h"

#define CPU_TESTING true

#ifdef CPU_TESTING
#undef __global__
#define __global__
using IdType = int64_t;
#else
using IdType = int;
#endif

namespace onnxruntime {
namespace cuda {

using onnxruntime::ResizeCoordinateTransformationMode;
using onnxruntime::UpsampleMode;

// Antialiasing filters
struct BilinearFilter {
  __device__ __host__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

struct BiCubicFilter {
  __device__ __host__ float operator()(float x, float cubic_coeff_a) const {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return ((cubic_coeff_a + 2.0f) * x - (cubic_coeff_a + 3.0f)) * x * x + 1;
    }
    if (x < 2.0f) {
      return (((x - 5.0f) * x + 8.f) * x - 4.f) * cubic_coeff_a;
    }
    return 0.0f;
  }
};

struct TriLinearFilter {
  __device__ __host__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

/// <summary>
/// This function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2]
/// 2. out_of_bounds: int64_t[output_size]
/// 3. scale_data: T[output_size * window_size]
///
/// Template parameter AccumType
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__device__ __host__ void SetupUpsampleFilterAnitAliasImpl(
    IdType id,
    int64_t input_size, int64_t output_size,
    float inv_scale,
    float roi_start, float roi_end,
    float scaled_support, int32_t window_size, bool exclude_outside,
    float cubic_coeff_a,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* scale_data) {
  Filter filter{};
  CudaFunctionOriginalCoordinate get_original_coordinate{};

  const float scale = 1.0f / inv_scale;
  const float center = 0.5f + (scale == 1.0f) ? static_cast<float>(id)
                                              : get_original_coordinate(static_cast<float>(id), inv_scale,
                                                                        static_cast<float>(output_size),
                                                                        static_cast<float>(input_size),
                                                                        roi_start, roi_end);

  if (center - 0.5f < 0 || center - 0.5f > static_cast<float>(input_size - 1)) {
    out_of_bounds[id] = id;
  } else {
    out_of_bounds[id] = -1;
  }

  AccumType total_weight{0};

  auto fmin = _Floor(center - scaled_support + 0.5f);
  auto fmax = _Floor(center + scaled_support + 0.5f);

  int64_t min_real = static_cast<int64_t>(fmin);
  int64_t max_real = static_cast<int64_t>(fmax);
  int64_t min_cut = std::max(min_real, 0LL);
  int64_t max_cut = std::min(max_real, input_size);

  auto min_val = exclude_outside ? min_cut : min_real;
  auto max_val = exclude_outside ? max_cut : max_real;
  bounds[id * 2] = min_cut;
  bounds[id * 2 + 1] = max_cut;

  auto* scale_buffer = &scale_data[id * window_size];

  int64_t x = 0;
  max_val -= min_val;
  for (; x < max_val; x++) {
    auto w = filter((x + min_val - center + 0.5f) * inv_scale, cubic_coeff_a);
    scale_buffer[x] = static_cast<AccumType>(w);
    total_weight += static_cast<AccumType>(w);
  }

  if (!exclude_outside) {
    int64_t neg_xsize = min_val < 0 ? -min_val : 0;
    for (x = 0; x < neg_xsize; x++) {
      scale_buffer[neg_xsize] += scale_buffer[x];
    }

    int64_t bound_size =
        max_val + min_val > input_size ? max_val + min_val - input_size : 0;
    for (x = max_val - bound_size; x < max_val; x++) {
      scale_buffer[max_val - bound_size - 1] +=
          scale_buffer[x];
    }

    for (x = 0; (neg_xsize | bound_size) > 0 && x < max_cut - min_cut; x++) {
      scale_buffer[x] = scale_buffer[x + neg_xsize];
    }
  }

  const AccumType total_weight_inv = (total_weight == 0) ? AccumType{1} : (AccumType{1} / total_weight);
  if constexpr (std::is_same<AccumType, int32_t>::value) {
    auto* scale_buffer_int = reinterpret_cast<int32_t*>(scale_buffer);
    for (x = 0; x < max_cut - min_cut; x++) {
      scale_buffer[x] *= total_weight_inv;
      // normalize the scale to 1 << 22 for int8/uint8
      scale_buffer_int[x] = static_cast<int32_t>(_Round(scale_buffer[x] * ConstValue::mag_factor * 2.f));
    }
  } else {
    for (x = 0; x < max_cut - min_cut; x++) {
      scale_buffer[x] *= total_weight_inv;
    }
  }
}

/// This kernel computes antialias filter for bilinear or bicubic upsampling.
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the two dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the two dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the two dimensions
/// Buffers layout [h_data, w_data]
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupBilinearUpsampleFilterAntiAlias(
#ifdef CPU_TESTING
    IdType id,
#endif
    std::tuple<int64_t, int64_t> input_dims,       // h, w
    std::tuple<int64_t, int64_t> output_dims,      // h, w
    std::tuple<float, float> inv_scale_vals,       // h, w
    std::tuple<float, float> roi_start_vals,       // h, w
    std::tuple<float, float> roi_end_vals,         // h, w
    std::tuple<float, float> dim_scaled_support,   // Pre-computed scaled support values h, w
    std::tuple<int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes h, w
    float cubic_coeff_a,
    bool exclude_outside,
    const size_t SumHW,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* weighted_coefficients) {  // computed weighted coefficients

#ifndef CPU_TESTING
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);
#endif

  // Setup for y
  int64_t input_size = std::get<0>(input_dims);
  int64_t output_size = std::get<0>(output_dims);
  float inv_scale = std::get<0>(inv_scale_vals);
  float roi_start = std::get<0>(roi_start_vals);
  float roi_end = std::get<0>(roi_end_vals);
  float scaled_support = std::get<0>(dim_scaled_support);
  int32_t window_size = std::get<0>(dim_window_size);

  // id >= output_height
  if (id >= std::get<0>(output_dims)) {
    // Setup for w
    // w = id - output_height
    id = id - std::get<0>(output_dims);
    input_size = std::get<1>(input_dims);
    output_size = std::get<1>(output_dims);
    inv_scale = std::get<1>(inv_scale_vals);
    roi_start = std::get<1>(roi_start_vals);
    roi_end = std::get<1>(roi_end_vals);

    scaled_support = std::get<1>(dim_scaled_support);
    window_size = std::get<1>(dim_window_size);

    // Adjust buffer positions
    bounds += (output_size * 2);
    out_of_bounds += output_size;
    weighted_coefficients += (output_size * window_size);
  }

  SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
      static_cast<int>(id),
      input_size, output_size,
      inv_scale,
      roi_start, roi_end,
      scaled_support, window_size,
      exclude_outside,
      cubic_coeff_a,
      bounds,
      out_of_bounds,
      weighted_coefficients);
}

/// <summary>
/// Compute AntiAlias filter for trilinear upsampling, all in one go
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the three dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the three dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the three dimensions
/// Each kind of buffer contains data for all 3 dims.
/// Buffers layout [d_data, h_data, w_data]
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupTrilinerarUpsampleFilterAntiAlias(
#ifdef CPU_TESTING
    IdType id,
#endif
    std::tuple<int64_t, int64_t, int64_t> input_dims,       // d, h, w
    std::tuple<int64_t, int64_t, int64_t> output_dims,      // d, h, w
    std::tuple<float, float, float> inv_scale_vals,         // d, h, w
    std::tuple<float, float, float> roi_start_vals,         // d, h, w
    std::tuple<float, float, float> roi_end_vals,           // d, h, w
    std::tuple<float, float, float> dim_scaled_support,     // Pre-computed scaled support values d, h, w
    std::tuple<int32_t, int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes d, h, w
    const size_t SumDHW,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* weighted_coefficients) {

#ifndef CPU_TESTING
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumDHW);
#endif

  const auto output_depth = std::get<0>(output_dims);

  // Setup for d by default (id < output_depth)
  int64_t input_size = std::get<0>(input_dims);
  int64_t output_size = std::get<0>(output_dims);
  float inv_scale = std::get<0>(inv_scale_vals);
  float roi_start = std::get<2>(roi_start_vals);
  float roi_end = std::get<0>(roi_end_vals);
  float scaled_support = std::get<0>(dim_scaled_support);
  int32_t window_size = std::get<0>(dim_window_size);

  if (id >= output_depth && id < (output_depth + std::get<1>(output_dims))) {
    // Setup for y - height

    // y = id - output_depth
    id = id - output_depth;
    input_size = std::get<1>(input_dims);
    output_size = std::get<1>(output_dims);
    inv_scale = std::get<1>(inv_scale_vals);
    roi_start = std::get<1>(roi_start_vals);
    roi_end = std::get<1>(roi_end_vals);

    // Adjust buffer positions
    scaled_support = std::get<1>(dim_scaled_support);
    window_size = std::get<1>(dim_window_size);

    bounds += output_size * 2;
    out_of_bounds += output_size;
    weighted_coefficients += (output_size * window_size);

  } else if (id > output_depth) {  // means we are out of bounds for the second for the first if on the right side
    // Setup for x

    // x = id - output_depth - output_height
    id = id - output_depth - std::get<1>(output_dims);
    input_size = std::get<2>(input_dims);
    output_size = std::get<2>(output_dims);
    inv_scale = std::get<2>(inv_scale_vals);
    roi_start = std::get<2>(roi_start_vals);
    roi_end = std::get<2>(roi_end_vals);

    // Adjust buffer positions
    scaled_support = std::get<2>(dim_scaled_support);
    window_size = std::get<2>(dim_window_size);

    bounds += (output_size * 4);
    out_of_bounds += (output_size * 2);
    weighted_coefficients += output_size * window_size * 2;
  }

  SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
      id,
      input_size, output_size,
      inv_scale,
      roi_start, roi_end,
      scaled_support, window_size,
      true,                       // exclude outside for trilinear
      onnxruntime::kCubicCoeffA,  // Default value for trilinear
      bounds,
      out_of_bounds,
      weighted_coefficients);
}

#define CASEA_COORD_ANTIALIAS(coordinate_mode, TransformCoordType, ...) \
  case coordinate_mode: {                                               \
    using coord_t = TransformCoordType;                                 \
    return __VA_ARGS__();                                               \
    break;                                                              \
  }

#define DISPATCH_ANTIALIAS_FILTER_SETUP(coord_enum, ...)                                                                                     \
  [&] {                                                                                                                                      \
    const auto the_type = coord_enum;                                                                                                        \
    switch (the_type) {                                                                                                                      \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::HALF_PIXEL, TransformCoordinate_HALF_PIXEL, __VA_ARGS__)                     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ASYMMETRIC, TransformCoordinate_ASYMMETRIC, __VA_ARGS__)                     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL, TransformCoordinate_PYTORCH_HALF_PIXEL, __VA_ARGS__)     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ALIGN_CORNERS, TransformCoordinate_ALIGN_CORNERS, __VA_ARGS__)               \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN, TransformCoordinate_TF_HALF_PIXEL_FOR_NN, __VA_ARGS__) \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE, TransformCoordinate_TF_CROP_AND_RESIZE, __VA_ARGS__)     \
      default:                                                                                                                               \
        ORT_THROW("unknown ResizeCoordinateTransformationMode");                                                                             \
    }                                                                                                                                        \
  }()

template <class T>
void ResizeAntiAliasImpl(
    cudaStream_t stream,
    int rank,
    const UpsampleMode upsample_mode,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    gsl::span<const int64_t> input_shape,
    gsl::span<const int64_t> output_shape,
    // const TArray<int64_t>& input_strides,
    const TArray<fast_divmod>& output_div_pitches,
    gsl::span<const float> roi_vals,
    gsl::span<const float> scales_vals,
    // const std::optional<T>& extrapolation_value,
    bool exclude_outside,
    std::tuple<float, float, float> scaled_support_vals,                       // d, y, h
    std::tuple<int32_t, int32_t, int32_t> window_sizes,                        // d, y, h
    gsl::span<int64_t> bounds_buffer,                                          // on device
    gsl::span<int64_t> out_of_bounds_buffer,                                   // on device
    gsl::span<typename onnxruntime::AccumulateType<T>::type> weighted_buffer,  // on device
    const T* input_data,
    T* output_data,
    const size_t N) {
  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  // We support a special case of bilinear or bicubic if the input data is 4D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_2D = (rank == 2 || rank == 4);

  // We support a special case of trilinear or tricubic if the input data is 5D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_3D = (rank == 3 || rank == 5);

  // Should not hit this as we have already validated input rank/scales and we provide verbose error messages
  // to the user.
  ORT_ENFORCE(is_2D || is_3D, "Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");

  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fast_divmod div_output_image;
  if (is_2D) {
    div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(gsl::narrow_cast<int>(N));
  } else if (is_3D) {
    div_output_image = (rank > 3) ? output_div_pitches[rank - 4] : fast_divmod(gsl::narrow_cast<int>(N));
  }

  const int64_t input_depth = is_3D ? input_shape[rank - 3] : 0;
  const int64_t input_height = input_shape[rank - 2];
  const int64_t input_width = input_shape[rank - 1];

  const int64_t output_depth = is_3D ? output_shape[rank - 3] : 0;
  const int64_t output_height = output_shape[rank - 2];
  const int64_t output_width = output_shape[rank - 1];
  int blocksPerDimsMappingGrid =
      static_cast<int>(ceil((output_depth + output_height + output_width) / 32.0));

  switch (upsample_mode) {
    case UpsampleMode::LINEAR: {
      if (is_2D) {
        float h_scaled_support, w_scaled_support;
        std::tie(std::ignore, h_scaled_support, w_scaled_support) = scaled_support_vals;
        int32_t h_window_size, w_window_size;
        std::tie(std::ignore, h_window_size, w_window_size) = window_sizes;

#ifdef CPU_TESTING

        for (int64_t id = 0, lim = output_height + output_width; id < lim; ++id) {
          DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
            _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                                  BilinearFilter,
                                                  coord_t>(
                narrow<int32_t>(id),
                std::make_tuple(input_height, input_width),
                std::make_tuple(output_height, output_width),
                std::make_tuple(scales_vals[rank - 2], scales_vals[rank - 1]),
                std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
                std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
                std::make_tuple(h_scaled_support, w_scaled_support),
                std::make_tuple(h_window_size, w_window_size),
                onnxruntime::kCubicCoeffA, exclude_outside,
                onnxruntime::narrow<size_t>(output_height + output_width),
                bounds_buffer.data(),
                out_of_bounds_buffer.data(),
                weighted_buffer.data());
          });
        }

        PrintAntiAliasBuffers(std::cout, bounds_buffer, out_of_bounds_buffer, weighted_buffer);

#else
        DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
        //  Data is d, h, w in tuples

          _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                                BilinearFilter,
                                                coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              std::make_tuple(input_height, input_width),
              std::make_tuple(output_height, output_width),
              std::make_tuple(scales_vals[rank - 2], scales_vals[rank - 1]),
              std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
              std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
              std::make_tuple(h_scaled_support, w_scaled_support),
              std::make_tuple(h_window_size, w_window_size),
              onnxruntime::kCubicCoeffA, exclude_outside,
              onnxruntime::narrow<size_t>(output_height + output_width),
              bounds_buffer.data(),
              out_of_bounds_buffer.data(),
              weighted_buffer.data());
        });
#endif
      } else if (is_3D) {
        float d_scaled_support, h_scaled_support, w_scaled_support;
        std::tie(d_scaled_support, h_scaled_support, w_scaled_support) = scaled_support_vals;
        int32_t d_window_size, h_window_size, w_window_size;
        std::tie(d_window_size, h_window_size, w_window_size) = window_sizes;
#ifdef CPU_TESTING
        for (int64_t id = 0, lim = output_height + output_width; id < lim; ++id) {
          DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
            _SetupTrilinerarUpsampleFilterAntiAlias<AccumType,
                                                    TriLinearFilter,
                                                    coord_t>(
                narrow<int32_t>(id),
                std::make_tuple(input_depth, input_height, input_width),
                std::make_tuple(output_depth, output_height, output_width),
                std::make_tuple(scales_vals[rank - 3], scales_vals[rank - 2], scales_vals[rank - 1]),
                std::make_tuple(roi_vals[rank - 3], roi_vals[rank - 2], roi_vals[rank - 1]),  // roi starts d, h, w
                std::make_tuple(roi_vals[rank - 3 + rank], roi_vals[rank - 2 + rank],         // roi ends d, h, w
                                roi_vals[rank - 1 + rank]),
                std::make_tuple(d_scaled_support, h_scaled_support, w_scaled_support),
                std::make_tuple(d_window_size, h_window_size, w_window_size),
                onnxruntime::narrow<size_t>(output_depth + output_height + output_width),
                bounds_buffer.data(),
                out_of_bounds_buffer.data(),
                weighted_buffer.data());
          });
        }

        PrintAntiAliasBuffers(std::cout, bounds_buffer, out_of_bounds_buffer, weighted_buffer);
#else
        DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
          _SetupTrilinerarUpsampleFilterAntiAlias<AccumType,
                                                  TriLinearFilter,
                                                  coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              std::make_tuple(input_depth, input_height, input_width),
              std::make_tuple(output_depth, output_height, output_width),
              std::make_tuple(scales_vals[rank - 3], scales_vals[rank - 2], scales_vals[rank - 1]),
              std::make_tuple(roi_vals[rank - 3], roi_vals[rank - 2], roi_vals[rank - 1]),  // roi starts d, h, w
              std::make_tuple(roi_vals[rank - 3 + rank], roi_vals[rank - 2 + rank],         // roi ends d, h, w
                              roi_vals[rank - 1 + rank]),
              std::make_tuple(d_scaled_support, h_scaled_support, w_scaled_support),
              std::make_tuple(d_window_size, h_window_size, w_window_size),
              onnxruntime::narrow<size_t>(output_depth + output_height + output_width),
              bounds_buffer.data(),
              out_of_bounds_buffer.data(),
              weighted_buffer.data());
        });
#endif
      }
    } break;
    case CUBIC: {
      if (is_2D) {
        // Compute scaled support values and windows sizes for the bilinear kernel
        float h_scaled_support, w_scaled_support;
        std::tie(std::ignore, h_scaled_support, w_scaled_support) = scaled_support_vals;
        int32_t h_window_size, w_window_size;
        std::tie(std::ignore, h_window_size, w_window_size) = window_sizes;

#ifdef CPU_TESTING

        for (int64_t id = 0, lim = output_height + output_width; id < lim; ++id) {
          DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
            _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                                  BiCubicFilter,
                                                  coord_t>(
                narrow<int32_t>(id),
                std::make_tuple(input_height, input_width),
                std::make_tuple(output_height, output_width),
                std::make_tuple(scales_vals[rank - 2], scales_vals[rank - 1]),
                std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
                std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
                std::make_tuple(h_scaled_support, w_scaled_support),
                std::make_tuple(h_window_size, w_window_size),
                onnxruntime::kCubicCoeffA, exclude_outside,
                onnxruntime::narrow<size_t>(output_height + output_width),
                bounds_buffer.data(),
                out_of_bounds_buffer.data(),
                weighted_buffer.data());
          });
        }
        PrintAntiAliasBuffers(std::cout, bounds_buffer, out_of_bounds_buffer, weighted_buffer);
#else
          DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
          _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                                BiCubicFilter,
                                                coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              std::make_tuple(input_height, input_width),
              std::make_tuple(output_height, output_width),
              std::make_tuple(scales_vals[rank - 2], scales_vals[rank - 1]),
              std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
              std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
              std::make_tuple(h_scaled_support, w_scaled_support),
              std::make_tuple(h_window_size, w_window_size),
              onnxruntime::kCubicCoeffA, exclude_outside,
              onnxruntime::narrow<size_t>(output_height + output_width),
              bounds_buffer.data(),
              out_of_bounds_buffer.data(),
              weighted_buffer.data());
        });
#endif
      } else {
        ORT_THROW("Resize supports only 2-D in CUBIC mode.");
      }
    } break;
    default:
      ORT_THROW("Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");
      break;
  }
}

#define SPECIALIZED_ANTIALIAS_IMPL(T)                                                        \
  template void ResizeAntiAliasImpl<T>(                                                      \
      cudaStream_t stream,                                                                   \
      int rank,                                                                              \
      const UpsampleMode upsample_mode,                                                      \
      ResizeCoordinateTransformationMode coordinate_transform_mode,                          \
      gsl::span<const int64_t> input_shape,                                                  \
      gsl::span<const int64_t> output_shape, /* const TArray<int64_t>& input_strides, */     \
      const TArray<fast_divmod>& output_div_pitches,                                         \
      gsl::span<const float> roi_vals,                                                       \
      gsl::span<const float> scales_vals, /* const std::optional<T>& extrapolation_value, */ \
      bool exclude_outside,                                                                  \
      std::tuple<float, float, float> scaled_support_vals,                                   \
      std::tuple<int32_t, int32_t, int32_t> window_sizes,                                    \
      gsl::span<int64_t> bounds_buffer,                                                      \
      gsl::span<int64_t> out_of_bounds_buffer,                                               \
      gsl::span<typename onnxruntime::AccumulateType<T>::type> weighted_buffer,              \
      const T* input_data,                                                                   \
      T* output_data,                                                                        \
      const size_t N);

SPECIALIZED_ANTIALIAS_IMPL(float)
SPECIALIZED_ANTIALIAS_IMPL(double)
SPECIALIZED_ANTIALIAS_IMPL(half)
SPECIALIZED_ANTIALIAS_IMPL(int32_t)
SPECIALIZED_ANTIALIAS_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime