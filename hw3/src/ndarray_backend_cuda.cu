#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t IndexToOffset(size_t idx, CudaVec shape, CudaVec strides, size_t offset) {
  for (int i = static_cast<int>(shape.size) - 1; i >= 0; --i) {
    size_t cur = idx % shape.data[i];
    idx /= shape.data[i];
    offset += cur * strides.data[i];
  }
  return offset;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    out[gid] = a[IndexToOffset(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[IndexToOffset(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[IndexToOffset(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

#define DEFINE_EWISE_BINARY_OP(KERNEL_NAME, FUNC_NAME, EXPR) \
  __global__ void KERNEL_NAME(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = (EXPR); \
  } \
  void FUNC_NAME(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    KERNEL_NAME<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
  }

#define DEFINE_SCALAR_BINARY_OP(KERNEL_NAME, FUNC_NAME, EXPR) \
  __global__ void KERNEL_NAME(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = (EXPR); \
  } \
  void FUNC_NAME(const CudaArray& a, scalar_t val, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    KERNEL_NAME<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
  }

#define DEFINE_EWISE_UNARY_OP(KERNEL_NAME, FUNC_NAME, EXPR) \
  __global__ void KERNEL_NAME(const scalar_t* a, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = (EXPR); \
  } \
  void FUNC_NAME(const CudaArray& a, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    KERNEL_NAME<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
  }

DEFINE_EWISE_BINARY_OP(EwiseMulKernel, EwiseMul, a[gid] * b[gid])
DEFINE_SCALAR_BINARY_OP(ScalarMulKernel, ScalarMul, a[gid] * val)
DEFINE_EWISE_BINARY_OP(EwiseDivKernel, EwiseDiv, a[gid] / b[gid])
DEFINE_SCALAR_BINARY_OP(ScalarDivKernel, ScalarDiv, a[gid] / val)
DEFINE_SCALAR_BINARY_OP(ScalarPowerKernel, ScalarPower, pow(a[gid], val))
DEFINE_EWISE_BINARY_OP(EwiseMaximumKernel, EwiseMaximum, a[gid] > b[gid] ? a[gid] : b[gid])
DEFINE_SCALAR_BINARY_OP(ScalarMaximumKernel, ScalarMaximum, a[gid] > val ? a[gid] : val)
DEFINE_EWISE_BINARY_OP(EwiseEqKernel, EwiseEq, static_cast<scalar_t>(a[gid] == b[gid]))
DEFINE_SCALAR_BINARY_OP(ScalarEqKernel, ScalarEq, static_cast<scalar_t>(a[gid] == val))
DEFINE_EWISE_BINARY_OP(EwiseGeKernel, EwiseGe, static_cast<scalar_t>(a[gid] >= b[gid]))
DEFINE_SCALAR_BINARY_OP(ScalarGeKernel, ScalarGe, static_cast<scalar_t>(a[gid] >= val))
DEFINE_EWISE_UNARY_OP(EwiseLogKernel, EwiseLog, log(a[gid]))
DEFINE_EWISE_UNARY_OP(EwiseExpKernel, EwiseExp, exp(a[gid]))
DEFINE_EWISE_UNARY_OP(EwiseTanhKernel, EwiseTanh, tanh(a[gid]))

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < P) {
    scalar_t sum = 0;
    for (uint32_t k = 0; k < N; ++k) {
      sum += a[row * N + k] * b[k * P + col];
    }
    out[row * P + col] = sum;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /// BEGIN SOLUTION
  dim3 block(16, 16, 1);
  dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t mx = a[gid * reduce_size];
    for (size_t j = 1; j < reduce_size; ++j) {
      scalar_t val = a[gid * reduce_size + j];
      mx = mx > val ? mx : val;
    }
    out[gid] = mx;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t sum = 0;
    for (size_t j = 0; j < reduce_size; ++j) {
      sum += a[gid * reduce_size + j];
    }
    out[gid] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
