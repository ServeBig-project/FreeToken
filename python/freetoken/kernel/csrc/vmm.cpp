// CUDA virtual memory management for the shared runtime pool: physical blocks of the
// device's allocation granularity, stable reserved address ranges, and per-block
// map/unmap. The Python owner decides which block backs which address.
#include <cstdint>
#include <cuda.h>
#include <torch/extension.h>

namespace {

void check(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, what, " failed: ", message ? message : "unknown error");
  }
}

// Driver calls need the device's primary context current on this thread; PyTorch
// creates it on first CUDA use but may not have made it current here.
void use_device(int64_t device) {
  check(cuInit(0), "cuInit");
  CUdevice dev;
  check(cuDeviceGet(&dev, static_cast<int>(device)), "cuDeviceGet");
  CUcontext ctx;
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  check(cuDevicePrimaryCtxRelease(dev), "cuDevicePrimaryCtxRelease");
}

CUmemAllocationProp properties(int64_t device) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device);
  return prop;
}

bool supported(int64_t device) {
  use_device(device);
  int value = 0;
  check(cuDeviceGetAttribute(
            &value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            static_cast<int>(device)),
        "cuDeviceGetAttribute");
  return value != 0;
}

int64_t granularity(int64_t device) {
  use_device(device);
  CUmemAllocationProp prop = properties(device);
  size_t size = 0;
  check(cuMemGetAllocationGranularity(&size, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
  return static_cast<int64_t>(size);
}

int64_t create(int64_t device, int64_t size) {
  use_device(device);
  CUmemAllocationProp prop = properties(device);
  CUmemGenericAllocationHandle handle;
  check(cuMemCreate(&handle, static_cast<size_t>(size), &prop, 0), "cuMemCreate");
  return static_cast<int64_t>(handle);
}

void release(int64_t handle) {
  check(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle)),
        "cuMemRelease");
}

int64_t reserve(int64_t device, int64_t size, int64_t alignment) {
  use_device(device);
  CUdeviceptr address = 0;
  check(cuMemAddressReserve(&address, static_cast<size_t>(size),
                            static_cast<size_t>(alignment), 0, 0),
        "cuMemAddressReserve");
  return static_cast<int64_t>(address);
}

void free_range(int64_t address, int64_t size) {
  check(cuMemAddressFree(static_cast<CUdeviceptr>(address),
                         static_cast<size_t>(size)),
        "cuMemAddressFree");
}

void map(int64_t device, int64_t address, int64_t size, int64_t handle) {
  use_device(device);
  check(cuMemMap(static_cast<CUdeviceptr>(address), static_cast<size_t>(size), 0,
                 static_cast<CUmemGenericAllocationHandle>(handle), 0),
        "cuMemMap");
}

// One access grant covers a run of freshly mapped blocks: the grant, not the map, is
// the expensive driver call.
void set_access(int64_t device, int64_t address, int64_t size) {
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(device);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  check(cuMemSetAccess(static_cast<CUdeviceptr>(address), static_cast<size_t>(size),
                       &access, 1),
        "cuMemSetAccess");
}

void unmap(int64_t address, int64_t size) {
  check(cuMemUnmap(static_cast<CUdeviceptr>(address), static_cast<size_t>(size)),
        "cuMemUnmap");
}

// A tensor over reserved addresses. It does not own them: the pool keeps the range
// reserved until every view is gone, and only mapped parts may be touched. The device
// is given, not queried, because the base address itself may be unmapped.
torch::Tensor view(int64_t address, std::vector<int64_t> sizes,
                   std::vector<int64_t> strides, at::ScalarType dtype,
                   int64_t device) {
  c10::Device target(torch::kCUDA, static_cast<c10::DeviceIndex>(device));
  return torch::for_blob(reinterpret_cast<void *>(address), sizes)
      .strides(strides)
      .deleter([](void *) {})
      .options(torch::TensorOptions().dtype(dtype).device(target))
      .target_device(target)
      .make_tensor();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supported", &supported);
  m.def("granularity", &granularity);
  m.def("create", &create);
  m.def("release", &release);
  m.def("reserve", &reserve);
  m.def("free_range", &free_range);
  m.def("map", &map);
  m.def("set_access", &set_access);
  m.def("unmap", &unmap);
  m.def("view", &view);
}
