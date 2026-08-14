#include "CudaDriver.h"

#include <array>
#include <cstring>
#include <mutex>
#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace graphforge::runtime::cuda {
namespace {

using Result = int;
using Device = int;
using Context = void *;
constexpr Result success = 0;
constexpr Result notReady = 600;

struct API {
  void *library = nullptr;
  Result (*init)(unsigned) = nullptr;
  Result (*deviceGet)(Device *, int) = nullptr;
  Result (*primaryContextRetain)(Context *, Device) = nullptr;
  Result (*contextSetCurrent)(Context) = nullptr;
  Result (*contextSynchronize)() = nullptr;
  Result (*memoryAllocate)(DevicePointer *, size_t) = nullptr;
  Result (*memoryFree)(DevicePointer) = nullptr;
  Result (*hostAllocate)(void **, size_t, unsigned) = nullptr;
  Result (*hostFree)(void *) = nullptr;
  Result (*copyHostToDevice)(DevicePointer, const void *, size_t) = nullptr;
  Result (*copyDeviceToHost)(void *, DevicePointer, size_t) = nullptr;
  Result (*copyHostToDeviceAsync)(DevicePointer, const void *, size_t, Stream) = nullptr;
  Result (*copyDeviceToHostAsync)(void *, DevicePointer, size_t, Stream) = nullptr;
  Result (*copyDeviceToDeviceAsync)(DevicePointer, DevicePointer, size_t, Stream) = nullptr;
  Result (*streamCreate)(Stream *, unsigned) = nullptr;
  Result (*streamDestroy)(Stream) = nullptr;
  Result (*streamSynchronize)(Stream) = nullptr;
  Result (*streamWaitEvent)(Stream, Event, unsigned) = nullptr;
  Result (*eventCreate)(Event *, unsigned) = nullptr;
  Result (*eventRecord)(Event, Stream) = nullptr;
  Result (*eventQuery)(Event) = nullptr;
  Result (*eventSynchronize)(Event) = nullptr;
  Result (*eventDestroy)(Event) = nullptr;
  Result (*moduleLoadDataEx)(Module *, const void *, unsigned, void *, void *) = nullptr;
  Result (*moduleUnload)(Module) = nullptr;
  Result (*moduleGetFunction)(Function *, Module, const char *) = nullptr;
  Result (*launchKernel)(Function, unsigned, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, Stream, void **,
                         void **) = nullptr;
  bool loaded = false;
};

template <typename T>
bool symbol(void *library, const char *name, T &destination) {
#if !defined(_WIN32)
  destination = reinterpret_cast<T>(dlsym(library, name));
  return destination != nullptr;
#else
  (void)library;
  (void)name;
  (void)destination;
  return false;
#endif
}

API &api() {
  static API instance;
  static std::once_flag once;
  std::call_once(once, [&] {
#if !defined(_WIN32)
    instance.library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!instance.library)
      return;
    bool ok = true;
    ok &= symbol(instance.library, "cuInit", instance.init);
    ok &= symbol(instance.library, "cuDeviceGet", instance.deviceGet);
    ok &= symbol(instance.library, "cuDevicePrimaryCtxRetain",
                 instance.primaryContextRetain);
    ok &= symbol(instance.library, "cuCtxSetCurrent", instance.contextSetCurrent);
    ok &= symbol(instance.library, "cuCtxSynchronize", instance.contextSynchronize);
    ok &= symbol(instance.library, "cuMemAlloc_v2", instance.memoryAllocate);
    ok &= symbol(instance.library, "cuMemFree_v2", instance.memoryFree);
    ok &= symbol(instance.library, "cuMemHostAlloc", instance.hostAllocate);
    ok &= symbol(instance.library, "cuMemFreeHost", instance.hostFree);
    ok &= symbol(instance.library, "cuMemcpyHtoD_v2",
                 instance.copyHostToDevice);
    ok &= symbol(instance.library, "cuMemcpyDtoH_v2",
                 instance.copyDeviceToHost);
    ok &= symbol(instance.library, "cuMemcpyHtoDAsync_v2",
                 instance.copyHostToDeviceAsync);
    ok &= symbol(instance.library, "cuMemcpyDtoHAsync_v2",
                 instance.copyDeviceToHostAsync);
    ok &= symbol(instance.library, "cuMemcpyDtoDAsync_v2",
                 instance.copyDeviceToDeviceAsync);
    ok &= symbol(instance.library, "cuStreamCreate", instance.streamCreate);
    ok &= symbol(instance.library, "cuStreamDestroy_v2", instance.streamDestroy);
    ok &= symbol(instance.library, "cuStreamSynchronize",
                 instance.streamSynchronize);
    ok &= symbol(instance.library, "cuStreamWaitEvent", instance.streamWaitEvent);
    ok &= symbol(instance.library, "cuEventCreate", instance.eventCreate);
    ok &= symbol(instance.library, "cuEventRecord", instance.eventRecord);
    ok &= symbol(instance.library, "cuEventQuery", instance.eventQuery);
    ok &= symbol(instance.library, "cuEventSynchronize", instance.eventSynchronize);
    ok &= symbol(instance.library, "cuEventDestroy_v2", instance.eventDestroy);
    ok &= symbol(instance.library, "cuModuleLoadDataEx", instance.moduleLoadDataEx);
    ok &= symbol(instance.library, "cuModuleUnload", instance.moduleUnload);
    ok &= symbol(instance.library, "cuModuleGetFunction",
                 instance.moduleGetFunction);
    ok &= symbol(instance.library, "cuLaunchKernel", instance.launchKernel);
    instance.loaded = ok && instance.init(0) == success;
#endif
  });
  return instance;
}

bool current(int ordinal) {
  API &driver = api();
  if (!driver.loaded || ordinal < 0)
    return false;
  Device device = 0;
  Context context = nullptr;
  return driver.deviceGet(&device, ordinal) == success &&
         driver.primaryContextRetain(&context, device) == success &&
         driver.contextSetCurrent(context) == success;
}

} // namespace

bool available() { return api().loaded; }
bool setDevice(int ordinal) { return current(ordinal); }

bool allocate(int ordinal, size_t bytes, DevicePointer &pointer) {
  pointer = 0;
  return current(ordinal) &&
         (bytes == 0 || api().memoryAllocate(&pointer, bytes) == success);
}

bool release(int ordinal, DevicePointer pointer) {
  return pointer == 0 || (current(ordinal) && api().memoryFree(pointer) == success);
}
bool hostAllocate(int ordinal, size_t bytes, void *&pointer) {
  pointer = nullptr;
  return current(ordinal) &&
         (bytes == 0 || api().hostAllocate(&pointer, bytes, 0) == success);
}
bool hostRelease(int ordinal, void *pointer) {
  return !pointer || (current(ordinal) && api().hostFree(pointer) == success);
}
bool copyHostToDevice(int ordinal, DevicePointer destination,
                      const void *source, size_t bytes) {
  return bytes == 0 ||
         (destination && source && current(ordinal) &&
          api().copyHostToDevice(destination, source, bytes) == success &&
          api().contextSynchronize() == success);
}
bool copyDeviceToHost(int ordinal, void *destination, DevicePointer source,
                      size_t bytes) {
  return bytes == 0 ||
         (destination && source && current(ordinal) &&
          api().copyDeviceToHost(destination, source, bytes) == success &&
          api().contextSynchronize() == success);
}
bool copyHostToDeviceAsync(int ordinal, DevicePointer destination,
                           const void *source, size_t bytes, Stream stream) {
  return bytes == 0 || (destination && source && stream && current(ordinal) &&
         api().copyHostToDeviceAsync(destination, source, bytes, stream) == success);
}
bool copyDeviceToHostAsync(int ordinal, void *destination,
                           DevicePointer source, size_t bytes, Stream stream) {
  return bytes == 0 || (destination && source && stream && current(ordinal) &&
         api().copyDeviceToHostAsync(destination, source, bytes, stream) == success);
}
bool copyDeviceToDeviceAsync(int ordinal, DevicePointer destination,
                             DevicePointer source, size_t bytes, Stream stream) {
  return bytes == 0 || (destination && source && stream && current(ordinal) &&
         api().copyDeviceToDeviceAsync(destination, source, bytes, stream) == success);
}

bool createStream(int ordinal, Stream &stream) {
  stream = nullptr;
  return current(ordinal) && api().streamCreate(&stream, 1) == success;
}
bool destroyStream(int ordinal, Stream stream) {
  return !stream || (current(ordinal) && api().streamDestroy(stream) == success);
}
bool synchronizeStream(int ordinal, Stream stream) {
  return stream && current(ordinal) && api().streamSynchronize(stream) == success;
}
bool createEvent(int ordinal, Event &event) {
  event = nullptr;
  return current(ordinal) && api().eventCreate(&event, 2) == success;
}
bool recordEvent(int ordinal, Event event, Stream stream) {
  return event && current(ordinal) && api().eventRecord(event, stream) == success;
}
bool queryEvent(int ordinal, Event event, bool &ready) {
  ready = false;
  if (!event || !current(ordinal))
    return false;
  Result result = api().eventQuery(event);
  ready = result == success;
  return result == success || result == notReady;
}
bool waitEvent(int ordinal, Event event) {
  return event && current(ordinal) && api().eventSynchronize(event) == success;
}
bool streamWaitEvent(int ordinal, Stream stream, Event event) {
  return stream && event && current(ordinal) &&
         api().streamWaitEvent(stream, event, 0) == success;
}
bool destroyEvent(int ordinal, Event event) {
  return !event || (current(ordinal) && api().eventDestroy(event) == success);
}
bool loadModule(int ordinal, const char *image, size_t bytes, Module &module) {
  module = nullptr;
  if (!image || bytes == 0 || !current(ordinal))
    return false;
  // CUDA accepts a nul-terminated PTX image. Preserve embedded bytes while
  // guaranteeing the terminator independently of the Python caller.
  std::string owned(image, bytes);
  owned.push_back('\0');
  return api().moduleLoadDataEx(&module, owned.data(), 0, nullptr, nullptr) == success;
}
bool unloadModule(int ordinal, Module module) {
  return !module || (current(ordinal) && api().moduleUnload(module) == success);
}
bool getFunction(int ordinal, Module module, const char *name,
                 Function &function) {
  function = nullptr;
  return module && name && current(ordinal) &&
         api().moduleGetFunction(&function, module, name) == success;
}
bool launch(int ordinal, Function function, Stream stream, uint32_t gridX,
            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
            uint32_t blockY, uint32_t blockZ, uint32_t sharedBytes,
            void **arguments) {
  return function && current(ordinal) &&
         api().launchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                            blockZ, sharedBytes, stream, arguments, nullptr) == success;
}

} // namespace graphforge::runtime::cuda
