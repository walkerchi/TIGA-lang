#include "tiga/Runtime/CAPI.h"
#include "CudaDriver.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>

namespace {

bool isValidDevice(GFRTDevice device) {
  return device.ordinal >= 0 && device.type >= GFRT_DEVICE_CPU &&
         device.type <= GFRT_DEVICE_PPU;
}

struct RefCounted {
  std::atomic<uint32_t> references{1};
};

} // namespace

struct GFRTBuffer : RefCounted {
  GFRTDevice device;
  void *data;
  size_t bytes;
  size_t alignment;
  GFRTExternalDeleter deleter;
  void *deleterContext;
  bool owned;
  bool pinned;
};

struct GFRTStream : RefCounted {
  GFRTDevice device;
  void *native;
  bool owned;
};

struct GFRTEvent : RefCounted {
  GFRTDevice device;
  std::atomic<bool> ready{true};
  void *native;
};

struct GFRTModule : RefCounted {
  GFRTDevice device;
  void *native;
};

struct GFRTKernel : RefCounted {
  GFRTModule *module;
  void *native;
};

extern "C" {

const char *gfrt_status_string(GFRTStatus status) {
  switch (status) {
  case GFRT_STATUS_OK:
    return "ok";
  case GFRT_STATUS_INVALID_ARGUMENT:
    return "invalid argument";
  case GFRT_STATUS_UNSUPPORTED:
    return "unsupported";
  case GFRT_STATUS_OUT_OF_MEMORY:
    return "out of memory";
  case GFRT_STATUS_INTERNAL:
    return "internal error";
  }
  return "unknown status";
}

const char *gfrt_runtime_version(void) { return "0.1.1"; }

GFRTStatus gfrt_buffer_allocate(GFRTDevice device, size_t bytes,
                                size_t alignment, GFRTBuffer **outBuffer) {
  if (!outBuffer || !isValidDevice(device) || alignment == 0 ||
      (alignment & (alignment - 1)) != 0)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outBuffer = nullptr;
  if (device.type != GFRT_DEVICE_CPU && device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_UNSUPPORTED;
  try {
    void *data = nullptr;
    if (device.type == GFRT_DEVICE_CPU) {
      data = bytes ? ::operator new(bytes, std::align_val_t(alignment))
                   : nullptr;
    } else {
      tiga::runtime::cuda::DevicePointer pointer = 0;
      if (!tiga::runtime::cuda::allocate(device.ordinal, bytes, pointer))
        return tiga::runtime::cuda::available()
                   ? GFRT_STATUS_OUT_OF_MEMORY
                   : GFRT_STATUS_UNSUPPORTED;
      data = reinterpret_cast<void *>(static_cast<uintptr_t>(pointer));
    }
    *outBuffer = new GFRTBuffer{
        {}, device, data, bytes, alignment, nullptr, nullptr, true, false};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

GFRTStatus gfrt_buffer_allocate_pinned(size_t bytes, GFRTBuffer **outBuffer) {
  if (!outBuffer)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outBuffer = nullptr;
  try {
    void *data = nullptr;
    if (!tiga::runtime::cuda::hostAllocate(0, bytes, data))
      return tiga::runtime::cuda::available()
                 ? GFRT_STATUS_OUT_OF_MEMORY
                 : GFRT_STATUS_UNSUPPORTED;
    *outBuffer = new GFRTBuffer{
        {}, {GFRT_DEVICE_CPU, 0}, data, bytes, alignof(std::max_align_t),
        nullptr, nullptr, true, true};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

GFRTStatus gfrt_buffer_wrap_external(GFRTDevice device, void *data,
                                     size_t bytes,
                                     GFRTExternalDeleter deleter,
                                     void *deleterContext,
                                     GFRTBuffer **outBuffer) {
  if (!outBuffer || !isValidDevice(device) || (bytes != 0 && !data))
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outBuffer = nullptr;
  try {
    *outBuffer = new GFRTBuffer{
        {}, device, data, bytes, alignof(std::max_align_t), deleter,
        deleterContext, false, false};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

void gfrt_buffer_retain(GFRTBuffer *buffer) {
  if (buffer)
    buffer->references.fetch_add(1, std::memory_order_relaxed);
}

void gfrt_buffer_release(GFRTBuffer *buffer) {
  if (!buffer ||
      buffer->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  if (buffer->owned && buffer->pinned)
    tiga::runtime::cuda::hostRelease(0, buffer->data);
  else if (buffer->owned && buffer->device.type == GFRT_DEVICE_CPU)
    ::operator delete(buffer->data, std::align_val_t(buffer->alignment));
  else if (buffer->owned && buffer->device.type == GFRT_DEVICE_CUDA)
    tiga::runtime::cuda::release(
        buffer->device.ordinal,
        static_cast<tiga::runtime::cuda::DevicePointer>(
            reinterpret_cast<uintptr_t>(buffer->data)));
  else if (buffer->deleter)
    buffer->deleter(buffer->data, buffer->deleterContext);
  delete buffer;
}

GFRTDevice gfrt_buffer_device(const GFRTBuffer *buffer) {
  return buffer ? buffer->device : GFRTDevice{GFRT_DEVICE_CPU, -1};
}

size_t gfrt_buffer_size(const GFRTBuffer *buffer) {
  return buffer ? buffer->bytes : 0;
}

void *gfrt_buffer_data(GFRTBuffer *buffer) {
  if (!buffer || buffer->device.type != GFRT_DEVICE_CPU)
    return nullptr;
  return buffer->data;
}

uintptr_t gfrt_buffer_address(GFRTBuffer *buffer) {
  return buffer ? reinterpret_cast<uintptr_t>(buffer->data) : 0;
}

GFRTStatus gfrt_buffer_write(GFRTBuffer *buffer, size_t offset,
                             const void *source, size_t bytes) {
  if (!buffer || (!source && bytes) || offset > buffer->bytes ||
      bytes > buffer->bytes - offset)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (buffer->device.type == GFRT_DEVICE_CPU) {
    if (bytes)
      std::memcpy(static_cast<char *>(buffer->data) + offset, source, bytes);
    return GFRT_STATUS_OK;
  }
  if (buffer->device.type == GFRT_DEVICE_CUDA)
    return tiga::runtime::cuda::copyHostToDevice(
               buffer->device.ordinal,
               static_cast<tiga::runtime::cuda::DevicePointer>(
                   reinterpret_cast<uintptr_t>(buffer->data)) + offset,
               source, bytes)
               ? GFRT_STATUS_OK
               : GFRT_STATUS_INTERNAL;
  return GFRT_STATUS_UNSUPPORTED;
}

GFRTStatus gfrt_buffer_read(GFRTBuffer *buffer, size_t offset,
                            void *destination, size_t bytes) {
  if (!buffer || (!destination && bytes) || offset > buffer->bytes ||
      bytes > buffer->bytes - offset)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (buffer->device.type == GFRT_DEVICE_CPU) {
    if (bytes)
      std::memcpy(destination, static_cast<char *>(buffer->data) + offset, bytes);
    return GFRT_STATUS_OK;
  }
  if (buffer->device.type == GFRT_DEVICE_CUDA)
    return tiga::runtime::cuda::copyDeviceToHost(
               buffer->device.ordinal, destination,
               static_cast<tiga::runtime::cuda::DevicePointer>(
                   reinterpret_cast<uintptr_t>(buffer->data)) + offset,
               bytes)
               ? GFRT_STATUS_OK
               : GFRT_STATUS_INTERNAL;
  return GFRT_STATUS_UNSUPPORTED;
}

GFRTStatus gfrt_buffer_copy_async(
    GFRTBuffer *source, size_t sourceOffset, GFRTBuffer *destination,
    size_t destinationOffset, size_t bytes, GFRTStream *stream,
    GFRTEvent **outEvent) {
  if (!source || !destination || !stream || !outEvent ||
      sourceOffset > source->bytes || bytes > source->bytes - sourceOffset ||
      destinationOffset > destination->bytes ||
      bytes > destination->bytes - destinationOffset)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outEvent = nullptr;
  const bool sourceCPU = source->device.type == GFRT_DEVICE_CPU;
  const bool destinationCPU = destination->device.type == GFRT_DEVICE_CPU;
  if (sourceCPU && destinationCPU) {
    if (stream->device.type != GFRT_DEVICE_CPU)
      return GFRT_STATUS_INVALID_ARGUMENT;
    if (bytes)
      std::memcpy(static_cast<char *>(destination->data) + destinationOffset,
                  static_cast<char *>(source->data) + sourceOffset, bytes);
    return gfrt_event_create_completed(stream->device, outEvent);
  }
  if (stream->device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_INVALID_ARGUMENT;
  const int ordinal = stream->device.ordinal;
  bool copied = false;
  if (sourceCPU && destination->device.type == GFRT_DEVICE_CUDA &&
      destination->device.ordinal == ordinal) {
    copied = tiga::runtime::cuda::copyHostToDeviceAsync(
        ordinal,
        static_cast<tiga::runtime::cuda::DevicePointer>(
            reinterpret_cast<uintptr_t>(destination->data)) + destinationOffset,
        static_cast<char *>(source->data) + sourceOffset, bytes, stream->native);
  } else if (source->device.type == GFRT_DEVICE_CUDA &&
             source->device.ordinal == ordinal && destinationCPU) {
    copied = tiga::runtime::cuda::copyDeviceToHostAsync(
        ordinal, static_cast<char *>(destination->data) + destinationOffset,
        static_cast<tiga::runtime::cuda::DevicePointer>(
            reinterpret_cast<uintptr_t>(source->data)) + sourceOffset,
        bytes, stream->native);
  } else if (source->device.type == GFRT_DEVICE_CUDA &&
             destination->device.type == GFRT_DEVICE_CUDA &&
             source->device.ordinal == ordinal &&
             destination->device.ordinal == ordinal) {
    copied = tiga::runtime::cuda::copyDeviceToDeviceAsync(
        ordinal,
        static_cast<tiga::runtime::cuda::DevicePointer>(
            reinterpret_cast<uintptr_t>(destination->data)) + destinationOffset,
        static_cast<tiga::runtime::cuda::DevicePointer>(
            reinterpret_cast<uintptr_t>(source->data)) + sourceOffset,
        bytes, stream->native);
  } else {
    return GFRT_STATUS_UNSUPPORTED;
  }
  if (!copied)
    return GFRT_STATUS_INTERNAL;
  return gfrt_event_record(stream, outEvent);
}

GFRTStatus gfrt_stream_create(GFRTDevice device, GFRTStream **outStream) {
  if (!outStream || !isValidDevice(device))
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outStream = nullptr;
  if (device.type != GFRT_DEVICE_CPU && device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_UNSUPPORTED;
  try {
    void *native = nullptr;
    if (device.type == GFRT_DEVICE_CUDA &&
        !tiga::runtime::cuda::createStream(device.ordinal, native))
      return tiga::runtime::cuda::available()
                 ? GFRT_STATUS_INTERNAL
                 : GFRT_STATUS_UNSUPPORTED;
    *outStream = new GFRTStream{{}, device, native, true};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

GFRTStatus gfrt_stream_wrap_external(GFRTDevice device,
                                     uintptr_t nativeHandle,
                                     GFRTStream **outStream) {
  if (!outStream || !isValidDevice(device) ||
      device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_INVALID_ARGUMENT;
  try {
    *outStream = new GFRTStream{
        {}, device, reinterpret_cast<void *>(nativeHandle), false};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

void gfrt_stream_retain(GFRTStream *stream) {
  if (stream)
    stream->references.fetch_add(1, std::memory_order_relaxed);
}

void gfrt_stream_release(GFRTStream *stream) {
  if (stream &&
      stream->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    if (stream->device.type == GFRT_DEVICE_CUDA && stream->owned)
      tiga::runtime::cuda::destroyStream(stream->device.ordinal,
                                                stream->native);
    delete stream;
  }
}

GFRTDevice gfrt_stream_device(const GFRTStream *stream) {
  return stream ? stream->device : GFRTDevice{GFRT_DEVICE_CPU, -1};
}

uintptr_t gfrt_stream_address(const GFRTStream *stream) {
  return stream ? reinterpret_cast<uintptr_t>(stream->native) : 0;
}

GFRTStatus gfrt_stream_synchronize(GFRTStream *stream) {
  if (!stream)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (stream->device.type == GFRT_DEVICE_CUDA &&
      !tiga::runtime::cuda::synchronizeStream(stream->device.ordinal,
                                                     stream->native))
    return GFRT_STATUS_INTERNAL;
  return GFRT_STATUS_OK;
}

GFRTStatus gfrt_stream_wait_event(GFRTStream *stream, GFRTEvent *event) {
  if (!stream || !event || stream->device.type != event->device.type ||
      stream->device.ordinal != event->device.ordinal)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (stream->device.type == GFRT_DEVICE_CPU)
    return event->ready.load(std::memory_order_acquire) ? GFRT_STATUS_OK
                                                        : GFRT_STATUS_INTERNAL;
  if (stream->device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_UNSUPPORTED;
  return tiga::runtime::cuda::streamWaitEvent(
             stream->device.ordinal, stream->native, event->native)
             ? GFRT_STATUS_OK
             : GFRT_STATUS_INTERNAL;
}

GFRTStatus gfrt_event_create_completed(GFRTDevice device,
                                       GFRTEvent **outEvent) {
  if (!outEvent || !isValidDevice(device))
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outEvent = nullptr;
  try {
    void *native = nullptr;
    if (device.type == GFRT_DEVICE_CUDA) {
      if (!tiga::runtime::cuda::createEvent(device.ordinal, native) ||
          !tiga::runtime::cuda::recordEvent(device.ordinal, native,
                                                   nullptr)) {
        if (native)
          tiga::runtime::cuda::destroyEvent(device.ordinal, native);
        return tiga::runtime::cuda::available()
                   ? GFRT_STATUS_INTERNAL
                   : GFRT_STATUS_UNSUPPORTED;
      }
    } else if (device.type != GFRT_DEVICE_CPU) {
      return GFRT_STATUS_UNSUPPORTED;
    }
    *outEvent = new GFRTEvent{{}, device, true, native};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

void gfrt_event_retain(GFRTEvent *event) {
  if (event)
    event->references.fetch_add(1, std::memory_order_relaxed);
}

void gfrt_event_release(GFRTEvent *event) {
  if (event && event->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    if (event->device.type == GFRT_DEVICE_CUDA)
      tiga::runtime::cuda::destroyEvent(event->device.ordinal,
                                               event->native);
    delete event;
  }
}

GFRTDevice gfrt_event_device(const GFRTEvent *event) {
  return event ? event->device : GFRTDevice{GFRT_DEVICE_CPU, -1};
}

int gfrt_event_is_ready(const GFRTEvent *event) {
  if (!event)
    return 0;
  if (event->device.type == GFRT_DEVICE_CUDA) {
    bool ready = false;
    return tiga::runtime::cuda::queryEvent(
               event->device.ordinal, event->native, ready) && ready;
  }
  return event->ready.load(std::memory_order_acquire);
}

GFRTStatus gfrt_event_wait(GFRTEvent *event) {
  if (!event)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (event->device.type == GFRT_DEVICE_CUDA)
    return tiga::runtime::cuda::waitEvent(event->device.ordinal,
                                                 event->native)
               ? GFRT_STATUS_OK
               : GFRT_STATUS_INTERNAL;
  return event->ready.load(std::memory_order_acquire) ? GFRT_STATUS_OK
                                                      : GFRT_STATUS_INTERNAL;
}

GFRTStatus gfrt_event_record(GFRTStream *stream, GFRTEvent **outEvent) {
  if (!stream || !outEvent)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outEvent = nullptr;
  try {
    if (stream->device.type == GFRT_DEVICE_CPU) {
      *outEvent = new GFRTEvent{{}, stream->device, true, nullptr};
      return GFRT_STATUS_OK;
    }
    if (stream->device.type != GFRT_DEVICE_CUDA)
      return GFRT_STATUS_UNSUPPORTED;
    void *native = nullptr;
    if (!tiga::runtime::cuda::createEvent(stream->device.ordinal, native) ||
        !tiga::runtime::cuda::recordEvent(
            stream->device.ordinal, native, stream->native)) {
      if (native)
        tiga::runtime::cuda::destroyEvent(stream->device.ordinal, native);
      return GFRT_STATUS_INTERNAL;
    }
    *outEvent = new GFRTEvent{{}, stream->device, false, native};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

GFRTStatus gfrt_module_load(GFRTDevice device, const void *image, size_t bytes,
                            GFRTModule **outModule) {
  if (!outModule || !image || bytes == 0 || !isValidDevice(device))
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outModule = nullptr;
  if (device.type != GFRT_DEVICE_CUDA)
    return GFRT_STATUS_UNSUPPORTED;
  try {
    void *native = nullptr;
    if (!tiga::runtime::cuda::loadModule(
            device.ordinal, static_cast<const char *>(image), bytes, native))
      return tiga::runtime::cuda::available() ? GFRT_STATUS_INTERNAL
                                                     : GFRT_STATUS_UNSUPPORTED;
    *outModule = new GFRTModule{{}, device, native};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

void gfrt_module_release(GFRTModule *module) {
  if (!module ||
      module->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  tiga::runtime::cuda::unloadModule(module->device.ordinal,
                                           module->native);
  delete module;
}

GFRTStatus gfrt_module_get_kernel(GFRTModule *module, const char *name,
                                  GFRTKernel **outKernel) {
  if (!module || !name || !outKernel)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outKernel = nullptr;
  try {
    void *native = nullptr;
    if (!tiga::runtime::cuda::getFunction(
            module->device.ordinal, module->native, name, native))
      return GFRT_STATUS_INTERNAL;
    module->references.fetch_add(1, std::memory_order_relaxed);
    *outKernel = new GFRTKernel{{}, module, native};
  } catch (const std::bad_alloc &) {
    return GFRT_STATUS_OUT_OF_MEMORY;
  }
  return GFRT_STATUS_OK;
}

void gfrt_kernel_release(GFRTKernel *kernel) {
  if (!kernel ||
      kernel->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  gfrt_module_release(kernel->module);
  delete kernel;
}

GFRTStatus gfrt_kernel_launch(
    GFRTKernel *kernel, GFRTStream *stream, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedBytes, void **arguments, size_t argumentCount,
    GFRTEvent **outEvent) {
  if (!kernel || !stream || !outEvent || !gridX || !gridY || !gridZ ||
      !blockX || !blockY || !blockZ || (argumentCount && !arguments) ||
      stream->device.type != kernel->module->device.type ||
      stream->device.ordinal != kernel->module->device.ordinal)
    return GFRT_STATUS_INVALID_ARGUMENT;
  *outEvent = nullptr;
  if (!tiga::runtime::cuda::launch(
          stream->device.ordinal, kernel->native, stream->native, gridX, gridY,
          gridZ, blockX, blockY, blockZ, sharedBytes, arguments))
    return GFRT_STATUS_INTERNAL;
  return gfrt_event_record(stream, outEvent);
}

GFRTStatus gfrt_kernel_launch_async(
    GFRTKernel *kernel, GFRTStream *stream, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedBytes, void **arguments, size_t argumentCount) {
  if (!kernel || !stream || !gridX || !gridY || !gridZ || !blockX ||
      !blockY || !blockZ || (argumentCount && !arguments) ||
      stream->device.type != kernel->module->device.type ||
      stream->device.ordinal != kernel->module->device.ordinal)
    return GFRT_STATUS_INVALID_ARGUMENT;
  if (!tiga::runtime::cuda::launch(
          stream->device.ordinal, kernel->native, stream->native, gridX, gridY,
          gridZ, blockX, blockY, blockZ, sharedBytes, arguments))
    return GFRT_STATUS_INTERNAL;
  return GFRT_STATUS_OK;
}

} // extern "C"
