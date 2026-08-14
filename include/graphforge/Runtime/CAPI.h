#ifndef GRAPHFORGE_RUNTIME_CAPI_H
#define GRAPHFORGE_RUNTIME_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(GRAPHFORGE_RUNTIME_EXPORTS)
#define GFRT_API __declspec(dllexport)
#else
#define GFRT_API __declspec(dllimport)
#endif
#else
#define GFRT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GFRTStatus {
  GFRT_STATUS_OK = 0,
  GFRT_STATUS_INVALID_ARGUMENT = 1,
  GFRT_STATUS_UNSUPPORTED = 2,
  GFRT_STATUS_OUT_OF_MEMORY = 3,
  GFRT_STATUS_INTERNAL = 4,
} GFRTStatus;

typedef enum GFRTDeviceType {
  GFRT_DEVICE_CPU = 0,
  GFRT_DEVICE_CUDA = 1,
  GFRT_DEVICE_HIP = 2,
  GFRT_DEVICE_METAL = 3,
  GFRT_DEVICE_PPU = 4,
} GFRTDeviceType;

typedef struct GFRTDevice {
  GFRTDeviceType type;
  int32_t ordinal;
} GFRTDevice;

typedef struct GFRTBuffer GFRTBuffer;
typedef struct GFRTStream GFRTStream;
typedef struct GFRTEvent GFRTEvent;
typedef struct GFRTModule GFRTModule;
typedef struct GFRTKernel GFRTKernel;

typedef void (*GFRTExternalDeleter)(void *data, void *context);

GFRT_API const char *gfrt_status_string(GFRTStatus status);
GFRT_API const char *gfrt_runtime_version(void);

GFRT_API GFRTStatus gfrt_buffer_allocate(GFRTDevice device, size_t bytes,
                                         size_t alignment,
                                         GFRTBuffer **out_buffer);
GFRT_API GFRTStatus gfrt_buffer_allocate_pinned(size_t bytes,
                                                GFRTBuffer **out_buffer);
GFRT_API GFRTStatus gfrt_buffer_wrap_external(
    GFRTDevice device, void *data, size_t bytes, GFRTExternalDeleter deleter,
    void *deleter_context, GFRTBuffer **out_buffer);
GFRT_API void gfrt_buffer_retain(GFRTBuffer *buffer);
GFRT_API void gfrt_buffer_release(GFRTBuffer *buffer);
GFRT_API GFRTDevice gfrt_buffer_device(const GFRTBuffer *buffer);
GFRT_API size_t gfrt_buffer_size(const GFRTBuffer *buffer);
GFRT_API void *gfrt_buffer_data(GFRTBuffer *buffer);
GFRT_API uintptr_t gfrt_buffer_address(GFRTBuffer *buffer);
GFRT_API GFRTStatus gfrt_buffer_write(GFRTBuffer *buffer, size_t offset,
                                      const void *source, size_t bytes);
GFRT_API GFRTStatus gfrt_buffer_read(GFRTBuffer *buffer, size_t offset,
                                     void *destination, size_t bytes);
GFRT_API GFRTStatus gfrt_buffer_copy_async(
    GFRTBuffer *source, size_t source_offset, GFRTBuffer *destination,
    size_t destination_offset, size_t bytes, GFRTStream *stream,
    GFRTEvent **out_event);

GFRT_API GFRTStatus gfrt_stream_create(GFRTDevice device,
                                       GFRTStream **out_stream);
GFRT_API GFRTStatus gfrt_stream_wrap_external(GFRTDevice device,
                                              uintptr_t native_handle,
                                              GFRTStream **out_stream);
GFRT_API void gfrt_stream_retain(GFRTStream *stream);
GFRT_API void gfrt_stream_release(GFRTStream *stream);
GFRT_API GFRTDevice gfrt_stream_device(const GFRTStream *stream);
GFRT_API uintptr_t gfrt_stream_address(const GFRTStream *stream);
GFRT_API GFRTStatus gfrt_stream_synchronize(GFRTStream *stream);
GFRT_API GFRTStatus gfrt_stream_wait_event(GFRTStream *stream,
                                           GFRTEvent *event);

GFRT_API GFRTStatus gfrt_event_create_completed(GFRTDevice device,
                                                GFRTEvent **out_event);
GFRT_API void gfrt_event_retain(GFRTEvent *event);
GFRT_API void gfrt_event_release(GFRTEvent *event);
GFRT_API GFRTDevice gfrt_event_device(const GFRTEvent *event);
GFRT_API int gfrt_event_is_ready(const GFRTEvent *event);
GFRT_API GFRTStatus gfrt_event_wait(GFRTEvent *event);
GFRT_API GFRTStatus gfrt_event_record(GFRTStream *stream,
                                      GFRTEvent **out_event);

GFRT_API GFRTStatus gfrt_module_load(GFRTDevice device, const void *image,
                                     size_t bytes, GFRTModule **out_module);
GFRT_API void gfrt_module_release(GFRTModule *module);
GFRT_API GFRTStatus gfrt_module_get_kernel(GFRTModule *module,
                                           const char *name,
                                           GFRTKernel **out_kernel);
GFRT_API void gfrt_kernel_release(GFRTKernel *kernel);
GFRT_API GFRTStatus gfrt_kernel_launch(
    GFRTKernel *kernel, GFRTStream *stream, uint32_t grid_x, uint32_t grid_y,
    uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_bytes, void **arguments, size_t argument_count,
    GFRTEvent **out_event);
GFRT_API GFRTStatus gfrt_kernel_launch_async(
    GFRTKernel *kernel, GFRTStream *stream, uint32_t grid_x, uint32_t grid_y,
    uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_bytes, void **arguments, size_t argument_count);

#ifdef __cplusplus
}
#endif

#endif
