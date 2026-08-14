#ifndef GRAPHFORGE_RUNTIME_CUDA_DRIVER_H
#define GRAPHFORGE_RUNTIME_CUDA_DRIVER_H

#include <cstddef>
#include <cstdint>

namespace graphforge::runtime::cuda {

using DevicePointer = uint64_t;
using Stream = void *;
using Event = void *;
using Module = void *;
using Function = void *;

bool available();
bool setDevice(int ordinal);
bool allocate(int ordinal, size_t bytes, DevicePointer &pointer);
bool release(int ordinal, DevicePointer pointer);
bool hostAllocate(int ordinal, size_t bytes, void *&pointer);
bool hostRelease(int ordinal, void *pointer);
bool copyHostToDevice(int ordinal, DevicePointer destination,
                      const void *source, size_t bytes);
bool copyDeviceToHost(int ordinal, void *destination, DevicePointer source,
                      size_t bytes);
bool copyHostToDeviceAsync(int ordinal, DevicePointer destination,
                           const void *source, size_t bytes, Stream stream);
bool copyDeviceToHostAsync(int ordinal, void *destination,
                           DevicePointer source, size_t bytes, Stream stream);
bool copyDeviceToDeviceAsync(int ordinal, DevicePointer destination,
                             DevicePointer source, size_t bytes, Stream stream);
bool createStream(int ordinal, Stream &stream);
bool destroyStream(int ordinal, Stream stream);
bool synchronizeStream(int ordinal, Stream stream);
bool createEvent(int ordinal, Event &event);
bool recordEvent(int ordinal, Event event, Stream stream);
bool queryEvent(int ordinal, Event event, bool &ready);
bool waitEvent(int ordinal, Event event);
bool streamWaitEvent(int ordinal, Stream stream, Event event);
bool destroyEvent(int ordinal, Event event);
bool loadModule(int ordinal, const char *image, size_t bytes, Module &module);
bool unloadModule(int ordinal, Module module);
bool getFunction(int ordinal, Module module, const char *name,
                 Function &function);
bool launch(int ordinal, Function function, Stream stream, uint32_t gridX,
            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
            uint32_t blockY, uint32_t blockZ, uint32_t sharedBytes,
            void **arguments);

} // namespace graphforge::runtime::cuda

#endif
