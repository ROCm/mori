#include <mpi.h>

#include "mori/application/memory/symmetric_memory.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

void* ShmemMalloc(size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::SymmMemObjPtr obj = states->memoryStates->symmMemMgr->Malloc(size);
  if (obj.IsValid()) {
    return obj.cpu->localPtr;
  }
  return nullptr;
}

void ShmemFree(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->memoryStates->symmMemMgr->Free(localPtr);
}

int ShmemBufferRegister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->memoryStates->mrMgr->RegisterBuffer(ptr, size);
  return 0;
}

int ShmemBufferUnRegister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->memoryStates->mrMgr->DeRegisterBuffer(ptr);
  return 0;
}

}  // namespace shmem
}  // namespace mori
