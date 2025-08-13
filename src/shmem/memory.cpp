#include <mpi.h>

#include "mori/application/memory/symmetric_memory.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

void* ShmemMalloc(size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  application::SymmMemObjPtr obj = states->memoryStates->symmMemMgr->Malloc(size);
  if (obj.IsValid()) {
    return obj.cpu->localPtr;
  }
  return nullptr;
}

void* ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  application::SymmMemObjPtr obj = states->memoryStates->symmMemMgr->ExtMallocWihFlags(size, flags);
  if (obj.IsValid()) {
    return obj.cpu->localPtr;
  }
  return nullptr;
}

void ShmemFree(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  states->memoryStates->symmMemMgr->Free(localPtr);
}

application::SymmMemObjPtr ShmemQueryMemObjPtr(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  return states->memoryStates->symmMemMgr->Get(localPtr);
}

int ShmemBufferRegister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  states->memoryStates->mrMgr->RegisterBuffer(ptr, size);
  return 0;
}

int ShmemBufferDeregister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  states->memoryStates->mrMgr->DeregisterBuffer(ptr);
  return 0;
}

}  // namespace shmem
}  // namespace mori
