// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cstddef>
#include <cstdint>

#include "mori/application/application_device_types.hpp"

namespace mori {
namespace collective {

struct CrossPeBarrier;

template <typename T>
struct CclAll2allArgs {
  int myPe;
  int npes;
  T* input;
  application::SymmMemObjPtr inputTransitMemObj;
  application::SymmMemObjPtr outputTransitMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
};

template <typename T>
struct CclAllgatherArgs {
  int myPe;
  int npes;
  T* input;
  application::SymmMemObjPtr srcMemObj;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t elementCount;
  size_t dstBaseOffset;
  uint64_t flagVal;
};

template <typename T>
struct CclAllreduceArgs {
  int myPe;
  int npes;
  const T* input;
  application::SymmMemObjPtr dstMemObj;
  application::SymmMemObjPtr flagsMemObj;
  CrossPeBarrier* barrier;
  size_t elementCount;
};

}  // namespace collective
}  // namespace mori
