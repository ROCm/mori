// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Based on PR #173 by Chao Chen <cchen104@amd.com>
// Adapted for the refactored architecture.
//
// Compile-time contract test: verifies that the FFI handler C symbols and the
// handle manager API exist with the expected signatures. If someone changes
// dispatch_combine.hpp or mori_xla_ffi_ops.cpp in an incompatible way, this
// test will fail to compile.

#include <cstdint>
#include <type_traits>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "src/ffi/mori_xla_ffi_handle_mgr.hpp"

// Verify HandleManager singleton API
static_assert(std::is_same_v<
    decltype(&mori::ffi::HandleManager::Instance),
    mori::ffi::HandleManager& (*)()>,
    "HandleManager::Instance() signature changed");

// Verify EpDispatchCombineConfig fields used by FFI handle creation
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::rank), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::worldSize), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::hiddenDim), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::scaleDim), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::maxNumInpTokenPerRank), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::numExpertPerRank), int>);
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineConfig::numExpertPerToken), int>);

// Verify EpDispatchCombineHandle key member types used by FFI ops
static_assert(std::is_same_v<decltype(mori::moe::EpDispatchCombineHandle::curRankNumToken),
                             mori::moe::index_t>);

// Verify extern "C" symbols exist (linker will catch if missing)
extern "C" {
int64_t mori_ffi_create_handle(int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
void mori_ffi_destroy_handle(int64_t);
void mori_ffi_register_kernel_module(int, const char*);
}

int main() {
  auto& mgr = mori::ffi::HandleManager::Instance();
  (void)mgr;

  auto create_fn = &mori_ffi_create_handle;
  auto destroy_fn = &mori_ffi_destroy_handle;
  auto register_fn = &mori_ffi_register_kernel_module;
  (void)create_fn;
  (void)destroy_fn;
  (void)register_fn;

  return 0;
}
