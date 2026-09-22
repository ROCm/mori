// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

// Host-only CQ-mode policy shared by CQ creation and both JIT compilers.
// Keep this independent of verbs, application, CCO, and the JIT: importing any
// of those here would make the compiler depend on the transport it compiles.
#include <dlfcn.h>

#include <array>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace mori {
namespace utils {

inline bool IonicCcqeDisabled(const char* value) {
  if (!value) return false;
  std::string lower(value);
  for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return lower == "1" || lower == "true" || lower == "on" || lower == "yes";
}

inline std::optional<std::array<int, 4>> ParseIonicFirmwareVersion(const std::string& text) {
  std::array<int, 4> version{};
  const char* cursor = text.data();
  const char* end = cursor + text.size();
  // Both spellings occur in sysfs. Reject trailing junk and unknown release tags.
  for (int i = 0; i < 4; ++i) {
    if (cursor == end || *cursor < '0' || *cursor > '9') return std::nullopt;
    const auto parsed = std::from_chars(cursor, end, version[i]);
    if (parsed.ec != std::errc{}) return std::nullopt;
    cursor = parsed.ptr;
    if (i < 2) {
      if (cursor == end || *cursor++ != '.') return std::nullopt;
    } else if (i == 2) {
      if (end - cursor < 2 || cursor[0] != '-' || cursor[1] != 'a') return std::nullopt;
      cursor += 2;
      if (cursor != end && *cursor == '-') ++cursor;
    }
  }
  if (cursor != end) return std::nullopt;
  return version;
}

// The arguments are also the CPU-test seam: temporary sysfs trees and a driver
// capability value cover old drivers/firmware without opening an RDMA device.
inline bool DetectIonicCcqe(const std::filesystem::path& ibRoot, bool driverSupportsCcqe,
                            const char* disable) {
  if (IonicCcqeDisabled(disable) || !driverSupportsCcqe) return false;
  constexpr std::array<int, 4> minimum{1, 117, 5, 58};
  std::optional<std::string> firstVersion;
  std::error_code error;
  std::filesystem::directory_iterator it(ibRoot, error), end;
  if (error) return false;
  for (; it != end; it.increment(error)) {
    if (error) return false;
    auto driver = std::filesystem::read_symlink(it->path() / "device/driver", error);
    if (error) {
      // An unclassified device may be an Ionic rail with unreadable sysfs;
      // enabling CCQE globally would also affect that rail's CQ constructor.
      return false;
    }
    if (driver.filename() != "ionic" && driver.filename() != "ionic_rdma") continue;
    std::ifstream firmware(it->path() / "fw_ver");
    std::string text;
    if (!std::getline(firmware, text)) return false;
    const auto first = text.find_first_not_of(" \t\r\n");
    const auto last = text.find_last_not_of(" \t\r\n");
    text = first == std::string::npos ? "" : text.substr(first, last - first + 1);
    const auto version = ParseIonicFirmwareVersion(text);
    if (!version || *version < minimum) return false;
    // A process compiles one CQ protocol. Use the existing v1 conservative
    // policy for mixed firmware instead of creating incompatible CQ modes on
    // different rails of the same communicator.
    if (firstVersion && *firstVersion != text) return false;
    firstVersion = text;
  }
  return !error && firstVersion.has_value();
}

inline bool IonicDriverSupportsCcqe() {
  // Exactly the library IonicDvApi loads; a different SONAME probe can select a
  // different installation and disagree with the CQ constructor.
  // Providers register callbacks with libibverbs when loaded. Retain the handle
  // for the process lifetime, like IonicDvApi: unloading it can leave those
  // callbacks dangling before the first ibv_get_device_list() call.
  static void* const library = dlopen("libionic.so", RTLD_LAZY | RTLD_LOCAL);
  if (!library) return false;
  return dlsym(library, "ionic_dv_create_cq_ex") != nullptr;
}

inline bool IonicCcqeEnabled() {
  // CQ creation and compilation must see the same startup environment. Changing
  // firmware, driver libraries, or this env var after initialization is not
  // supported; rebuild the process/communicator to change modes.
  static const bool enabled = [] {
    const char* disable = std::getenv("MORI_DISABLE_IONIC_CCQE");
    // Do not load a provider solely to discover that the user disabled CCQE.
    if (IonicCcqeDisabled(disable)) return false;
    return DetectIonicCcqe("/sys/class/infiniband", IonicDriverSupportsCcqe(), disable);
  }();
  return enabled;
}

}  // namespace utils
}  // namespace mori
