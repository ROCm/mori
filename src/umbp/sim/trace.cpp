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
#include "umbp/sim/trace.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <stdexcept>
#include <string_view>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

SimTrace::SimTrace(const UMBPSimConfig& config) {
  google::protobuf::Value profile;
  profile.set_string_value(config.profile_path);
  const auto status = google::protobuf::util::MessageToJsonString(profile, &profile_json_);
  if (!status.ok()) throw std::invalid_argument(status.ToString());
  if (!config.trace_path.empty()) {
    output_.open(config.trace_path, std::ios::app);
    if (!output_) throw std::invalid_argument("UMBP sim: cannot open trace " + config.trace_path);
  }
}

void SimTrace::Write(uint64_t id, const char* api, size_t keys, uint64_t bytes, uint64_t start_ns,
                     uint64_t elapsed_ns, int64_t result_count, bool exception) {
  static auto logger = MORI_GET_LOGGER(mori::modules::UMBP);
  if (!output_.is_open() && !logger->should_log(spdlog::level::info)) return;
  // API names are fixed internal identifiers; the user-supplied path is escaped once.
  const auto json = fmt::format(
      R"({{"call_id":{},"api":"{}","key_count":{},"bytes":{},"start_ns":{},)"
      R"("elapsed_ns":{},"result_count":{},"exception":{},"profile_path":{}}})",
      id, api, keys, bytes, start_ns, elapsed_ns, result_count, exception, profile_json_);
  std::lock_guard<std::mutex> lock(mutex_);
  if (output_.is_open()) {
    output_ << json << '\n';
    if (std::string_view(api) == "flush" || std::string_view(api) == "close") output_.flush();
    if (!output_) throw std::runtime_error("UMBP sim: trace write failed");
  } else {
    MORI_UMBP_INFO("[UMBP sim] {}", json);
  }
}

SimCall::~SimCall() {
  if (!trace_) return;
  const auto end = std::chrono::steady_clock::now();
  using Ns = std::chrono::nanoseconds;
  try {
    trace_->Write(id_, api_, keys_, bytes_,
                  std::chrono::duration_cast<Ns>(start_.time_since_epoch()).count(),
                  std::chrono::duration_cast<Ns>(end - start_).count(), result_count_,
                  std::uncaught_exceptions() > exceptions_);
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("{}", e.what());
  }
}

}  // namespace mori::umbp
