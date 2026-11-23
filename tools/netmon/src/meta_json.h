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
// Minimal write-only JSON generator (meta.json, `list --json`). Reading meta is done by the
// Python report tool, so no parser is needed here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace netmon {

class JsonWriter {
 public:
  explicit JsonWriter(bool pretty = true) : pretty_(pretty) {}

  JsonWriter& BeginObject();
  JsonWriter& EndObject();
  JsonWriter& BeginArray();
  JsonWriter& EndArray();
  JsonWriter& Key(const std::string& k);
  JsonWriter& String(const std::string& v);
  JsonWriter& Uint(uint64_t v);
  JsonWriter& Int(int64_t v);
  JsonWriter& Double(double v);
  JsonWriter& Bool(bool v);
  JsonWriter& Null();
  JsonWriter& StringArray(const std::vector<std::string>& v);

  const std::string& str() const { return out_; }
  static std::string Escape(const std::string& s);

 private:
  void BeforeValue();
  void Newline();

  bool pretty_;
  std::string out_;
  // Per nesting level: number of elements written so far.
  std::vector<int> counts_;
  bool after_key_ = false;
};

}  // namespace netmon
