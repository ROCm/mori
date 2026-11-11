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
