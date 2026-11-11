#include "meta_json.h"

#include <cmath>
#include <cstdio>

namespace netmon {

std::string JsonWriter::Escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        o += "\\\"";
        break;
      case '\\':
        o += "\\\\";
        break;
      case '\n':
        o += "\\n";
        break;
      case '\r':
        o += "\\r";
        break;
      case '\t':
        o += "\\t";
        break;
      case '\b':
        o += "\\b";
        break;
      case '\f':
        o += "\\f";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += static_cast<char>(c);  // UTF-8 passes through unchanged
        }
    }
  }
  return o;
}

void JsonWriter::Newline() {
  if (!pretty_) return;
  out_ += '\n';
  out_.append(counts_.size() * 2, ' ');
}

void JsonWriter::BeforeValue() {
  if (after_key_) {
    after_key_ = false;
    return;
  }
  if (!counts_.empty()) {
    if (counts_.back() > 0) out_ += ',';
    counts_.back()++;
    Newline();
  }
}

JsonWriter& JsonWriter::BeginObject() {
  BeforeValue();
  out_ += '{';
  counts_.push_back(0);
  return *this;
}

JsonWriter& JsonWriter::EndObject() {
  int n = counts_.back();
  counts_.pop_back();
  if (n > 0) Newline();
  out_ += '}';
  return *this;
}

JsonWriter& JsonWriter::BeginArray() {
  BeforeValue();
  out_ += '[';
  counts_.push_back(0);
  return *this;
}

JsonWriter& JsonWriter::EndArray() {
  int n = counts_.back();
  counts_.pop_back();
  if (n > 0) Newline();
  out_ += ']';
  return *this;
}

JsonWriter& JsonWriter::Key(const std::string& k) {
  BeforeValue();
  out_ += '"' + Escape(k) + (pretty_ ? "\": " : "\":");
  after_key_ = true;
  return *this;
}

JsonWriter& JsonWriter::String(const std::string& v) {
  BeforeValue();
  out_ += '"' + Escape(v) + '"';
  return *this;
}

JsonWriter& JsonWriter::Uint(uint64_t v) {
  BeforeValue();
  out_ += std::to_string(v);
  return *this;
}

JsonWriter& JsonWriter::Int(int64_t v) {
  BeforeValue();
  out_ += std::to_string(v);
  return *this;
}

JsonWriter& JsonWriter::Double(double v) {
  BeforeValue();
  if (!std::isfinite(v)) {
    out_ += "null";
  } else {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", v);
    out_ += buf;
  }
  return *this;
}

JsonWriter& JsonWriter::Bool(bool v) {
  BeforeValue();
  out_ += v ? "true" : "false";
  return *this;
}

JsonWriter& JsonWriter::Null() {
  BeforeValue();
  out_ += "null";
  return *this;
}

JsonWriter& JsonWriter::StringArray(const std::vector<std::string>& v) {
  // Compact single-line arrays keep long counter-name lists readable.
  BeforeValue();
  out_ += '[';
  for (size_t i = 0; i < v.size(); i++) {
    if (i) out_ += pretty_ ? ", " : ",";
    out_ += '"' + Escape(v[i]) + '"';
  }
  out_ += ']';
  return *this;
}

}  // namespace netmon
