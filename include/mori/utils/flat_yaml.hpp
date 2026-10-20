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

// A dependency-free reader for the small subset of YAML used by mori's
// human-written config and catalog files: a list of flat maps.
//
//   # comments and blank lines
//   - id: first          "- " at column 0 starts an entry
//     count: 3           indented "key: value" continues it
//     note: "a: b"       values may be quoted ('...' or "...", no escapes)
//
// Keys are [A-Za-z0-9_]. Anything else (nested maps, lists, flow syntax,
// multi-line values, anchors, tabs, an unquoted ": " in a value) is rejected
// with its line number rather than misread, so a file this accepts is also
// valid YAML that any other tool reads the same way.

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <istream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mori {
namespace yaml {

using FlatMap = std::map<std::string, std::string>;

namespace detail {

inline std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(' ');
  size_t e = s.find_last_not_of(' ');
  return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

inline std::string ParseScalar(const std::string& raw, const std::string& where) {
  std::string v = Trim(raw);
  if (!v.empty() && (v[0] == '"' || v[0] == '\'')) {
    size_t close = v.find(v[0], 1);
    std::string rest = close == std::string::npos ? "" : Trim(v.substr(close + 1));
    if (close == std::string::npos || (!rest.empty() && rest[0] != '#')) {
      throw std::runtime_error(where + ": unterminated quote or text after a quoted value");
    }
    return v.substr(1, close - 1);
  }
  size_t comment = v.find(" #");
  if (comment != std::string::npos) v = Trim(v.substr(0, comment));
  if (v.empty())
    throw std::runtime_error(where + ": missing value (nested maps are not supported)");
  if (std::strchr("[{&*!|>", v[0]) != nullptr || v.find(": ") != std::string::npos) {
    throw std::runtime_error(where + ": unsupported value '" + v +
                             "' (quote it if it contains ': ')");
  }
  return v;
}

}  // namespace detail

// Parses a list of flat maps from `in`; `source` names the input in errors
// ("<source>:<line>: ..."). Throws std::runtime_error on anything outside the subset.
inline std::vector<FlatMap> ParseFlatList(std::istream& in, const std::string& source) {
  std::vector<FlatMap> list;
  std::string line;
  for (int lineNo = 1; std::getline(in, line); ++lineNo) {
    std::string where = source + ":" + std::to_string(lineNo);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find('\t') != std::string::npos) throw std::runtime_error(where + ": tab character");
    std::string body = detail::Trim(line);
    if (body.empty() || body[0] == '#') continue;

    if (line.rfind("- ", 0) == 0) {
      list.emplace_back();
      body = detail::Trim(line.substr(2));
    } else if (line[0] != ' ' || list.empty()) {
      throw std::runtime_error(where + ": expected '- key: value' or an indented 'key: value'");
    }
    size_t colon = body.find(':');
    bool keyOk = colon != std::string::npos && colon > 0 &&
                 (colon + 1 == body.size() || body[colon + 1] == ' ');
    for (size_t i = 0; keyOk && i < colon; ++i) {
      keyOk = std::isalnum(static_cast<unsigned char>(body[i])) || body[i] == '_';
    }
    if (!keyOk) throw std::runtime_error(where + ": expected 'key: value'");
    std::string key = body.substr(0, colon);
    if (!list.back().emplace(key, detail::ParseScalar(body.substr(colon + 1), where)).second) {
      throw std::runtime_error(where + ": duplicate field '" + key + "'");
    }
  }
  return list;
}

inline std::vector<FlatMap> LoadFlatList(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(path + ": cannot open");
  return ParseFlatList(in, path);
}

// Typed access. `where` prefixes error messages (e.g. "catalog.yaml: entry 3").

inline void CheckKeys(const FlatMap& map, const std::set<std::string>& allowed,
                      const std::string& where) {
  for (const auto& field : map) {
    if (allowed.count(field.first) == 0) {
      throw std::runtime_error(where + ": unknown field '" + field.first + "'");
    }
  }
}

inline std::string GetString(const FlatMap& map, const std::string& key,
                             const std::string& fallback = "") {
  auto it = map.find(key);
  return it == map.end() ? fallback : it->second;
}

inline bool GetBool(const FlatMap& map, const std::string& key, bool fallback,
                    const std::string& where) {
  auto it = map.find(key);
  if (it == map.end()) return fallback;
  if (it->second == "true") return true;
  if (it->second == "false") return false;
  throw std::runtime_error(where + ": " + key + " must be true or false, got '" + it->second + "'");
}

inline long GetInt(const FlatMap& map, const std::string& key, long fallback,
                   const std::string& where) {
  auto it = map.find(key);
  if (it == map.end()) return fallback;
  const char* s = it->second.c_str();
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    throw std::runtime_error(where + ": " + key + " must be an integer, got '" + it->second + "'");
  }
  return v;
}

}  // namespace yaml
}  // namespace mori
