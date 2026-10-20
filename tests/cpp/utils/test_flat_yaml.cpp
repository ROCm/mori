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
#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "mori/utils/flat_yaml.hpp"

using mori::yaml::FlatMap;

namespace {

std::vector<FlatMap> Parse(const std::string& text) {
  std::istringstream in(text);
  return mori::yaml::ParseFlatList(in, "t.yaml");
}

// Returns the error message, or "" if `text` parsed.
std::string ParseError(const std::string& text) {
  try {
    Parse(text);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(FlatYaml, ParsesEntriesCommentsAndQuotes) {
  auto list = Parse(
      "# header comment\n"
      "\n"
      "- id: first   # trailing comment\n"
      "  rule: cqe_drop:op=write:skip=3\n"
      "  note: \"a: b # not a comment\"\n"
      "\n"
      "  # comment inside an entry\n"
      "- id: second\r\n"
      "  quoted: 'single'\n");
  ASSERT_EQ(list.size(), 2u);
  EXPECT_EQ(list[0].at("id"), "first");
  EXPECT_EQ(list[0].at("rule"), "cqe_drop:op=write:skip=3");
  EXPECT_EQ(list[0].at("note"), "a: b # not a comment");
  EXPECT_EQ(list[1].at("id"), "second");
  EXPECT_EQ(list[1].at("quoted"), "single");
}

TEST(FlatYaml, EmptyInputIsEmptyList) { EXPECT_TRUE(Parse("# only a comment\n\n").empty()); }

TEST(FlatYaml, TypedGetters) {
  FlatMap m = Parse("- on: true\n  off: false\n  n: -7\n  s: x\n")[0];
  EXPECT_TRUE(mori::yaml::GetBool(m, "on", false, "w"));
  EXPECT_FALSE(mori::yaml::GetBool(m, "off", true, "w"));
  EXPECT_TRUE(mori::yaml::GetBool(m, "missing", true, "w"));
  EXPECT_EQ(mori::yaml::GetInt(m, "n", 0, "w"), -7);
  EXPECT_EQ(mori::yaml::GetInt(m, "missing", 5, "w"), 5);
  EXPECT_EQ(mori::yaml::GetString(m, "s"), "x");
  EXPECT_EQ(mori::yaml::GetString(m, "missing", "d"), "d");
  EXPECT_THROW(mori::yaml::GetBool(m, "s", false, "w"), std::runtime_error);
  EXPECT_THROW(mori::yaml::GetInt(m, "s", 0, "w"), std::runtime_error);
  EXPECT_THROW(mori::yaml::CheckKeys(m, {"on", "off", "n"}, "w"), std::runtime_error);
  EXPECT_NO_THROW(mori::yaml::CheckKeys(m, {"on", "off", "n", "s"}, "w"));
}

TEST(FlatYaml, RejectsAnythingOutsideTheSubsetWithLineNumber) {
  struct Case {
    const char* text;
    const char* expect;  // substring of the error
  };
  const Case cases[] = {
      {"- id: a\n  rule:\n    kind: x\n", "t.yaml:2: missing value"},  // nested map
      {"- id: a\n  list: [1, 2]\n", "t.yaml:2: unsupported value"},    // flow list
      {"- id: a\n  m: {k: v}\n", "t.yaml:2: unsupported value"},       // flow map
      {"- id: a\n  text: |\n", "t.yaml:2: unsupported value"},         // block scalar
      {"- id: a\n  ref: *anchor\n", "t.yaml:2: unsupported value"},    // alias
      {"- id: a\n\tn: 1\n", "t.yaml:2: tab character"},                // tab
      {"- id: a\n  note: a: b\n", "t.yaml:2: unsupported value"},      // unquoted ": "
      {"- id: a\n  note: \"open\n", "t.yaml:2: unterminated quote"},   // quote
      {"- id: a\n  id: b\n", "t.yaml:2: duplicate field 'id'"},        // duplicate
      {"id: a\n", "t.yaml:1: expected '- key: value'"},                // top-level map
      {"  n: 1\n", "t.yaml:1: expected '- key: value'"},               // no entry yet
      {"- id: a\n  bad key: 1\n", "t.yaml:2: expected 'key: value'"},  // key chars
      {"- id: a\n  k:v\n", "t.yaml:2: expected 'key: value'"},         // no space
  };
  for (const Case& c : cases) {
    std::string err = ParseError(c.text);
    EXPECT_NE(err.find(c.expect), std::string::npos) << "input:\n"
                                                     << c.text << "error: '" << err << "'";
  }
}

TEST(FlatYaml, MissingFileThrows) {
  EXPECT_THROW(mori::yaml::LoadFlatList("/nonexistent/flat_yaml.yaml"), std::runtime_error);
}
