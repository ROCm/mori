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

// Small helpers shared by both RESP client implementations (RespClient on
// hiredis, RespClusterClient on redis-plus-plus), so the SHA cache, the EVALSHA
// argv layout, and the const-char*/len splitting live in ONE place instead of
// being copy-pasted per client. Header-only (all tiny / on the hot path).

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mori::umbp::redis {

// Content-addressed EVALSHA SHA cache, keyed by the script object's ADDRESS (not
// its ~1.5KB text): every Lua script is a single `inline const std::string`
// (lua_scripts.h), so `&script` is stable and unique per script. This keeps the
// read hot path from rehashing/comparing the whole script body under the lock on
// every EVALSHA. Both clients own one of these.
class ScriptCache {
 public:
  // Return the cached SHA for `script`, or run `loader(script)` once (outside the
  // lock) to obtain it, cache it, and return it. `loader` does the SCRIPT LOAD
  // and may throw (transport error / bad reply); nothing is cached if it throws.
  template <typename Loader>
  std::string GetOrLoad(const std::string& script, Loader&& loader) {
    const void* key = static_cast<const void*>(&script);
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end()) return it->second;
    }
    std::string sha = loader(script);
    {
      std::lock_guard<std::mutex> lk(mu_);
      cache_[key] = sha;
    }
    return sha;
  }

  // Drop `script`'s cached SHA (after a NOSCRIPT, so the next call reloads it).
  void Invalidate(const std::string& script) {
    std::lock_guard<std::mutex> lk(mu_);
    cache_.erase(static_cast<const void*>(&script));
  }

 private:
  std::mutex mu_;
  std::unordered_map<const void*, std::string> cache_;  // &script -> sha1
};

// Split `args` into the parallel const-char*/length arrays hiredis' *Argv APIs
// want. `ptrs`/`lens` are cleared and refilled; they must outlive the call that
// consumes them (they alias into `args`).
inline void ToArgv(const std::vector<std::string>& args, std::vector<const char*>& ptrs,
                   std::vector<std::size_t>& lens) {
  ptrs.clear();
  lens.clear();
  ptrs.reserve(args.size());
  lens.reserve(args.size());
  for (const auto& a : args) {
    ptrs.push_back(a.data());
    lens.push_back(a.size());
  }
}

// Build the argv for one EVALSHA: [EVALSHA, sha, nkeys, keys..., args...].
inline std::vector<std::string> BuildEvalshaArgv(const std::string& sha,
                                                 const std::vector<std::string>& keys,
                                                 const std::vector<std::string>& args) {
  std::vector<std::string> cmd;
  cmd.reserve(3 + keys.size() + args.size());
  cmd.push_back("EVALSHA");
  cmd.push_back(sha);
  cmd.push_back(std::to_string(keys.size()));
  for (const auto& k : keys) cmd.push_back(k);
  for (const auto& a : args) cmd.push_back(a);
  return cmd;
}

}  // namespace mori::umbp::redis
