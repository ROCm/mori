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
//
// Host-only plumbing for the JIT: run a compiler, publish a cache directory,
// digest a string. Three small concerns in one header because each is ~50 lines
// and they have exactly one consumer between them (compiler.cpp).

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mori {
namespace jit {
namespace v2 {

// ---------------------------------------------------------------------------
// Subprocess
// ---------------------------------------------------------------------------

struct CommandResult {
  int exitCode = -1;
  std::string output;  // stdout and stderr, interleaved
};

// fork + execvp. No shell, so a path with spaces or metacharacters is data.
CommandResult RunProgram(const std::vector<std::string>& argv);

// /bin/sh -c, for fixed internal command lines only.
CommandResult RunCommand(const std::string& commandLine);

// Shell-quote for display. Not used to build anything that gets executed.
std::string Quote(const std::string& s);
std::string JoinArgv(const std::vector<std::string>& argv);
std::vector<std::string> SplitWhitespace(const std::string& s);

// ---------------------------------------------------------------------------
// Filesystem: lock-free cache publication
//
// Build into a temp directory, fsync it, rename the whole directory into place.
// Directory rename is atomic on local and distributed filesystems, so a reader
// sees a complete entry or nothing. Losing the race to another rank is fine --
// its copy is byte-equivalent, since the directory name IS the content hash.
// ---------------------------------------------------------------------------

bool MakeDirs(const std::string& path);

// Write then fsync: close() alone does not guarantee the bytes are visible to a
// subsequently forked compiler on a networked filesystem.
bool WriteFileSynced(const std::string& path, const std::string& data);

bool ReadFile(const std::string& path, std::string* out);

void FsyncPath(const std::string& path);
void FsyncDirRecursive(const std::string& dir);

// Unique created subdirectory of `root`; empty string on failure.
std::string MakeUniqueTempDir(const std::string& root, const std::string& prefix);

// rename(tmpDir -> finalDir). True if this call created finalDir, false if
// someone else won (in which case tmpDir is removed). finalDir exists either way.
bool PublishDir(const std::string& tmpDir, const std::string& finalDir);

// remove_all that does not throw.
void SafeRemoveAll(const std::string& path);

// ---------------------------------------------------------------------------
// SHA-256
//
// A cache-key collision loads the wrong kernel, so this is a real digest rather
// than a cheap hash. It runs once per distinct config.
// ---------------------------------------------------------------------------

class Sha256 {
 public:
  Sha256() = default;

  void Update(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (len--) {
      buf_[bufLen_++] = *p++;
      if (bufLen_ == 64) {
        Transform(buf_.data());
        bitLen_ += 512;
        bufLen_ = 0;
      }
    }
  }

  void Update(const std::string& s) { Update(s.data(), s.size()); }

  // Lowercase hex, truncated to `chars` (default: the full 64).
  std::string HexDigest(size_t chars = 64) {
    std::array<uint8_t, 32> out{};
    Final(out.data());
    static const char* kHex = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (uint8_t b : out) {
      s.push_back(kHex[b >> 4]);
      s.push_back(kHex[b & 0xf]);
    }
    return chars >= s.size() ? s : s.substr(0, chars);
  }

 private:
  static uint32_t Rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

  void Transform(const uint8_t* chunk) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(chunk[i * 4]) << 24) | (uint32_t(chunk[i * 4 + 1]) << 16) |
             (uint32_t(chunk[i * 4 + 2]) << 8) | uint32_t(chunk[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  void Final(uint8_t* out) {
    uint64_t total = bitLen_ + uint64_t(bufLen_) * 8;
    size_t i = bufLen_;
    buf_[i++] = 0x80;
    if (i > 56) {
      while (i < 64) buf_[i++] = 0;
      Transform(buf_.data());
      i = 0;
    }
    while (i < 56) buf_[i++] = 0;
    for (int j = 7; j >= 0; --j) buf_[i++] = uint8_t(total >> (j * 8));
    Transform(buf_.data());
    for (int j = 0; j < 8; ++j) {
      out[j * 4 + 0] = uint8_t(h_[j] >> 24);
      out[j * 4 + 1] = uint8_t(h_[j] >> 16);
      out[j * 4 + 2] = uint8_t(h_[j] >> 8);
      out[j * 4 + 3] = uint8_t(h_[j]);
    }
  }

  std::array<uint8_t, 64> buf_{};
  size_t bufLen_ = 0;
  uint64_t bitLen_ = 0;
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
};

inline std::string HexDigest(const std::string& s, size_t chars = 64) {
  Sha256 h;
  h.Update(s);
  return h.HexDigest(chars);
}

}  // namespace v2
}  // namespace jit
}  // namespace mori
