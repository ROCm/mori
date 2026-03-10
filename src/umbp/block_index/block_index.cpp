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
#include "umbp/block_index/block_index.h"

#include <openssl/evp.h>

#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>

std::string BlockIndexClient::HashKVBlock(const std::vector<int>& token_ids,
                                          const std::string& prior_hash) {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);

  // If prior_hash is provided, feed its raw bytes (hex-decoded)
  if (!prior_hash.empty()) {
    // Decode hex string to bytes (same as Python: bytes.fromhex(prior_hash))
    std::vector<unsigned char> prior_bytes;
    prior_bytes.reserve(prior_hash.size() / 2);
    for (size_t i = 0; i + 1 < prior_hash.size(); i += 2) {
      unsigned int byte_val;
      std::sscanf(prior_hash.c_str() + i, "%2x", &byte_val);
      prior_bytes.push_back(static_cast<unsigned char>(byte_val));
    }
    EVP_DigestUpdate(ctx, prior_bytes.data(), prior_bytes.size());
  }

  // Feed token IDs as 4-byte little-endian (same as Python: t.to_bytes(4, 'little'))
  for (int token : token_ids) {
    uint32_t val = static_cast<uint32_t>(token);
    unsigned char buf[4];
    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;
    buf[2] = (val >> 16) & 0xFF;
    buf[3] = (val >> 24) & 0xFF;
    EVP_DigestUpdate(ctx, buf, 4);
  }

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hash_len = 0;
  EVP_DigestFinal_ex(ctx, hash, &hash_len);
  EVP_MD_CTX_free(ctx);

  // Convert to hex string
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < hash_len; ++i) {
    oss << std::setw(2) << static_cast<int>(hash[i]);
  }
  return oss.str();
}

bool BlockIndexClient::MayExist(const std::string& key) const {
  std::shared_lock lock(mu_);
  return index_.count(key) > 0;
}

std::optional<LocalLocation> BlockIndexClient::Lookup(const std::string& key) const {
  std::shared_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  return it->second;
}

void BlockIndexClient::Insert(const std::string& key, const LocalLocation& loc) {
  std::unique_lock lock(mu_);
  index_[key] = loc;
}

std::optional<LocalLocation> BlockIndexClient::Remove(const std::string& key) {
  std::unique_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  auto loc = it->second;
  index_.erase(it);
  return loc;
}

bool BlockIndexClient::UpdateTier(const std::string& key, StorageTier new_tier) {
  std::unique_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  it->second.tier = new_tier;
  return true;
}

size_t BlockIndexClient::Count() const {
  std::shared_lock lock(mu_);
  return index_.size();
}

void BlockIndexClient::Clear() {
  std::unique_lock lock(mu_);
  index_.clear();
}
