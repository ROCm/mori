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

// The ONE gRPC message-size limit for every UMBP surface.
//
// gRPC defaults to 4 MiB on RECEIVE and unlimited on send, and it enforces that
// at the receiver: an oversized message is rejected whole, with
// "Received message larger than max (N vs. 4194304)", after the sender has
// already serialized and shipped it.  There is no partial delivery.
//
// Setting it per call site is how UMBP ended up with the limit configured on
// every surface that talks to the MASTER and on none of the surfaces that talk
// to a PEER -- so master-side traffic could grow to 64 MB while peer-side
// traffic silently kept a 4 MiB ceiling.  A single EvictKey carrying 155,157
// victim keys (18.7 MB) was refused on every one of 81 consecutive eviction
// rounds, the tier never reclaimed a byte, and writes then failed with
// NO_SPACE until the deployment was restarted.  Nothing logged the connection
// between the two.
//
// So the limit lives here, one value, applied through the same two helpers by
// every channel and every server.  A surface that forgets to call them is a
// surface running on gRPC's default, which is the bug above.
//
// This header needs gRPC, so it is only includable from umbp_common sources.
// It sits in common/ rather than distributed/ because both distributed/ and
// standalone/ dial and serve.

#include <grpcpp/grpcpp.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "umbp/common/env_time.h"

namespace mori::umbp {

// 64 MiB.  Chosen to match what the master surfaces already used, so this
// change lowers no existing limit.  It is a ceiling for a control-plane
// message, not a data path: bulk bytes move over RDMA or shared memory, and
// anything approaching this size is a batch that should be chunked
// (see GrpcMaxItemsPerBatch).
inline constexpr uint32_t kDefaultGrpcMaxMessageBytes = 64u * 1024u * 1024u;

// Floor of 1 MiB: below that the handshake and registration messages
// themselves start failing, which presents as a node that cannot join rather
// than as a size limit, and is a far worse thing to debug than a slow one.
inline constexpr uint32_t kMinGrpcMaxMessageBytes = 1u * 1024u * 1024u;

// Resolved once per process.  Static so every channel and server in this
// process agrees even if the environment is mutated later -- two surfaces
// disagreeing about the limit is exactly the failure this header exists to
// prevent.
inline uint32_t GrpcMaxMessageBytes() {
  static const uint32_t value = GetEnvUint32("UMBP_GRPC_MAX_MESSAGE_BYTES",
                                             kDefaultGrpcMaxMessageBytes, kMinGrpcMaxMessageBytes);
  return value;
}

// Apply to a client channel.  Both directions: a client that can send 64 MB
// but only receive 4 MiB fails on the response instead of the request, which
// is the same bug wearing a different hat.
inline void ApplyGrpcLimits(grpc::ChannelArguments* args) {
  const int bytes = static_cast<int>(GrpcMaxMessageBytes());
  args->SetMaxReceiveMessageSize(bytes);
  args->SetMaxSendMessageSize(bytes);
}

// Sugar for the common case of a channel with no other arguments.
inline grpc::ChannelArguments GrpcChannelArgs() {
  grpc::ChannelArguments args;
  ApplyGrpcLimits(&args);
  return args;
}

// Apply to a server.  The receive limit is the one that actually rejects
// traffic, so a server that skips this is the site that produces the
// "Received message larger than max" above.
inline void ApplyGrpcLimits(grpc::ServerBuilder* builder) {
  const int bytes = static_cast<int>(GrpcMaxMessageBytes());
  builder->SetMaxReceiveMessageSize(bytes);
  builder->SetMaxSendMessageSize(bytes);
}

// How many strings fit in one message, given the limit above.
//
// Raising the ceiling moves the wall; it does not remove it, because the
// batches that approach it grow with the deployment (an eviction round frees a
// FRACTION of the tier, so a bigger tier means more victims) while the limit is
// fixed.  A caller shipping a `repeated string` therefore has to split, and
// this is the split point.
//
// Measured rather than assumed: each key costs its own bytes plus a field tag
// and a length varint.  kPerItemOverhead is 8 rather than the 3 a <16 KiB
// string actually needs, and kFrameReserve holds back room for the rest of the
// message, because being one byte over costs the WHOLE batch -- the asymmetry
// between "slightly smaller chunks" and "nothing is evicted, ever" is not
// close.
//
// Returns at least 1: a single item larger than the whole budget still has to
// be attempted, and failing on it is more honest than silently sending an
// empty batch forever.
inline size_t GrpcMaxItemsPerBatch(const std::vector<std::string>& items, size_t start) {
  constexpr size_t kPerItemOverhead = 8;
  constexpr size_t kFrameReserve = 64 * 1024;

  const size_t limit = static_cast<size_t>(GrpcMaxMessageBytes());
  const size_t budget = limit > kFrameReserve ? limit - kFrameReserve : limit / 2;

  size_t bytes = 0;
  size_t count = 0;
  for (size_t i = start; i < items.size(); ++i) {
    const size_t cost = items[i].size() + kPerItemOverhead;
    if (count > 0 && bytes + cost > budget) break;
    bytes += cost;
    ++count;
  }
  return count == 0 ? 1 : count;
}

}  // namespace mori::umbp
