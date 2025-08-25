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

#include <msgpack.hpp>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/enum.hpp"

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
  namespace adaptor {

  template <>
  struct pack<mori::io::BackendType> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        mori::io::BackendType loc) const {
      o.pack(static_cast<uint32_t>(loc));
      return o;
    }
  };

  template <>
  struct convert<mori::io::BackendType> {
    const msgpack::object& operator()(const msgpack::object& o, mori::io::BackendType& loc) const {
      loc = static_cast<mori::io::BackendType>(o.as<uint32_t>());
      return o;
    }
  };

  template <>
  struct pack<mori::io::MemoryLocationType> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        mori::io::MemoryLocationType loc) const {
      o.pack(static_cast<uint32_t>(loc));
      return o;
    }
  };

  template <>
  struct convert<mori::io::MemoryLocationType> {
    const msgpack::object& operator()(const msgpack::object& o,
                                      mori::io::MemoryLocationType& loc) const {
      loc = static_cast<mori::io::MemoryLocationType>(o.as<uint32_t>());
      return o;
    }
  };

  template <>
  struct pack<mori::application::TCPContextHandle> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        const mori::application::TCPContextHandle& ctx) const {
      o.pack_array(2);
      o.pack(ctx.host);
      o.pack(ctx.port);
      return o;
    }
  };

  template <>
  struct convert<mori::application::TCPContextHandle> {
    const msgpack::object& operator()(const msgpack::object& o,
                                      mori::application::TCPContextHandle& ctx) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 2) throw msgpack::type_error();
      ctx.host = o.via.array.ptr[0].as<std::string>();
      ctx.port = o.via.array.ptr[1].as<uint16_t>();
      return o;
    }
  };

  template <>
  struct pack<mori::application::RdmaMemoryRegion> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        const mori::application::RdmaMemoryRegion& m) const {
      o.pack_array(4);
      o.pack(m.addr);
      o.pack(m.lkey);
      o.pack(m.rkey);
      o.pack(m.length);
      return o;
    }
  };

  template <>
  struct convert<mori::application::RdmaMemoryRegion> {
    const msgpack::object& operator()(const msgpack::object& o,
                                      mori::application::RdmaMemoryRegion& m) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 4) throw msgpack::type_error();
      m.addr = o.via.array.ptr[0].as<uintptr_t>();
      m.lkey = o.via.array.ptr[1].as<uint32_t>();
      m.rkey = o.via.array.ptr[2].as<uint32_t>();
      m.length = o.via.array.ptr[3].as<size_t>();
      return o;
    }
  };

  template <>
  struct pack<mori::application::InfiniBandEndpointHandle> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, mori::application::InfiniBandEndpointHandle const& v) const {
      o.pack_array(1);
      o.pack(v.lid);
      return o;
    }
  };

  template <>
  struct convert<mori::application::InfiniBandEndpointHandle> {
    msgpack::object const& operator()(msgpack::object const& o,
                                      mori::application::InfiniBandEndpointHandle& v) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 1) throw msgpack::type_error();
      v.lid = o.via.array.ptr[0].as<uint32_t>();
      return o;
    }
  };

  template <>
  struct pack<mori::application::EthernetEndpointHandle> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        mori::application::EthernetEndpointHandle const& v) const {
      o.pack_array(2);
      o.pack_bin(sizeof(v.gid));
      o.pack_bin_body(reinterpret_cast<char const*>(v.gid), sizeof(v.gid));
      o.pack_bin(sizeof(v.mac));
      o.pack_bin_body(reinterpret_cast<char const*>(v.mac), sizeof(v.mac));
      return o;
    }
  };

  template <>
  struct convert<mori::application::EthernetEndpointHandle> {
    msgpack::object const& operator()(msgpack::object const& o,
                                      mori::application::EthernetEndpointHandle& v) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 2) throw msgpack::type_error();
      auto gid_bin = o.via.array.ptr[0].as<msgpack::type::raw_ref>();
      auto mac_bin = o.via.array.ptr[1].as<msgpack::type::raw_ref>();
      if (gid_bin.size != sizeof(v.gid) || mac_bin.size != sizeof(v.mac))
        throw msgpack::type_error();
      std::memcpy(v.gid, gid_bin.ptr, sizeof(v.gid));
      std::memcpy(v.mac, mac_bin.ptr, sizeof(v.mac));
      return o;
    }
  };

  template <>
  struct pack<mori::application::RdmaEndpointHandle> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                        mori::application::RdmaEndpointHandle const& v) const {
      o.pack_array(5);
      o.pack(v.psn);
      o.pack(v.qpn);
      o.pack(v.portId);
      o.pack(v.ib);
      o.pack(v.eth);
      return o;
    }
  };

  template <>
  struct convert<mori::application::RdmaEndpointHandle> {
    msgpack::object const& operator()(msgpack::object const& o,
                                      mori::application::RdmaEndpointHandle& v) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 5) throw msgpack::type_error();
      v.psn = o.via.array.ptr[0].as<uint32_t>();
      v.qpn = o.via.array.ptr[1].as<uint32_t>();
      v.portId = o.via.array.ptr[2].as<uint32_t>();
      v.ib = o.via.array.ptr[3].as<mori::application::InfiniBandEndpointHandle>();
      v.eth = o.via.array.ptr[4].as<mori::application::EthernetEndpointHandle>();
      return o;
    }
  };

  }  // namespace adaptor
}  // MSGPACK_API_VERSION_NAMESPACE
}  // namespace msgpack
