#pragma once

#include <msgpack.hpp>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/enum.hpp"

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
  namespace adaptor {

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
      o.pack_array(3);
      o.pack(ctx.host);
      o.pack(ctx.port);
      o.pack(ctx.listenFd);
      return o;
    }
  };

  template <>
  struct convert<mori::application::TCPContextHandle> {
    const msgpack::object& operator()(const msgpack::object& o,
                                      mori::application::TCPContextHandle& ctx) const {
      if (o.type != msgpack::type::ARRAY || o.via.array.size != 3) throw msgpack::type_error();
      ctx.host = o.via.array.ptr[0].as<std::string>();
      ctx.port = o.via.array.ptr[1].as<uint16_t>();
      ctx.listenFd = o.via.array.ptr[2].as<int>();
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