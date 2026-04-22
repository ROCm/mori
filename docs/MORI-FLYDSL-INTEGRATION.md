# Mori FlyDSL Integration

See [`python/mori/ir/flydsl/README.md`](../python/mori/ir/flydsl/README.md) for the
full integration guide, including:

* recommended import patterns for `@flyc.kernel` code that uses mori shmem,
* the three-piece integration contract (bitcode path, ABI metadata, post-load
  module initialiser) that mori exposes to any host DSL,
* the pickling / on-disk JIT cache contract for `module_init_fn` callables.

For the framework-agnostic shmem bitcode ABI and the complete list of device
functions, see [`python/mori/ir/README.md`](../python/mori/ir/README.md).
