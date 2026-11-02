`sgl_kv_indexer.proto` is a mutation-only subset of SGLang's
[`sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`](https://github.com/wuyl1/sglang/blob/818364e613564ad51ae7618f16cac3c77e6d1af8/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto)
at commit `818364e613564ad51ae7618f16cac3c77e6d1af8`.
Shared-pool mode requires the protocol in
[SGLang #46995](https://github.com/sgl-project/sglang/pull/46995).
Either PR can merge first; enabling reporting requires shared-pool support in the Indexer.
It is Copyright (c) 2026 The SGLang Authors, licensed under Apache-2.0;
see [LICENSE.sgl-kv-indexer](LICENSE.sgl-kv-indexer).
Unused query/worker-spec definitions and comments were removed; retained wire types, field numbers,
enum values, reserved fields and the mutation RPC path are unchanged.
