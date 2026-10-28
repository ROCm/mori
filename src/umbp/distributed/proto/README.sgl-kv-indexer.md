`sgl_kv_indexer.proto` is a mutation-only subset of SGLang's
[`sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`](https://github.com/wuyl1/sglang/blob/1bb575f1d5c49eff2e4021fea7b776faa46508ee/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto)
at commit `1bb575f1d5c49eff2e4021fea7b776faa46508ee`.
Shared-pool mode requires the protocol in
[SGLang #46995](https://github.com/sgl-project/sglang/pull/46995), which should land first.
It is Copyright (c) 2026 The SGLang Authors, licensed under Apache-2.0;
see [LICENSE.sgl-kv-indexer](LICENSE.sgl-kv-indexer).
Unused query definitions and comments were removed; wire types, field numbers,
enum values, reserved fields and the mutation RPC path are unchanged.
