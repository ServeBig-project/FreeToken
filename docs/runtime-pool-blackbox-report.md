# 共享 runtime 池：服务级黑盒验收报告

依据 `docs/runtime-pool-public-contract.md` 第 2–6 节和 `docs/runtime-pool-acceptance-config.md`。本报告只记录公开可复现的条件和观测，不推测内部原因。套件在 `blackbox_tests/runtime_pool/test_service_*.py`。

## 结论

被测实现 `feat/runtime-pool@65b7c48` 上，B–F 五个服务模块 25/25 通过；合并前补充的功能组合模块 H 在 `e6f6d90` 上 8/8 通过，DFlash＋ReplaySSM 压力模块 G 在 `d3d13fb` 上 3/3 通过。启动期 A 模块：6 项无需 GPU 的用例在 65b7c48 上通过；2 项需 GPU 的用例只在 793e48d 上跑过，结果为通过。TP、多模态和第 7 节性能对比本轮未验收。

最新一轮（PR 头 `7155bcf`，GPU1 独占）：C 9/9、E 6/6、G 4/4（含共享前缀版）通过。详见“7155bcf 复跑”。

## 环境

- 单卡 GPU1 `GPU-b8a2a927-…`（RTX 4090）。服务进程用 `taskset -c 8-15,24-31` 启动。
- 公共参数：`--model-path Qwen3.6-35B-A3B-NVFP4 --moe-backend offload --moe-cache-size 2048 --attention-backend fi`。
- 请求一律 temperature 0；除早停用例外都带 `ignore_eos`。
- 原分池基线为 `main@732f1ee`。

| 模块 | 额外启动参数 |
|---|---|
| A | 启动期错误（见下） |
| B | `--runtime-cache-gib 4 --enable-cache-report`，省略并发数 |
| C | `--runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-cache-report` |
| D | `--runtime-cache-gib 0.5 --max-running-requests 64 --prefix-cache-host-gib 0 --cache-type naive --enable-cache-report` |
| E | `--runtime-cache-gib 4 --max-running-requests 4 --speculative-num-steps 4 --speculative-draft-model-path <DFlash> --enable-cache-report` |
| F | 基线 `--num-tokens 65536 --max-running-requests 4`，SD 参数与 E 相同 |

## 关键观测（65b7c48）

**启动期错误（A）**
- 与共享模式冲突的 `--num-pages`、`--num-tokens`、`--gdn-state-budget-bytes`、`--kv-reserve-tokens`，以及 `--runtime-cache-gib 0` 和 `-1`：在加载模型之前退出，并给出可读文本。
- `--max-seq-len-override 262144`（0.5 GiB 预算）：ready 前以 "0.50 GiB of runtime holds one request of 18430 tokens; 262144 are required" 拒绝。
- `--runtime-cache-gib 30`：ready 前报错。

**生效并发与上限**

| 模块 | requested | resource | 生效 max | context_tokens |
|---|---|---|---|---|
| B | null | 65 | 65 | 198654 |
| C | 6 | 6 | 6 | 16382 |
| D | 64 | 6 | 6 | 18430 |
| E | 4 | 27 | 4 | 161790 |

**真正共享（C）**：三个阶段按顺序跑，每 0.25 s 采样一次 runtime 状态。
- 短阶段 `gdn_state` 峰值 420 MiB。
- 长阶段 `kv` 峰值 240 MiB；同一采样时刻 `gdn_state` 为 240 MiB。
- 第三阶段 `gdn_state` 回到 420 MiB。
- 240 + 420 MiB 大于 512 MiB 预算，说明物理容量在组件间转用了。
- 全程 `held ≤ budget`，`0 ≤ evictable ≤ held`；专家容量 2048 不变，没有发生重建。

**暂停与恢复**
- C（host 4 GiB）：6 条 1200 token 的枚举请求全部完整、连续，usage 只计真实输入输出。paused 5、restored 5、recompute 0，paused_ms 18.4 s。暂停期间维护请求返回 `status: busy`。
- D（host 0，naive）：同样 6 条全部完整。paused 3、recompute 3、recomputed_tokens 1959、restored 0，paused_ms 69 s，host 分配为 0。
- 暂停期间取消（C、D）：断开最新的 2 条，其余请求完成，`requests.active` 归零，之后新请求照常服务。

**准入与公开错误**
- max_tokens 8000、约 186 token 时早停的请求重叠运行（B 6 条、C 3 条、D 6 条），都没有触发暂停，说明准入没有按输出上限整段预留。
- 超过 `context_tokens` 的请求：非流式返回 HTTP 400，code 为 `context_length_exceeded`，文本为 "prompt is too long: …"。流式返回 error 块。之后服务继续可用。

**维护**
- 有请求运行时返回 busy。
- `num_pages`、`num_mamba_slots`、`num_swa_pages`、`swa_full_tokens_ratio` 被 rejected，前后均可服务。
- `runtime_cache_gib` 设为 0 或 100 被 rejected。
- 省略并发数的服务（B）上设为 3 GiB 被 rejected，文本为 "requests at their minimum"。这符合公开语义：生效并发在服务生命周期内不变。
- B 上 `4 GiB + moe 1536` 和复原都成功，之后 Graph 仍回放。
- 显式并发 6 的服务（C）上，0.5 → 1 → 0.5 GiB 都成功，并发和专家容量不变。
- E 上 4 → 3 GiB 成功，之后 SD 仍真实执行。

**DFlash（E）**
- 组件包含 `draft_full` 和 `draft_swa`，与目标 KV、GDN 状态共享同一预算。
- 纯 decode 时真实执行了 SD（19 轮，起草 76 个，接受 76 个），Graph 回放计数增加。
- 4 条并发的 300 token 枚举请求全部完整，起草 865 个，接受 858 个。
- 同一配置下两次全新 prefill 逐字一致。同组热命中的输出在后段不同：执行计划不同，只记录。

**与原分池对比（F）**
- 约 2991 token 的 prompt，max_tokens 160，单请求串行。两边 usage 相同，文本在第 242/582 字符处分叉，分叉后两边都通顺。
- 按契约 §3 和协调者结论，两种模式是不同的执行计划，这里只记录分叉位置，不判为失败。

## 7155bcf 复跑

本轮按公开行为的 4 条变化新增判据，在 GPU1 上逐模块运行，每个模块起一次服务。

| 模块 | 结果 | 新增判据的观测 |
|---|---|---|
| C | 9/9 | 两条约 3000 token 的长 prompt 同时运行：首 token 分别在 1.0 s、2.3 s，都早于两者中先结束的那条（3.4 s）。7000 token 请求挤压预算后，evictable 为 402 MiB，之前暖过的 prompt 命中 960 token，全部来自 GPU |
| E | 6/6 | 无压力下跑 4 条 DFlash 请求：evictable_bytes 不减，暖过的 prompt 命中量不降 |
| G | 4/4 | 共享前缀版见下 |

所有模块的每个采样点都满足 `used ≤ held ≤ budget`。

**共享前缀版 G**（DFlash＋ReplaySSM，`--runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-gdn-replayssm`，生效并发 2）
- 负载：4 条请求共用 3000 token 的枚举前缀 `item1000…item1599`，各接 40 项不同后缀，各输出 4500 token，同时发出。
- 对照：同一 prompt 在另一个 cache_group 中逐条单独运行，事先暖好同一前缀。
- 计数：paused 3、restored 2、recompute 1、recomputed_tokens 200、compactions 2。
- 结果：4 条输出与单独运行逐字一致（8100 字符）。枚举连续，没有跳号，没有混入 prompt 文本；usage 均为 3200/4500。
- 输出改为 2500 token 时，两条请求能同时放下，没有发生暂停；那一轮 4 条同样与单独运行逐字一致。
- e6f6d90 上曾出现的“跳号并混入 prompt 文本”，当时的 prompt 是散文前文加枚举，与本轮负载不同；本轮未复现。

**GPU 独占说明**
- C 服务加载专家权重期间，NoWAG 服务在同卡上运行了约 1–2 分钟。
- C 启动时的显存测量（23.10 GiB）早于该重叠；共享预算推导和全部测试都在重叠结束后完成。
- 21:25 起每 10 秒检查一次 GPU1，没有发现外部进程。
- 因此表中结果都视为在独占 GPU1 时得到。

## 补充模块 G、H

**H 功能组合（e6f6d90，GPU2 容器，cpuset 0-7,16-23）**，每种组合起一次服务：

| 服务 | 组合 | 观测 |
|---|---|---|
| H1 | legacy＋DFlash 非 compact（`--no-dflash-compact-kv`）＋Graph 关（`--cuda-graph-max-bs 0`），4 GiB，并发 4 | 生效 batching 为 legacy，组件为 `draft_kv`；4 条并发的 300 token 枚举完整，SD 跑了 60 轮，Graph 回放增量为 0 |
| H2 | legacy＋AR＋Graph 开，0.5 GiB，并发 6，host 4 GiB | 4 条并发时 Graph 回放 299 次；压力轮 paused 8、restored 8，6×1200 token 全部完整连续，暂停中维护返回 busy |
| H3 | 显式 layered-pipeline＋DFlash 非 compact＋Graph 开，0.5 GiB，并发 6，host 4 GiB | 组件为 `draft_kv`；并发时 SD 93 轮、Graph 回放 468 次；压力轮 paused 4、restored 4，全部完整 |

三个服务的每次采样都满足 `held ≤ budget`。

**G：DFlash＋ReplaySSM 压力（d3d13fb）**
- 配置：`--runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-gdn-replayssm` 加 DFlash N4。
- 就绪：生效并发为 2，context_tokens 为 8190，组件含 `draft_full`、`draft_swa`、`replay_*`。
- 压力轮：2 条约 3000 token 的私有枚举历史，各输出 3000 token。
  - 计数增量：paused 1、recompute 1、recomputed_tokens 3000、paused_ms 23 s。后到的一条在 prefill 阶段让路，按公开语义重算。
  - 两条 usage 都是 3000/3000，输出完整连续，SD 跑了 1194 轮；之后的新请求 SD 照常执行。
- 暂停期取消：取消正在让路的那条后，另一条完整完成，服务回到空闲。被取消的一条记为 recompute 1、recomputed_tokens 0，因为它在重新准入前就被取消了。
- 修复前（e6f6d90）的公开问题：
  - 让路后被重算的请求，`usage.prompt_tokens` 报成实际值的两倍（6400 对 3200，total 9400，超过 context_tokens）。
  - 同时 `recomputed_tokens` 为 0。
  - 两者都由 d3d13fb 修复。

## 过程中发现并已修复的公开问题

| 首次发现 | 条件 | 观测 | 修复 |
|---|---|---|---|
| 793e48d | 共享模式省略 `--max-running-requests` | ready 前崩溃，TypeError | 修复后 B 通过 |
| 793e48d | `geometry` 字段 | 缺少 `runtime_cache_bytes` 和 `address_*` | 已补；在 dd8bf5e 上 `address_*` 为 0，1d6df11 修复 |
| dd8bf5e | 超过 `context_tokens` | HTTP 400，但 `code` 为 null | 849edc8 |
| dd8bf5e | 重算路径暂停 | `paused_ms` 一直为 0 | 已修，D 已验证 |
| dd8bf5e | DFlash 组件名 | 报告为 `full`/`swa`，无法辨认属于 drafter | 65b7c48 改为 `draft_*` |
| e6f6d90 | DFlash＋Replay 压力下，prefill 阶段让路的请求 | usage.prompt_tokens 翻倍，recomputed_tokens 为 0 | d3d13fb |

验收过程中，公开配置有 5 处按实际接口作了更正，判据随之调整，每处都已单独提交：
- busy 状态码按 `status` 字段判断；
- 新增 `evictable_bytes`；
- 超过上限时沿用既有长度错误文本；
- 重建时保持生效并发；
- drafter 组件名统一为 `draft_*`。

另有 3 个测试缺陷已修复：取消请求时没有真正断开连接；客户端方法名写错；共享判据改为在同一时刻比较。

## 未覆盖

- TP 路径（本轮只有单卡）。
- 多模态拒绝。
- Self-SD、hybrid 专家后端、DFlash 自适应、有限窗口 cap 的组合。
- 第 7 节性能对比。
