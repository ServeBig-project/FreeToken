# 共享 runtime 池：服务级黑盒验收报告

依据 `docs/runtime-pool-public-contract.md` 第 2–6 节和 `docs/runtime-pool-acceptance-config.md`。本报告只记录公开可复现的条件和观测，不推测内部原因。套件在 `blackbox_tests/runtime_pool/test_service_*.py`。

## 结论

被测实现 `feat/runtime-pool@65b7c48` 上，B–F 五个服务模块 25/25 通过。启动期 A 模块：6 项无需 GPU 的用例在 65b7c48 上通过；2 项需 GPU 的用例只在 793e48d 上跑过，结果为通过。TP、多模态和第 7 节性能对比本轮未验收。

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

## 过程中发现并已修复的公开问题

| 首次发现 | 条件 | 观测 | 修复 |
|---|---|---|---|
| 793e48d | 共享模式省略 `--max-running-requests` | ready 前崩溃，TypeError | 修复后 B 通过 |
| 793e48d | `geometry` 字段 | 缺少 `runtime_cache_bytes` 和 `address_*` | 已补；在 dd8bf5e 上 `address_*` 为 0，1d6df11 修复 |
| dd8bf5e | 超过 `context_tokens` | HTTP 400，但 `code` 为 null | 849edc8 |
| dd8bf5e | 重算路径暂停 | `paused_ms` 一直为 0 | 已修，D 已验证 |
| dd8bf5e | DFlash 组件名 | 报告为 `full`/`swa`，无法辨认属于 drafter | 65b7c48 改为 `draft_*` |

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
- 可选模块 G：DFlash + Replay 在压力下的暂停恢复。
- layered/legacy、compact 开关、Graph 关闭等功能组合矩阵。
- 第 7 节性能对比。
