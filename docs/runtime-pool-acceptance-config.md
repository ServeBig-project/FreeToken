# 共享 runtime 池：独立验收运行配置

协调者提供给独立黑盒测试作者的公开运行信息。配合 `docs/runtime-pool-public-contract.md` 使用；不含实现细节。

## 环境

- Python：`/home/nengneng/miniconda3/envs/freetoken-dev/bin/python`。
- 被测实现：`PYTHONPATH=/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/runtime-pool/python`。只能通过 CLI、HTTP 和本文列出的公开入口使用，不得阅读其中源码、diff 或笔记。
- 原分池基线（对照用）：`PYTHONPATH=/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/dflash-mainline/python`。
- 启动命令：`python -m freetoken --port <端口> --gpu <GPU UUID> --model-path <模型> ...`；就绪判断：`GET /v1/cache/status` 的 `state == "serving"`。模型加载约 4–5 分钟。
- GPU：只能使用 GPU1 `GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd`（RTX 4090，24 GiB），设置 `CUDA_VISIBLE_DEVICES=<UUID>` 并传 `--gpu <UUID>`。GPU0／GPU2 属于其他用户，禁止使用。每个服务进程必须用 `taskset -c 8-15,24-31` 启动（其他核归另一组实验，混用会让 CPU 专家计算慢 5～10 倍）。GPU1 分时共享：只有协调者消息允许时才能启动 GPU 进程；每轮结束退出全部进程并向协调者报告。
- 模型：`/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4`（hybrid_linear：GDN 线性层＋全注意力，MoE，上下文 262144）。
- DFlash 草稿模型：`/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/f181eece646affea2c38b2765f1aaa01a9734ccd`。
- 公共参数：`--moe-backend offload --moe-cache-size 2048 --attention-backend fi`。专家容量由 `--moe-cache-size` 固定。
- 只有单 GPU 可用，TP 路径本轮无法验收。

## 启动参数

共享模式：
- `--runtime-cache-gib R`（R > 0）：启用共享模式，R 为每个 GPU worker 的 runtime 预算。
- `--max-running-requests N`：可选；共享模式下与自动推导值取较小者。
- `--prefix-cache-host-gib H`：host 数据预算（冷前缀与暂停副本共用）；0 表示无副本，暂停走重算。
- `--cache-type naive|radix`：默认 radix（本模型实际为 hybrid_radix）；naive 关闭公共前缀缓存。
- SD：`--speculative-num-steps 4 --speculative-draft-model-path <草稿模型>`；可加 `--speculative-phase outwave|all|inwave`。
- `--batching-policy auto|layered-pipeline|legacy`（auto 对本模型解析为 layered-pipeline）；`--max-extend-tokens`（默认 8192）。
- 与共享模式冲突、启动前报错退出：`--num-pages`、`--num-tokens`、`--gdn-state-budget-bytes`、`--kv-reserve-tokens`（以及只在 Python 配置里存在、没有 CLI 开关的 `linear_state_cache_ratio`）；`--runtime-cache-gib 0` 或负数同样报错。错误文本出现在进程输出中（例如 `--runtime-cache-gib shares one budget; it conflicts with the fixed pool sizes ...`、`--runtime-cache-gib must be > 0`）。
- 总预算放不下（例如 24 GiB 卡上 `--runtime-cache-gib 30`）或显式 `--max-seq-len-override` 超过单请求可执行上限：在权重加载之后、ready 之前以可读文本报错退出（`needs ... a GPU has only ... left after weights and experts`、`... of runtime holds one request of N tokens; M are required`）。
- 原模式对照：`--num-tokens 65536 --max-running-requests 4` 等旧参数，不带 `--runtime-cache-gib`。

验证过的参考配置（单卡）：
- 普通：`--runtime-cache-gib 4 --max-running-requests 4`。
- 压力：`--runtime-cache-gib 0.5 --max-running-requests 6`，可加 `--prefix-cache-host-gib 4`；6 条 1200 token 输出的请求会触发暂停。
- DFlash：`--runtime-cache-gib 4 --max-running-requests 4 --speculative-num-steps 4 --speculative-draft-model-path <草稿模型>`。

## HTTP 接口

- 生成：`POST /v1/completions`、`POST /v1/chat/completions`（OpenAI 兼容：`stream`、`max_tokens`、`temperature`、`stop`、`ignore_eos`、`cache_group`）；`GET /v1/models`。
- 状态：`GET /v1/cache/status`、`GET /v1/stats`。
- 维护：`POST /v1/cache/rebuild`，JSON 体。共享模式接受 `runtime_cache_gib`（可同时给 `moe_cache_size`）；`num_pages`、`num_mamba_slots`、`num_swa_pages`、`swa_full_tokens_ratio` 在共享模式被拒绝。生效并发（启动时显式给出或自动推导的 `max_running_requests`）在服务生命周期内保持不变：新的 `runtime_cache_gib` 必须能让该并发数的请求各自以最小占用同时放下，否则以 `rejected`（文本含 `requests at their minimum; N are required`）拒绝且旧缓存继续服务；要允许更小的 runtime，需以更小的 `--max-running-requests` 启动。只支持 `mode="if_idle"`：调度器不空闲时立即拒绝，不等待；请求体的 `timeout`（默认 300 秒）只是等待调度器答复的上限。响应 `status` 为 `ok`（HTTP 200）、`rejected`（HTTP 503，`error` 为文本）或 `busy`：另一次重建或停机进行中为 HTTP 409，有请求在运行、暂停、保存或恢复中为 HTTP 503（沿用既有维护接口）。以 `status` 字段判断，不以状态码区分 busy 与 rejected。
- 生成错误：非流式返回 OpenAI 风格错误体 `{"error": {"message", "type": "invalid_request_error", "code"}}`（HTTP 4xx）；流式先发一个带 `error` 的 SSE 块再发 `[DONE]`。超过公布的 `context_tokens`（prompt＋生成）的请求在准入前即以 `code` 为 `context_length_exceeded`、文本 `prompt is too long: N tokens > M maximum ...` 拒绝（沿用既有长度错误）；已准入但运行中放不下共享 runtime 的请求以同一 `code`、文本含 `does not fit the shared runtime` 或 `no longer fits the shared runtime even alone` 结束；共享模式的多模态请求以文本 `multimodal requests are not supported with --runtime-cache-gib` 拒绝。

## 共享模式的公开状态字段

`GET /v1/cache/status`：
- `geometry.runtime_cache_bytes`：共享预算；`geometry.address_pages`、`geometry.address_mamba_slots`：只是地址空间上限；`geometry.num_pages`、`geometry.num_mamba_slots` 在共享模式为 0；`geometry.moe_cache_size`：专家容量。
- `prefix_cache.runtime`：物理占用 `budget_bytes`、`granularity_bytes`、`free_bytes`、`held_bytes`、`used_bytes`、`waste_bytes`、`idle_bytes`、`protected_bytes`、`components{<组件名>: held_bytes/used_bytes/waste_bytes/idle_bytes/protected_bytes/address_bytes}`（组件名：目标 KV `kv`，GDN 状态 `gdn_conv` 与 `gdn_state`，ReplaySSM 记录 `replay_*`，DFlash 历史 `draft_kv`）、`map_count`、`unmap_count`、`map_ms`；可回收缓存 `evictable_bytes`（没有请求持有的前缀 KV 与 GDN 检查点，任何申请都可回收；`held_bytes` 包含它）；生效上限 `context_tokens`（单请求可执行上限）、`max_running_requests`（生效并发）、`requested_running_requests`（省略 `--max-running-requests` 时为 null）、`resource_running_requests`、`requested_context_tokens`、`model_context_tokens`、`execution_bytes`；暂停统计 `paused`、`restored`、`recompute`、`recomputed_tokens`、`paused_ms`、`short_decode`、`short_prefill`、`compactions`。
- `prefix_cache.components[]`：各组件 `device_allocated_bytes` 与 host 字节；`prefix_cache.host_*`：host 预算与使用。
- `GET /v1/stats`：既有字段（`kv.used_pages/total_pages`、`cuda_graph`、`speculative`、`throughput`、`requests` 等）。

## 协调规则

- 只写黑盒测试；不读实现源码、diff、内部测试或实现笔记。
- 测试放在本工作树 `blackbox_tests/runtime_pool/` 下（P0 算子套件已在此），服务级文件名 `test_service_*.py`；按阶段提交到 `test/runtime-pool` 分支，每个提交报告测试代码增加／删除行。
- 每个失败只报告公开可复现条件：启动参数、请求、观测到的输出或状态字段、期望行为（引用契约条款）。
