# Runtime 共享池：独立黑盒补验

状态：本轮四组请求补验完成。`d912bbe` 同配置仍留有一次 Graph 捕获 OOM 的启动失败；成功重放不证明该启动问题已消除，不能据此宣布整体达到可交付标准。TP、多模态按本轮用户决定暂缓。

测试作者只读取公开契约、运行配置、请求数据和公开输出，未读取生产源码、diff、内部测试或实现笔记。协调者启动服务并运行测试；本报告不以实现说明替代观察。

## 版本与运行条件

| 组别 | 实际受测版本 | 配置 |
| --- | --- | --- |
| 分池 AR 基线 | `732f1ee` | 专家 2048，KV 65536 tokens，并发上限 4 |
| 共享 AR 对照 | `6556e13` | 专家 2048，runtime 2.75 GiB，并发上限 4 |
| Self-SD 两轮 | `187dc9a` | offload，专家 2048，runtime 0.5 GiB，并发上限 6，host 实际预算 0，SD4，不给 draft 路径 |
| 自适应 DFlash | `187dc9a` | legacy/offload，专家 2048，runtime 0.75 GiB，并发上限 6，host 4 GiB，SD4，full-attention cap 128 |
| 原 58 请求复验 | `d912bbe` | hybrid，专家 5000，并发上限 8，runtime 8.626953125 GiB，上下文 49152，prefill 上限 8192，Graph 上限 8，ReplaySSM，layered-pipeline，DFlash4 |

共同条件：Qwen3.6-35B-A3B-NVFP4，GPU1 `GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd`，CPU 绑定 `8-15,24-31`，attention backend `fi`，开启 cache report，sampling defaults `none`。DFlash 使用公开配置中的 `f181eece646affea2c38b2765f1aaa01a9734ccd` checkpoint。

宿主机另有 GPU0、GPU2 任务，不是独占性能实验。各组按实际版本报告，不把未重跑的组改写为最新版本实测。

证据根目录：`/data2/servebig-envs/runtime_pool_closeout_20261009/`。各 `*-server.json` 保存完整启动参数及 source；每个服务测试目录保存实际请求、SSE 事件、usage、逐阶段 stats 和资源采样。

## 固定资源短→长→短续写

基线与候选请求文件完全相同：4 条短请求→2 条长请求→4 条短续写→4 条重复前缀请求。长输入为 5555/5556 tokens；两轮均 14/14 完成，共输出 1408 tokens。

基线实际 KV+GDN 为约 2815.176 MiB；共享预算 2816 MiB，相差约 0.824 MiB。整个序列不重建、不改预算，专家容量不变。

| 完整请求指标 | 分池基线 | 共享候选 |
| --- | ---: | ---: |
| 全序列耗时 | 26.800 s | 27.324 s |
| 输出吞吐，包含 prefill 与等待 | 52.537 token/s | 51.529 token/s |
| 请求吞吐 | 0.5224 请求/s | 0.5124 请求/s |
| 平均 / p95 请求延迟 | 6.246 / 8.450 s | 6.329 / 8.647 s |
| 平均 / p95 TTFT | 2.244 / 4.273 s | 2.295 / 4.456 s |
| 平均 / p95 TPOT | 40.401 / 56.277 ms | 40.702 / 56.480 ms |
| 最大流式输出间隔 | 275.073 ms | 205.897 ms |

约 2% 的总耗时差异不作实现因果归因，也未使用未经确认的退化阈值。

- 初始 4 条请求均观察到两两输出交叠；最终四条原 prompt 均复用 128 tokens。
- long 两条均复用 267 tokens。候选 short-again 的命中为 267/267/128/128，基线均为 267：后两条候选先前生成文本不同，而续写输入固定使用基线文本，因此不能复用完整生成前缀。这不是缓存被无故清空的证据。
- 候选持有量从初始 102 MiB，经各阶段变为 594/1040/1448/1694 MiB；所有采样满足 `0 <= used <= held <= budget`、`waste >= 0`。
- 此序列未发生暂停、重算或解除映射，不能单凭本项证明压力下跨组件回收。

证据：`baseline-sequence-v2/`、`candidate-ar-sequence/`。

## Self-SD：回退、暂停与重算

两轮均先发一条 64-token 参考请求，再发 6 条并发请求，最后发相同的 64-token 参考请求；输入均为 140 tokens。

| 观察 | 每条输出 1200 的第一轮 | 每条输出 2400 的补充轮 |
| --- | ---: | ---: |
| 完整请求 / 总输出 tokens | 8/8，7328 | 8/8，14528 |
| 压力阶段耗时 | 89.133 s | 198.165 s |
| 压力阶段 draft / accepted / verify 增量 | 15 / 12 / 8 | 981 / 794 / 491 |
| 压力阶段暂停 / 重算 | 0 / 0 | 4 / 4 |
| 重算 tokens | 0 | 6374 |
| 压力后参考的 draft / accepted / verify 增量 | 46 / 40 / 23 | 46 / 40 / 23 |

第一轮触及预算并发生 6994 次 `kv_capacity` 回退，随后恢复真实 SD；不能把它写成暂停恢复验收。补充轮实际完成暂停与重算，最长流式输出间隔 22.101 s，全部请求最终完成，重算没有增加用户 usage。

补充轮 202 个资源样本的最大 held 为 510 MiB，不超过 512 MiB。host 预算、分配、使用均为 0。所有请求均只有一个 usage，completion_tokens 等于要求值，total_tokens 等于 prompt 加 completion，finish_reason 为 length。

文本边界：第一轮前后参考逐字一致；补充轮开始参考命中 128-token 缓存，与本轮结束的无缓存参考不同。补充轮结束文本与第一轮无缓存参考逐字一致，不能声称补充轮自身前后逐字一致。

证据：`self-sd-pressure/`、`self-sd-pressure-2400/`。

## 自适应 DFlash 与显式 cap

8/8 完整，共输出 7328 tokens；六条压力请求各输出 1200 tokens。公开 `geometry.dflash.attention_window=128`，所有输入均为 140 tokens，超过 cap。`window_tokens=4095` 保持原生 4096 滑窗语义；cap 128 只约束草稿的 full-attention 历史，不写成所有层或目标上下文的上限。

| 阶段 | 实际 nominal 0 / 2 / 4 增量 | draft / accepted / verify 增量 |
| --- | --- | --- |
| reference | 58 / 1 / 1 | 6 / 2 / 2 |
| pressure | 2673 / 11 / 10 | 45 / 27 / 21 |
| after | 62 / 0 / 0 | 0 / 0 / 0 |

公开 `dflash_control=adaptive`。使用 DFlash 自身的实际窗口计数判断 AR；generic `cost_ar_requests=0` 不能代表未执行 AR。reference 已出现 AR，后续 pressure 又真实执行 SD；最后 after 选择 AR，不能声称最后一条还执行了 SD。

压力阶段发生 3 次暂停和 3 次 host 恢复，重算为 0。93 个 GPU 样本最大 held 为 764 MiB，不超过 768 MiB；host 样本均满足 used≤allocated≤4 GiB。after 复用 128 tokens，host_reused_tokens 同增 128，证明自然 AR 选择后仍保留可恢复的冷前缀。前后参考文本不同，分别为无缓存与命中 128 tokens 的路径；保留观察，不判作逐字等值。

证据：`adaptive-cap-pressure/`。

三组候选均实际执行 Graph：AR 的 target_decode 增 412；Self-SD 2400 的 target_decode/draft/verify 增 4661/1070/535；自适应组增 2797/23/23。运行中 capture_seconds 增量均为 0，speculative_eager 均为 0。

## 原 58 请求长历史负载

原始数据为 `trace-benchmark/outputs/agent-5u5f/prepared-1of16/`：58 请求，含 12 warmup 与 46 scored；输入 4076–45475 tokens，输出上限 2–196。直接重放现成数据，不进一步缩放输入、输出或删请求。

公开固定池容量参照为含 compact 的 main732：KV 2,013,286,400 B，GDN+Replay 5,986,222,228 B，draft full 402,657,280 B，draft window 860,835,840 B，总计 9,263,001,748 B。共享按 2 MiB 向上取整为 9,263,120,384 B，多 118,636 B。历史 OOM 基座早于 DFlash 内存优化，不能把此次结果全部归因共享池。

| 已保留的失败 | 公开观察 |
| --- | --- |
| `187dc9a`，首次启动 | ready 前报 `a 1-token prefill does not fit the 20.71 GiB left beside runtime...`，未发送负载 |
| `271ce06`，v2 | ready，公开 prefill_tile_tokens=1661；warmup 申请 14 MiB 时 CUDA OOM，随后 HTTP 503；产物仅有 4 条 warmup 记录，1 条 503、3 条客户端取消，0 条完成 |
| `d912bbe`，v3 | CUDA Graph 捕获时 OOM，未 ready，未发送负载 |

v4 使用同一 `d912bbe` 和完全相同参数，58/58 完成，12 warmup、46 scored，0 错误。请求 ID 集合与原 manifest 一致，全部 prompt_tokens、completion_tokens 逐条等于原输入和输出上限；total_tokens 均正确。共处理 1,244,452 prompt tokens，输出 1847 tokens，没有缩短请求。

| 指标 | v4 观察 |
| --- | ---: |
| 包含 warmup 的全序列耗时 | 214.585 s |
| 全序列输出吞吐 / 请求吞吐 | 8.607 token/s / 0.2703 请求/s |
| 46 条 scored 平均 / p95 延迟 | 23.955 / 43.500 s |
| scored 平均 / p95 TTFT | 22.401 / 38.507 s |
| scored 平均 / p95 TPOT | 58.361 / 113.146 ms |
| 最大流式输出间隔 | 920.968 ms |

322 个 serving 资源样本的预算均为 9,263,120,384 B，专家均为 5000，所有 runtime 与组件均满足字节关系；最大 held 为 9,258,926,080 B。Replay 的 `replay_u/k/g/window` 字节均合法，末次 stats 记录实际 Replay AR 1703 tokens、verify 142 tokens、35 次 flush，并非仅声明 active。

实际 DFlash draft/accepted/verify 为 113/49/29，全部 verify 在 outwave；Graph draft/verify 各 29，capture_seconds 未增长。prefill 配置上限仍为 8192，公开生效 tile 为 1661。KV 达到峰值时持有 4600 MiB、GDN state 2220 MiB；结束时分别为 3280/3180 MiB，固定预算下观察到用途变化，累计 unmap 2315 次，无显式重建。

GPU 前缀复用 811710 tokens；runtime 暂停、恢复、重算均为 0。顶层 prefix_cache.recomputed_tokens 为 432742，恰为总 prompt 减去复用 tokens，不能把它混称为暂停恢复重算。

资源采样在 `hybrid-trace-v4-status/cache-status.jsonl`，请求、stats 与 summary 在 `hybrid-trace-v4/`。保留上表三轮失败，不因 v4 完成而删除。协调采集器曾占用输出目录导致外部 runner 拒绝，已修正，不计生产故障。

## 复现入口与计量范围

在对应配置服务的 `/v1/cache/status` 为 serving 后，从项目根目录运行；每次使用新的输出目录。完整服务命令见证据根目录的 `baseline-server.json`、`candidate-ar-server.json`、`self-sd-server.json`、`adaptive-cap-server.json`、`hybrid-trace-v4-server.json`。

启动时 `CUDA_VISIBLE_DEVICES` 固定为上述 GPU1 UUID。基线 `PYTHONPATH=.worktrees/runtime-pool-base/python`，候选为 `.worktrees/runtime-pool/python`；复现历史轮次需使用表中相应 source，不能把当前工作树当作历史版本。

```bash
PY=/home/nengneng/miniconda3/envs/freetoken-dev/bin/python
DRIVER=.worktrees/runtime-pool-final-blackbox/blackbox_tests/runtime_pool/test_service_final.py
RESULTS=/data2/servebig-envs/runtime_pool_closeout_20261009

"$PY" "$DRIVER" --url http://127.0.0.1:31940 --case sequence --output /tmp/baseline-sequence-new
"$PY" "$DRIVER" --url http://127.0.0.1:31940 --replay "$RESULTS/baseline-sequence-v2/requests.json" --output /tmp/candidate-sequence-new
"$PY" "$DRIVER" --url http://127.0.0.1:31940 --case pressure --output /tmp/pressure-1200-new
"$PY" "$DRIVER" --url http://127.0.0.1:31940 --case pressure --pressure-output 2400 --output /tmp/pressure-2400-new
"$PY" trace-benchmark/replay.py run --prepared trace-benchmark/outputs/agent-5u5f/prepared-1of16 --case multiuser --base-url http://127.0.0.1:31940 --timeout-seconds 900 --output /tmp/hybrid-trace-new
```

吞吐包含完整请求的 prefill、排队与生成；最大间隔测量可见 SSE 文本事件。SSE 身份、终止、usage 与资源关系通过，不把任意自然语言重复当作 token 重发，也不据跨调度文本差异声称数学错误或证明逐 token 数学等值。

新增独立测试提交为 `ee6343f`（测试 +242/−0）和 `15682e9`（测试 +2/−1）；相对本补验开始前基线，测试代码新增 243 行、删除 0、净增 243；生产代码 +0/−0。文档和生成结果不计测试代码。
