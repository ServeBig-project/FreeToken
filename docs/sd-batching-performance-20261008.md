# SD＋batching P5 性能报告

范围（用户 10-08 确认）：只验两项——默认路径回归（main 默认 AR vs 新默认 layered＋AR）与主要收益路径（NVFP4 DFlash 下 legacy vs layered）。负载为真实 multi-agent trace 与对 layered 不利的 lab 同步突发；服务配置固定，不逐负载调参。self-SD、自适应、驻留／预取不展开性能矩阵。

## 结论

- **默认路径**：真实 trace 上新默认总耗时快 3–4%，最长输出停顿 11–14 s → <1 s；平均延迟慢 9–14%，TTFT 约 3 倍。lab 同步突发上慢 15%。原因见“归因”。
- **DFlash（NVFP4）**：lab 上 layered 与 legacy 持平（34.0 vs 34.3 s，两组顺序效应约 2 s）；trace 上 layered＋DFlash 在紧显存配置下 OOM，未得到对比（已知问题，HBM 预算由后续 PR 处理）。
- 与 main 比较：layered＋DFlash 在 lab 上与 main 默认持平（34.2 vs 34.4 s），平均延迟快 11%、TTFT 快 29%。

## 环境

GPU1（24 GB），Qwen3.6-35B-A3B NVFP4，`--moe-backend hybrid --moe-cpu-threads 7 --moe-cache-size 5000`，ReplaySSM，独占 cpuset 0-7,16-23。时段间存在漂移，只比较同时段结果；外部容器占用核心的一组已隔离不计。原始数据 `/data2/servebig-envs/sd_batching_ship_p5_20261008`。

## 1. 默认路径回归（最终代码 `59e0071`）

真实 trace：agent-5u5f multiuser 1/16 长度，58 请求、约 97 万输入 token（82–84% 命中缓存）、1475 输出 token；`--max-running-requests 8 --num-tokens 196608 --max-seq-len-override 49152 --cuda-graph-max-bs 8`；正序、反序各一组。

| 配置 | makespan s | 平均延迟 s | TTFT 均值 s | TTFT p95 s | 最长停顿 s | vram GB |
|---|---|---|---|---|---|---|
| main 默认（legacy AR） | 117.1 / 115.3 | 9.69 / 9.05 | 2.51 / 3.11 | 8.55 / 8.53 | 14.1 / 11.3 | 18.75 |
| 新默认（layered AR） | 113.0 / 111.4 | 10.53 / 10.33 | 8.76 / 8.66 | 14.8 / 14.9 | 0.93 / 0.23 | 19.16 |

输出极短，TPOT 不具代表性，未列。

lab_agent_burst_v1 main 档（4 用户×5 轮同步，512 输出／轮；`--max-running-requests 4 --num-tokens 20480 --cuda-graph-max-bs 4`），正反两组均值：

| 配置 | makespan s | 平均延迟 s | p95 延迟 s | TTFT s | TPOT ms | vram GB |
|---|---|---|---|---|---|---|
| main 默认 | 34.43 | 6.77 | 7.08 | 1.42 | 10.48 | 15.15 |
| 新代码 legacy AR | 34.61 | 6.81 | 7.01 | 1.44 | 10.49 | 15.15 |
| 新默认 layered AR | 39.44 | 7.72 | 8.70 | 0.96 | 13.22 | 14.61 |

### 归因

- **同时到达的请求串行 prefill**：默认 `--prefill-wave-max-chunks 1`，每个波次只放一个 prompt 分块；legacy 会把同时到达的 prompt 合并成一次 prefill。lab 每轮 4 个请求同时到达，layered 拆成 4 个波次，解码要陪跑多个波次（TPOT 13.2 vs 10.5 ms）；trace 上表现为 TTFT 变长。main 自身的 layered（`63156df`）同样慢，非移植引入。
- **batch=3 的 Graph 覆盖**（已修）：P6 曾只录 [1,2,4]，分层解码在 batch=3 走 eager；同时段 layered AR 46.3–47.0 → 39.1 s。`59e0071` 改为 min(并发,8) 内每个尺寸都录，约 1–2 MB／尺寸。
- **收益**：layered 让已在输出的请求不再被别人的长 prefill 卡住，trace 最长停顿 11–14 s → <1 s。

## 2. NVFP4 DFlash：legacy vs layered（最终代码）

lab，正序／反序：

| 配置 | makespan s | 平均延迟 s | TTFT s | TPOT ms | vram GB |
|---|---|---|---|---|---|
| legacy DFlash4 | 35.20 / 33.40 | 6.23 / 5.83 | 1.23 / 1.03 | 9.78 / 9.39 | 15.43 |
| layered DFlash4 | 32.82 / 35.16 | 6.03 / 6.28 | 1.03 / 1.00 | 9.78 / 10.34 | 14.85 |

两组均为后跑的一项更快，平均持平。接受率约 72–76%。

trace（为放下 DFlash 上下文改为 `--num-tokens 98304 --gdn-state-budget-bytes 6e9`，其余同上）：legacy DFlash4 makespan 222.8 s、平均延迟 19.8 s；layered DFlash4 在第 2 个波次 CUDA OOM（GDN prefill 内核申请 64 MiB 失败），同配置 layered AR 与 main layered AR 正常。启动后剩余显存与 layered AR 相同，说明 layered＋DFlash 运行期额外显存未计入 prefill 显存预算。用户决定由后续 HBM 预算 PR 解决。

## 正确性与资源（GPU）

- SD Graph 上限 3/5/7：启动与重建后 batch=N 的 draft/verify 都走 Graph；`execution.resources` 在 ready、回复与重建后都有值（重建后 SD Graph 新增保留量可能偏低，已在文档说明）。
- greedy：SD 与 AR、首次与缓存命中之间都可能在后半段分叉（近似并列 token），不开 SD 时同样存在；按契约量化报告，非门槛。
- 独立黑盒第 2 轮（`d0af4f4`）15 个模块全部通过。

## 未覆盖

- trace 上 layered＋DFlash（OOM，见上）；BF16 不在本次范围（此前测得 BF16 下 DFlash 与 layered 均更慢）。
- 冷前缀压力下的性能。
