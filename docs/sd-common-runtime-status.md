# 公共 SD 基座：当前状态

Qwen3 MoE 与 Qwen3.6-35B-A3B 使用公共SD流程。2026-09-29 已按[实现协议](replayssm-implementation-protocol.md)实现 GDN ReplaySSM（`--enable-gdn-replayssm`，默认关闭）。**独立黑盒验收尚未完成，不能称为达到可交付标准。**

## 当前实现

- 公共流程使用实际路由、attention与状态组件，不设模型名字白名单；SD要求`legacy`调度。保留缓存路由、补缺加载、逐步成本自适应、verify专家预取和Graph。
- 关闭Replay时沿用原路径：draft每请求一份临时状态，verify按`Σ(N_i+1)`申请验证状态，stop／EOS确定保留长度后提交。
- 开启Replay时，请求的正式状态＝完整状态槽（checkpoint）＋按绝对位置存放的更新记录环（长度R，`--gdn-replay-buffer-len`，默认32）。target AR追加记录；draft写临时尾部；verify从target历史用完整模型重算并覆盖尾部；提交只选卷积状态。draft／verify的卷积走独立小窗口，target卷积状态只在提交时改变。不再为draft或verify分配完整状态槽。
- 合并（flush）由host在记录放不下之前决定，不读回GPU数值；不越过尚待导出的工具调用位置；请求结束捐献前先合并。公共前缀仍保存完整快照，命中后从空记录开始。
- `--gdn-state-budget-bytes`统一计价完整状态、记录和卷积窗口；未指定时沿用关闭Replay时的状态池字节。Qwen3.6、C16、R32、N8：记录182 MiB、卷积窗口90 MiB，可用完整状态槽96→91。naive默认预算放不下Replay缓冲时启动即报错，需显式预算。
- 公开字段见[公开契约](replayssm-public-contract.md)：`/v1/cache/status`的`geometry.gdn_replayssm`、`/v1/stats.gdn_replayssm`、自适应模式的`cost_gpu_ms.state`；数值验收入口见契约5.1节。

代码量相对`41b50c5`：生产 **+805/−68/净+737** 行，分四个phase提交（`a100b42`计算核心239、`39c1628`生命周期与预算354、`be6f735`服务接口88、`70d3f7e`成本与精度54，另有收尾计时调整＋1）。独立测试尚未提交，0行。

## 本轮性能（2026-09-29）

GPU2（4090），Qwen3.6 BF16，9 GiB专家池，GDN预算6245744640字节，Graph，k3缓存路由＋补缺，预取关闭。冻结请求：预热16 token，再两轮C16×64 token，计分2048 token。每臂单次运行，同一提交`70d3f7e`，开关两侧仅差`--enable-gdn-replayssm`。

| 臂 | 旧 token/s | 新 token/s | 差异 | 说明 |
| --- | ---: | ---: | ---: | --- |
| AR | 25.99 | 26.40 | +1.6% | |
| SD 固定N4 | 28.17 | 28.49 | +1.1% | 实际平均草稿3.82，两侧相同 |
| SD 固定N8 | 28.27 | 26.72 | −5.5% | 旧版受状态槽限制，B16最多80个查询；新版B16×144查询重放22次，平均草稿7.38，但接受率53%→36% |
| SD 自适应N8 | 27.35 | 28.30 | +3.5% | 平均草稿1.65→1.60；每轮状态工作GPU时间612→65 ms |

同预算容量目标已达成（真实满B16、实际query=144的N8）。固定深起草本身不划算；新版的收益来自状态工作和verify状态复制的消除，短测幅度与单次波动相当，不能宣称稳定提速。

其他冒烟：AR与原AR的贪心文本在96 token内2/4逐字一致，另2条在约第80 token分叉；fp32记录诊断显示分叉来自近平局token。前缀命中与冷算逐字一致；空闲重建96→80槽后公开字节与预算即时更新并继续生成；layered AR、naive显式预算SD N4可运行。

## 仍未解决的事项

- 独立黑盒：数值验收agent中途停止，未提交测试；服务与性能验收未由独立agent执行，需重新安排。
- 工具调用位置跨SD窗口的导出已实现，未经实际工具输出触发验证。
- joint／layered-pipeline驻留波次未验证，其解码不计入`ar_tokens`；Qwen3（无GDN）回归本轮未运行。
- 新AR臂记录到一次flush计时异常（80次共1127 ms），复测为约0.4 ms／次，未复现。
- 已知旧差异保持：stop／EOS／取消后4组命中与从头计算全文差异、AR／SD质量题差异，本轮未重做。完整agent trace与分块长输入的生命周期成本未测。

## 证据位置

本轮配对运行：`/data2/servebig-envs/replayssm_ab_20260929_gpu2/`（`run_ab.py`、`summarize.py`、`summary.json`、各臂命令／日志／公开输出）。上一阶段（状态槽阶段复用）证据：`/data2/servebig-envs/state_phase_20260925_gpu1/`、`/data2/servebig-envs/state_slots_20260925_gpu1/`。
