# 新一轮 Harness 交接

新实验从已合入公共 SD、ReplaySSM 和 DFlash 的 `main@c3b637f` 开始。使用新的 campaign、候选工作区和性能标定；历史结论作为调查线索，旧候选代码按需查阅。本文供实现和优化 agent 快速理解基座，不代替正式 campaign 配置。

## 基座和支持范围

- 公共流程负责验证、接受／拒绝、提交和缓存生命周期；SelfDrafter 与 DFlashDrafter 只负责提出候选。路由、注意力和状态差异由对应组件处理，不增加模型名白名单。
- 少专家 self-SD 已覆盖 Qwen3 MoE 与 Qwen3.6 MoE。缓存路由允许补缺加载；逐步成本自适应、专家预取、Replay 与 Graph 分别有开关。
- DFlash 首轮覆盖 Qwen3.6-35B-A3B 的匹配 BF16 检查点，整块并行起草，固定长度与2／4／8步按块自适应均可用。上下文 KV 参与公共前缀复用，额外持久内存从 GDN 状态预算扣除。
- Replay 用检查点和有限长度记录表示当前 GDN 状态，减少多步 SD 的临时完整状态需求；公共前缀仍保存完整检查点。它解决容量约束，不保证草稿越长就越快。
- **当前 SD 要求单 GPU、legacy 调度和 GPU 专家执行；已验收组合为 offload。Hybrid AR 可用，hybrid／layered SD 仍需源码适配。** 不能通过静默回退 AR 宣称完成耦合。

实现入口：`python/freetoken/speculative/`、`scheduler/speculative.py`、`engine/speculative_cost.py`、`engine/speculative_graph.py`、`kvcache/linear_state_pool.py`、`kvcache/gdn_replay.py`、`moe/routing.py`。后六项均相对 `python/freetoken/`。

公开行为见 [DFlash 契约](dflash-public-contract.md)、[Replay 契约](replayssm-public-contract.md)；当前验收范围见[基座状态](sd-common-runtime-status.md)。

## 历史实验能提示什么

下表来自远程单张4090、Qwen3.6 BF16、C16短请求实验：专家池9 GiB、目标KV 4096 token、GDN与drafter合计预算6245744640字节。宿主CPU未独占，模型输入和输出长度远短于真实agent trace。它不是新campaign的性能基线。

| 配置 | 输出吞吐 token/s |
| --- | ---: |
| Offload AR，两次 | 25.30／26.38 |
| Hybrid AR，两次 | 45.83／64.86 |
| Self-SD，自适应＋Replay | 30.11 |
| DFlash，固定4步＋Replay | 35.02 |

- **优先调查每个有效输出 token 的专家搬运。** DFlash固定4步一轮实验搬运1.446 TB，按实测PCIe带宽估算约56–58秒，接近端到端58.48秒。Verify按需加载专家，但整块候选会涉及更大的专家并集。
- **DFlash起草已经很便宜。** 该实验的草稿logits准备与模型计算约占端到端0.35%，不含过滤、采样和候选写入。只优化这段计算的收益有限；仍需观察CPU调度和完整起草阶段。
- **长草稿与自适应都要保留固定长度对照。** 固定8步和当时的按块自适应未超过固定4步；接受率、验证搬运和控制器历史共同影响结果。
- **Graph与内存容量相互影响。** 过去录制全部验证尺寸额外常驻约1.8 GB并OOM。比较时要同时记录真实／物理验证位置、进程显存峰值和缓存容量，不能只看PyTorch张量占用。
- **预取和Replay的收益有条件。** 历史预取组合没有稳定改善吞吐；hybrid AR的逆序复跑也未证实独立Replay提速。这些是特定配置下的结果，新负载可以重新检验。

上述数字、计时口径和局限见[性能报告](dflash-performance-20260930.md)。旧S2相关机制只是部分复现，不能将整篇论文的收益归到当前实现。

## 新一轮如何开始

1. 固定新main的提交、模型、硬件、输入trace、采样和工作量约束，独立标定。旧58请求、长度缩为八分之一的trace可以复用输入；旧延迟、带宽标定、控制器历史和候选成绩不直接沿用。
2. 分清两种对照：原版FreeToken固定提交用于衡量项目总收益；新main上的AR和现有SD用于衡量本轮改动。原版对照曾固定为 `9db1a39`。保留offload AR及hybrid AR，各臂重新测量。
3. 先让新版公共SD与CPU／GPU混合专家执行、分层调度真正协作，再优化组合。原候选 `9eb44ab` 的耦合基于旧架构，仅作参考；不要整体合回，或恢复旧的内联起草流程。SelfDrafter和DFlashDrafter都应复用同一目标验证与状态管理。
4. 每项改动独立提交并做A/B，明确是否改变专家、KV、GDN、drafter或Graph的资源预算。收益同时看用户延迟、尾部停顿、完成工作量，以及每个有效输出token的专家加载字节。

分段计时区分每个草稿token、每个草稿块和每轮验证；同时记录实际起草比例、接受数量、AR回退、前缀命中及Graph执行。CUDA事件时间可能包含等待，搬运时间已包含在forward中，不能重复相加。

用户此前选择性能优先，不以逐字一致或额外语义probe门控campaign；请求完成量、输出工作量和公开错误仍必须检查。实现与独立黑盒测试保持分工，已有BF16数值差异与完整任务质量边界不能抹去。

## 历史代码和工作区

主目录 `FreeToken/` 保持main。人工任务在 `.worktrees/<任务名>` 使用单独分支；harness在新campaign自己的目录创建候选。需要旧实现时按提交或归档恢复一份，不默认恢复全部旧worktree。

本机完整工作区快照位于 `/data2/servebig-envs/worktree_archive_20260930/`：包含归档时全部137个worktree的目录内容、共享 `.git`、各工作区暂存区和未提交／未跟踪／被忽略文件；清单与恢复说明放在归档旁。外部模型、共享环境及历史实验结果仍保留原处，符号链接按原样保存。新campaign使用自己的配置与结果目录，只有查阅旧实验时才需要归档。
