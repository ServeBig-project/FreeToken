# 公共 SD 基座：当前状态

Qwen3 MoE 与 Qwen3.6-35B-A3B 使用公共 SD 验证、提交和缓存流程。少专家起草与 DFlash 起草已拆成独立组件；DFlash 首轮目标为 Qwen3.6-35B-A3B。本轮实现、远端部署和独立验收已完成；完整数据及统一风格的图见[性能报告](dflash-performance-20260930.md)。

## 已实现

- 不使用模型名字白名单。公共流程根据路由、注意力、状态和目标特征导出能力组织；模型组件处理架构差异。
- 不指定 `--speculative-draft-model-path` 时沿用少专家 SD：缓存路由、补缺加载、逐步成本自适应、预取及 Graph 开关保持独立。
- 指定 DFlash 目录时，用小模型并行提出整块候选，复用目标 embedding／输出层；完整目标模型仍负责验证、拒绝采样和最终提交。固定长度及按块自适应均可用，首轮最多8个草稿 token。
- DFlash 上下文 KV 与目标 token 页共同分配和回收，使用同一公共前缀树。被拒绝位置不属于有效历史；验证重新写入完整目标模型的特征。
- DFlash 权重、持久上下文和固定工作区从既定 GDN 状态预算中扣除，专家池和目标 KV 预算不变。Graph 可执行对象另通过进程显存观察。
- 修复了 GDN 最后一个短 prefill 分块没有保留块起点快照的问题：544-token 输入在512处分块后，可复用前512 token；不再为每个请求预留两份快照工作槽。

## Replay 状态表示

- 关闭 Replay：target 当前状态保存在完整矩阵中，少专家 draft 另用工作状态，verify 需要逐位置状态。DFlash draft 不申请目标递推状态。
- 开启 Replay：当前状态由完整检查点和长度有限的更新记录共同表示。少专家 draft 使用临时记录；target verify 覆盖临时尾部；提交只保留接受部分。记录环快满时合并到检查点。
- 位置游标常驻 GPU；合并在写入新记录前执行，避免环覆盖竞争。公共前缀仍保存完整检查点，命中后从空记录环开始。
- verify 保留逐 token 递推实现。此前同精度窗口矩阵算法没有测得优势；不能把这一实现描述为已复现论文全部 kernel。
- verify Graph 沿用公共尺寸策略：常用的2次幂batch额外录制每请求2–5个输入的尺寸，回放与成本估计使用能容纳本轮的最小尺寸。此前录制全部倍数会额外占用约1.8 GB并OOM，未采用；尺寸A/B证据在 `/data2/servebig-envs/sd_graph_sizes_ab_20260929b/`。

## 本轮验收与比较

基线为[远端 ReplaySSM PR #3](https://github.com/ServeBig-project/FreeToken/pull/3)的 `8422e1b`，其生产 Python 与本轮起点 `54b8ab1` 相同。新增生产代码截至 `481994f` 为 **+952／−128／净+824** 行。

- 独立小模型数值参考：CPU、CUDA各8/8通过，FP32／BF16误差已量化，未调整门槛。
- ReplaySSM计算核心此前在`f184217`上通过24项独立数值验收，包含环满场景重复1600次无错误；记录见PR #3，后续DFlash接入未修改该计算核心。
- 六组服务验收90/90通过：DFlash固定N8／自适应＋Replay、DFlash关闭Replay、Qwen3.6原self-SD、Qwen3原self-SD固定／自适应。覆盖C1/C4/C16、尾批、前缀、停止／取消、重建；DFlash实际执行C16×144位置Graph。
- 同一台4090、相同专家/KV/状态＋drafter预算：DFlash固定N4的C16为35.02 token/s，当前self-SD自适应30.11，PR原路径29.83；C1暂无收益。
- offload AR两次25.30／26.38，hybrid AR两次45.83／64.86，hybrid＋Replay两次65.48／65.06。不能将首轮hybrid差异归因于Replay；目前DFlash仍慢于hybrid AR。现有SD不支持hybrid专家执行。
- 当前主要成本是目标验证时的专家传输；DFlash固定N4的草稿logits准备与模型计算累计约占端到端时间0.35%（不含后续采样）。8步和当前按块自适应均未超过固定4步。
- Qwen3验收的无GDN参数和长输入样本问题已在独立测试中修复；生产代码无需因此改变。测试最终版本`c6e5bcc`。

## 接口清理与合入门控

- DFlash 预算改为显式传参，移除未用 logits 回调；块状起草通过成本组件的方法收集、估价和计数，不再访问其私有方法。成本公式、资源分配和模型支持范围保持原有行为。
- 仓库单测迁移到按需单快照、当前 Context／路由和重建消息；包括原 main 上已过期的 NoWAG 能力判断及输出参数测试。CPU 命令为 `pytest tests --ignore=tests/e2e -k 'not real_server'`，隐藏 CUDA并加载本地 NoWAG 插件。
- 最终 PR3 `9343d45`、PR4 `10f4303` 各 **1294通过、0失败**，335跳过、8排除。同一命令下 main 原有2项失败、PR分支原有29项失败；本轮全部消除。GPU／真实服务用例不由该CPU门控证明。
- 独立 HTTP 回归：本机 GPU0，DFlash 自适应N8＋Replay＋Graph；9请求正常，实际起草107 token、验证15轮。KV 4096→3072后 DFlash 预留894591360→869425536字节，非法重建保留旧资源。四条重复请求中一条尾部文字不同，已记录；不宣称逐字一致。
- 原性能报告仍是清理前的测量，本次没有重跑吞吐矩阵。单测与服务证据在 `/data2/servebig-envs/sd_interface_cleanup_20260930/`。

| 提交 | 范围 | 生产 +/− | 测试 +/− |
| --- | --- | ---: | ---: |
| `60c3889` | PR3未用模型属性 | 0/2 | 0/0 |
| `53fe4ad` | 公共SD旧测试迁移 | 0/0 | 173/77 |
| `9343d45` | NoWAG与快照测试契约 | 0/0 | 24/5 |
| `3bd66f5` | DFlash预算与成本接口 | 45/34 | 0/0 |
| `10f4303` | DFlash重建测试消息 | 0/0 | 1/0 |
| `4f05acd` | 独立HTTP回归 | 0/0 | 204/0 |

相对本轮清理前 `a622cdc`，生产 **+45/−36/净+9**，测试 Python **+401/−81/净+320**；不含文档与结果文件。Replay 的空 `prepare_verify` 是公共接口中的有效空操作，保留。

## Harness 下一轮基座与任务

- PR3与PR4已合入 `main@c3b637f`。新一轮从该基座固定提交、新建campaign并重新标定；入口、历史线索及对照要求见[Harness交接](harness-handoff.md)。测量用的上游AR对照与候选起点分别记录，保留真实agent trace、相同输入和工作量约束。
- 原harness候选`9eb44ab`的hybrid／layered-pipeline耦合仅作为参考。它基于早期公共SD，未包含Replay与SelfDrafter／DFlashDrafter拆分，不整体合回新版。
- 耦合是源码适配任务，不只是开关搜索：复用公共draft／verify／提交和状态管理，适配CPU/GPU专家执行及分层调度，覆盖已有SelfDrafter与DFlashDrafter。不得恢复模型名白名单或退回旧的内联起草流程。
- 当前验收范围是legacy／offload SD；CPU/GPU hybrid只测过AR。新组合需独立黑盒和真实trace A/B，报告实际draft／verify执行、接受量、专家传输和内存；不能以静默退回AR代替耦合完成。工具调用特殊检查点及其他调度组合不由现有短测证明。

公开边界见 [DFlash 公开契约](dflash-public-contract.md)和 [Replay 公开契约](replayssm-public-contract.md)。旧 Replay 配对实验、窗口算法证据保留在 `/data2/servebig-envs/replayssm_ab_20260929b_gpu2/`；本轮原始结果在 `/data2/servebig-envs/dflash_integration_20260930/remote-results/`。完整长上下文 agent 任务质量不由64-token性能短测证明。
