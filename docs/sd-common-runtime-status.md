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

## 本轮验收与比较

基线为[远端 ReplaySSM PR #3](https://github.com/ServeBig-project/FreeToken/pull/3)的 `8422e1b`，其生产 Python 与本轮起点 `54b8ab1` 相同。新增生产代码截至 `481994f` 为 **+952／−128／净+824** 行。

- 独立小模型数值参考：CPU、CUDA各8/8通过，FP32／BF16误差已量化，未调整门槛。
- 六组服务验收90/90通过：DFlash固定N8／自适应＋Replay、DFlash关闭Replay、Qwen3.6原self-SD、Qwen3原self-SD固定／自适应。覆盖C1/C4/C16、尾批、前缀、停止／取消、重建；DFlash实际执行C16×144位置Graph。
- 同一台4090、相同专家/KV/状态＋drafter预算：DFlash固定N4的C16为35.02 token/s，当前self-SD自适应30.11，PR原路径29.83；C1暂无收益。
- offload AR两次25.30／26.38，hybrid AR两次45.83／64.86，hybrid＋Replay两次65.48／65.06。不能将首轮hybrid差异归因于Replay；目前DFlash仍慢于hybrid AR。现有SD不支持hybrid专家执行。
- 当前主要成本是目标验证时的专家传输；DFlash固定N4的草稿logits准备与模型计算累计约占端到端时间0.35%（不含后续采样）。8步和当前按块自适应均未超过固定4步。
- Qwen3验收的无GDN参数和长输入样本问题已在独立测试中修复；生产代码无需因此改变。测试最终版本`c6e5bcc`。

公开边界见 [DFlash 公开契约](dflash-public-contract.md)和 [Replay 公开契约](replayssm-public-contract.md)。旧 Replay 配对实验、窗口算法证据保留在 `/data2/servebig-envs/replayssm_ab_20260929b_gpu2/`；本轮原始结果在 `/data2/servebig-envs/dflash_integration_20260930/remote-results/`。完整长上下文 agent 任务质量不由64-token性能短测证明。
