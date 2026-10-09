# DFlash 主线交付：独立公开验收契约

状态：目标行为已确认，待验收。实现基线为 SD batching（`feat/sd-batching-ship`，合入后以 main 为准）：公共 SD 已支持 layered-pipeline 调度与 hybrid 专家后端。

本轮交付固定长度 DFlash。`--speculative-adaptive-cost` 与 observe-only 保留、默认关闭，其策略收益和同执行开销门槛（第 6 节第 1、3 项）另开 issue，不作为本轮交付条件；功能正确性仍需验收。

本文可单独交给独立黑盒 agent。测试作者只能获得本文、公开模型格式/数学、CLI/API 和运行权限；不得读取实现源码、diff、内部测试或实现协议。公开失败由协调者转给实现者，不向实现者开放黑盒测试源码。

## 1. 范围、默认值与配置

- 单 GPU；legacy 或 layered-pipeline 调度（后者按 `--speculative-phase outwave|inwave|all` 选择在 prefill 波次外、内或两者起草）；专家后端 offload 或 hybrid；目标权重 BF16/NVFP4；DFlash 自身使用公开支持的 BF16 权重/激活。Graph 覆盖并发与自然尾批。
- 保留现有 DFlash 的 FlashInfer、BF16 激活和 page_size=1 要求；NVFP4 目标权重不改变此 attention/KV 支持边界。
- 全 CPU 专家层的 SD 不在支持范围；adaptive/observe-only 只支持 legacy 调度，layered 下在 ready 前拒绝。不兼容配置在 ready 前明确拒绝。普通 AR 的既有后端支持不变。
- 不按模型名或路径名限制功能；目标必须有匹配 drafter、支持的 features/输出头和状态/attention 能力。真实配对至少覆盖 Qwen3.6-35B-A3B 与匹配 DFlash。
- 无 GDN 目标不必指定 GDN 预算；如果没有可用的真实匹配权重，可以用合法小模型检查公开能力/资源行为，但不得冒称真实模型性能验收。
- `--speculative-draft-model-path PATH` 指定外部 DFlash；不指定时保留现有 self-SD。`--speculative-num-steps 0` 关闭 SD 并忽略 draft 路径；正值为最多草稿 token 数，不含已有输入，当前上限 8；给出 draft 路径但不指定步数时为 4。
- 固定长度也受单请求剩余长度/容量约束。`--speculative-adaptive-cost` 为 DFlash 选择一整块的 AR/2/4/8，受配置上限裁剪；块内不逐 token 决策。
- 同批请求可以有不同草稿长度。一个请求只剩 1 token 或无起草容量，不应把其他请求统一缩为零草稿。
- `--dflash-compact-kv` 默认开启，仅对 DFlash 生效；`--no-dflash-compact-kv` 为全容量存储 A/B。两者不改变注意力数学。
- compact 的窗口池按并发请求定额：窗口紧张时，新的长 prompt 会等待或分更小的块准入，正在 decode 的请求不受影响，服务不得因此退出。调度相同（如单请求或无窗口压力）时两种存储输出逐 token 一致；调度不同时，切块差异带来的数值差可能使并发输出不同。
- `--dflash-attention-window W` 默认 0；正值只截断原本全注意力层的历史，原生窗口保持原定义，完整本轮 noise block 保留。目标模型始终完整验证。
- compact 与历史 cap 相互独立；关闭 compact 只是多留存储，不能悄悄取消已配置的 cap。
- `--dflash-adaptive-observe-only` 依赖 adaptive：先执行一次真实动作初始化，随后控制器照常计算，但执行固定配置长度及正常逐请求裁剪；初始化单列，正式阶段用于同执行开销对照。
- 新自适应使用近期真实观测，不提供旧原型 `--dflash-recent-acceptance` 开关；过时参数明确报错。
- Replay 开关保持独立。请求 Graph 时不能静默用 eager 代替，真实位置与物理填充位置必须可观测。
- 未使用 DFlash 时，其 compact 默认值不影响 AR/self-SD；显式不兼容的 DFlash 专用参数正常报错。

## 2. 数学、输出与服务行为

沿用现有 OpenAI 兼容接口、温度/top-k/top-p、流式、usage、最大输出、stop、EOS、取消和 cache_group。Completion prompt 为字符串，不支持 token ID 数组。

- 原生窗口紧凑存储与完整存储具有相同数学；近似 cap 只改变 drafter 的历史，不改变目标 attention/GDN/验证。
- cap 的数学：已提交历史结束位置为 H（右开）时，原本全注意力层读取 `[max(0,H-cap),H)`，同一块所有 noise 位置使用这一历史范围；整个 noise block 仍按原来可见性计算。原生窗口层保留原始逐查询窗口定义，不套用此近似规则。绝对位置不因窗口滑动重置。
- 随机 SD 使用真正 proposal 概率完成接受和修正采样，不能直接把目标分布采样当作接受规则。近似 draft 不能改变所声称的目标采样分布。
- 正式历史只保留有效提交内容。拒绝、EOS/stop 后的暂存、取消请求不得污染后续前缀或其他请求。
- 输出 token 数、草稿数、接受数、验证轮数分别定义；接受数不是“最终输出长度减一”在任意 stop 截断后的盲算。
- 相同公共前缀允许多人复用，cache_group 隔离。目标历史与 DFlash 历史共同复用；不能报告完整命中却缺失需要的 draft history。
- 需要的 CPU 冷窗口存在时只恢复所需部分；不存在时按正确 miss/recompute 行为运行，不读已释放 GPU 数据。
- 退 AR 时仍维护 DFlash 上下文，允许后续恢复起草。这种 AR 必须与没有 drafter 的纯 AR 分开报告。
- Greedy 的 Graph/eager 或不同块长差异需量化定位并做质量回归；不能要求不同浮点执行逐字恒等，也不能自动豁免乱码、污染、稳定错误。

## 3. GPU/CPU 内存公开行为

- DFlash 独占权重、持久草稿历史、固定工作区和元数据独立计价，不绑定 GDN 池。
- 明确配置的专家、目标 KV、GDN 容量保持不变；自动容量沿用现有资源优先级。放不下在 ready 前报所需资源，不静默缩池。
- 省下显存不强制填回 GDN；显式容量 A/B 下可以表现为更大的实际空闲显存。
- 原生有限窗口层的 GPU 历史不随目标总 KV 容量无限增长；全注意力层在未额外 cap 时仍按完整历史付费。
- 随长生成推进，逐步释放本请求不再需要的旧前缀窗口保护，并降低实际物理预留。不得在每个请求上永久留“一个旧窗口＋一个新窗口”。
- 别的共享请求、工具锚点、在途复制仍可保护旧窗口；实际占用不保证永远等于并发数×窗口长。不得为省内存提前复用这些位置。
- CPU 冷窗口复用现有 `--prefix-cache-host-gib`、物理计价和淘汰，不新建无预算缓存。相同物理副本共享时不重复算字节。
- `prefix-cache-host-gib=0`、小预算、缓存满、冷窗口被淘汰均是有效配置场景；服务继续正确运行，性能/命中可变化。
- 源数据在异步备份完成前受保护，恢复目的数据在完成前不能用于生成。取消最后一个等待请求不会泄漏或双重归还；空闲时最终报告在途归零。
- 空闲重建后可继续服务和前缀复用；忙时拒绝、非法资源请求不破坏原资源。状态重建和仅专家池重建都要验收。

`/v1/cache/status.geometry.dflash` 应报告 active、weight_bytes、context_bytes（含 full/window 拆分）、metadata_bytes、workspace_bytes、reserved_bytes（上述四项之和），以及窗口总/空闲槽和保护情况。报告中的引用计数与物理槽数不可混淆。CPU 物理占用与 Graph/激活及进程总显存另报，不能只凭 PyTorch 张量量断言没有额外占用。

完整状态容量仍为 `geometry.num_mamba_slots`。0 表示无 GDN 池，重建时不传相应字段。非法容量（如 num_pages=-1）按现有接口返回 HTTP 503、status=rejected 及具体错误，原资源继续可用；只有明确 status=busy 才表示等待空闲重试。

## 4. 统计与可观测开销

`/v1/stats` 至少可区分：

- drafter 类型、配置最大步数、实际每请求草稿长度分布、accepted 和最终 emitted。
- 实际/物理 draft、verify 位置；请求尾部和资源裁剪，不能只报告最大宽度（`clipped_requests` 按 tail/capacity 统计执行了 SD 但草稿被缩短的请求）。verify 位置只计 prefill 波次外的轮次；波次内外的验证轮数与请求数分别报告，退 AR 按原因报告。
- 固定执行、成本选择、初始化、探测、退 AR 次数；观察模式下“建议选择”与“实际执行”分别报告。
- 完整轮累计时间、完整 proposal 时间（包括采样/候选准备）、verify forward 与接受/修正部分；所有计时说明是否重叠，不能无条件相加。
- 控制器 CPU 决策时间、有效成本样本数、丢样数、初始化与探测实耗；空闲时已结束样本最终可见。
- 上两项（计时与控制器计数）只在 `--speculative-adaptive-cost`（含观察模式）下提供；固定模式不采集每轮 GPU 计时，不增加额外开销。

原 `dflash_block_gpu_ms` 若改为完整 proposal 口径，必须显式标明 timing_scope，不能直接与旧 forward-only 数字相比。完整轮时间也不等于端到端请求延迟，后者另测。

启用 `--enable-cache-report` 时，命中数大于零才有 `usage.prompt_tokens_details.cached_tokens`；缺省表示零命中。

## 5. 独立功能矩阵

只测试项目真实支持的公开输入。CPU-only 资源/算术测试、小模型独立数学、真实 GPU 服务三类结果分别列出，不能相互替代。

| 维度 | 必测内容 |
| --- | --- |
| 格式/模型 | 匹配 Qwen3.6 BF16 与 NVFP4；无 GDN 的公开能力案例；原有 Qwen3/Qwen3.6 self-SD 和 AR 回归 |
| 执行 | 固定 N2/4/8、adaptive、observe-only；Graph/eager 数值；Replay 开/关；legacy 与 layered-pipeline（outwave/inwave/all）；offload 与 hybrid |
| 批次 | C1/C4/C16 及中间/自然尾批；实际长度不一、单请求不能起草、真实/物理位置不等；有填充的较大 N8 图 |
| 输入 | 短于窗口、跨窗口边界、远长于窗口、分块 prefill、长生成多次滑出旧公共窗口 |
| 共享 | 同前缀多请求、不同分叉/继续长度、组隔离、一个结束另一个继续、旧分支再次访问 |
| 冷缓存 | host=0/小/满、热命中/冷恢复/已淘汰重算、复制期间出现共享分叉、多个复制计划重叠 |
| 终止 | 拒绝、max_tokens、EOS、stop、取消在 draft/verify/等待恢复、取消最后等待者后服务空闲 |
| 生命周期 | 槽多轮循环使用、内存压力尾批、重建目标 KV/GDN/专家池、无效重建、随后继续命中/生成 |

每种格式需有代表性端到端组合和风险交叉，不要求盲目运行所有维度的笛卡尔积。C16 长输入必须先选择能容纳的显式容量，并保持该对照组资源一致；不能用不可容纳输入制造 OOM 后宣称失败，或用缩短输入掩盖本应支持的路径失败。

独立数学参考验证原生紧凑与完整存储相同、cap 后的明确注意力定义、位置连续性、随机接受/修正分布。先固定数值容差和统计检验方法；输出数列/短句冒烟不能替代完整 coding/research 任务质量。

若共享窗口分配组件也用于已支持的其他 SWA 模型，需补相应公开缓存行为回归；这是共享组件回归，不宣称本轮新增这些模型的 DFlash 支持。

## 6. 性能与资源验收

先冻结请求、硬件/格式、资源、种子、最大输出与计分区间。记录 GPU/CPU 外部负载，不停止他人任务。使用同一总预算和同一显式专家/目标 KV/GDN 容量；纯 AR 没有 DFlash 时省下的内存单列，不能偷偷换成更多专家来混作算法 A/B。

必要对照：纯 AR；DFlash 固定 N2/N4/N8；adaptive；固定 N8 对 adaptive+observe-only 的 A–B–A；完整存储/原生 compact/额外 cap。各子实验只改其研究变量，不强求运行一个巨大的全组合矩阵。

1. **同执行开销**：固定与 observe-only 的实际草稿/验证/接受/输出及形状匹配后再算，目标额外时间≤2%，CPU 决策 p95≤50 μs。按每种计分负载完成匹配的初始化/热身，报告实际比较了有效候选费用的次数和有效候选数量；不能测一个始终因未知数据/探测建议早退、或只给当前动作自身打分的控制器。控制组漂移超过效应则结论未确定，不得声称达标。
2. **窗口收益**：同样数学/资源下原生 compact 的 decode 时间相对完整存储目标不退化超过2%；证明目标容量继续增长时有限窗口部分保持封顶。长生成后省下的是实际分配，不只是统计“可淘汰”。复制/释放开销应随本次位置量变化，而不是随池总容量扫描。
3. **策略收益**：短/长、C1/C4/C16、混合请求与负载变好/变差。稳定低接受负载计入探测后，相对保持 DFlash 上下文的 AR，稳态时间损失目标≤5%；初始化单列。高接受负载必须能恢复有效 SD。与固定 N2/4/8 的差异逐项报告，不承诺压过事后最优。
4. **真实场景**：至少一组 coding/research 多轮长上下文轨迹，含工具续写/共享前缀；合成重复 filler 仅作容量压力，不作为质量结论。
5. **完整账本**：TTFT、decode ms/token/吞吐、端到端时间、每轮 draft/verify/有效产出、专家 H2D（如开启计量）、GPU 进程峰值、CPU 物理占用、前缀命中/恢复量、Graph 真/物理位置。带 profiler 的诊断运行与正式计时分开。

上述阈值为交付目标，不是已测事实。若达不到，报告失败原因并继续修复/讨论，不静默放宽门槛、关闭功能或更换基线。不得靠永远 AR 来通过自适应验收。最终只有独立矩阵、性能与内存验收、生产清理和代码量核算都完成，才称为可交付。

## 公开模型计算入口

入口模块：`freetoken.speculative.dflash_model`。

```python
read_dflash_config(path: str | Path) -> SimpleNamespace
DFlashModel(path: str | Path, *, dtype: torch.dtype, device: torch.device | str)
model.project_context(features, positions, store) -> None
model.forward(noise_embeddings, positions, attend) -> Tensor
```

`path` 是本地目录，包含公开 DFlash 格式的 `config.json` 与 safetensors 权重。
配置读取不分配张量；模型加载支持该检查点的 SiLU、默认全头 RoPE 和配置指定的注意力类型，
不根据目录名判断模型。缺少 mask token、无法支持的激活／RoPE／注意力类型或不匹配的权重应报错。
数学参考可使用同一格式、较小维度的 FP32 检查点；实际服务使用 BF16。

公开属性：`config`、`target_layer_ids`、`mask_token_id`、`hidden_size`、`block_size`、
`attention_modes`、`input_embedding_scale`、`output_multiplier`、`final_logit_softcapping`、
`weight_bytes`。`weight_bytes` 是实际参数字节数，不包含运行时缓存与工作区。
`attention_modes[i]` 为 `(causal, window_left)`：窗口左侧最多可见的位置数，`-1` 表示无限制。
本轮真实检查点为前五层 `(True, 4095)`、末层 `(False, -1)`。

输入输出均按 token 展平，批次内不同请求的边界由回调掌握：

- `features`: `[T, len(target_layer_ids) * hidden_size]`。按指定层号顺序，拼接目标模型对应层
  **输出**，T 个非空有效输入位置。`positions`: `[T]`，各请求的绝对位置。
- `project_context` 对每层调用 `store(layer, k, v)`，k/v 为 `[T, num_key_value_heads, head_dim]`。
  k 已经完成逐头归一化和 RoPE；v 只经过线性投影。回调负责保存这些目标上下文。
- `noise_embeddings`: `[Q, hidden_size]`，Q 个非空待起草位置，通常每请求为已有输入 token
  的 embedding 加若干 mask embedding。`positions`: `[Q]`。输入 embedding 的可选比例由调用者应用。
- `forward` 每层调用 `attend(layer, q, k, v)`，q 为 `[Q, num_attention_heads, head_dim]`，
  k/v 为 `[Q, num_key_value_heads, head_dim]`。回调将这些临时位置与该请求已有上下文做 attention，
  应用 `head_dim**-0.5`、GQA 分组及该层的因果／窗口约束，返回 `[Q, num_attention_heads, head_dim]`。
- `forward` 返回最终归一化的 `[Q, hidden_size]`，不包含输出词表投影、采样或接受判断。
  调用者使用目标输出层，并应用 `output_multiplier` 和配置中的可选 logit softcap。
- 两个方法不改写输入张量、位置或模型权重；只有回调可以写入其拥有的缓存。
  不同请求不得在回调中互相读取历史。临时 noise K/V 不等同于可复用的目标上下文 K/V。

独立数学参考如下；所有线性层使用检查点对应权重和配置中的 bias，RMSNorm 的 eps 来自配置。

1. 定义 `RMS(x, w) = w * cast(x_fp32 / sqrt(mean(x_fp32²) + eps), input_dtype)`。
   context 投影为 `C = RMS(features @ fc.weight.T, hidden_norm.weight)`，所有层共享同一个 C。
2. 第 i 层 context：`Kc = RoPE(RMS(reshape(C @ k_proj.T), k_norm.weight), positions)`，
   `Vc = reshape(C @ v_proj.T)`。**C 不经过该层 input_layernorm。**
3. 第 i 层 noise：令输入为 H，`X = RMS(H, input_layernorm.weight)`；从 X 计算 Q、Kn、Vn。
   Q 和 Kn 分别经过 q_norm/k_norm 与 RoPE；Vn 不经过逐头归一化或 RoPE。
4. `A = attend(i, Q, Kn, Vn)`，`H1 = H + reshape(A) @ o_proj.T`；
   `Y = RMS(H1, post_attention_layernorm.weight)`，
   `H2 = H1 + (SiLU(Y @ gate_proj.T) * (Y @ up_proj.T)) @ down_proj.T`。
   H2 成为下一层输入。最终输出是 `RMS(H_final, norm.weight)`。
5. RoPE 频率为 `theta**(-2*j/head_dim)`，按绝对位置求相位；cos/sin 转为输入 dtype，
   使用前半／后半配对的旋转：`x*cos + concat(-x_second_half, x_first_half)*sin`。

比较 BF16 结果应报告数值误差，不能把不同 GEMM／attention 计算顺序的最后几位差异直接认定为错误。
FP32 小规模独立参考、真实 BF16 模型和 Graph 回放应分别报告。
