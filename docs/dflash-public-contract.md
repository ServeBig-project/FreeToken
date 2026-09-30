# DFlash 公共验收契约

本轮基线：`54b8ab1`。此文只描述公开输入、输出、资源和性能行为，供独立黑盒验收使用。

## 范围与开关

- 原有 Qwen3-30B-A3B 与 Qwen3.6-35B-A3B 的少专家 SD 保留。
- 新增 `--speculative-draft-model-path PATH`：使用指定 DFlash 草稿模型；不指定时沿用少专家 SD。目录名称不决定能力。
- 首轮匹配的公开权重：`z-lab/Qwen3.6-35B-A3B-DFlash`，目标为 Qwen3.6-35B-A3B，BF16、单 GPU、legacy 调度；本轮验收使用 offload 专家后端。沿用原 SD 的限制，CPU／hybrid 专家执行在启动时明确拒绝；hybrid AR 仅作为性能对照。
- `--speculative-num-steps` 始终表示最多起草的 token 数，不含验证使用的已有输入；0 关闭 SD。首轮上限仍为8。
- 固定模式沿用指定长度，在请求剩余长度、缓存容量限制下缩短。`--speculative-adaptive-cost` 开启时，DFlash 在一轮开始前按成本选择2／4／8个草稿 token（受配置上限和实际容量约束），或使用普通生成；不逐 token 重做整块起草。
- 原有少专家模式继续使用原来的逐步自适应。
- DFlash 自己的草稿不使用目标专家数、缓存路由或专家补缺参数。显式不兼容配置应在启动正常就绪前报错，不静默忽略或回退。
- 目标侧 Replay 开关保持独立；Graph 开启时覆盖C1、C4、C16及自然尾批。不得静默以 eager 代替请求的 Graph 功能。

## 服务行为

沿用现有 OpenAI 兼容接口、温度／top-k／top-p、流式响应、usage、最大输出长度、stop、EOS、取消和 cache_group。
Completion 的 prompt 使用文本字符串；本服务明确不支持 token ID 数组。

- 每轮候选经完整目标模型验证；随机采样必须使用对应 proposal 概率完成接受和修正采样。
- 正式历史只包含实际保留的输入。被拒绝、stop 后、取消后的暂存内容不得污染后续请求。
- 目标历史和 DFlash 历史共同参与前缀复用；命中已有前缀后，不因缺失草稿历史而重跑完整目标前缀。
- 同一公共前缀可被多个请求复用，cache_group 之间隔离。
- 冷输入、命中前缀、分块长输入、不同长度混合批次、请求终止后重新使用资源均应可运行。
- 空闲缓存重建后继续服务；忙时拒绝重建、非法重建保留旧资源的语义不变。
- 重建请求只指定模型实际存在的池。状态中的 `num_mamba_slots=0` 表示没有GDN池，此时省略该重建字段；503响应也可能是参数／资源错误，只有明确的 `status="busy"` 才表示忙时可重试。
- 非法容量（如 `num_pages=-1`）返回 HTTP 503、`status="rejected"` 和具体错误；旧资源保持有效，服务继续。
- Greedy 文本差异需报告，不自动等同于状态错误或自动豁免；数值验证与质量回归分别报告。

## 资源与可观测性

- 本轮 DFlash 权重与持久上下文缓存从原 GDN 状态预算中分配；保持实验的专家池与目标 KV 容量，总预算不增加。完整状态槽数可以下降。
- 不为 DFlash 起草申请目标模型的临时递推状态；目标验证仍按所选状态表示获得所需资源。
- 权重、持久草稿缓存、工作区和 Graph 的实际占用应在报告中分开说明；进程总 GPU 占用和负载峰值是验收依据，不能仅凭 PyTorch 分配量断言没有额外占用。
- `/v1/stats` 保留 SD 的草稿数、接受数、验证轮数、草稿长度分布以及 Graph 实际／物理输入数量。DFlash 增加可识别的起草类型及起草成本观察。
- `/v1/cache/status.geometry.dflash` 报告 `active`、`weight_bytes`、`context_bytes`、`metadata_bytes` 和三项之和 `reserved_bytes`；完整状态容量为 `geometry.num_mamba_slots`。Graph 可执行对象的占用另以进程总 GPU 显存观测，不能混作上述张量字节。
- `/v1/stats.speculative.dflash_block_gpu_ms` 是草稿logits准备与模型计算的累计CUDA计时间隔，不含后续概率过滤、采样和候选写入；`dflash_block_samples` 按 `"batch_size:实际最大草稿长度"` 计数。它不是单个草稿 token 的时间。`dflash_block_choices[0]` 记录选择 AR 的次数，其余位置记录选择对应整块长度的次数。
- 启用 `--enable-cache-report` 时，命中数大于零才返回 `usage.prompt_tokens_details.cached_tokens`；该字段缺省表示零命中。

## 独立验收与对照

- 测试作者不得读取生产实现、实现 diff、内部测试或实现笔记；可依据此契约、公开服务行为与上游数学独立构造参考。
- 第一阶段：现有少专家 SD 在重构前后的公开回归，覆盖两种目标模型、固定／自适应、Graph／尾批，以及缓存和终止行为。
- DFlash 阶段：增加模型算子的独立数值参考、固定／按块自适应、Replay开关、前缀和重建生命周期。
- 性能对照使用相同请求与总显存预算，至少比较 AR、少专家 SD 和 DFlash；分别给出起草／验证成本、有效输出、加载量、显存峰值。
- 不停止外部任务。GPU和端口由协调者分配，未分配前只编写测试或做 CPU 公开接口检查。

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
