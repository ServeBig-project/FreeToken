# GDN ReplaySSM：独立黑盒公开契约

日期：2026-09-29。状态：计划实现的公开行为；不代表功能已经存在。

本文可交给独立测试agent。测试agent不得读取实现协议、生产源码、生产diff、内部测试或实现笔记；实现agent不得代写测试。可以读取本文、公开模型配置和论文数学定义，调用公开CLI／HTTP／已公开契约的实际生产算子接口，并分析这些接口的输出。

## 1. 功能与基线

本功能优化含GDN层模型的状态保存与计算，适用于普通完整模型生成（AR）以及现有self-speculative decoding（SD）。开启后须实际覆盖target AR、draft、verify；不能只优化verify或因启用优化而暗中关闭SD、Graph、前缀缓存。

生产基线为`41b50c5`。主要公开模型路径：

- `/data1/lmcache_kv/models/Qwen3.6-35B-A3B`
- `/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B`

首轮BF16、单GPU，C1／C4／C16及自然尾批。Qwen3.6是主要功能模型，Qwen3用于无GDN组件时的回归。支持依据实际组件能力，不能依据模型目录名字；可在隔离目录改变公开路径名称检查不存在名字白名单，不复制大权重文件。

## 2. 配置接口

| CLI | 默认 | 公开含义 |
| --- | --- | --- |
| `--enable-gdn-replayssm` | 关闭 | 对实际GDN组件启用该方法；一个开关涵盖AR／draft／verify |
| `--gdn-replay-buffer-len R` | 32 | 每个活跃请求的记录容量；R为至少4的2次幂，且不得小于最大草稿长度加1 |
| `--gdn-state-budget-bytes M` | 未指定 | 启动时GDN状态相关存储总字节预算；优化开启／关闭都可指定 |

其他参数继续使用现有接口，包括：`--speculative-num-steps`、`--speculative-draft-experts`、`--speculative-draft-residency`、`--speculative-draft-load-missing`、`--speculative-adaptive-cost`、`--speculative-verify-prefetch`、Graph与cache配置。

重要组合：

- 优化开＋SD关：执行优化后的target AR。
- 优化开＋SD开：实际draft和verify都执行优化路径。
- 优化关：保留原行为，不分配新方法的记录／卷积工作缓冲。
- 无GDN的模型：该开关不改变模型计算，公开报告未激活，新增GDN存储为0。
- 存在GDN但实际参数不受kernel支持：明确配置错误，不能静默回退AR或旧后端并声称优化已生效。
- 配置错误包括非法R、R装不下最大窗口、非正M、M不足以启动所需状态。错误应在服务正常就绪前可观测。

naive默认状态容量可能不足以容纳新方法的固定存储；足够显式M时应支持。不要把失败于不足预算解读为某个模型名字不支持。

## 3. HTTP和资源语义

沿用现有OpenAI兼容请求、流式输出、stop字符串／数组、EOS、最大输出长度、cache_group、取消以及公开资源API。

- `GET /health`：正常就绪后才能开始测试。
- `GET /v1/models`：客户端使用实际返回的模型ID，不写死服务别名。
- `POST /v1/completions`／已有chat接口：完成状态、usage、终止原因和cache-report语义不变。
- `GET /v1/stats`：保留原SD和Graph统计，新增本节规定的最少方法证据。
- `GET /v1/cache/status`：报告实际资源几何和字节数。
- `POST /v1/cache/rebuild`：仍是idle-only；没有单独的cache-clear接口。现有同尺寸重建可用于清缓存。

`geometry.num_mamba_slots`和重建输入`num_mamba_slots`继续表示真实可用完整状态槽数，不能改成等价预算单位。开启优化后，为保持总字节预算，实际槽数可以变化。

同一启动预算下，应比较总字节和真实池容量，不能要求开关两侧`num_mamba_slots`相等。显式重建仍按目标实际槽数定价并接受总缓存预算校验；非法重建应保留旧服务可用。同尺寸重建保持预算口径，改变状态容量时如实更新公开预算和实际字节。

### 3.1 静态资源字段

建议在`/v1/cache/status`的`geometry.gdn_replayssm`报告以下字段；若复用已有等价字段，实施前在本文列明唯一映射，不提供重复别名：

```text
active                  是否实际激活，不仅是CLI请求开启
buffer_len              配置R
request_capacity        可容纳的活跃请求数，不是公共前缀数
state_budget_bytes      当前状态资源分配的预算口径
checkpoint_bytes        包括padding的大状态存储字节
record_bytes            更新记录存储字节
conv_workspace_bytes    原大状态之外的短卷积工作存储
metadata_bytes          该方法专用GPU元数据字节
state_workspace_bytes   其他确有用途的专用固定状态工作区
reserved_bytes          上述各存储项之和
```

不允许把专用状态缓冲藏进不明的“其他内存”并宣称满足预算。权重、专家池、KV以及Graph共有工作区与上述口径分开。

关闭路径的新增记录、额外卷积及专用元数据为0；无GDN时全部GDN存储为0。最小padding和对齐按实际字节计入，不能只报有效载荷。

### 3.2 动态方法证据

建议`/v1/stats.gdn_replayssm`至少提供：

```text
active
ar_tokens, draft_tokens, verify_tokens
flushes, flushed_records, snapshot_exports
```

token计数按真实逻辑位置计一次，不按GDN层数／head数或Graph填充重复计数；AR项只计target decode，不混入prefill。flushes和snapshot_exports按请求级操作计数。已有SD长度直方图、接受数、Graph回放形状继续复用。

统计允许按现有生成回报时机更新，不承诺任意时刻的强同步读数；应注明采样时机。不得仅因idle后的`mamba.used_slots`非零判泄漏，公共缓存可能持有状态，该字段也可能是上一批的快照。

Graph公开指标中`query_tokens`是实际查询数，`physical_query_tokens`是补齐后的执行大小。C16／N8的容量证据必须来自**B16且实际query_tokens=144**，不能来自物理144或小尾批的N8。

## 4. 公开语义不变量

1. 请求从共同的正确历史开始。少专家draft不能改变完整target的历史或污染其他请求。
2. verify仍由完整target重新计算候选输入；不能以draft的状态结果替代验证。
3. 完整模型生成、起草、验证、终止后继续、命中前缀后的继续，应在相同输入历史下满足相同GDN数学语义，允许按精度契约量化的浮点差异。
4. 已保留前缀的状态必须对应那个前缀的确切位置；不能保存更晚状态并标为较早位置。
5. 公共前缀在被其他请求命中后仍可被第三个请求复用。cache_group隔离不变。
6. 生成长度、EOS、stop和取消以实际公共请求语义为准；不能多发终止后的内容，也不能因长度控制而提交错误的target位置。
7. 请求结束、取消和重建后，后续请求仍可运行；旧请求内容不能通过复用资源污染新请求。
8. 固定预算下空间不足可以沿用明确的资源行为，但不能暗改专家、KV、并发或忽略配置。新方法能通过正常处理继续时，不能沿用旧大状态需求人为限制起草深度。

## 5. 数学验收契约

仅比较生成文本不足以排查状态错误。应使用实际生产计算入口的公开输入／输出契约做独立数值验收，禁止读取其实现。

实现agent在该入口形成后，须在本文补充：实际导入路径与签名、tensor形状／dtype、哪些输入允许被修改、返回值、导出状态的确切位置语义。该入口必须是模型实际调用的生产入口；不新增只供测试的wrapper或调试HTTP接口。

公开GDN参考数学（每个value head，S形状为V×K）：

```text
decay = exp(g)
u = beta * (v - decay * S_prev @ k)
S_next = decay * S_prev + outer(u, k)
y = S_next @ q
```

q/k归一化、q缩放、门控转换、卷积和dtype规则由生产入口的公开说明给定，须与模型已有定义一致。独立agent可依据此数学写自己的高精度参考；不得把待测实现本身或其内部测试当oracle。

至少验证：

- 连续AR、起草后验证、不同接受位置导出的状态及输出。
- 用不同输入模拟draft／target计算，确认target结果不依赖被放弃的draft值。
- 多次合并历史前后、空历史和长历史下，同一输入轨迹的结果。
- 请求行排列变化、混合实际长度、合法padding及不同请求之间的隔离。
- 单步、小窗口、最大窗口；连续轨迹足够长，确实经过不止一次flush。

容差必须依据公开dtype和数学，在正式验收前确定并记录。报告最大／相对／RMS误差及误差随步数变化，不允许看到失败后只调大阈值。非有限值、位置错一、串请求或使用错误分支不是可豁免的浮点差异。

若入口契约尚未公布，暂停该部分测试编写并向协调者索取契约，不读取实现源码补齐。

### 5.1 已公布的生产入口

```python
from freetoken.kernel.triton.gdn_replay import gdn_replay, gdn_replay_fold, gdn_replay_conv
```

记号：H个key头、HV个value头（HV整除H，value头j使用key头`j // (HV/H)`），头维K、V；R为记录环长度（2的幂）；m为请求记录行；slot为完整状态槽。每个value头的状态S是V×K矩阵，`y = S @ q`，`state[slot, j]`按行v、列k存放。

记录按**绝对输入位置**p存放在环中：`u[m, j, p & (R-1), :]`（长度V）、`k[m, h, p & (R-1), :]`（归一化后的key，长度K）、`g[m, j, p & (R-1)]`（fp32）。位置p指已消费输入的序号：p之前的输入已经进入状态。

**`gdn_replay(qkv, a, b, A_log, dt_bias, state, u, k, g, cu_seqlens, slots, cursors, scale) -> out`**

| 参数 | 形状／dtype | 说明 |
| --- | --- | --- |
| qkv | [tokens, 2·H·K + HV·V]，激活dtype | 卷积＋silu之后的每个token依次为q(H×K)、k(H×K)、v(HV×V)；最后一维连续，行stride可更大 |
| a, b | [tokens, HV]，激活dtype | 原始门控输入；最后一维连续 |
| A_log, dt_bias | [HV] fp32 | |
| state | [slots, HV, V, K] 连续，默认fp32 | 只读 |
| u / k / g | [rows, HV, R, V] / [rows, H, R, K] / [rows, HV, R]；u、k为激活dtype，g为fp32，连续 | 记录 |
| cu_seqlens | [n+1] int32 | 序列i的输入为`[cu[i], cu[i+1])`，长度T |
| slots | [n] int32 | 序列i的完整状态槽 |
| cursors | [n, 3] int32 | `(m, b, p)`：记录行、完整状态所在位置b、本次第一个输入的位置p |

语义（序列i）：起始状态＝`state[slot]`依次叠加位置`[b, p)`的记录；随后对T个输入逐个执行
`q̂ = q/√(Σq²+1e-6)·scale`，`k̂ = k/√(Σk²+1e-6)`，`g = -exp(A_log)·softplus(a+dt_bias)`（x>20时softplus(x)=x），`β = sigmoid(b)`，`S ← e^g·S`，`u = β(v − S k̂)`，`S ← S + u k̂ᵀ`，`y = S q̂`。
返回`out[tokens, HV, V]`（qkv的dtype），并写入位置`[p, p+T)`的记录`(u, k̂, g)`。不修改state、其他位置或其他行的记录。要求`0 ≤ p−b`且`p+T−b ≤ R`；同一次调用中各序列的记录行互不相同。`m < 0`表示padding：输出行为0，不写任何记录。

**`gdn_replay_fold(state, u, k, g, plan) -> None`**

state为[L, slots, HV, V, K]，u/k/g为带层维的[L, rows, …]；`plan`为[n, 5] int32，每行`(源slot, 目标slot, m, b, count)`，`count ≤ R`。对每一层写入`state[l, 目标] =` 源状态（位于位置b）依次叠加位置`[b, b+count)`的记录后的完整状态，即长度为`b+count`的前缀对应的状态。目标可以等于源（原地合并）；记录不被修改；各行的目标互不相同且不等于其他行的源。

**`gdn_replay_conv(x, weight, window, cu_seqlens, cursors) -> out`**

x为[tokens, D]的原始卷积输入（激活dtype），weight为[D, KW]，window为[rows, W, D]，按绝对位置存原始卷积输入：`window[m, q % W]`为位置q的输入。调用前须保证位置`[p−KW+1, p)`的输入已在窗口中。输出`out_t = silu(Σ_j weight[:, j]·x(p+t−KW+1+j))`（fp32累加，输出x的dtype），其中`q ≥ p`的输入取自x，`q < p`的取自窗口；之后`window[m, (p+t) % W] = x_t`。要求`W ≥ KW−1+T`；`m < 0`时输出0、不写窗口。cursors的b列不使用。

精度：所有乘加在fp32中完成（矩阵乘为IEEE fp32）；u、k按激活dtype存储，g为fp32，完整状态保持其存储dtype。单次调用内部的逐token递推使用未舍入的fp32 u、k，因此与普通逐token递推相比，只有此前调用写入的u、k记录存在舍入。

模型中的用法：每个GDN层先做卷积（target AR沿用原有逐token卷积并原地更新卷积状态；draft和verify使用`gdn_replay_conv`），再以该层的state／记录视图调用`gdn_replay`；`gdn_replay_fold`用于合并记录和导出完整状态。`state`在模型中是状态池的一层视图，K=V时与上表布局一致。

## 6. 最小服务验收集合

不做所有参数的全笛卡尔积。每个case必须说明会检测什么实际失败，失败后改变什么交付结论。

| 范围 | 需要真实触发的行为 |
| --- | --- |
| A/B关闭回归 | 两个本地模型沿用原接口、原计算路径；新方法存储为0 |
| GDN AR | 优化开启、SD关闭；Graph运行，标准重叠调度下连续请求和终止可用 |
| 固定SD | C1／C4／C16，N1／N4／N8选取代表性组合，k3为主；Graph与必要eager对照 |
| 自适应SD | 实际深度变化和AR回退；不能仅配置了开关而未执行对应路径 |
| 混合／尾批 | 不同输出上限、请求陆续结束、零草稿请求与有草稿请求同批时可继续 |
| 长度／flush | R16／R32至少两种配置，长生成经历多次真实flush，容量不膨胀 |
| 公共前缀 | 冷请求、重复／生成后续请求、共享前缀并发、不同cache_group控制组 |
| 终止 | stop字符串、stop数组、EOS、输出上限；随后命中该历史继续 |
| 取消 | prefill或decode实际在途时取消，后续请求与资源复用仍正常 |
| 分块／重建 | 长输入跨真实prefill分块；空闲重建后冷启动、Graph继续和原容量恢复 |
| 已有其他路径 | 有足够预算的naive、代表性layered AR；不扩大为新调度算法 |
| 工具位置 | 若模型公开输出可触发工具标记，验证跨SD窗口位置的复用；没有触发就明确未覆盖 |

在同GPU、同输入和同字节预算的C16短请求中，关闭自适应、给予足够KV及允许补缺加载，配置N8的新方法应能实际运行满批N8。不能用短尾批、实际N1或纯AR替代该验收。

覆盖未发生时报告`uncovered`及原因，不能把汇总计数拼成某个具体分支已运行的证据。正常完成、预算变化、Graph运行和质量观察分别记录。

若自然生成无法稳定触发工具标记，可通过已公开且确有生产调用的状态导出接口验证窗口内合法位置；HTTP工具路径未触发仍单独标未覆盖，不能以算子检查冒充端到端覆盖。

## 7. 性能协议

- 主硬件：经协调者确认可用的4090；测试过程中不停止外部任务。前后使用同一GPU，记录共享负载。
- 主资源：Qwen3.6 BF16、专家池9 GiB、相同KV、相同最大并发及状态字节预算，Graph开启；k3缓存路由＋补缺，预取关闭。
- 主C16短协议可复用已有冻结请求：16-token预热，随后两轮每请求64-token，计分2048token。使用真正相同的公开请求体，不强行替换响应内容。
- 先关闭自适应，分别比较同实际N4，以及同预算配置N8。另测旧AR／新AR。之后才比较自适应，不能把三种效果混为一谈。
- 状态预算按实际字节比较。旧96可用槽的测试配置不能在新方法中简单要求也保留96个完整槽。
- 记录吞吐、TTFT／decode延迟、实际draft长度和接受长度、实际Graph形状、峰值显存及各存储项。
- 分段成本至少覆盖AR、单步draft、整轮verify、flush、提交和公共快照导出；注明计时范围和是否存在包含关系，不能重复求和。可复用既有统计／公开profile产物，不必为每项增加一个服务开关。
- 需要一组包含前缀命中的代表性长输入／agent请求，观察导出和复用成本；短C16不是完整agent trace。
- 短测波动如实报告。实现不提速也应输出结果，不要求达到论文倍率，不通过悄悄换资源制造收益。

## 8. 已知基线与判定

`41b50c5`之前已记录4组stop／EOS／取消后“命中前缀 vs 从头计算”的全文差异；前后旧版本已经逐字复现。更早固定质量题也存在AR／SD差异。

保持以下分类：

- **失败**：公开行为、数学／状态位置、资源或已规定数值容差不满足要求。
- **待调查**：新增文本／质量差异，尚无证据判断来源；保留完整输入输出，进行有目的的旧版／新版本对照。
- **未覆盖**：请求完成了，但需要验证的实际路径没有发生。
- **通过**：明确约定的检查得到满足，不代表所有质量问题均已解决。

不要将历史差异改判为无害，也不要在没有对照的情况下把相同旧问题归因于新优化。本轮性能campaign不以文本逐字一致为门槛，但状态正确性不是可跳过的项目。

## 9. 交付给协调者

提供可公开复现的启动配置、请求、结果、失败条件及状态；测试代码独立commit并统计增加／删除／净增。复用自己已有客户端且无新代码时，明确报告新增0行，不为凑commit复制一套测试。

原始证据保留机器可读格式；人工总结只列结论、关键数据、未解决项和入口，不堆叠全日志。实现agent只接收公开失败条件及修复后的复验结果，不接收测试源码。
