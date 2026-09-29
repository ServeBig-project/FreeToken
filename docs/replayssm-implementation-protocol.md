# FreeToken GDN ReplaySSM 实现协议

日期：2026-09-29。状态：**交给实现 agent 的设计与交付协议，功能尚未实施**。

本文面向实现 agent 和协调者，包含内部实现约束，**不得交给独立黑盒测试 agent**。测试 agent 只读取 [公开验收契约](replayssm-public-contract.md)、公开模型资料、论文数学定义和公开接口结果。实现 agent 可以读取公开契约，但不得读取、编写或修改测试源码。

## 1. 目标、基线与优先级

在现有公共SD组件上接入GDN ReplaySSM：**target AR、少专家draft、target verify全部使用checkpoint＋小记录；draft与verify按执行顺序复用同一段记录尾部。** 不再为draft恢复一份完整工作矩阵，也不为verify每个位置分配完整状态。

- 实施代码仓库：本工作树 `../`，绝对路径 `/home/nengneng/AIPrometheus/servebig/servebig-project/.sd-worktrees/freetoken-sd-portability`。
- 生产起点：`41b50c5`，分支 `feat/sd-common-runtime`。该起点已包含按需前缀快照、draft／verify大状态槽的阶段复用。
- 不要误改外层 `FreeToken/` 旧工作树，也不要初始化外层项目的占位 `.git`。开工时先确认当前分支和用户改动；若基线已有后续提交，保留它们，先说明与本协议有关的变化。
- 首轮主要验收：本地Qwen3.6-35B-A3B、BF16、单张4090、C16、最多8步；Qwen3-30B-A3B用于无GDN模型回归。
- 必须保持组件通用性。根据实际GDN组件、张量形状和dtype选择路径，不增加模型名、目录名、checkpoint名白名单。
- 用户已同意一次完整实现，包括Graph、AR／SD切换、公共前缀和终止／取消；不能将“只做verify、draft仍恢复大矩阵”作为最终交付。
- 本协议不要求复现论文的加速倍数。要求实现正确的状态协议、达到同预算容量目标，并报告真实性能。

执行约束优先级：用户最新指令、适用AGENTS.md、本协议、参考仓库。本文不授权扩大模型支持范围、停止其他人的进程、改动harness基线或推送／合并远端。

## 2. 已定设计与本协议采用的落地默认值

### 2.1 已定设计

1. 一个请求的正式记忆由GDN checkpoint、其后的有效target更新记录、相应短卷积历史共同组成。
2. 同一请求先draft后verify；这期间没有另一个计算分支同时修改该请求的记录尾部。沿用现有SD调度限制，不新增同请求draft／verify并行执行。
3. draft只共享已接受历史；其新记录来自少专家计算，不能提交为target结果。verify必须用完整target重新计算、覆盖临时尾部。
4. checkpoint在draft和本轮verify候选计算期间保持不变。flush只处理安全、已确认的target历史。
5. 公共前缀树仍保存完整且不可变的GDN快照，KV仍由原attention缓存管理。公共快照不附带一份记录缓冲。
6. 新记录、卷积缓冲、GPU元数据从同一GDN状态总预算中支出；不能保留原全部大状态槽，再额外加小缓冲而不记账。
7. GPU大块内存在启动或空闲重建时分配；运行时只分配逻辑槽位、更新固定地址中的数据。
8. 保留当前实现作为A/B基线，开关关闭时不执行Replay kernel、不分配Replay缓冲。

### 2.2 落地默认值

以下是本协议为明确实现和实验接口采用的默认值，不是声称已测得的最优参数：

| 参数 | 默认／语义 | 实际用途 |
| --- | --- | --- |
| `--enable-gdn-replayssm` | 默认关闭；开启后同时作用于target AR、draft、verify | 唯一算法A/B开关 |
| `--gdn-replay-buffer-len` | 默认32；每个请求、每个GDN层的物理记录容量R，包含临时尾部 | 实测16／32／64容量取舍，不再额外暴露独立flush阈值 |
| `--gdn-state-budget-bytes` | 可选；启动时GDN相关GPU常驻存储的字节预算，开关两侧都识别 | 同字节预算A/B，并让naive等配置能显式提供足够容量 |

名字统一映射到配置字段 `enable_gdn_replayssm`、`gdn_replay_buffer_len`、`gdn_state_budget_bytes`。不要再加draft-only、verify-only、prefix-off、自动降级、调试后端等开关。

R必须为正的2次幂、至少4，且开启SD时不得小于配置最大草稿长度加1。校验发生在可执行配置形成时；不静默截小最大草稿长度来迁就错误的R。首轮交付至少覆盖现有Graph支持的N=1…8；对现有更宽的eager配置，不得无说明地改变旧路径。

无GDN的模型：开关对计算无作用，Replay状态报告为未激活，新增GDN显存为0。存在GDN但实际dtype／形状无法运行时，由GDN组件明确报错，不能用模型白名单或悄悄改走AR替代。不要借接入改写已有权重量化路径；以投影后实际激活和GDN状态dtype判断兼容性。

## 3. 参考实现与复用边界

只读参考仓库：`/data2/servebig-envs/references/ReplaySSM`，固定提交 `a84849410ab56cc2b23432969eb2ecfc42a13d9c`。

- [作者说明](https://tridao.me/blog/2026/replayssm/)
- [GDN verify／flush kernel及游标操作](https://github.com/Johnny-Liou/ReplaySSM/blob/a84849410ab56cc2b23432969eb2ecfc42a13d9c/vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py)
- `vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/layers/mamba/mamba_utils.py`
- `vllm/model_executor/layers/mamba/ops/replayssm_config.py`

复用GDN更新记录的数学、窗口输出计算、flush和Triton计算主体。保留移植代码的Apache-2.0/SPDX归属，记录参考提交；只带入实际调用的代码，不复制整个vLLM或其平台／配置系统。不得引入vLLM运行时依赖。

不能原封照搬的地方：

- 参考kernel把checkpoint和记录用同一个state索引定位；我们的大状态池包含公共快照，记录区只按活跃请求容量分配，两种索引必须区分。
- 参考配置的普通Replay与spec Replay互斥，且要求关闭GDN前缀状态缓存；我们需要统一表示并支持前缀复用。
- 参考提交时机在下一轮metadata构建；我们SD必须先经过本轮stop／EOS裁剪，再提交正确前缀。
- 参考flush可能与窗口计算融合；不能让draft的临时记录通过这条路径合入共享checkpoint。
- 参考运行参数主要面向较大batch。单token draft／AR可采用同一数学核心的编译特化，但不能假定多token verify的launch参数最适合4090。

旧文档中“target Replay＋独立完整draft工作状态”的建议已被用户否决，不得作为本次实现方案。

## 4. 数学与位置定义

GDN缓存的是更新向量u、归一化key k和log-decay g；参考代码把u称为d。不是token ID，也不是简单缓存原始v。

按单个value head、矩阵布局 `[V,K]` 写出等式；q/k按组件原有规则归一化，q的缩放也保持一致：

```text
a_t = exp(g_t)
u_t = beta_t * (v_t - a_t * S_(t-1) @ k_t)
S_t = a_t * S_(t-1) + u_t @ k_t^T
y_t = S_t @ q_t

S_(b+m) = exp(sum(g_1..g_m)) * S_b
          + sum_i exp(sum(g_(i+1)..g_m)) * u_i @ k_i^T
```

沿用现有门控：`g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)`，以及既有归一化epsilon、scale、dtype语义。不要把不同kernel的矩阵轴仅因为K=V就当作已验证相同；核对实际存储方向。

FP32 checkpoint存储不代表所有中间计算都用FP32。参考kernel会对部分矩阵和向量转BF16；移植时说明实际cast位置与累加精度。不要把存储dtype当作逐位等价的证据，也不要在没有误差记录的情况下进一步降低精度。

**所有位置都指已经消费的输入位置，不是已经采样但尚未送入模型的输出token。** N个草稿对应N+1个target查询。现有SD的`retained[i]`是在stop／EOS处理后确定的保留查询数；必须沿用它与状态位置的关系，不能直接套参考实现中同名`num_accepted`的含义。

每个请求逻辑上需要区分：

| 名称 | 含义 |
| --- | --- |
| checkpoint位置b | 大矩阵覆盖的输入边界 |
| target有效记录长度c | 从b开始，可以作为target历史使用的记录数量 |
| draft工作长度w | 本轮起草临时扩展到的记录数量；起始等于c |
| 本轮起点c0 | verify必须回到的target历史边界 |
| 最终保留量r | host裁剪后本轮真正提交的target查询数 |
| 最早仍需恢复的位置 | AR重叠调度、待处理输出或待发布快照仍可能读取的边界 |

这是语义要求，不要求机械地为每一项都创建字段。能从已有不可变批次信息推导的就推导；不能把GPU已算到的位置和host最终发布的位置混成一个数。

## 5. GPU存储与所有权

### 5.1 大状态池

- 沿用共享大状态池，存活跃请求的checkpoint与公共完整快照。
- 每个活跃请求一份checkpoint，原正式状态槽直接承担这个角色。
- Replay模式下，不为draft申请完整状态槽，不为verify按N+1申请完整状态槽。
- 公共快照仍按实际保存需求领槽，捐献、去重、取消后及时释放；不恢复已经删除的固定双缓冲预留。
- 保留现有padding／naive固定live槽隔离规则。公共缓存持有的槽位始终不可被活跃请求原地修改。

### 5.2 活跃请求记录区

按最大活跃请求容量分配，不按大状态池槽数、公共树节点数或全部历史请求分配。每个实际GDN group的典型布局：

```text
checkpoint: [GDN层数, checkpoint槽数, value_heads, V, K]  FP32
u:          [GDN层数, 请求行数, value_heads, R, V]        BF16
k:          [GDN层数, 请求行数, key_heads, R, K]          BF16
g:          [GDN层数, 请求行数, value_heads, R]           FP32
```

最终stride可按kernel访问调整，但逻辑区分不变。公共快照不携带u/k/g。不要扩大为每个checkpoint槽一份记录，也不要给draft和verify各分配一份完整历史副本。

首选复用参考环形表示：`base`指向逻辑记录0，逻辑位置j映射为`(base+j) & (R-1)`。环形仅解决固定空间的重用，不代表保留无限历史。只保留一种布局，不同时维护环形和线性两套路径。

### 5.3 短卷积

GDN的短卷积也有状态，必须单独处理：

- target当前卷积历史必须在draft期间保持不变。
- draft可以有一份小的可写卷积工作窗口；这不等于额外完整GDN矩阵。
- verify从target的原始窗口开始，处理本轮全部真实输入；提交时选择与r对应的最后`kernel_size-1`个原始卷积输入。
- 需要保留足以导出本轮任一允许提交位置的短历史，并覆盖AR重叠调度仍未处理完的边界。不能只保留最末尾窗口，随后发现stop位置的窗口已经丢失。
- 缓冲按最大并发和最大验证窗口预分配；选择一个清楚的窗口布局，避免同时保留多套重复历史。实际字节全部记入预算。

### 5.4 元数据

checkpoint槽号、请求记录行号、base、有效长度、draft长度、flush范围等用固定地址GPU数组表达。CPU仍负责既有调度和字符串终止判断，但不为此增加逐层／逐token的D2H读长度或全设备同步。

索引数组的内容可以改变，地址不能在Graph回放之间改变。padding不能更新真实请求的记录或游标；若需要专用padding记录行，其字节也必须计入预算。

## 6. 一轮SD的完整协议

以下操作由已有状态组件管理；公共SD继续负责候选token、接受／拒绝和采样，不知道u/k/g内部布局。

### 6.1 起草前

1. 沿用现有KV、输出长度、batch容量及成本策略确定各请求本轮最大N_i。
2. 为最大可能target窗口`T_i=N_i+1`留出记录尾部。不能用Graph补齐后的物理token数充当实际窗口长度。
3. 若target有效历史＋T_i超出R，在起草前flush安全的target历史，腾出足够空间。完整窗口能通过flush放下时，不应仅因记录接近满而把N缩成1。
4. 保存本轮target边界c0，令draft工作长度w=c0；初始化小卷积工作窗口。没有完整GDN矩阵复制或恢复步骤。

### 6.2 draft

每步读取checkpoint＋当前draft有效记录，计算少专家模型的一步GDN输出，将u/k/g追加到临时尾部。所有GDN层使用相同的该步逻辑长度。

- 一整次模型forward完成后，才推进该批请求的工作长度；不能每经过一个GDN层就推进一次。
- target有效边界保持c0，不把临时记录变成target历史。
- 不flush draft历史、不写共享checkpoint。通过起草前的容量保证避免中途需要扩大记录区。
- 自适应提前停止时，保留实际已生成的候选token与概率；未写的记录无效，不靠清零整个缓冲表示有效性。

### 6.3 verify

1. 将计算使用的历史边界恢复到c0，卷积输入历史恢复为target版本；不用清零或搬走draft记录尾部。
2. 使用完整target计算候选窗口。每层重新产生自己的u/k/g，覆盖draft的临时尾部。
3. 读历史时严格掩码到c0；旧draft尾部不能作为target历史参与计算。
4. 使用窗口算法处理GDN输出，不能继续用现有`_run_verify`的逐位置完整状态复制循环。
5. 返回所有真实位置的logits，接受／拒绝仍由现有公共采样流程完成。此时新target记录仍是待提交内容。

### 6.4 提交

在`Scheduler._process_last_data`确定最终保留量后，状态组件提交r个target查询：有效长度变为c0+r，短卷积位置与其一致。被拒、超过stop或超过输出上限的记录留在物理内存中但不再有效。

- r=0时不提交本轮候选；完成／取消路径只释放自己拥有的资源。
- 包含零草稿请求的验证批次中，该请求仍有一个target查询，不是零查询。
- commit完成并建立正确stream顺序后，才允许前缀发布、请求结束捐献和槽位复用。
- 记录长度按实际r推进，不按计划N、接受率统计或补齐行数推进。
- 下一轮draft复用同一请求记录区；不复制整段target记录来创建第二套draft历史。

最小状态转移示意：

```text
TARGET(c)
  -> reserve/flush confirmed history
  -> DRAFT(work=c, target=c)
  -> DRAFT(work=c+actual_N, target=c)
  -> VERIFY(read history through c; overwrite temporary tail)
  -> COMMIT(target=c+r)
  -> TARGET(c+r)
```

## 7. flush与AR／prefill

### 7.1 flush的精确定义

flush q条安全target记录：用这q条更新checkpoint，checkpoint位置增加q，base前移q，剩余有效记录仍表示新checkpoint之后的历史。只在记录空间需要回收时触发；不在每轮结束、每次AR／SD切换或每次起草前无条件flush。

- 没有待处理输出时，可以一次合并全部已确认记录。
- 存在AR重叠或待发布历史位置时，只合并不会越过这些位置的前缀，保留仍需回退／导出的尾部。
- 先完成所有相关GDN层的checkpoint更新，再推进共享元数据。不要在一个layer／head更新base后，让其他layer／head读到新游标和旧checkpoint。
- flush可以融合到合适的计算kernel，也可以先用独立的按层／请求并行kernel实现。不能为了融合牺牲上述边界，也不能用CPU逐请求重算矩阵。
- GDN数学需要覆盖实际可能出现的全部历史长度。参考kernel的BC由其自有阈值推导，移植后必须重新匹配我们R、draft追加长度及AR历史，不能截掉有效记录。

### 7.2 target AR

AR使用同一checkpoint和记录表示，一次追加一个完整target查询的记录。自适应回退AR也走这条路径，不恢复完整矩阵再运行旧GDN decode。

**不能把SD的host延迟commit机械套到普通AR重叠调度。** 现有`overlap_loop`会先发起下一批，再处理上一批输出：

- 下一次AR forward必须读到上次GPU已经算好的target历史；GPU可读位置必须在同一engine stream中及时推进。
- 待处理输出可能触发终止、取消或快照导出；checkpoint不得flush越过仍可能需要恢复的位置。
- 用已有在途批次的不可变边界确定保留范围，保留必要的小记录和卷积尾部。不能通过给AR补一套完整矩阵快照解决，也不能静默全局关闭overlap。
- `Req`字段会在后续forward推进，不能把一个随后被修改的`Req.cached_len`引用当作上一批不可变的提交位置。
- 当前SD调度本身不交错同请求draft／verify；不要扩大为新的流水并行算法。

### 7.3 prefill

保留现有chunked prefill计算：prefill产出完整状态后，以它作为checkpoint，对应记录长度为0。命中公共完整快照时同样从空记录开始。

若一个实际路径要在带有效记录的状态上进入prefill，先把该正确target状态整合成prefill所需输入；不能直接读取落后的checkpoint。不要给本项目没有的路径新增迁移框架。

已有layered／mixed等AR路径按组件接入，部分layer group完成不等于整次forward完成；游标不得每组重复推进。SD保持现有调度限制。

## 8. 公共前缀、工具调用、取消与重建

### 8.1 前缀导出／恢复

- 公共前缀仍是完整GDN状态＋对应卷积历史，与该位置的普通attention KV共同构成可复用前缀。
- 保存时，按指定的已确认位置从checkpoint和有效记录生成完整快照；不能简单`copy_from`那块可能落后的checkpoint。
- 活跃请求继续运行时，导出不能改变它的逻辑记忆；请求结束时，可以将已恢复到正确位置的自身槽直接捐献，避免多一次无意义复制。
- 只在已有实际保存事件发生时导出；flush不自动新增公共节点。不缓存每个token，也不让历史trace无限增长。
- 命中完整快照：复制到请求自己的checkpoint槽，重置记录长度和相关游标，恢复正确短卷积历史；公共源槽保持不可变。
- 新请求复用旧请求的table行／记录行时，必须清有效元数据；无需为清理而写零整个记录数组。
- 两个公共快照之间缺少历史记录时，维持从较早可用快照补算的语义；不能假装较晚快照可逆，也不新增CPU历史trace存储。

### 8.2 工具调用位置

保持现有保存策略，不扩大为密集快照。已知旧代码用`cached_len == anchor`触发，SD跨过anchor可能漏存；该缺口此前只是代码推导，不能当作已验证正确。

Replay接入必须保证已有能触发的工具保存事件导出完整逻辑状态。对于本轮最终接受窗口内新发现的anchor，应在记录／卷积历史仍可访问、且下一轮flush前处理到该精确边界；不得写错位置。把它作为接入边界补齐并单列验收结果，不借此重写工具协议。若需改变对外保存策略，先向用户说明。

### 8.3 取消与错误

取消不能把未接受的draft／verify尾部发布到公共缓存。沿用现有在途请求回收时机，在GPU使用结束后回收checkpoint、可选快照和记录行；不能在另一个stream仍读旧行时把行重发给新请求。

OOM、非法配置或重建拒绝不能被改判成成功AR。继续沿用已有公开错误机制，不增加通用重试／回滚框架。

### 8.4 空闲重建

继续支持已有idle-only资源重建。先验证目标几何和字节预算，再销毁Graph、重建相关池、清无效缓存／游标、重新捕获。不能在活跃请求上换tensor地址，也不能扩大headroom来掩盖预算错误。

## 9. 内存预算与公开API语义

设S为一份完整GDN状态的字节数（递推状态＋原有短卷积状态），P为实际完整状态槽数（含padding），A为记录、额外卷积、元数据及状态专用固定工作区的字节数：

```text
GDN reserved bytes = P*S + A <= M
P = floor((M-A)/S)，再遵守现有padding与可运行容量约束
```

R和记录行数按启动最大并发决定。增加公共快照槽只增加S，不增加一份R长度记录。Graph共有的执行工作区单独报告；不能把Replay私有缓存藏到“其他”中规避M。

- 未指定M：以同配置、Replay关闭时原本的GDN状态池预算作为启动M。
- 显式指定M：开关两侧都按该预算规划状态池，关闭时A=0；不能单独给开启侧更多专家／KV／状态预算。
- 若M不足以放下必须的活跃状态、padding和小缓冲，启动明确报错。特别是naive默认仅给live槽时，可能需要显式M；不能悄悄借用专家池或缩小并发。足够预算下naive应能运行并保持固定live槽隔离。
- `num_mamba_slots`和重建请求中的同名字段继续表示**实际可用完整状态槽数量**，不改成“等价预算单位”。Replay开启后启动实际槽数可以更少，必须如实报告。
- 显式运行时重建仍按调用者请求的实际槽数定价：`(usable+padding)*S+A`，纳入原总缓存预算检查。不要把启动M当成不能由操作员显式重建改变的隐藏第二限制。
- 同尺寸重建不改变状态预算口径；显式改变状态容量后，报告该次操作的预算和实际字节，不能让状态信息仍假称旧预算未变。
- 缓存状态、单位价格、可调整上限和harness使用的资源信息都要识别固定A；不能UI宣称能分配、实际必定超预算。
- 启动、重建和公开几何共用一处字节计算。A不能既算入权重占用又再从缓存预算扣一次，也不能在Graph工作区中重复计费。
- Qwen3.6单卡旧96可用槽加1个padding的参考物理预算为`6245744640`字节；该值只用于已核对配置的A/B命令，不允许硬编码进实现。

## 10. 组件归属与clean slate要求

在既有公共流程中替换GDN状态表示，不创建第二套SD调度器。

| 现有入口 | 本轮职责 |
| --- | --- |
| `engine/config.py`、`server/args.py` | 三个有实际用途的配置参数、按能力校验 |
| `kvcache/linear_state_pool.py` | 保留大状态分配；构造Replay状态组件；真实字节计费 |
| 新的单一GDN Replay状态模块（如`kvcache/gdn_replay.py`） | 小缓冲、请求映射、开始／提交／导出等状态生命周期；不含路由和采样 |
| 新的GDN Replay kernel模块 | port后的计算、flush／导出、小游标操作；数学主体共享，允许编译特化 |
| `models/qwen3_5_moe/gdn.py`与`attention/linear.py` | 根据实际GDN状态组件接计算与metadata，保持现有投影／norm／MoE |
| `scheduler/speculative.py` | 继续公共起草与采样，只调用状态生命周期；不能含u/k/g或模型名字判断 |
| `scheduler/cache.py`与`scheduler/scheduler.py` | 按准确位置导出／恢复、终止后提交、overlap安全边界 |
| `engine/graph.py`、`engine/speculative_graph.py` | 通过组件准备固定metadata和Graph生命周期；不硬编码GDN层数 |
| engine预算、服务资源统计、`engine/speculative_cost.py` | 完整记账、真实阶段成本和最少必要公开证据 |

文件名是建议，职责边界是要求。禁止为未接入的Mamba/KDA、树状SD、跨机cache做预留接口；相同GDN架构自然复用本组件。

最小实现要求：

- clean slate指新路径清楚、没有临时补丁堆叠，不是重写整库、删除A/B基线或覆盖用户其他改动。
- 不加空壳backend registry、无调用方protocol、通用transaction／migration框架。
- 相同数学第二次出现就复用；不要复制AR、draft、verify三套GDN更新代码。
- 旧实现保留作A/B，但只保留这一条旧路径；新路径内部不能还藏“每步大矩阵快照”的替代方案。
- 不加每token日志、指纹、checksum、全记录清零、为了断言而D2H同步。
- 清除试验代码、未用参数、临时调试分支。更新注释中的旧N+1槽公式，但保留其在关闭路径上的正确含义。

## 11. Graph与真实执行长度

- 覆盖现有AR、draft和verify Graph；实际N=0…8、混合长度、部分请求结束和C1／C4／C16及自然尾批。
- checkpoint、记录、卷积缓冲、游标／索引、输出buffer地址固定；capture使用隔离哨兵，不能写坏真实缓存。
- 记录更新只统计真实请求和真实查询；Graph填充行既不能写记录，也不能推进长度或消费容量。
- 组件明确区分draft、target AR、target verify，普通decode Graph不允许误用draft工作游标。
- 不在GPU回放期间根据`.item()`决定flush。按GPU标志和固定捕获路径选择实际工作。
- 不改变已有Graph形状策略来掩盖Replay收益。本轮先保留实际query与物理padding的独立报告；进一步缩小物理verify Graph是另一个优化，不混入主A/B。
- 捕获／重建后检查实际Replay路径被执行；仅配置显示Graph开启不算完成。

## 12. 成本与最少必要观测

现有成本模型分别记录AR、draft、verify，但新引入的起草准备、flush、提交、快照导出可能在原计时区间外。必须识别这些成本，避免自适应把它们当作免费操作。

- 起草前只做一次的准备／flush计入本轮固定成本，不误当成每个draft token都重复的成本。
- verify窗口计算按真实查询和实际Graph执行尺寸记账。提交／短卷积选择的成本计入对应轮次，不能漏算，也不能与包含它的forward时间重复相加。
- 公开快照导出属于服务生命周期成本，应单列，不能悄悄只在Replay关闭侧发生。
- 保持原MoE传输估计与策略；本轮不重写自适应目标函数或专家预取算法。
- 详细GPU计时复用已有统计／事件机制及`--moe-collect-stats`，不用新加多个profile开关。计时不额外逐层同步，A/B两边使用同一统计设置。

公开静态资源信息至少包含：是否激活、R、记录请求行容量、实际checkpoint槽数、checkpoint／记录／额外卷积／元数据字节和总常驻字节。动态信息至少能证明AR／draft／verify各实际走过Replay、flush与快照导出实际发生；已有SD计数和Graph形状继续复用，不重复设计一套相同计数。

建议公开字段的稳定命名见公开验收契约。计数应以请求／逻辑位置为单位，避免每个head或每层重复计数；统计快照可能滞后必须如实注明。

## 13. 分phase提交

每phase一个生产commit；修复未完成phase可在本地整理后提交。测试由独立agent按对应phase另交commit。中间phase不代表功能交付。

1. **计算核心**：移植GDN小记录、窗口输出、flush／完整状态导出，接实际调用入口和精度说明。发布供独立数值验收使用的真实生产接口契约，不新增只为测试存在的API。
2. **完整状态生命周期与预算**：target AR、draft、verify一起接新表示；单尾部复用、卷积状态、字节预算和A/B参数。禁止把target-only版本当作本phase终点。
3. **服务集成**：Graph、prefix、stop／EOS／取消、AR overlap、重建、真实调用路径全部接通。保持旧路径回归。
4. **成本、验收和整理**：补齐真实计时与必要统计，运行有意义的A/B；删除临时代码，更新简短状态文档，记录结果和未解决问题。

每次提交分别报告生产增加／删除／净增，测试单列。第三方原样移植行数、文档和生成文件另列；在第三方代码上自写的适配不能全部隐藏为“vendor”。不要为了行数指标压缩可读性，也不要以“以后通用”为由扩大代码量。

## 14. 独立验收与性能实验要求

协调者只向独立测试agent提供公开验收契约，不给本文、生产源码、diff或内部调试笔记。实现agent可以构建和说明公开接口，但不能代写测试。已有合格独立agent可复用其自己编写的客户端；新的测试agent不得通过阅读其他内部测试获取实现信息。

验收分开报告：数学／位置正确性、公开服务行为、性能。允许已量化的浮点差异，不要求所有文本逐字一致；但不能用“BF16误差”掩盖错位、串请求、接受draft状态或漏掉历史。数值容差由独立agent按公开精度契约事先确定，不因看到失败再任意放宽。

最小有意义的实验：

- 同N对照：旧／新SD均实际N4，隔离状态算法与kernel变化。
- 同预算容量对照：配置最多N8、C16短输入，关闭自适应；旧版受状态容量限为N4，新版应有真实满B16、实际query=144的N8证据。
- 四臂：旧AR、新AR、旧SD、新SD；同输入、专家／KV容量、GDN字节预算、Graph、统计设置和同一块GPU。
- 然后才比较自适应，记录实际草稿长度分布、接受长度、fallback与flush成本，不能用configured N当actual N。
- 用包含前缀命中和分块输入的代表性agent请求补测生命周期开销；不把短C16结果当完整agent trace结论。

可复用的既有原始证据目录：`/data2/servebig-envs/state_phase_20260925_gpu1/`、`/data2/servebig-envs/state_slots_20260925_gpu1/`。旧数据用于理解基线，正式比较优先本轮成对运行，保留原始请求、实际命令、提交、GPU、共享负载及结果。不得停止外部GPU／CPU任务；资源不足时说明并请求协调。

**已知质量基线不能丢失**：截至`41b50c5`，stop字符串／数组／EOS／取消后的命中与从头计算有4组全文差异，前后版本已逐字复现；更早Python copy题也有AR／SD差异。新增差异必须独立报告／定位，旧差异不能改判成质量等价。本轮不顺带重做harness的质量评分策略。

## 15. 完成标准与需要停下来讨论的变化

完成必须同时满足：三个计算角色都实际走Replay、没有独立draft大矩阵和逐位置verify大矩阵、状态／前缀／终止位置正确、Graph和重建可用、字节预算可核对、独立验收完成、性能数据真实、生产代码和提交干净。

新方法即使不提速，也要交付真实结果及成本分解；不能删掉不利case、静默改资源、关闭公共前缀或只展示kernel微基准。未完成的真实路径必须明确列为未完成，不能称为达到可交付标准。

以下变化须先与用户讨论：恢复完整draft矩阵作为默认、扩大状态总预算／减少专家或KV来腾空间、禁用已有前缀或AR overlap、改变公共保存策略或模型支持范围、引入新的默认fallback、追加与本方案无关的优化。R、launch参数、同一职责内的函数拆分可按证据自行选择，并记录实际测量配置。

## 16. 启动模板与交接指令

下面是实现后的目标CLI，不是说这些新参数已在基线存在。GPU须由协调者确认可用，端口须选择未占用值；不要照抄一张当前已被其他任务占用的卡。

```bash
cd /home/nengneng/AIPrometheus/servebig/servebig-project/.sd-worktrees/freetoken-sd-portability
REPLAY_GPU_UUID='<协调者确认的GPU UUID>'
REPLAY_PORT='<空闲端口>'
PYTHONPATH="$PWD/python" \
CUDA_VISIBLE_DEVICES="$REPLAY_GPU_UUID" \
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  -c 'from freetoken.cli import main; main()' serve \
  --model-path /data1/lmcache_kv/models/Qwen3.6-35B-A3B \
  --gpu "$REPLAY_GPU_UUID" --host 127.0.0.1 --port "$REPLAY_PORT" \
  --dtype bfloat16 --attention-backend fi --moe-backend offload \
  --moe-cache-size 1536 --num-tokens 4096 \
  --max-seq-len-override 1024 --max-prefill-length 512 \
  --max-running-requests 16 --cuda-graph-max-bs 16 \
  --cache-type radix --batching-policy legacy --num-tokenizer 0 \
  --sampling-defaults none --reasoning-parser off \
  --served-model-name replayssm-ab --enable-cache-report --moe-collect-stats \
  --speculative-num-steps 8 --speculative-draft-experts 3 \
  --speculative-draft-residency router --speculative-draft-load-missing \
  --gdn-state-budget-bytes 6245744640 --gdn-replay-buffer-len 32 \
  --enable-gdn-replayssm
```

- 旧SD对照：仅移除Replay开关，保留同一字节预算，不能退回另一份未包含预算参数的旧安装包而忽略资源核对。
- 同N实验：两边都把最大草稿改成4；实际是否达到4仍核对公开记录。
- AR两臂：steps改0、residency改off、移除load-missing及其他依赖SD的选项；正常AR验收不携带强制关闭overlap的环境变量。
- 自适应两臂：完成固定模式后，双方同时加`--speculative-adaptive-cost`。
- 前缀／分块验收：双方一致调整context和KV，使工作量实际能装下，并使用真实分块。不能用这个短请求模板宣称覆盖长agent trace。

可直接交给实现agent的任务摘要：

> 在当前FreeToken工作树上按本文完成GDN ReplaySSM。实现AR、少专家draft和target verify统一表示及同尾部复用；保持公共SD架构、公共完整快照、固定显存预算和Graph。只做生产代码、构建和公开接口，不读取／编写测试。按四个phase独立提交，提供每次生产代码量和公开契约变化。将数值／服务／性能验收交给未读实现的独立agent；无合格测试agent时完成可做的实现工作并向协调者请求安排，不代写测试。不要把target-only、eager-only或关闭前缀／overlap的版本作为最终完成。出现本文第15节的设计变化先说明，不擅自扩大范围。
