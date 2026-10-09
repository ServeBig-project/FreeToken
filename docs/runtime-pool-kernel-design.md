# 共享 runtime 的布局与内核实施协议

范围：共享 runtime feature 的必要组成部分，不单独作为交付终点。2026-10-09最终复核对齐`main@c320fbb`及DFlash合流版`32646f6`；下面两处FLA、状态／Replay布局与VMM修改仍未由前置功能完成。KV数学、GDN更新、Replay和SD接受规则保持不变。用户已确认采用VMM及状态布局调整，完成全部内核适配；QSA／DSA及逐轮CPU KV streaming不属于本协议。

## 1. 已核对的环境和结论

2026-10-08核对的开发环境：PyTorch `2.11.0+cu130`、FlashInfer `0.6.17`、`sglang-kernel 0.4.5+cu130`。三张RTX4090，驱动 `580.126.09`。当时通过只读Driver查询确认，三卡均支持VMM，最小和推荐分配粒度均为 **2 MiB**；没有为此分配GPU数据或运行模型，本轮未重跑设备检查。

结论：KV继续使用现有按层布局；GDN状态与Replay记录改为同一槽／记录行的各层相邻。两处FLA状态寻址需要修改，其余路径按实际stride接通，不能通过运行时整池`contiguous()`复制来规避。

源码检查与实际GPU验收是不同证据。本文件锁定要实现的布局、修改点及验收要求，不声称尚未运行的内核已通过。

## 2. 物理块和虚拟布局

设Driver最小粒度为G。启动创建 `floor(R/G)` 个大小G的物理分配，所有组件共用；设备粒度运行时查询，不写死4090数值。每块独立映射，offset=0；物理块不能被同时当成两份独立可写数据。地址范围和tensor视图固定，块用途在最后一次访问完成后可变。

不采用多种物理块大小，避免把不可拆的较大handle再分给小映射；不引入每token的Driver分配。普通同类空槽先复用，跨类型缺空间时才撤销整块映射、归还公共空块并重新映射。

共享模式的逻辑页／窗口槽分配在CPU控制侧完成，CPU在launch前知道需要哪些块及其虚拟地址范围。GPU页表和full→window表批量更新，保留attention查询方式；不沿用GPU-only free ring后再同步回读槽号。前缀索引在CPU可追踪，不能只增加一个物理allocator而遗漏分裂、释放和恢复路径。

CPU只决定所有权和块分配，不逐token执行寻址。由GPU根据块ID、请求位置和有效长度生成slot_mapping、展开页内位置并更新计算表；优先使用已有GPU操作，仅为实际调用路径增加必要kernel。VMM映射是host Driver API操作，与attention kernel内部读表是两件事。

### KV

保留目标KV的 `[2, layers, pages, page_size, heads, dim]` 逻辑布局，以及前置DFlash功能的全历史／窗口分组布局。各层K/V内部的token／head布局不变。逻辑页只映射覆盖它的实际字节范围。

这样不改变attention访问模式，也不在每次attention前打包整段KV。代价是各bank尾部可能浪费不足一个G，dummy所在块也要常驻；这部分按物理块记账，进入最小可用预算和上下文推导，不能按理想bytes/token隐去。

### GDN

recurrent和conv仍是两种dtype的独立视图，共用物理块来源；不把conv塞进跨几十MiB的recurrent槽间隔。

```text
recurrent底层顺序：[state_slot, layer, value_head, key_dim, value_dim]
对组件暴露的视图：[layer, state_slot, value_head, key_dim, value_dim]
conv底层顺序：    [state_slot, layer, conv_dim, kernel_minus_one]
对组件暴露的视图：[layer, state_slot, conv_dim, kernel_minus_one]
```

槽涵盖该模型所有GDN层，可以是活跃状态、前缀快照或普通SD暂存，并非固定归某一用户。既有逻辑槽编号继续有效；视图中的槽stride变为全部层的字节跨度。头维／矩阵维仍连续。相邻槽可以共享边界物理块，不能因一个槽释放就撤销另一个仍在用的块。

### ReplaySSM

`u/k/g/window`分别采用 `[record_row, layer, ...]` 底层顺序，向调用方提供原来的 `[layer, record_row, ...]` 视图。各字段仍独立，避免小记录跨越完整大状态的地址跨度。每请求内部的head、ring和feature维连续，现有ring数学不变。

记录行不再按最大并发全部映射。取得行时映射必要字段，重新设置checkpoint位置和有效记录长度；没有有效记录的历史不能读旧内容。Graph行描述符与小计数器属于有界固定元数据，不为其重复预留完整记录数据。

### 为什么调整布局

用本机模型参数作一个解析例子：Qwen3.6-35B-A3B，BF16 conv／记录、FP32 recurrent，30层GDN，ring32，SD上限4；虚拟状态槽上限128、记录行上限64，实际只有1个活跃状态＋1个dummy状态、1行记录。按G=2 MiB计算最低映射量：

| 内容 | 保留逐层布局直接套VMM | 上述按槽／行集中布局 |
|---|---:|---:|
| recurrent | 120 MiB | 120 MiB |
| conv | 60 MiB | 4 MiB |
| Replay u | 60 MiB | 8 MiB |
| Replay k | 60 MiB | 4 MiB |
| Replay g | 8 MiB | 2 MiB |
| Replay window | 60 MiB | 4 MiB |
| 合计 | 368 MiB | 142 MiB |

这是指定逻辑布局与占用下的**映射粒度算术**，不是当前main的显存实测，也不是吞吐结果；KV、权重、Graph等未包含。它说明不能照搬逐层小字段布局后宣称共享空间已经充分利用。

## 3. 内核修改清单

| 路径 | 源码结论 | 必须完成的改动／保留行为 |
|---|---|---|
| `kernel/csrc/jit/store.cu`、`kernel/store.py` | 写KV按实际行stride寻址，byte偏移使用`size_t`；没有负页号保护 | 保持接口与数学；只提供真实已映射位置或已映射dummy，禁止用-1代替dummy。 |
| `attention/fi.py`及FI paged attention | 当前逐层K/V视图可继续使用，FI以页表与stride访问；新方案不改变KV布局 | 保持既有布局、dtype、page-size能力；只更新页表内容和有效长度，不复制整池、不重录每轮Graph。 |
| `attention/triton.py` | 从组件取逐层K/V后构建token视图 | 同样保留KV布局；支持该后端原有合法page-size，不能把FI的page-size=1限制变成全局限制。 |
| `kernel/fla/fused_sigmoid_gating_recurrent.py` | `h0_source + idx*HV*K*V`硬编码槽间隔 | 包装层传`initial_state_source.stride(0)`；读和写统一用实际槽stride，索引乘法先转64位。矩阵内部布局、gate、归一化、更新和中间verify缓冲算法不变。 |
| `kernel/fla/chunk_delta_h.py` | 同一个`stride_h`目前同时用于池状态及临时chunk状态h | 单独传池的槽stride；只改h0/ht池地址，保留临时h的紧凑stride，避免把prefill中间结果一起改坏。slot乘法使用64位。 |
| `kernel/causal_conv1d.py`、Triton conv fallback | wrapper传状态tensor；Triton已接收状态stride，indexed路径已将槽号转为64位 | 这部分寻址是正确的，保留并覆盖新conv视图最后两维连续的路径，padding仍提前跳过。不得增加每轮全状态gather／scatter或重复修改已正确的64位转换。 |
| `kernel/triton/gdn_replay.py` | replay/fold已有状态、行、层stride；window也有行stride | 用新视图的真实stride；将行／槽的偏移乘法统一为64位，ring索引及更新数学不变，负记录行仍不读写状态。 |
| `kvcache/linear_state_pool.py`、GDN模型调用 | `index_select/index_copy`、按层视图和槽复制是主要访问方式 | 改存储创建及映射生命周期；新槽只初始化自己的已映射范围，清空/复制/快照/commit都按视图stride，不能整张虚拟tensor清零。 |
| `kvcache/prefix_store.py`复制 | 逐family行视图、`index_select/index_copy`支持stride | 保留按实际逻辑行搬运。padding/未映射洞不进入复制范围；不能把新布局当连续旧布局做裸memcpy。 |
| DFlash全历史／窗口 | 使用既有store及attention，各组位置映射不同 | 保留分组、绝对位置、窗口解锁和reject语义，各组申请物理块统一计价；不再启动映射整份历史容量。 |
| KV／DFlash窗口分配与映射更新 | 最新前置实现仍由GPU tensor/free ring保存具体空闲编号 | 共享模式改由CPU分配具体编号，映射完成后批量上传页表／窗口表；GPU scatter按已有流依赖执行，不引入GPU→CPU编号查询。 |

SGL conv的上游接口按tensor stride传递状态，末维仍需遵守原访问约定；其索引字段含32位类型，不能把任意64位跨度传入并假定安全。[SGL接口源码](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/kernels/aot/csrc/mamba/causal_conv1d.cu)、[参数类型](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/kernels/aot/csrc/mamba/causal_conv1d.h)。本协议将conv与大recurrent分开，因此不产生“大状态槽stride强加给conv”的放大；仍按实际虚拟寻址范围检查后端能力，独立验收必须调用安装的0.4.5算子，上游源码不能替代wheel运行证据。

主线两处FLA硬编码在原连续布局下是正确的；它们是新布局必须适配的点，不作为旧版缺陷报告。64位偏移是为本次支持的较大池／新槽间距服务，不能用悄悄限制可用槽数替代。

## 4. Graph、元数据与分配时序

- native所有者提供Driver物理块、地址预留及带明确deleter的tensor视图；接入现有原生构建方式，不替换全局PyTorch allocator，不引入IPC／跨卡物理借用。
- 参数tensor、形状、stride和基地址在同一几何版本内固定。Graph引用的dummy和实际读写范围在launch前已映射、初始化并具备访问权限；修改页表／状态索引的GPU写入与下一次回放按原流顺序提交。
- 占用统计按原生物理块计算；PyTorch tensor的虚拟numel不作为真实分配量。退出／维护先结束计算与复制、销毁Graph，再撤销映射／释放地址和handle。
- 准入先取得整轮目标KV、状态、Replay、draft及必要临时容量；部分申请不能先进入模型。新空块映射不引用旧计算；跨类型重用必须等相关最后使用事件完成。
- `pool.free()`不再等于立即可以`cuMemUnmap()`。在途引用记作待回收，不供另一类型消费；缺空间时可以等待已提交操作完成，不靠全设备每token同步解决。
- DFlash不能为1到C每种batch各建一份`batch×max_seq_len`索引数组。按允许同时在途的执行路径保留有界最大描述符，序列执行的wrapper／Graph使用同一固定存储的视图；真正重叠的路径使用各自缓冲或事件依赖，不能覆盖旧plan。
- 最新DFlash使用`full_stores/protect_capture`区分全历史与窗口；保留这些hooks，并在clone／warmup／capture访问前映射实际临时页。不能用旧版`context.kv`假设覆盖所有层，也不能把全窗口模式的空`paged_views`判断为没有缓存能力。
- 捕获集合沿用有界尺寸及合法padding，不记录所有并发、长度、阶段组合。未捕获合法shape走声明的eager路径；总体内存不会随历史出现过的shape无限增长。
- 继承最新AR小batch覆盖（默认1到`min(C,8)`）与SD独立集合／分层verify，不退回旧的1/2/4集合。显式上限与重建后的`graph_bs_limit`保持；物理转用不重录、不把当前可用页数当成新的图上限。捕获scratch的虚拟范围必须先纳入R计价并映射，再clone／warmup。
- Graph新增reserved统计仅作诊断，不能代替E总占用。分层GDN prefill的临时h等仍属执行workspace；只改池状态stride，不把临时张量误搬进状态池或假定KV共享已经消除了执行峰值。

## 5. Kernel验收要求

独立测试作者只接收公开契约，不读取本文件。需要验证连续／合法跨槽stride的等值输入、非连续槽号、首末槽、padding、非满batch、prefill及decode、N1/4/8、Replay开关、BF16/NVFP4目标、TP局部shape和实际Graph回放。

必须覆盖新槽stride使地址跨过32位乘法边界的合法池几何，验证未选中的状态／邻接缓冲不被修改；只能使用已声明可映射的地址，不制造未映射输入调用算子。数值容差沿用原算子契约，不因新布局放宽。

共享池级验收还必须证明：同一Graph继续回放，已释放的状态块能够转给KV，再转回状态而输出正确；冷复制在途时不会提前改用途；取消／拒绝尾部没有泄漏，峰值计价包含实际对齐。

映射吞吐、块边界延迟、冷启动、长时间碎片和端到端结果属于完整交付验收。即使算子通过，缺少持续serving和TP验收也不能称为本feature完成。

## 6. 与当前 vLLM 的分工对照

核对vLLM主线源码后，不能把“换用其他attention backend”描述成“不再分页”。其[FlashAttention后端](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)仍向内核传递block_table；[BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)在host侧管理块分配／引用／回收；[GPU block_table](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py)包含Triton的slot mapping生成与表整理。我们的CPU分配约定采用这个职责区分，并不把本可由GPU生成的逐token映射搬回CPU。

首版明确不实现GPU自主全局分配器及其与host的所有权协调协议。CPU承担分配／回收决策和VMM调用，GPU承担映射计算和数据操作；相关GPU kernel适配仍是必做，不因排除GPU自主分配而退回逐token的Python地址计算。
