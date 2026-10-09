# NoWAG 在线独立接入设计

状态：设计交付，未实施、未通过运行验收。基线：`main@732f1ee`，2026-10-09。

本文供实现者和协调者使用。测试作者只接收[独立公开契约](nowag-runtime-public-contract.md)，不得读取本文、源码、diff 或内部测试。

## 1. 已确认范围

用户已确认：先把 FreeToken 的在线专家接口整理干净，将 NoWAG 作为独立权重格式接入；本轮同时补齐已有公共执行路径中的 NoWAG 支持。离线校准、量化、恢复训练及后续训练方法留在原仓库；GGUF 不牵引本设计，也不纳入实现范围。

- 只改变路由专家的权重表示和计算实现。模型路由、激活、bias、共享专家、attention、KV、请求调度的数学与策略由原组件负责。
- 接入按专家组件的实际能力判断，不增加模型名字白名单；“每个架构”指 FreeToken 已注册模型的专家组件，不是增加模型解析器或承诺没有权重的模型已完成质量验收。
- 覆盖原生权重目录和 FTW、全驻 GPU、offload／cpu／hybrid、预取／分层执行、AR／self-SD／DFlash、已有 CUDA Graph 路径及 TP。
- 组合范围取各组件已有能力的交集。例如主线 SD 仅 TP=1、全 CPU 专家 SD 不受支持、部分 attention 尚无 SD；本轮消除 **NoWAG 新增的障碍**，不顺带实现这些公共组件本来没有的能力。
- 保留公开选项及既有非 NoWAG 默认行为。不新增“启用新架构”开关，不另建专家调度器、CPU 线程池、显存分配策略或在线训练流程。

本 PR 只有设计。后续实现单独分支，按 phase 提交；设计合入不表示功能合入。

## 2. 现状与可复用边界

| 当前组件 | 判断 | 本轮处理 |
| --- | --- | --- |
| [ExpertBanks](../python/freetoken/moe/expert_banks.py)、公共 host banks | 已有权重来源到缓存的边界 | 演进现有记录和加载入口，复用分配／并行读取 |
| [专家缓存](../python/freetoken/moe/offload_cache.py) | 按 bank 的实际行字节搬运；不是 BF16 专用 | 保留 slot 归属、淘汰、预取、事件及重建策略 |
| [MoE 执行](../python/freetoken/layers/moe.py) | 路由、搬运、计算已经部分分离 | 把格式布局和计算分支收至格式模块，公共执行只调用绑定的方法 |
| [NoWAG 读取](../python/freetoken/moe/nowag.py) | 有持久化格式，但数学规则按两个模型名字选择 | 读取只验证权重契约；计算语义由专家组件提供 |
| [NoWAG 算子桥接](../python/freetoken/moe/fused_nowag.py) | 依赖 `nowag_vllm`；已有 out／workspace 参数 | 将在线依赖闭包纳入 FreeToken，并接公共缓冲管理 |
| [SD 配置](../python/freetoken/engine/config.py) | Graph 与部分控制项按格式字符串排除 NoWAG | 完成实现后改为组件能力判定，不能只删除检查 |
| [FTW](../python/freetoken/checkpoint/ftw.py) | 有格式化 bank，但没有 NoWAG codebook 的完整往返 | 纳入共享权重及格式元数据，原生／FTW 使用同一运行描述 |

当前 NoWAG GPU 支持 D4/B12 和 D6/B12；CPU 路径仅支持 D6/B12。全驻 GPU、FreeToken 内的 NoWAG TP 切片、完整 FTW 和 SD Graph 均需本轮补齐。插件在 vLLM 中有 TP 经验，不构成 FreeToken 的 TP 验收。

## 3. 目标边界

```text
模型专家描述 ─────────────┐
                         ├─ 启动时解析并绑定专家执行方法
checkpoint → 格式读取 ───┘              │
                           ┌────────────┴────────────┐
                           │                         │
                    公共驻留／缓存／搬运       格式私有运行参数
                           │                         │
                           └──── 权重视图＋输入 ──────┘
                                         │
                           NoWAG GPU／CPU kernel
```

公共调度继续决定何时运行哪一层、处理哪个 batch、何时预取／暂停／恢复。NoWAG 只声明布局、共享数据、运行能力及临时空间，并实现专家计算。

### 3.1 三个数据边界

以下名称是计划中的小型数据记录，不是动态插件框架；复用现有 `ExpertBanks`，不保留两套同义对象。

| 记录 | 必需内容 | 所有者 |
| --- | --- | --- |
| 专家数学描述 | H／I／E／top-k、真实 MoE 层编号、激活及参数、bias、路由权重作用位置、输入／中间值舍入语义、TP 分区 | 模型的专家组件 |
| 专家权重记录 | 公共几何、bank dtype／stride、每专家字节、共享张量；编码细节由格式私有记录承载 | 格式加载器；公共存储持有分配 |
| 已绑定执行方法 | GPU／CPU 入口、实际布局、支持的执行模式、workspace 需求、输出写入约定 | 格式模块，初始化时绑定 |

模型类型仍可用于检查 checkpoint 是否属于目标模型，但不能用来决定“允许 NoWAG 的模型名单”。输入属于错误模型时仍须拒绝，不能以移除白名单为由忽略权重对应关系。

公共记录不定义D／B、codeword lane或normalizer放置等NoWAG专属字段；模型只声明原有计算语义。NoWAG模块解释自己的编码并决定如何满足该数学，公共层只认识可搬运数据、资源需求和计算契约。

### 3.2 最小调用面

沿用现有加载入口，只收口确有调用方的四类操作；具体名字、输入／输出和错误约定固定在[接口约定](nowag-runtime-interface.md)，公开计算部分同步给黑盒作者：

1. `load_expert_banks`：读取／切片／准备布局，返回专家权重记录；转换时支持逐层写出。
2. 启动时绑定执行方法：根据数学描述、实际权重格式、设备与配置检查能力；不靠模型名或路径名选择。
3. 空间请求：给定物理token行数、top-k和实际bank行数，报告workspace的真实大小／布局，由engine计价和分配；不能把模型专家数误当成缓存槽数。
4. 专家执行：输入 x、当前层、有效路由、权重视图、预分配 workspace 和 out；返回写入后的 out。

执行方法不拥有 batch 调度、CUDA stream 选择、缓存淘汰或资源预算。CPU 使用既有 executor／线程池，格式模块提供布局描述与计算入口。不存在热路径 import、模型实例 monkey patch 或按专家逐个 Python 调度。

## 4. 只改变专家权重，不改变专家数学

NoWAG 每个投影的实数含义为 `y = ((x * input_norm) @ C[A].T) * output_norm + bias`，展开时裁掉编码尾部 padding。实际 dtype、舍入和激活顺序见公开契约；不能用实数等价掩盖 BF16／FP8 舍入位置变化。

本轮覆盖现有专家组件实际使用的数学族：

| 组件语义 | NoWAG 处理 |
| --- | --- |
| 普通 SiLU 门控 | 复用现有融合路径 |
| GELU／tanh-GELU 门控 | grouped NoWAG 投影＋原激活；可融合但先保持数学 |
| GPT-OSS／MiniMax 的带参数门控、clamp | 使用组件给出的 alpha／limit 和对应公式，不降级成普通 SiLU |
| 投影 bias | 从原 checkpoint 的模型映射读取并保留；NoWAG v1 不含 bias 时不可当作零 |
| DSV4 的输入与中间 FP8 round-trip | 保留组件声明的舍入及 clamp 顺序；不能将 down normalizer 移过舍入点 |
| 路由权重乘在输入或输出、特殊 router／专家缩放 | 路由由原组件计算；执行方法保留其作用位置与尺度 |
| 前若干层为 dense、只有后续层为 MoE | 由模型提供真实层映射，不假设所有 decoder 层都有专家 |

现有融合 kernel 不覆盖的数学，补 grouped 压缩投影路径，再调用现有激活；只在 tile 中解码，不展开完整 BF16 专家权重。不得用 CPU 逐专家循环或整层反量化作为不告知用户的默认实现。

无路由专家的模型不适用 NoWAG expert 选项。新模型若具有已有数学族、有效权重及可用公共执行能力，应直接通过绑定；缺失的是具体能力时明确报该能力，不报“模型名未支持”。

## 5. Streaming、共享参数和资源生命周期

### 5.1 三类存储

- **专家行**：assignment、normalizer，以及原模型需要的 bias。expert 为最外维、每行连续；公共 cache 继续按行复制，prefill／decode 使用同一份压缩表示。
- **共享参数**：一个模型共享 codebook，GPU 每个 TP rank 一份；cpu／hybrid 保留主机视图。声明为共享不可变张量，不按专家或 cache slot 复制；空闲重建只重绑视图。
- **workspace**：路由整理、中间激活、分路输出、任务描述和 padding。按并发执行域与捕获形状分配，不能让可能同时执行的图／层段共享可写临时区。

沿用已有容量政策，精确计入专家行、codebook、workspace、Graph 与对齐成本。减少每专家字节后，由原自动预算规则产生新容量；NoWAG 不额外抢占 KV／状态预算，也不增加固定保底槽政策。

### 5.2 沿用所有既有搬运入口

decode miss 按需H2D、整层prefill streaming、双缓冲预取、命中D2D、joint／layered／layered-pipeline驻留组均使用同一权重记录。统一计算入口只接受当前bank的行号；公共存储路径将逻辑专家id转换一次，格式模块无需猜当前是slot还是逻辑id。若内部kernel需要按逻辑专家排序，由格式方法局部整理，不能把该要求泄露成公共格式名单。

并发复制的源、目的、映射、out／workspace 必须保留至完成事件。缓存槽、执行缓冲和 codebook 各有清楚的所有者；格式模块不另加全设备同步。继续复用公共输入存活期、取消、重建及流间依赖协议。

### 5.3 全驻 GPU 与 CPU／hybrid

全驻 GPU 直接持有所有专家的同格式 bank，逻辑专家 id 即行号，不伪装成一个超大卸载缓存；普通 MoE 层调用同一执行方法，保留原 TP 合并。显式选择全驻 GPU 时容量不够正常报错，不自动改为卸载。

CPU／hybrid 补齐 D4/B12 与 D6/B12；保留原 CPU 队列、flag 握手及 hybrid 分路。CPU 只读主机压缩行和 codebook，每条有效路由恰好计算一次。GPU／CPU 必须使用相同数学描述，不能各自保存一份模型名字规则。

## 6. CUDA Graph 与投机解码

- AR 完整图、已有范围图、self-SD draft／verify 图和 DFlash target verify 图都调用相同专家执行方法；NoWAG 不新增自己的图管理器。
- 编译、profile读取、最终kernel计划及workspace建立在capture前完成。公共预算先结合物理token行数与bank容量确定几何，再完成kernel准备；不能先按逻辑专家数准备，再靠replay临时补空间。根据batch形状选择已有准备好的kernel可以保留；不能在replay中作主机取数、动态分配或离线调优。
- 物理行数按实际 Graph padding 和 `batch × query_width` 计算；尾批、无效专家 id、零权重路由写确定结果，不能读取未初始化或越界槽。
- self-SD 起草缓存命中、补缺加载和 verify 预取使用原缓存接口。按真实压缩行字节测量搬运成本，不能沿用 BF16 成本常数；不改控制器的政策、启用条件或默认值。
- fixed／adaptive SD 的接受、回滚、GDN／KV 快照以及 DFlash drafter 数学不变。NoWAG 只替换 target 专家；DFlash 自己的权重方案不跟随 target 改变。
- `speculative_graphs` 和控制项检查改为执行组件能力交集；只在相应路径实际完成后开放。不能通过删除 `nowag_expert_path` 检查宣称已支持。
- Graph 输出与 workspace 地址、池重建后重新捕获、layered 范围图独立回放均纳入验收。显式请求不能静默变成 eager 或 AR。

SD 的 TP=1、attention／状态支持范围、legacy-only 控制项等公共限制仍生效；不得借本轮扩大这些组件的支持承诺。

## 7. TP：切片压缩权重，保留全局 codeword 分组

复用模型已有 TP 分区与通信。对普通门控专家，gate/up 按输出 I 切片，down 按输入 I 切片：

- gate/up 的 assignment 输出行、output normalizer 和 bias 一起切片；输入 normalizer 保留 H。
- down 读取覆盖全局输入区间 `[a,b)` 的完整 codeword 组，记录起始 lane 和逻辑长度；边界组可能被相邻 rank 重复读取，但只计算本 rank 的有效 lane。
- 不重新训练 codebook、不重新量化边界、不把局部输入从零开始重新分组。I=512／TP2 时分界256不是6的倍数，是当前真实形状必须覆盖的情况。
- codebook 每 rank 保持原值；normalizer 按对应逻辑轴切片。共享参数不参与专家输出 all-reduce。
- 每 rank 计算局部 down 贡献，沿用公共 all-reduce。down bias 仅由rank0贡献，再归约一次；gate/up bias按分片生效。路由权重在模型规定的位置使用，包括其与bias的先后关系。
- prefill、decode、CPU／hybrid 在同一 rank 使用同一分片和数学。缓存计价是本 rank 的实际大小，边界重复字节也计入。

已有插件的分组边界计算可移植为纯函数；不移植 vLLM 层对象或 parameter monkey patch。TP2 必须真实双卡验收，不能用两个独立单卡服务代替；SD+TP 不因本设计自动成为合法组合。

## 8. 原生权重与 FTW

保留 `ft serve --model BASE --nowag-expert-path SIDE`。原模型仍提供 tokenizer、非路由权重及必要专家 bias；NoWAG 目录只提供其声明的压缩矩阵。

新增转换入口沿用同名选项：`ft checkpoint --model BASE --nowag-expert-path SIDE --out DEST ...`。转换结果必须自包含，隔离两个源目录后可直接 `ft serve --model DEST`，不再需要 NoWAG 仓库或 sidecar 路径。

- 导出全局逻辑编码及明确的 D／B／layout、逐层映射、共享 codebook、normalizer、必要 bias；TP 切片在运行加载阶段完成，文件不绑定转换机 rank。
- 转换按层交给现有 writer，读取按现有 host bank 流程执行。共享参数单独写一次；不把 codebook 当成普通每专家 bank。
- FTW 与原生路径最终返回同一专家权重记录和执行方法；格式与 layout 必须可从产物确定，不依赖目录名、转换时 Python 对象或原仓库绝对路径。
- 已存在的两种 v1 manifest（通用与 DSV4 历史格式）都继续读取；这是保留真实存量输入，不引入泛化迁移框架。缺少新模型需要的 bias／数学信息时由原模型描述补齐，无法对应则拒绝。
- 专家格式选择不改变 attention／共享专家／lm_head 的精度，也不改变 KV 编码。整份权重支持某个 TP／后端，还须满足这些非专家组件自身的能力。

## 9. 文件职责与移植范围

| 位置（目标） | 职责 |
| --- | --- |
| `moe/expert_format.py` | 现有专家格式的布局与执行绑定；集中现有分散注册，不建立自动插件发现框架 |
| `moe/expert_banks.py` | 公共加载协调与权重记录；不存放 NoWAG 解码数学 |
| `moe/nowag/weights.py` | v1 读取、bank 准备、共享参数与 TP 切片 |
| `moe/nowag/method.py` | 数学能力、GPU／CPU 绑定、workspace 与执行适配 |
| `kernel/nowag/` 及现有 CPU kernel 目录 | grouped 压缩投影、融合快速路径、CUDA／Triton 与 CPU 计算 |
| 原模型组件 | 权重名称映射、专家数学、router、非专家权重与 TP 语义 |

迁移后删除失去调用方的旧 `moe/nowag.py`／`fused_nowag.py` 桥接与模型名字规则；更新调用点，不保留两套接口。checkpoint 的旧格式读取保留。

在线来源固定为 `servebig-nowag-plugin@4883f1d`：`moe_ops`、`cuda_ops`、`assignment_layout`、`moe_activation`、必要的执行选择／实测配置和对应 MoE CUDA 源。只提取在线可达部分，保留来源和现有第三方声明；不带入 vLLM 注册、模型 adapter、dense 实验路径、校准器、训练器或黑盒候选。

源码可借鉴已有 TP 边界工具，但重新通过 FreeToken 接口绑定。已有 kernel 和实测配置先保持数值／选择行为；职责整理与算法优化分开提交。构建、wheel 包数据和清洁安装纳入验收，运行不再 import `nowag_vllm` 或原始 `NoWag` 仓库。

## 10. 分阶段实施与交付门槛

| Phase | 交付内容 | 进入下一步的证据 |
| --- | --- | --- |
| P0：边界 | 固定公开契约、格式／数学／存储描述，迁移已有注册与调用 | 原非 NoWAG 行为保持；未新增无消费者抽象 |
| P1：在线独立 | 提取内核、原生加载、共享参数、D4/D6 GPU／CPU | 清洁安装与已存在 Qwen／DSV4 路径通过独立数学验收 |
| P2：组件与驻留 | 各专家数学族、bias、真实层映射、全驻与全部现有 streaming 模式 | 模型组件矩阵、缓存生命周期和非专家精度隔离通过 |
| P3：SD／Graph | 固定／自适应控制、预取、完整／范围图、尾批及输出缓冲 | 独立 SD 状态／数值／公开服务验收通过 |
| P4：TP／FTW | 分组边界切片、通信、原生与自包含转换 | 真实 TP2、源目录隔离和往返矩阵通过 |
| P5：交付整理 | 删除旧分支与重复逻辑、整合文档、性能和代码量 | 完整矩阵、同资源对照、最终代码审计及计量完成 |

每 phase 生产与独立测试分别提交。实现者不读写测试；独立测试作者只获得公开契约、合法输入及运行资源；协调者运行构建与验收，将公开失败条件反馈实现者。

性能以同一份 NoWAG 权重、同设备和显存预算对照重构前后，区分同槽数的执行开销与同字节预算的服务收益。既有路径默认允许的中位延迟回退不超过5%；使用三个交错配对 block，同时记录背景负载。超过门槛须定位并修复，测量被污染则结论未完成；不得无限复测挑最好值。新增路径单独报告，无既有 NoWAG 基线时不虚构加速比。

计算与搬运采用实际字节、有效／物理 token 行数和既有公开统计；新增状态固定为 `/v1/cache/status` 的 `geometry.experts`，字段见公开契约，不逐 token 打日志。不增加摘要文件或无消费者的校验码。

每个实现 commit 报生产增加／删除／净增，测试另计；最终相对功能开始前基线计量，不累计重复搬动。已有项目自有 kernel 移入 FreeToken 的增加行仍须报告，真正第三方原样代码另列。**本设计 PR：生产0、测试0。**

## 11. 并行工作与未决依赖

- 本设计基于干净主线，不把尚未合入的共享 runtime／Flash-Next 复制过来。实施开始或集成时记录实际基线；只核对相关接口变化。
- 共享 runtime 快照 `da96cd7` 仍在独立验收；此前审计问题的修复提交不等于交付。NoWAG 复用最终公共分配与生命周期协议。
- Flash-Next 快照 `85d93e4` 的模型／状态验收尚未完成。其48层、512专家、top-10等数学／形状列入组件验收，真实 NoWAG 质量验收须等待模型基座及匹配产物。不能拿旧 Qwen3.6 权重冒充。
- 现有真实产物优先使用 Qwen3.6、DSV4；其他已注册专家数学族由独立合法小模型覆盖。生产权重尚无、双卡未分配或基座未交付的项须明确标未完成，不能以组件验收代替端到端验收。
- 本轮不启动离线量化训练，不挑选未来 GGUF 格式，不更改默认资源分配／回退政策。若必须改变后者，先报告具体影响并取得用户确认。

设计验收点是边界、语义和实施／验证路径完整；功能交付点是独立矩阵真实通过。两者不得混称。
