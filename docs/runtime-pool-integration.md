# 共享 runtime：当前基座与实现交接

当前实现基线为 `main@732f1ee`，已经包含 batching＋SD 和 DFlash。生产实现位于 `.worktrees/runtime-pool`（`feat/runtime-pool`，PR #9），正在收尾及独立验收。

本文保留基座行为与实现要求，配合[主设计](runtime-pool-design.md)与[内核协议](runtime-pool-kernel-design.md)使用；其中的历史失败不等于当前验收结论。独立测试作者只接收[公开契约](runtime-pool-public-contract.md)。

## 1. 已解决的差异与保留边界

此前五项接口差异在DFlash合流版已收口，不再是共享runtime的待办：

| 接点 | `32646f6`中的现状／必须保留的行为 |
|---|---|
| 配置 | 无草稿路径／SD意图时AR；给DFlash路径默认N4；N0忽略路径及DFlash专属设置。固定SD支持layered／hybrid，通用SD控制仍按其原能力校验。 |
| 预算 | `_load_dflash`先计独占权重，`DFlashLayout`计context／metadata／workspace；GDN不再替DFlash付钱，失效的`draft_bytes`调用已清除。 |
| 特征 | `record/flush(..., features)`和`LayerGroupState.draft_features`已与full／window存储结合，不能重新改回一份全局特征。 |
| 窗口 | `_draft_lengths`同时裁剪目标页与窗口；提交保留实际接受数、输出数和拒绝尾部释放语义。 |
| Graph | 独立SD覆盖、分层verify、重建上限和`full_stores/protect_capture`均已保留；普通AR也已覆盖1到`min(C,8)`的小批次。 |

源码入口均在[当前实现](../python/freetoken/)：`engine/{config,engine,graph,speculative_graph}.py`、`speculative/{dflash,dflash_cache}.py`、`scheduler/speculative.py`。

第六项自适应交叉已明确范围：**保留、默认关闭，仅legacy且沿用现有后端限制（当前为offload）；layered自适应不在这次交付范围。** 不再等待前置功能补齐，也不由共享runtime顺手实现或调参。自适应收益／决策开销优化已有单独后续事项，不能把它变成本feature的性能门槛。

已经转交内存预算工作的 OOM 纳入第6节的共享模式验收；不能用源码收口替代 GPU 运行证据。

**目标模型SWA后续接入，不等于DFlash的窗口不接入。** 最新DFlash用`HybridSWAKVCache`承载草稿窗口，CacheManager也会暴露`swa_paged`；必须按目标／drafter组件角色判断支持，不能看到这个标志或类名就拒绝共享runtime。

用户已明确本轮先不接稀疏attention：QSA／DSA（包括Flash-Next稀疏缓存适配）和活跃请求逐轮KV streaming都不加入本PR。GLM的KDA新组件也不因概念上类似GDN就自动算作已支持；不为这些后续组件预留空接口。

## 2. 开启 SD 后的内存账

| 内存 | 归属 | 必须避免的问题 |
|---|---|---|
| 目标驻留权重、DFlash独占权重 | 固定模型成本 | DFlash已先加载并计入权重测量；不能再从GDN或R重复扣。共享embedding／输出头不重复计价。 |
| 目标KV、GDN完整状态／Replay记录 | runtime物理池R | 不再应用独立GDN槽预算来判断共享模式能否SD。 |
| DFlash全历史组、有限窗口组，包括本轮临时位置 | runtime物理池R | 分组保留；按实际有效位置绑定物理块，不按虚拟页数／最大并发全量映射。 |
| 页表、full→window映射、窗口空闲元数据、Graph描述符 | 有界控制／执行额度E | 逻辑容量和物理数据分开；不能把`tensor.numel`诊断当作已使用的数据显存。 |
| drafter输入／位置缓冲、logits、概率、verify工作区 | 执行额度E | 必须按实际物理padding和同时存活的张量计峰值，不只算有效query数量。 |
| layered hidden／residual／DFlash target features | 波次执行额度E | SD＋batching已加入`feature_bytes_per_token`预估，合并后保留；不能与实测残留状态重复计价，也不能遗漏后续层才产生的特征。 |
| CPU冷副本和暂停快照 | 同一个host数据预算 | 保存窗口所需内容，不能把草稿所有旧位置重新备份成全历史。 |

新DFlash的 `DFlashLayout.total_bytes(P)` 包含全量context、metadata、workspace，适合原分池模式的启动规划；共享模式必须拆开使用。不能先保留整个R，又额外扣一份`full_context_bytes + window_context_bytes`。`geometry().reserved_bytes`是诊断总计，也不能整体再扣一次。

同样，`execution.resources.speculative_graph_reserved_bytes`记录捕获带来的PyTorch reserved增量，不是Graph全部存活字节，重建复用已有保留空间时可偏低；不能把它当作硬预算或把0解释成没有Graph成本。新增物理账保持这些旧字段的原口径。

## 3. 窗口容量与生命周期

`DFlashLayout.window_capacity(P)` 当前用最大请求数、窗口W、释放粒度、草稿上限、prefill额度、复制额度与工具锚点计算容量。共享模式保留这些语义中的**真实持有量与上界约束**，不保留预占固定物理窗口池的做法。

当前所有有限窗口层共用一个窗口组，按最大窗口保留；例如原生4k层＋额外8k cap，不是各层分别只占4k／8k。首版继承这个真实布局、映射与保护范围计价，不按理想的多窗口组公式少算。独立窗口组优化已另列后续事项，本feature不暗中拆组；若届时main已改变该接口，再按其实际组件计划接入。

- 窗口编号空间按R能承载的最大窗口单位和目标位置上界规划；当前合法绑定数由联合物理预算限制。不能仅因旧`C×W`分区满了就声称整个runtime没有空间。
- 只恢复／保存该模式需要的窗口，加上原策略仍要求保留的锚点。无限层保留完整目标历史对应的草稿上下文；native窗口和显式近似cap分别按其原数学工作。
- `--no-dflash-compact-kv`仍是原有全历史存储对照模式，不能借共享池悄悄改成紧凑留存；总账按它的实际策略推导，因此可支持上下文可能不同。
- 渐进解锁按**已提交token数**推进，不按forward轮数。SD一轮提交多个token不能拖延释放；拒绝尾部、暂停、取消、共享分叉都只解除自己的绑定／引用。
- 单纯KV备份不锁住无关草稿窗口；一个窗口最后使用结束才可回收其物理块。copy额度是对在途保护的限制，不是另外一块不计入R的显存。
- 恢复即使只搬GDN，也必须保护恢复点依赖的已驻留窗口，直到全部必要组件发布；窗口复制量为0不等于没有窗口引用。请求结束不主动删除整条共享prompt头部窗口，保留其他恢复点，压力下按真实引用和淘汰规则回收。
- 私有暂停快照复用现有有界复制提交。需要保存多个窗口范围时可以按额度分段，完成前保留整个请求原件；不能绕过复制额度，也不能因一个窗口额度小于整份暂停记录就永久无法卸载。CPU无法取得完整快照预算时明确走重算。

## 4. 分配控制必须与 VMM 接通

当前两项实现的KV空闲页是GPU tensor；DFlash窗口是GPU空闲环，CPU仅保留head/count。它们在已全量分配的池中是正确的，但VMM在提交计算前必须知道具体会访问哪些地址。

用户已确认：共享模式由CPU决定逻辑KV页／窗口槽／状态槽的归属与回收，GPU负责slot_mapping、页表及full→window表的批量更新与查表。首版不另做GPU自主全局分配器，但保留VMM共享物理块。禁止逐轮`.cpu()`／`.item()`读取GPU分配结果，再补做VMM映射；原分池保留既有分配方式。

实施要求：

1. CPU知道申请的具体编号与范围，按这些范围先计价并映射；再在正确stream上发布GPU描述符，之后才执行store／attention。
2. 请求、公共前缀分裂／共享、拒绝尾部、私有窗口释放和冷恢复都保留CPU可追踪的位置元数据。可复用同一前缀树的CPU索引载荷，不建立第二棵树；不能只在首次分配知道编号，释放时又退回GPU查询。
3. CPU元数据只保存索引和所有权，不复制KV／状态数据。按新增／释放范围批处理，优先复用已映射空单位；不逐token遍历整棵树或整条GPU空闲表。
4. 一轮内的页表、窗口映射和实际被映射块来自同一申请记录。CPU上传分配变更和稳定的有界描述符，GPU展开逐token位置并更新计算表；上传／复制读取结束前不复用host槽。
5. `free_swa`的GPU ring返回动作不能成为共享模式的唯一权威。CPU释放绑定、GPU清映射、在途事件与物理块回收保持同一顺序；不新建每轮全设备同步。

这项不是当前DFlash的bug修复，而是共享runtime必要的控制层变更。

## 5. SD 开关、资源裁剪和自适应

**关闭SD与这一轮执行AR必须区分。** 显式`steps=0`不加载drafter／不保留草稿历史；SD已启用但阶段策略、自适应或临时容量选择AR时，仍维护DFlash上下文，供后续轮次继续起草。不能因此归还仍需保存的draft历史。

batching最新默认已改为：无草稿路径／SD意图时AR，提供DFlash路径时默认上限4；显式0优先。合流保留此约定，共享runtime不恢复早期“默认self-SD”的设计。

SD＋batching的静态能力检查当前仍通过`_linear_pool_num_slots`等独立GDN预算路径判断资源。共享模式改用实际组件布局下可执行的一轮最低资源；按已配置阶段判断是否存在合法SD轮次，不要求预先放下最大并发的全部SD暂存，也不要求用户再给被禁止的GDN硬预算。

资源裁剪沿用阶段规则：legacy／波次外可逐请求缩短；波次内仍要求整批达到配置最大草稿宽度，否则本轮整批AR。不改`SpeculativeDecoder.admit(full_width=True)`与分层verify图的宽度语义，不新增波次内自适应或可变宽度调度。

每轮先形成候选，由已支持的 drafter／控制器决定是否 SD；选中 SD 后按联合预算取得资源，先回收冷缓存，不足再裁剪。选择 AR 不为草稿回收缓存。最终 Graph padding、`batch×(max(lengths)+1)`概率缓冲及 verify 状态一起计价；不只检查页／槽数。控制器记录资源裁剪后的实际执行长度，取消／恢复／重算不能冒充完成样本。现有`verify_positions/verify_physical_positions`只覆盖波次外，波次内另有phase计数和Graph形状；不能在内存验收中混用统计范围。

物理块改供另一用途不会改变数学／Graph几何，不重置控制器全部历史。明确池几何或Graph集合重建仍遵守原控制器的失效规则；暂停等待／冷恢复／重算时间另报，不直接记成对应不了执行形状的普通AR／SD样本。

## 6. 实现agent必须完成的接入

| 修改位置 | 共享模式需要改变的职责 | 继续复用 |
|---|---|---|
| `engine/config.py`、`engine.py`启动 | shared路径在旧分池容量检查／分配前解析R、E、C；不调用要求独立GDN余额的`_sd_state_shortfall`或再用旧`_fit_dflash_pages`给R重复扣费 | 组件能力校验、N0、默认和显式错误；独占权重测量 |
| `DFlashLayout`、各pool、`linear_state_pool/gdn_replay` | 真实数据按R分配，元数据／工作区按E；不根据C一次物理建满；CPU控制绑定，必要kernel stride适配 | 当前full／单window组、状态数学、Replay和已有索引接口 |
| `SpeculativeDecoder.admit/start`、CacheManager | 在任何状态复制／草稿写入前取得同一轮R与E额度，含下一次AR及在途工作；不足按阶段规则裁剪 | `start/finish`、采样、目标验证及统一提交出口 |
| `PrefillMemoryBudget`、`LayeredPipelineExecutor` | 准入传递真实可执行tile／波次大小，保证已开启波次可到安全边界；见下文 | 分层执行、独立features、正常AR／SD交错 |
| `HostTier`、`PrefixTransfer`、请求所有者 | 增加私有暂停快照和无采样恢复，同一host预算；暂停队列／在途引用进入runnable、取消、维护判断 | 公共前缀树、已有window保护与复制完成事件 |
| Graph与重建 | capture前映射scratch／dummy；保留当前捕获集合及重建上限；R内物理转用不换GraphRunner、不清窗口映射 | `full_stores/protect_capture`；真正几何重建才沿用`reset_window`及控制器失效规则 |

### 必须处理的运行期峰值

[基座性能报告](sd-batching-performance-20261008.md)已有实际失败：Qwen3.6 NVFP4、hybrid、C上限8、DFlash4/outwave、Replay、prefill8192、目标98304 tokens、GDN 6e9；后续prefill中申请64 MiB失败。报告尚未隔离完整峰值根因，不能说只是漏算features，也不能声称新窗口已经修好。

现有`PrefillMemoryBudget.token_budget`在未测量时直接放行配置tile，测量后仍以`max(tile_tokens, ...)`兜底；其`record`只测PyTorch，并且波次内SD的`start`在`memory.start`之前。**这些测量可用于估算，不能作为共享模式的容量保证。**

实现要求：

1. 根据组件真实shape计入同时存活的波次hidden／residual／features、GDN prefill临时结果、草稿概率、验证logits和Graph padding；已保留部分只计一次。不能将各阶段峰值取最大而漏掉跨阶段保留项，也不把互斥临时项全部相加。
2. 无法放下配置tile时缩小实际tile／准入波次。`max_extend_tokens`是上限，不是必须执行的下限；不能继续强行放行8192或靠OOM后重试。最小合法tile仍放不下时先回收／暂停，单请求不可执行则按启动／长度契约报错。
3. 波次开启时保证到下一安全边界所需R和E，decode／SD不得花掉这部分额度；SD先于层组workspace测量产生的保留张量也要计入。不得暂停半完成GDN状态以临时腾位置。
4. R内的空物理块仍属于R，不会自动成为PyTorch执行余量；缩短草稿／暂停只在确实释放对应E占用时解决执行余量不足。Graph专用保留空间也不能凭`reserved-allocated`一律当成可复用workspace。
5. 将这组长历史、后续prefill、混合decode负载列入共享模式验收。保持请求／输出预算、专家池和总GPU预算，旧分池硬配额换成R；允许设计内的动态tile／准入／暂停，不减少到达请求、不缩短输入、不关闭DFlash来宣称修复。记录生效C及真实SD执行。此处不宣称原分池模式的旧配置已被修复。

## 7. 必须有的组合证据

- 同一R与专家池，AR-only、self-SD、DFlash固定各自报告；legacy/offload自适应验证资源裁剪与恢复正确性，不要求重做其策略优化。不能以开启后全程AR通过SD验收。
- DFlash compact开／关、native窗口与显式cap、窗口全部有限／仍有无限层，分别核对真实占用与恢复历史范围。
- layered波次内AR但波次外SD时，波次内仍维护draft上下文；窗口推进与多token提交后能继续真实SD。
- 开启SD后暂停／CPU恢复／重算，再产生新输出；拒绝尾部、stop与取消没有草稿窗口泄漏，缓存关闭的私有恢复也能执行。
- 先产生大量窗口／状态占用，再释放供目标KV使用，反向亦然；在途复制保护和工具锚点开启时仍保证推进。
- 对metadata、概率／logits峰值、保留features、原生物理块分别核算，证明没有重复扣钱、未申报分配或误报虚拟容量。
- 小批次AR及SD图覆盖、显式图上限3/5/7和真正重建后的上限保持；日常跨类型转用不重录Graph。保留当前统计口径，不把Graph新增reserved量当其总占用。

上述实现与验收落实主设计P0–P5。多窗口分组、layered自适应、模型embedding／输出头接口泛化及self-SD提速仍是另外的任务；不为它们建设预留抽象。
