# Flash-Next 上游移植清单

用途：第二版源码取用地图。配合[主设计](flash-next-design.md)及[精度／主机KV设计](flash-next-kv-tier-design.md)执行；不得交给独立黑盒作者。本文路径除明确绝对路径外均相对FreeToken根目录。

## 1. 固定来源

- 我们的起点：`732f1ee39c6c1046d760f72645e8e2b970ae9f33`。
- 上游冻结点 U：`9b585b72d0838e0e977e029c427a827029397a8d`。
- 上游首次接入 A：`bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40`，[原提交](https://github.com/FlashML-org/FreeToken/commit/bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40)。其 Python 生产差异为42文件、+5877/−66；这是依赖地图，绝非建议整颗引入的最小补丁。
- 源权重：`/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4`，RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`。专家NVFP4、PLE FP8、dense BF16；显式dense FP8策略在加载／FTW转换时处理。依据见[本地证据](flash-next-resource-evidence.md)。

已有本地 Git 对象可直接 `git show <commit>:<path>` 读取。只在对象缺失时 fetch 对应 upstream 引用；fetch 不改变 main 工作树。不要执行整目录 `git checkout upstream/main -- python/freetoken`，也不要更新正在开发的 runtime 工作树。

```bash
git show bd8f3d5 -- python/freetoken/models/qwen4_exp
git diff bd8f3d5 9b585b7 -- python/freetoken/models/qwen4_exp
git diff bd8f3d5 9b585b7 -- python/freetoken/attention/qsa_sparse.py python/freetoken/kvcache/qsa_pool.py
```

先阅读差异的目的：找出已有修正及新增依赖，决定哪些实际用于已选 checkpoint；不是反复重审已确定的模型数学。取源码时保留原 SPDX、版权和来源，修改之处按普通生产改动审查。

## 2. 文件级取用

| 上游位置 | 取用内容 | 本地接法／禁止覆盖 |
| --- | --- | --- |
| `models/qwen4_exp/config.py` | 几何、PLE一基转零基、QSA组、输出gate、零中心norm | 使用本地ModelConfig；精度交给公共量化模块，不在此按模型名指定FP8 |
| `models/qwen4_exp/model.py` | 四路 decoder 的 mix／compute／combine 与最终 mixer | 实现本地 layer-group 方法；不复制其整模型 forward 的全局 PLE prefetch／末尾 commit 顺序 |
| `models/qwen4_exp/hc.py` | 四路残差数学与投影融合 | 对接独立Linear方法；融合对应scale，保留一次 `1+w` |
| `models/qwen4_exp/attention.py` | QSA 权重、双宽 q 投影、index原始投影、部分RoPE | 后端接当前 Batch／metadata；text-only不带三轴视觉位置 |
| `models/qwen4_exp/ple.py` | 模型哈希、scale、查询、gate及9步卷积 | 内存 lookup；历史更新与快照接我们的 FLAPathMetadata；删无调用方的磁盘／视觉／跨批次 pending 分支 |
| `models/qwen4_exp/weight.py` | dense融合、专家键规则、PLE分片读取 | A版匹配本地RadixArk布局；量化由公共计划处理，接Nvfp4ExpertSourceSpec与serial／parallel／layer_sink |
| `models/qwen4_exp/gdn.py` | sigmoid 输出门控、投影顺序的对照 | 不落第二份GDN；复用本地 mixed／verify／Replay／slot-stride核心 |
| `models/qwen4_exp/moe.py` | 共享专家 gate 的数学／可选融合 | 继续调用公共 MoE；不得绕过 route_experts 或重写专家驻留 |
| `attention/qsa_sparse.py` | 压缩、打分、选组、稀疏attention、普通Graph | 接codec和按GPU驻留表读取选中K/V；新增主机gather，不能照搬全GPU寻址 |
| `kvcache/qsa_pool.py` | 逻辑K/V页与压缩index关系 | 拆分GPU索引、K/V驻留与主机句柄；pending纳入状态槽；不是把原pool注册进runtime即完成 |
| `kernel/triton/qsa/{compress,score,topk,expand,attend}.py` 及包入口 | QSA计算核心，含既有mask、int64寻址和top-k处理 | 取U的文本所需路径；指针、槽stride和padding必须匹配本地布局 |
| `kernel/triton/hc.py`、`kernel/triton/ple.py` | 残差混合与CPU表gather核心 | 取U适用实现；实际CPU表必须已pin，输出是按批次计价的BF16工作区 |
| `kernel/triton/moe_router.py`、`moe_shared_gate.py` | top-10／共享gate融合（如采用） | 有真实调用才引入；当前Torch top-10是有效路径，不把它报告成不支持 |
| `attention/linear.py`、`kvcache/linear_state_pool.py` 的 A 版差异 | 边界输入行与声明附加状态的思路 | 手工接入本地按需快照、decode/prefill分路径及共享槽布局；不得恢复上游旧ping-pong状态管理 |
| `attention/{base,__init__}.py`、`kvcache/{base,__init__}.py`、`models/register.py` | QSA类型、后端、pool、模型注册和实际字节 | 小范围增加能力条目，保留现有混合注意力、SD和默认解析规则 |
| `moe/host_banks.py::read_range_into` | PLE分片的分段读入 | 本地缺该函数；仅带实际读表依赖，不能整文件覆盖现有HostBank |
| 本地 `models/glm_moe_dsa/weight.py::_quant_fp8_per_row`、`models/quant_linear.py`、`kernel/triton/fp8_pertensor_linear.py` | 逐行FP8转换、现有W8A16算子 | 抽公共量化函数和能力选择；源／FTW／构造／计价共用计划，旧调用保留数值 |
| 本地 `moe/expert_banks.py`、`moe/nowag.py` | NVFP4／NoWAG专家provider及实际算子语义 | 与dense／KV选择分开，禁止为新模型追加名字白名单；不宣称新NoWAG产物已验收 |
| `kernel/aot_models.py` | 新模型实际专家形状（若现有发布构建要求） | 仅补512专家、H2560、I640对应项；不用加入本阶段不支持的格式 |

取A骨架不代表使用过时数学；按U差异带入适用修正。上游该快照没有本任务的INT8 KV／活跃主机K/V能力，这部分按专项设计实现。对接量化API时权重和scale都必须正确消费，不能丢scale来凑接口。

## 3. 后续提交的处理

| 提交 | 与本阶段关系 |
| --- | --- |
| `3d919e9`：FTW旁保存PLE表 | 必须移植相关 converter hook／side files；否则转换目录不完整 |
| `fa814ab`：从实际量化方案识别专家格式 | 保留其结论；本地按选定 ModelOpt `quantized_layers` 识别NVFP4，不要求移植整个QuantConfig |
| `477c860`：全仓量化架构重构 | 参考其配置／编码／方法边界，只取与本地真实调用匹配部分，不整颗重构全仓 |
| `ddd2e3a`：dense block-FP8 | 作为加载／scale处理参考；本地源本身没有这类dense，新增方案选择现有逐行W8A16，不把block-FP8硬套非128维矩阵 |
| `4c0bad3`、`f5b9700`、`d3512b4`：磁盘表及其缓冲修复 | 磁盘模式不在本阶段；不能为了拿内存lookup把磁盘线程／host-dispatch上下文全带入 |
| `08d728d`、`cade1a9`：视觉与转换修正 | 不引入视觉服务；FTW不得误读／混入视觉权重，保留文本转换必需的元数据 |
| `235e201`：YaRN与额外配置覆盖 | 保留固定checkpoint的原生RoPE；本阶段不新增上下文扩展或覆盖接口 |

## 4. 与本地 API 的明确差异

1. 本地 `FLAMetadata` 包含 `decode/prefill/verify`，上游 PLE 直接读取扁平 `cu_seqlens/cache_indices` 不可照搬。AR混合批次逐路径处理，新的边界字段加在实际所属的 `FLAPathMetadata`。
2. 本地 `LinearStatePool` 已含按需快照、短尾起点快照、Replay 和（共享池合入后）按槽布局。新状态进入现有copy／clear／views／banks；不能重建旧pool或第二份allocator。
3. 本地 NVFP4 bank 通用入口通过模型导出的 `load_nvfp4_expert_sources[_parallel]` 工作；保持 `layer_sink` 给FTW转换器使用。无需重写 provider。
4. U的Linear带 `quant_config/prefix`。本地把当前工厂接到公共量化计划，模型仍只描述算子；保留A的键映射／融合关系，但不能把BF16作为不可关闭的模型内分支。所有实际权重／scale由已解析方法消费。
5. QSA原始pending以请求表行索引，现有恢复会更换表行；改按状态槽索引后，QSA metadata里页表行和状态槽必须分开。
6. 本地基础服务默认 AR；PR #7 已移除 DFlash 对 GDN 预算的依赖。第一阶段不再修这些已解决问题，也不带入上游覆盖我们 SD／layered 的 Engine 大改。
7. `cached_load_hf_config` 已有原始JSON读取路径。新配置解析要同时接收原始 `full_attention` 和 Transformers规范化后的 `qwen_sparse_attention`，不能要求用户手改config或安装任意最新Transformers。
8. 本地已有 `GemmaPlusOneRMSNorm`、`kernel.pinned.device_ptr` 和 FP8 gather所需的 `e4m3_compat`，直接复用。U 的 `GatedRMSNorm` 对接抽取后的本地GDN门控实现；`embed_input_ids`／`freetoken.mm`／Qwen视觉mixin在本阶段替换为现有文本embedding调用，不为消除import错误引入整套多模态包。

## 5. I1 的直接执行顺序

1. 在功能worktree固定main起点和本文权重revision，记录现有依赖版本与可用硬件。只取源码和配置；下载大权重／占GPU按已有运行安排执行。
2. 添加模型配置和注册、QSA及数学模块；抽取共用GDN组件并保持旧门控。生产代码中既有调用方一并改用它，不保留废弃导入壳。
3. 接入附加状态、HF加载、内存词表、公共专家bank及独立dense策略；用显式FP8和关闭额外量化的配置得到AR入口。未接共享池的里程碑不宣称未知负载交付。
4. 将公开构造参数／输入输出和运行命令交协调者，由独立作者进行I1验收；实现者不读取测试源码。先修数值／格式问题，再做性能调整。
5. 删除临时诊断、未用参考实现及磁盘／视觉依赖；I1记录来源、验证和代码量。I2接INT8 codec，I3接共享池与主机KV，I4–I5完成分层／Graph／157K serving与交付。

验收权重是样例与冻结参考，不得在代码里写Hub ID或目录名白名单。同一支持格式和架构的合法本地目录、重命名目录及自包含FTW目录通过相同入口。
