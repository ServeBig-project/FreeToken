# NoWAG 在线接入：独立公开验收契约

状态：验收契约；实现已进入独立验收，未宣称完整矩阵通过。2026-10-09。

本文可独立交给黑盒测试作者。只提供本文、公开模型数学／权重格式、CLI/API、固定输入产物和获准的运行资源；不得提供生产源码、实现设计、diff、内部测试或实现笔记。测试作者不修改生产代码；实现者不读取黑盒源码。

## 1. 范围与配置

NoWAG 只替换路由专家的三组投影权重；模型的路由、激活、bias、共享专家、attention、KV、采样和请求调度语义保持原定义。离线校准／量化／训练与 GGUF 不在本轮验收内。

公开入口：

```bash
ft serve --model BASE --nowag-expert-path SIDE --moe-backend offload ...
ft checkpoint --model BASE --nowag-expert-path SIDE --out DEST ...
ft serve --model DEST --moe-backend offload ...
```

Python 使用现有 `freetoken.llm.LLM` 的相应参数；HTTP 继续沿用现有服务 API。省略 NoWAG 路径且加载非 NoWAG checkpoint 时保持原行为。重命名有效目录不改变解析结果；FTW 源目录隔离后仍可独立运行。

BASE也可为本项目已有FTW。显式SIDE决定路由专家投影来源，不能被BASE保存的专家覆盖；非专家和模型必需bias仍来自BASE自身。无SIDE时按FTW自身的格式与权重加载。适用于已有专家bank和全驻普通weight布局；不能因替换专家而忽略未知或不匹配的非专家权重。

本轮目标格式为专家共用一个 codebook、BF16激活／codebook／normalizer、12-bit assignment，D=4或6。量化位宽由权重元数据决定；不得偷偷改精度、重新训练或把所有专家展开成 BF16。

`--moe-backend fused` 全驻 GPU，`offload` 按需搬运，`cpu` CPU decode＋GPU prefill，`hybrid` CPU/GPU 合作，已有 CPU 层配置继续可用。各模式只在模型及公共组件支持时组合；显式配置在 ready 前得到实际能力检查。

支持范围按专家数学、形状、设备和公共执行组件决定。不得新增模型名白名单；同时不得跳过 checkpoint 与目标模型的一致性验证。

## 2. 独立输入与数学参考

### 2.1 NoWAG v1 格式

原生目录包含 `manifest.json`、一个全局 codebook 文件、各 MoE 层的 tensor 文件及 manifest 引用的索引。已有通用 `nowag_expert_sidecar_v1` 和历史 `deepseek_v4_nowag_expert_sidecar_v1` 输入继续可读。

通用 manifest 至少声明：`format`、`scope="expert_only"`、`codebook_sharing="global_all"`、`d`、`assignment_bits=12`、`assignments_packed=true`、模型类型／逻辑几何、`matrix_count`、`codebook` 位置以及 `layers` 的实际层编号与文件。模型几何应与基座一致，真实 MoE 层不必覆盖所有 decoder 层。

对逻辑投影 `[N,K]`：

- `C` 是 BF16 `[4096,D]`；codebook tensor 名为 `global_all.codebook`。
- 每行有 `ceil(K/D)` 个 assignment。每个 id 占12bit，按 id 顺序从低位写入 int32 流，跨32bit边界时延续到下一 word；每个输出行重新从bit0开始。
- `row_major` 存 `[N, ceil(ceil(K/D)*12/32)]`；`word_major` 是这两个轴交换，数值语义不变。缺省布局沿用已有 row-major 语义。
- 每个投影有 BF16 `input_norm[K]` 和 `output_norm[N]`；尾部 codeword 的无效输入 lane 不参与结果。
- 层文件键沿用 `layers.L.ffn.experts.E.wP.assignments`、`...wP.normalizer.norms.0`、`...wP.normalizer.norms.1`；`w1=gate`、`w3=up`、`w2=down`。
- bias 若模型需要，从 BASE 的对应专家参数读取并保持原值；原生 NoWAG 不含 bias 不能被解释成零。FTW 必须带齐。

独立作者可依据上述定义构造小型合法权重，不调用生产 encoder／decoder。小模型遵循模型公开结构，覆盖真实 H／I 尾部及 TP 边界；不以不存在的数学族制造反例。

### 2.2 数学语义

未舍入的单投影参考是 `((x * input_norm) @ C[A].T) * output_norm + bias`。实际计算按模型公开激活／舍入定义执行：

- 保留 gate/up 后的激活、clamp、bias 和乘法顺序；普通 SiLU、GELU、tanh-GELU、GPT-OSS／MiniMax 带参数门控均是不同参考。
- 路由权重乘在输入或输出的位置由原模型规定；不能跨非线性移动。
- DSV4 保留 gate/up 输入与 down 输入的 E4M3 分组舍入；down normalizer 位于该舍入之后。路由权重按 DSV4 参考乘在 down 输入上（激活之后、down 输入舍入之前），down 输出不再乘权重。其他模型不能因为用了 NoWAG 就得到此额外舍入。
- CPU、GPU、全驻、卸载、Graph、TP 均与同一组压缩权重的独立参考比较。NoWAG 与原未压缩模型的质量差异另行报告，不要求两者逐字相同。

在看到候选结果前冻结容差，依据相同 dtype／舍入的参考与基准 kernel 确定，记录最大绝对／相对误差及累计误差；不得遇到失败后扩大容差。语义正确的精确小样本还应逐值核对。

## 3. 公共执行模式的边界

| 维度 | 本轮要求 |
| --- | --- |
| 权重 | D4/B12、D6/B12，两种既有布局；原生与 FTW；相同 checkpoint 重命名 |
| 专家计算 | 全驻 GPU、offload、cpu、hybrid，以及已支持的逐层 CPU 分配 |
| 搬运 | 冷／热缓存、按需 miss、整层 streaming、重叠预取、命中 D2D、驻留层组 |
| batching | 对该模型公开支持的 legacy／mixed／layered／layered-pipeline（joint 已废弃，不在本契约内）；对应合法配置与拒绝行为 |
| Graph | eager、普通 decode 图、已有层段图；反复回放时换 token、路由、前缀长度和自然尾批 |
| SD | self-SD 和匹配 DFlash；fixed与已有 adaptive／缓存起草／补缺／预取控制；N=1/2/4/8及真实尾部裁剪 |
| TP | TP1、真实 TP2；合法分片及 D6 边界组，原生 checkpoint 各 rank 使用同一份；FTW 只支持 TP1，TP>1 在 ready 前报公开错误 |

不是所有行都能笛卡尔组合。主线 SD 的 TP=1、全 CPU 专家 SD 的拒绝、attention／状态／drafter限制、legacy-only 控制项、并行 prefill 的公共限制继续适用。格式本身不能再排除一个已具备所需组件能力的合法组合。

新 attention／模型的基座未交付时，不能要求 NoWAG 绕过检查。Flash-Next 需先有合格基座与匹配权重；其形状／专家数学可独立做组件验收，不能据此声称已通过真实模型服务。

## 4. TP 的公开结果

TP 只改变执行分区，不改 codeword 编码与模型数学。gate/up 输出切片，down 输入切片；跨 rank 边界的 codeword 只计各 rank 的有效 lane，不能重新从局部索引0分组。

必测形状包含项目已有 I=512、D6、TP2（边界256），以及 I=2048 的 DSV4 型投影；Flash-Next 型 I=640／H=2560 在合法组件夹具中覆盖。gate/up bias切片、down bias恰好加入一次、路由权重作用位置一致。

TP1/TP2 输出与各自独立参考在冻结容差内一致，无遗漏、重复专家贡献或采样／终止差异。真实双卡必须记录 rank／设备／collective 执行；不得用单进程切片计算冒充通信验收。

## 5. 输出缓冲、并发与资源

除CLI／HTTP外，engine实际使用的公共专家计算入口固定为 `freetoken.moe.expert_format.bind_expert_method(math, layout, format_state, device=..., backend=...)`，返回对象提供：

```python
method.workspace_spec(rows, top_k, bank_rows=...)  # dict: name -> (shape_tuple, torch.dtype)
method.run(x, expert_rows, route_weights, banks, shared, workspace=..., out=...)
```

`math`是原组件数学记录 `freetoken.moe.expert_format.ExpertMath`：activation、activation_alpha／activation_limit、router_weight_on_input（乘在 gate/up 输入）、router_weight_on_down_input（乘在 down 输入）以及gate/up与down输入舍入（`None` 或 `E4M3_GROUP128_UE8M0`）。`layout`含格式名和全局H/I/E；bank张量提供实际形状／dtype，MoE层映射及TP分片由公开加载入口完成。`format_state`由该入口返回，具体格式负责解释。独立作者通过合法模型和上文权重文件取得这些公开输出，不读取实现私有对象字段；参考数学直接来自本契约的权重定义。

`rows`包含Graph padding；`bank_rows`是本次kernel实际可寻址的专家行数，缓存模式取实际槽数，全驻／整层通常取逻辑专家数。空间查询没有分配、编译和文件读取副作用；改变缓存容量后必须使用与新几何对应的workspace及图。

`x`为连续BF16 `[T,H]`；`expert_rows`为连续int32 `[T,K]`，直接索引当前传入bank；`route_weights`为连续float32 `[T,K]`。`banks`是当前层或缓存的具名权重tensor，`shared`包含当前设备codebook；workspace按返回规格分配。`out`为连续BF16 `[T,H]`，不与其他实参重叠。TP调用返回本rank贡献，公共通信归约；仅rank0贡献down bias。格式绑定和workspace建立在Graph捕获之前。

- out写入调用方指定的缓冲，返回同一输出存储；T=0返回空out。不得覆盖输入和codebook／assignment／normalizer；不得依赖workspace或out原先为零。
- 基础非重叠 out／workspace 是必须路径。若不支持别名须在执行前明确拒绝，不能静默给出错误结果；不能为测试任意扩展支持的别名组合。
- padding、无效路由及空有效子批次产生确定输出，不读越界槽、不保留上次回放残值。只构造公共调度能产生的 padding／尾批。
- 公共池计入压缩专家行、bias、codebook、workspace、Graph存储和对齐。每 rank 一份 codebook，不随 cache slot 数复制。
- offload 流水线保持压缩数据搬运；cpu／hybrid 无全量 BF16 专家副本。局部 tile／中间激活不是全量权重副本。
- 缓存小容量、边界容量、大容量，以及每模式公开最小容量上下边界均覆盖；不为避免失败擅自增加预算。
- 异步复制期间的输入／源权重／输出存活，多个合法执行域的临时区隔离，取消／空闲／重建后资源回收，都通过公开行为验收。
- 空闲调整专家／KV／状态容量后继续生成；忙时维护按原协议拒绝或等待。NoWAG 不创建第二套维护接口。

新建实例、退出实例、反复加载与重建后，不能持续累积共享 codebook、旧图或失效的锁页权重。

## 6. HTTP、状态与可复现失败

保留流式／非流式、stop／EOS／max_tokens、usage、温度与采样、取消、cache_group和多轮会话语义。正常 stop、长度截断与错误必须正确区分。

确定性HTTP对照显式设置 `temperature=0, top_p=1`，避免继承模型采样默认值；流式／非流式比较使用独立冷cache_group并核对usage。温度采样另测，不能通过改变用户显式采样参数或放宽文本／数值阈值消除失败。

`/v1/cache/status`在原有geometry中增加 `experts`，含 `format`、`format_parameters` 和 `ranks`。NoWAG的 `format_parameters`包含 `d`、`assignment_bits`，由格式模块提供，公共状态层只转发。每个rank记录 `rank`、`device`、`compute_backend`、`kernel_backend`、`storage_mode`、`expert_host_bytes`、`expert_device_bytes`、`shared_host_bytes`、`shared_device_bytes`、`workspace_device_bytes`；按实际存储计量，不把共享codebook重复算进每个专家。多种kernel时 `kernel_backend`为实际已绑定名称的列表。

Graph与SD的实际执行继续由既有配置、Graph／speculative统计和独立profiler证据判定；仅发布“支持”不证明本轮使用。上述状态在初始化／重建后更新，读取状态不改变执行。不要求逐token日志或新性能分析服务。

以下是必须覆盖的公开错误：不匹配的权重几何／层映射、缺少共享 codebook或必要bias、无专家模型使用该选项、未实现的数学／设备组合、不合法TP分区、非法资源容量、FTW缺必要数据，以及公共组件本身不支持的组合。错误在可判定的最早阶段报告，配置问题在 ready 前报告；不能悄悄改格式、改模型数学或改成另一后端后宣称通过。

不沿用按目录名／模型名猜测的支持范围。改目录名不影响有效输入；将错误模型权重换个目录名不能使其通过。

## 7. 独立验收组织

每一阶段至少报告：实现commit、公开入口／依赖、固定权重及模型、实际设备、支持配置、逐项结果、失败复现命令和未完成原因。测试 commit 与实现 commit 分离。

| 验收组 | 最少覆盖 |
| --- | --- |
| CPU 格式／数学 | D4/D6、布局、尾部、bias、门控族、层映射、错误输入；不调用生产解码器作参考 |
| GPU 算子 | 真实 H/I/top-k、B=1/4/16与非整批尾部、输出缓冲、压缩权重数值、图回放 |
| 缓存／调度 | 冷热、容量边界、预取重叠、不同层／请求的路由、驻留组、长短混合与取消 |
| SD | 正常／零接受、不同草稿长度、拒绝后状态、范围图、尾批、合法控制项与错误组合 |
| TP／FTW | 双卡边界组、bias与归约；转换后隔离原目录、FTW 配 TP>1 被拒绝、重建与复用 |
| 真实服务 | 现有Qwen3.6和DSV4权重各覆盖其合法执行模式；新模型仅在匹配产物／基座齐备后计为通过 |
| 非NoWAG回归 | 受公共边界修改影响的BF16、NVFP4等已有配置，尤其默认后端、精度和缓存行为 |

不同专家数学族、真实模型、执行模式分别列出证据；不要求所有维度盲目全组合，但每一公共路径及风险交叉都必须有明确覆盖。模型别名测试不能代替不同数学族。

对重构前已支持路径，使用相同 NoWAG 权重／设备／输出工作量，三个交错配对 block，记录TTFT、decode及端到端耗时、HBM和搬运字节；默认中位延迟回退门槛5%。共享背景污染时标未完成，不选择有利轮次。新增路径报告绝对值与合理对照，不能承诺量化模型必然快于原模型。

缓存状态一致的确定性对照可逐token比较；不同算子／TP的近平局差异需按预先冻结数值协议解释，并报告任务级输出，不以“量化本来有误差”豁免乱码、循环或稳定错误。

## 8. 尚需协调者提供的材料

以下运行材料由协调者按受测版本提供；材料已准备不代表对应验收已通过：

- 每phase实现commit、公开入口的可运行环境；接口与字段以上文为准，若需改变须同步契约并说明影响。
- 合法独立小模型参考、真实NoWAG产物、DFlash配对权重、可复现运行环境。
- 获准的单卡／双卡时间、CPU／内存预算；没有资源就报告未运行，不占用其他实验。
- 共享runtime／Flash-Next合入后的最终基座及已知限制；不能用旧快照假定通过。

所有必须矩阵完成、生产代码整理完毕、相对功能开始前基线的生产／测试代码量分别核算后，才可称达到可交付标准。CPU检查、HTTP冒烟、设计合入均不能代替这些条件。

## 9. 实施期补充说明（2026-10-09）

以下回答黑盒作者提出的公开问题，与上文同等有效。

- **取得 bind 输入**：先 `freetoken.distributed.set_tp_info(rank=0, size=1)`（每进程一次；engine 启动时由它设置），再 `cfg = freetoken.engine.config.EngineConfig(model_path=BASE, nowag_expert_path=SIDE, tp_info=DistributedInfo(0, 1))`；`banks = freetoken.moe.expert_banks.load_expert_banks(BASE, cfg.model_config, device=..., dtype=torch.bfloat16)`。`banks.sources[name][l]` 是第 `l` 个 MoE 层的 `[E, ...]` 主机张量，行号即逻辑专家号；`l = 解码器层号 − first_k_dense_replace`。`banks.shared` 是共享张量（`codebook`），`banks.format_state` 传给 bind。`layout = ExpertLayout("nowag", H, I, E)`；`backend` 为 `--moe-backend` 取值（`offload`、`cpu`、`hybrid`、`fused`）。bank 名：九个基础 bank（`{gate,up,down}_{assignments,input_norm,output_norm}`），模型有专家 bias 时另有 `gate_bias`、`up_bias`、`down_bias`（`[E, I]`/`[E, H]`）。
- **布局**：manifest 的 `assignment_layout` 为 `row_major`（缺省，每专家 `[N, W]`）或 `word_major`（每专家 `[W, N]`）。
- **层覆盖**：sidecar 必须恰好覆盖基座的全部 MoE 解码器层 `[first_k_dense_replace, num_layers)`，层号为解码器层号；部分覆盖拒绝。MTP 层不在其中。
- **状态**：NoWAG 的 `format` 为 `"nowag"`，`format_parameters` 为 `{"d", "assignment_bits"}`；其他格式报实际绑定格式名（如 `bf16`、`fp8_block`、`nvfp4_marlin`、`ds_fp4`），`format_parameters` 为 `{}`。`workspace_device_bytes` 是 decode 流（decode、SD 起草与 verify）预留的专家临时空间，按 `max(并发上限, 图最大批) × (草稿步数 + 1)` 行和 `workspace_spec` 计算，计入共享 runtime 的执行额度或分池模式的固定成本；prefill tile 的临时空间按次分配，计入实测的 prefill 峰值。非 NoWAG 格式为 0。
- **容量**：沿用现有 CLI 语义（`ft serve --help`）；`--moe-cache-size` 以专家槽计（一槽 = 一层的一个专家的全部 bank）。非法容量以 ready 前的公开错误为准。
- **不支持的组合**：NoWAG 不支持路由权重乘在 gate/up 输入、上述以外的激活或舍入。gpt-oss 全驻 GPU 和 CPU／hybrid TP 均纳入本轮要求；实现完成不代替各路径的独立验收结果。
- **数学族与模型**：SwiGLU-OAI（MiniMax-M3 `swigluoai`、gpt-oss）先令 `g=clamp(gate, max=L)`、`u=clamp(up, ±L)`，结果为 `g * sigmoid(α·g) * (u + 1)`；tanh-GELU 对应 Gemma4；erf-GELU 目前无已注册模型，不要求覆盖。
- **padding**：CUDA Graph 补齐行的路由 id 为 `-1`，`run` 对其贡献为零；这是公共调度实际产生的输入。
- **合成 gpt-oss sidecar**：沿用通用 v1 键（`format="nowag_expert_sidecar_v1"`、`model_type="gpt_oss"`、`hidden_size`、`moe_intermediate_size`、`num_experts`、`num_moe_layers`）；sidecar 不含 bias，bias 取自基座。
- **无公开手段**：强制 SD 零接受、观测 collective 执行均无公开入口；不能构造时如实记为未覆盖。
- **设备**：FreeToken 需要可见的 CUDA 设备。加载会把主机 bank 锁页（需要 CUDA），`method.run` 在 CUDA 张量上计算。CPU 上的专家计算由 engine 的 CPU 执行器完成，只经 `--moe-backend cpu`／`hybrid`／`--moe-cpu-layers` 的公开服务验收，不通过 `run` 调用。无 GPU 时这些测试记为未运行。
- **activation 字符串**：`silu`、`gelu`、`gelu_tanh`（或 `gelu_pytorch_tanh`）、`swigluoai`（或 `gpt_oss_swiglu`）。clamp 不是单独的激活名，由 `activation_limit` 表达。
- **DSV4 的 `ExpertMath`**：`activation="silu"`、`activation_limit` 取基座 config 的 `swiglu_limit`、`router_weight_on_down_input=True`，两处舍入均为 `E4M3_GROUP128_UE8M0`。

## 10. 服务验收与 TP 调用补充（2026-10-10）

- **合法服务组合**：Qwen3.6 的 AR/offload 支持 legacy、mixed、layered、layered-pipeline；layered-pipeline 使用 Triton prefill。SD 保持公共组件已有的 legacy／layered-pipeline 限制。DSV4 的 AR/cpu 与 AR/hybrid 属于要求通过的路径。Qwen3.6 和 DSV4 基座加载器仍为 TP1；TP2 服务使用支持 TP 的合法模型。合法配置启动失败是失败，不能改记为跳过。
- **专家缓存下限**：令 E 为每层专家数。legacy／mixed 无重叠为 E、开重叠为 2E；layered 为 3E；layered-pipeline 为 2E。cpu 使用固定 2E，不由手动专家缓存参数改变。检查上下边界时保持其他资源足够。
- **维护**：`GET /v1/cache/status`；`POST /v1/cache/rebuild`，请求使用 `mode="if_idle"` 与 `timeout`（秒，默认300）。分池模式可调 `moe_cache_size`、`num_pages`、`num_mamba_slots`、`num_swa_pages`；共享模式可调 `runtime_cache_gib`、`moe_cache_size`。忙时返回 `status="busy"`，非法容量返回 `status="rejected"` 并保留原服务；其他 mode 返回 HTTP 422。没有 `drain` 模式。
- **实际执行**：`GET /v1/stats` 的 `cuda_graph.target_decode`、`draft`、`verify`、`verify_range` 是实际回放计数，`cuda_graph.speculative_eager` 下对应字段是 eager 次数。以请求前后差值确认，不能只看 enabled；`replay_shapes` 提供实际形状。`execution.effective.batching_policy` 为实际策略，`execution.fallback_reasons` 说明回退；`requests.active` 为未结束请求数，忙时维护需先观察其大于0，完成后归零。`speculative.draft_tokens`、`accepted_draft_tokens`、`verify_steps` 等沿用现有 [DFlash 公开契约](dflash-public-contract.md)。
- **TP2 的 bind 调用**：两进程各先 `torch.cuda.set_device(local_rank)`、`set_tp_info(rank=rank, size=2)`，再以 `tp_info=DistributedInfo(rank, 2)` 构造 `EngineConfig`。两进程读取同一 BASE／SIDE；`load_expert_banks(..., device=torch.device("cuda", local_rank), dtype=torch.bfloat16)` 返回本 rank 的 bank 和格式状态。`ExpertLayout("nowag", H, global_I, E)` 保留全局 I；输入 x 保留完整 H，路由与权重两 rank 相同。`run` 返回本 rank 贡献，由调用方执行公开 all-reduce，down bias 仅 rank0 加入。测试使用获准的两个 GPU UUID 固定可见顺序，local_rank 为0或1；由测试建立 NCCL 通信，不把单进程切片当作 TP2。真实服务再检验 engine 的公共通信路径。
- **共享预算读数与联合维护**：正常完成短请求后，`geometry.runtime_cache_bytes` 与 `prefix_cache.runtime.budget_bytes` 发布runtime物理预算；后者另有 `granularity_bytes`、`free_bytes`、`held_bytes`、`used_bytes`、`waste_bytes`、`idle_bytes`、`protected_bytes`、`evictable_bytes`、`execution_bytes` 及逐组件 `components`。`geometry.unit_bytes.moe_per_expert` 为每个专家槽的实际字节，`geometry.cache_budget_bytes` 为扣权重后的总缓存预算上限，不是实时空闲量。API不提供单一固定成本或实时GPU空闲字段，可用获准GPU的 `nvidia-smi` 总量／空闲读数作外部观测。同次减小runtime总量、增加专家容量，只要目标总预算与执行边界合法，应重建成功并继续服务；不能因中间分配顺序而OOM。非法目标仍须在破坏原服务前拒绝。
