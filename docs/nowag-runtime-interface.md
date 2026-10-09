# NoWAG 接口与集成约定

状态：设计接口，尚未实现。配套[主设计](nowag-runtime-design.md)。本文供实现者／协调者，黑盒作者只读[公开契约](nowag-runtime-public-contract.md)。

## 1. 模型提供数学，格式提供权重与执行

在 `moe/expert_format.py` 定义小型数据记录，不建立类继承树或自动发现系统：

- `ExpertMath`：`activation`、`activation_alpha`、`activation_limit`、`router_weight_on_input`、`gate_up_input_rounding`、`down_input_rounding`、`down_norm_placement`。字段来自真实组件，模型没有的参数使用None，不推测模型名字。
- `ExpertLayout`：`format`、`d`、`assignment_bits`、`assignment_layout`、全局H/I/E、`moe_layer_ids`、`tp_rank`、`tp_size`；每个投影的全局／局部形状、全局分片起点、首组有效lane及物理bank布局。非NoWAG记录只带其实际使用的格式参数。
- 继续使用 `ExpertBanks` 表达 `sources`，增加 `layout` 和 `shared`。`shared` 是具名只读张量，例如NoWAG的 `codebook`；所有者和设备副本由公共初始化负责。现有NVFP4 alpha也通过实际作用域明确归属，迁移时保持其逐专家索引语义。

`sources` 是具名bank到逐MoE层tensor的映射；每个tensor以expert为最外维。bank表只描述物理布局，不决定router、调度或预算。bias作为逐专家数据与对应权重一起索引。

原生加载与FTW都先完成模型映射，再产生同一记录。源checkpoint的非专家精度先独立解析，设置NoWAG专家格式不能覆盖原本的attention／共享专家精度字段。

## 2. 绑定和执行

共同入口：

```python
bind_expert_method(math, layout, *, device, backend) -> method
method.workspace_spec(rows, top_k) -> dict[str, tuple[tuple[int, ...], torch.dtype]]
method.run(x, expert_rows, route_weights, banks, shared, *, workspace, out) -> out
```

这里的 `method` 是三个操作的已绑定实现，不要求每个格式写一个类。`backend` 使用FreeToken原有后端选择结果；NoWAG GPU具体kernel由设备／layout／math和batch形状决定，CPU走既有C++ executor。

各实参含义：

- `x`：连续BF16 `[T,H]`，路由权重尚未施加；TP输入H保持模型公开分区语义。
- `expert_rows`：连续int32 `[T,K]`，**直接索引当前传入bank的行**。GPU缓存映射在调用前完成；CPU选择当前层主机bank后，逻辑专家编号即其行号。
- `route_weights`：连续float32 `[T,K]`，由原router提供；执行方法不能再选专家或重新归一化。
- `banks`：当前层或当前GPU缓存的具名tensor视图；NoWAG九个基础bank及必要bias，不传整个模型对象。
- `shared`：当前设备的具名共享tensor，NoWAG要求 `codebook`。
- `workspace`：严格按 `workspace_spec` 分配的连续tensor映射。内容在调用前不保证为零，实现必须初始化会读的元数据。公共执行域管理其存活和复用。
- `out`：连续BF16 `[T,H]`，调用前内容任意；不得与x、banks、shared、workspace重叠。函数写满有效结果并返回同一存储，不另分输出。

`T=0`返回空out且不启动无意义计算。真实padding路由允许 `expert_rows=-1`，对应贡献为零；不能借零权重访问无效行。host/device、shape、workspace容量及不支持的math/layout在可判定阶段报错。

此接口是实际engine调用面，不为测试另做kernel包装。CPU异步提交仍由公共executor负责；`run`定义的是计算与输出契约，不把Python函数放入CUDA Graph的主机回调。

TP时 `run`返回本rank的局部贡献；仅rank0贡献down bias，公共调用者进行既有all-reduce。中间维边界mask来自layout，不由kernel重新推测分组。

## 3. 初始化与资源顺序

1. 解析模型组件、原权重元数据和用户配置；检查NoWAG产物的模型对应关系。
2. 读取共享codebook与专家布局，解析TP切片；读取／重排压缩bank并保留必要bias。
3. 绑定实际kernel和数学能力；准备编译与已存在的实测profile。不可用组合在ready前报错。
4. 报告每专家物理字节、共享参数大小、workspace需求。公共预算确定槽数／执行形状，不能反向由格式模块抢占预算。
5. 公共管理器建立resident或offload存储，分配每个实际并发执行域的workspace；安装共享参数。
6. warmup、捕获相应Graph，再发布ready与实际状态。

重建沿用同一顺序中的资源准备／重绑／重新捕获步骤；不重新读取离线模型或训练codebook。异步旧资源必须在公共完成点后释放。

## 4. 能力检查与默认值

能力检查只包含已有消费者需要的问题：当前math/layout是否可算、设备是否可用、是否可capture、是否支持该TP分片、CPU是否可计算、所需workspace。查表／profile选择允许按shape和数学身份缓存，不按模型名缓存支持结论。

`auto`沿用既有FreeToken政策；新增格式信息使其准确计价，不新增自动转BF16／自动改变驻留模式。kernel内部的既有CUDA／Triton选择保留并发布实际结果；需要改变已确认的回退政策时另行说明并取得确认。

全驻路径绕开slot cache，仍使用相同bank和执行方法；offload／cpu／hybrid共享原缓存、预取与线程池。显式NoWAG保持指定权重，不能为了通过能力检查读取BASE里的原始专家替代。

## 5. 迁移清单

| 现有位置 | 最终职责 |
| --- | --- |
| `engine._adjust_config` | 调公共能力解析；不保留NoWAG模型名单与独立CPU／SD格式排除 |
| `make_moe_layer`及模型专属专家构造 | 提供ExpertMath／几何；绑定格式方法；保留router与公共通信 |
| `OffloadMoELayer._expert_gemm`／resident分支 | 使用同一已绑定执行方法；保留外部搬运流程 |
| `_BANK_SCHEMAS`／`_PROVIDERS` | 迁移为各格式的单一声明与加载入口，公共代码不重复维护同义表 |
| `CpuMoeExecutor` | 保留调度／握手；NoWAG bank解析和数学适配放到其格式模块；原生计算仍在公共扩展构建中 |
| `speculative_graphs`／SD控制检查 | 查询目标方法与attention／drafter能力；费用按实际行字节计 |
| FTW转换／读取 | 保存、恢复布局和shared；使用同一逐层sink及公共存储 |
| 构建与包数据 | 打包NoWAG所需Python、CUDA头／源、实测profile；清洁安装可用 |

只移动或抽取真实在线调用闭包；需要新数学能力时新增对应grouped计算。不要把整个离线插件复制进FreeToken，也不要为“统一”提前改写其他量化算法。

## 6. FTW与对外状态

FTW新增的NoWAG数据使用现有tensor存储和metadata机制：`quant_format="nowag"`、`expert_layout`记录逻辑编码／层映射，`expert_shared`列出具名共享tensor条目；bank继续逐层保存。codebook只写一次，必要bias不遗漏。旧非NoWAG FTW的字段语义不改变。

NoWAG FTW存全局编码，不存转换时的TP局部切片；运行时按目标TP生成layout。`BASE`配置／tokenizer／非专家权重一并保留，源NoWAG绝对路径只可作来源说明，不能成为加载依赖。

状态接口固定见公开契约。状态里的物理字节来自实际分配，不能把逻辑压缩率换算值当实际显存；多rank分别报告，不能用某rank乘TP假定所有rank相同。
