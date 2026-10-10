# Flash-Next 接手说明

2026-10-10：用户要求交回原 agent，当前停止新增测试。**I3a／I3b已实现，单请求和三请求157K关键路径通过；第一阶段完整验收尚未完成。** FTW暂缓，SD未开始。

本文给实现者／协调者，包含实现审计信息；独立黑盒作者只接收[公开契约](flash-next-public-contract.md)与运行材料，不读本文或生产源码。

## 代码在哪里

- 正式功能树：`.worktrees/flash-next`，分支 `feat/flash-next`，[PR #10](https://github.com/ServeBig-project/FreeToken/pull/10)。生产版本 `0abb433`；后续提交只有文档。
- 本轮隔离开发树：`.worktrees/flash-next-i3b` / `feat/flash-next-i3b`，代码已合回上述功能分支并推送。保留两棵树，未强制清理任何工作树。
- `FreeToken/main` 保持 `732f1ee`，没有在主线开发，也没有合并PR。
- 失败候选 `7139b31` 保留在 `validation/flash-next-i3b-7139b31`，便于对应旧失败记录；不能把它当当前代码。

## 本轮实现

| 提交 | 内容 | 生产增加／删除／净增 |
| --- | --- | ---: |
| `2cc8ef4` | I3a：QSA载荷／scale／索引及额外状态进入共享bank；固定索引暂存单独计价；按池能力启用共享runtime；模型报告四路残差保留量 | +132／−22／+110 |
| `cb26f83` | I3b：活跃主机K/V、共用HostStore引用、选中组gather、预算不足时回收载荷、尾页／恢复生命周期、读取计数及下表修复 | +617／−63／+554 |
| `a8e1f87` | 修正已有原生NVFP4 dense被状态接口误报为BF16；计算不变 | +3／−1／+2 |

本轮三个生产提交累计 **+752／−86，净+666**，不含公共共享池合入。先以 `c0ba02a` 合入 `a189310`，再以 `f645ee2` 合入 `d912bbe` 的执行余量、波次容量、Graph临时计价与SD选择修复；没有复制共享池未提交内容。

采用已确认的方案：GPU先选组，仅把选中的主机K/V和scale取入有界显存暂存，再做attention。只有申请不足时，按**冷缓存→有完整主机副本的活跃载荷→既有暂停**回收。活跃历史、冷前缀和私有暂停副本共用一个主机预算。没有新增独立调度器或模型名字白名单。

## 审计与修复结论

| 已确认问题 | 修复和证据 |
| --- | --- |
| 分层prefill完成路径漏备份，157309输入在已公布262144范围内仍报 `context_length_exceeded` | 在完整波次提交后备份完成页；全部完成页都标为可迁移，不只新建副本的页。原输入、原R2/H4/2048已精确通过 |
| 资源不足时把尚未结束的有效备份当成没有可回收容量 | 压力回收等待对应已提交复制事件，再释放载荷；不做每步全局同步 |
| 调度流可能在已发出的gather和attention之间改变驻留标志 | 仅实际回收时等待当前计算流，再修改标志；独立源码复核关闭，Graph数值和长服务通过 |
| 把dummy页和远处尾页按相邻页计价，少算实际物理块 | 按两个页号触及的块并集计价；当前INT8几何过第64页时原先可少算48 MiB |
| 公共精度上报把原生NVFP4 dense写成BF16 | `a8e1f87`按来源方案上报；Flash-Next的auto／BF16／FP8构造、加载和上报也经复核一致 |

独立源码审查还覆盖了HostSeries引用、冷恢复、非64对齐尾页、取消、重建和host页计数，未发现其他确认缺陷；未发现I3b需要删除的明确死代码或无用抽象。**源码复核不能替代尚未跑的暂停／恢复与配置矩阵。**

## 已完成的独立验证

- I3a `c0ba02a`：88个服务请求通过，实际暂停2次、重算2次；不是当前所有配置的证明。
- 原失败精确复验：R2 GiB、H4 GiB、2048专家槽、原请求body和4096输出上限不变；157309输入／1792输出，128条早段记录逐字正确，零暂停。首内容109.577秒，总160.933秒；host使用峰值约2.05 GiB，累计选中主机读取41,113,435,200字节。[汇总](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-exact-repro-summary.json)。
- 主配置M1：固定R2/H8/2048、CPU8、FP8 dense／INT8 KV／tiered、hybrid／layered、Graph4、Replay关、AR0，自动推导C16和262144上下文。**45个请求通过**：短边界、前缀复用、取消、维护、157K单请求及重访；三份157309／157310／157311输入各返回6656输出，512条记录逐字正确，SSE确有A→B→A交错，三请求段暂停／恢复／重算均为0，结束后活跃请求和状态均为0。
- 数值：11项gather／attention／计数器，6项CPU量化编码／scale，2项真实Hq24/Hkv2/D256的BF16／INT8集成全部通过；GPU项含eager和Graph，原容差未放宽。
- CPU定向421通过／24跳过；扩展1264通过／338跳过，另14失败／4错误涉及缺失NoWAG依赖或无GPU环境，未算通过。两个旧单token offload用例仍使用短prefill改动前的假设，留给测试维护者处理。

长记录值含等差规律，因此45项结果用于长输出、状态隔离和交错证据，**不单独证明中／尾位置不可推导事实的检索能力**。补充事实已冻结，但未发送。

## 当前运行环境

- 保留空闲服务供接手：容器 `ft-flash-next-m1-gpu2`，`http://127.0.0.1:18230`，GPU2 `GPU-847e9c75-56a9-1090-4f4f-7d70a71792dd`，CPU `0-7,16-23`（8物理核）。没有运行中的测试客户端。
- **该进程加载的是 `.worktrees/flash-next-i3b/python`**。修改生产前先停／重启，避免测试版本混淆；停止命令 `docker stop -t 30 ft-flash-next-m1-gpu2`。
- Python：`/home/nengneng/miniconda3/envs/freetoken-dev/bin/python`；镜像 `freetoken:555efd8`。native扩展需要容器环境；宿主CPU检查使用空 `CUDA_VISIBLE_DEVICES`。
- 权重：`/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4`，RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`，直接读safetensors。未转换FTW，未应用 `/tmp/flash-next-ftw-read-fallback.patch`，不要顺手应用它。
- NoWAG已明确归还GPU2并转GPU1；不要占GPU0。后续GPU协调记录在根 `RESEARCH_PLAN.md`；可用 `codex queue --thread <UUID> --message ...`。NoWAG线程为 `01a121f3-2ddd-7ba2-a7d5-120632e361b8`。
- 启动参数、服务日志、交接时cache/stats均在 `/data2/servebig-envs/flash_next_i3_acceptance/`，主配置启动记录为 `i3b-0abb433-m1-launch.json`。

## 下一步，从这里继续

1. 先用保留的M1服务跑已冻结的三位置独立事实短答案，再跑I5已有质量任务；不要重跑已通过的45项长负载。事实文件：`i3b_blackbox/m1-independent-positions.fixtures.json`。
2. 跑余下M2–M8，按覆盖差集选择用例。还需实际tiered暂停／CPU恢复，含非4／64对齐位置，以及Replay环满后的历史复用。M1零暂停不能替代这些项。
3. 质量与量化对照、实际容量／性能报告、已有模型AR／self-SD／DFlash回归都未完成。冻结上游参考树 `.worktrees/flash-next-upstream-reference` / `research/flash-next-upstream-reference@9b585b7` 已准备；参考容器 `ft-flash-next-reference-gpu2`仅创建、未启动。只用同权重短任务比较质量，不冒充同资源性能对照。
4. 若新测试发现问题，修复后只补相关复验；生产与新增测试继续由不同agent负责。第一阶段完整验收通过后才开始SD；FTW保持暂缓。

| 配置 | dense／KV | 驻留 | 后端／调度 | Graph／Replay | 前缀 |
| --- | --- | --- | --- | --- | --- |
| M1，已跑上述范围 | FP8／INT8 | tiered，H8 | hybrid／layered | 开／关 | 开 |
| M2，未跑 | BF16／INT8 | tiered，H8 | hybrid／layered | 开／关 | 开 |
| M3，未跑 | FP8／BF16 | tiered，H8 | hybrid／layered | 开／关 | 开 |
| M4，未跑 | BF16／BF16 | tiered，H8 | hybrid／layered | 开／关 | 开 |
| M5，未跑 | FP8／INT8 | gpu，H0 | offload／legacy | 开／开 | naive |
| M6，未跑 | FP8／INT8 | tiered，H8 | offload／layered | 关／开 | 开 |
| M7，未跑 | FP8／INT8 | tiered，H8 | hybrid／legacy | 关／关 | naive |
| M8，未跑 | FP8／INT8 | tiered，H8 | hybrid／layered | 开／开 | 开 |

均固定R2、专家2048；Graph开=4、关=0，Replay开时记录长度8，不预设未来并发／输出。余下启动命令保存在 `remaining-variant-launch.json`；已切至标准功能树 `.worktrees/flash-next/python`，执行前记录实际受测commit。

## 独立测试与代码量

测试源码与生产分开，均留在已登记工作树：I3a `test/flash-next-i3a@8caae70`；I3b／事实 `test/flash-next-i3b@50fb256`（已跑 `716a1a8`，未跑事实 `4cb06db`）；数值 `test/flash-next-tiered-kernels@98662c2`；未跑I5入口 `test/flash-next-i5@e913c16`。对应目录在根 `RESEARCH_PLAN.md`。服务测试的[公开交接说明](../../flash-next-i3b-blackbox/blackbox_tests/FLASH_NEXT_HANDOFF.md)包含准确续跑命令，数值运行说明在其 `blackbox_tests` README中；实现者不要阅读测试源码。

本轮独立测试代码累计 **+1485／−5，净+1480**：服务／事实／I5作者+883／−5，数值作者+602／−0。此口径排除README、结果JSON和此前已存在的旧质量任务；两位作者均未改生产。

整个Flash-Next相对公共共享池 `d912bbe`：原始生产+5761／−269；扣1489行第三方原样部分后 **+4272／−269，净+4003**。相对功能开始前 `732f1ee`，含共享池且扣第三方后+6683／−581，净+6102。FreeToken自有top-k和改写集成代码全部计入；旧2147行第三方扣除口径过大，已修正。测试与生成结果不计生产代码。
