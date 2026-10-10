# Flash-Next 第一期状态

当前：独立验收进行中，**第一阶段尚未交付**。PR #10 头含 I5 修复（`2133e7e`、`c3799b1`）并跟上 main `ea3b9df`；配置矩阵在 `8be9796` 上跑完，tiered 内核改动后的复验见下。FTW 按用户决定暂缓，本期直接加载现有 RadixArk safetensors。入口：[交接说明](flash-next-handoff.md) · [主设计](flash-next-design.md) · [公开契约](flash-next-public-contract.md)。

## 已实现与已有证据

- 模型、dense FP8、INT8 KV、分层执行、普通 Graph、附加状态快照已接入。精度与驻留各自选择，默认保持来源与组件原有语义。
- I3a `2cc8ef4`：QSA、压缩索引、每请求附加状态使用共享 runtime；模型报告四路残差的执行保留量。合流 `c0ba02a` 的独立真权重服务共88请求通过，包含实际暂停2次、重算2次；不是完整配置矩阵结论。
- I3b `cb26f83`：同一主机缓存预算保存活跃K/V、前缀与暂停副本。GPU只取得选中的主机组；申请不足时先冷缓存、再已备份活跃载荷、最后暂停。完整页备份与回收时序已通过独立源码复核。
- 合流定向CPU回归421通过、24跳过。扩展回归1264通过、338跳过；另14失败、4错误涉及未安装的NoWAG依赖或无GPU环境，未当作通过。
- 固定R2/H8的M1配置共45个请求通过；三份157309／157310／157311输入各输出6656 tokens，512条记录逐字正确，真实内容帧交错成立，三请求段零暂停／恢复／重算。数值验证共19项通过，包含真实Hq24几何。
- I5：`/v1/cache/status` 的主机字节按活跃K/V、暂停副本、冷缓存分别报告（`2133e7e`）；BF16 K/V＋tiered 的 QSA attention 改为按 token 选取行地址，修复 RTX 4090 共享内存超限导致的预热退出（`c3799b1`，独立 tiered 算子 11/11 通过）。

## 配置矩阵（生产 `8be9796`，独立黑盒套件）

| 配置 | dense / KV / 放置 / 主机 | MoE / 批处理 / Graph / Replay | 结论 |
| --- | --- | --- | --- |
| M1 | fp8 / int8 / tiered / 8 GiB | hybrid / layered / 4 / 关 | I3b 45项、gaps、三条157K独立事实、压力（含CPU副本恢复）通过 |
| M2 | bf16 / int8 / tiered / 8 | hybrid / layered / 4 / 关 | I3b 45项、I5质量（GSM8K 31/32）通过 |
| M3 | fp8 / bf16 / tiered / 8 | hybrid / layered / 4 / 关 | 预热超共享内存退出；`c3799b1` 修复，待复验 |
| M4 | bf16 / bf16 / tiered / 8 | hybrid / layered / 4 / 关 | 同 M3 |
| M5 | fp8 / int8 / gpu / 0 | offload / legacy / 4 / 开 | I3a、gaps（重算恢复、Replay环绕、naive缓存压力）通过 |
| M6 | fp8 / int8 / tiered / 8 | offload / layered / 关 / 开 | I3b、gaps（Replay环绕后前缀复用、共享主机字节只计一次）通过 |
| M7 | fp8 / int8 / tiered / 8 | hybrid / legacy / 关 / 关 | I3b（2次合法miss）、gaps 通过 |
| M8 | fp8 / int8 / tiered / 8 | hybrid / layered / 4 / 开 | I3b 45项、gaps、I5质量（31/32）通过 |
| M9 | fp8 / int8 / tiered / 1 | hybrid / layered / 4 / 关 | 压力：真实暂停3次、重算恢复、非对齐位置，输出逐字正确 |

共同参数：runtime 2 GiB、2048专家槽、radix 缓存（M5/M7 为 naive）。M1 首轮因测试机主机内存被其他进程占满导致后端退出，结果作废并保留证据，同代码重跑通过。仍未覆盖：暂停副本复制途中取消、恢复途中取消、Replay 下记录环绕后的暂停恢复。报告在 `/data2/servebig-envs/flash_next_i3_acceptance/`。

## 已发现、待定位

- 同一请求的贪心输出会随此前处理过的请求变化：纯 AR、offload 配置下，同一 prompt 的首个 token 在不同历史后翻转。已排查 GDN 卷积初始状态、专家合并顺序、NVFP4 缩放与 FP8 线性层，未发现状态泄漏；原因未定位。它影响与上游参考的逐 token 对照，需先定位再做质量结论。

## 157K原失败已关闭

旧候选 `7139b31` 在 R2 GiB／H4 GiB／2048专家槽下拒绝唯一的157309-token输入；`0abb433` 用原请求与原预算通过（1792输出、128条早段记录逐字正确、零暂停），选中主机读取持续增长。[原公开复现](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/7139b31-157309-public-repro.json) · [精确复验](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-exact-repro-002.jsonl)

## 接下来

1. 在 `c3799b1` 上复验 M3、M4，并补一次 M1，确认改动后的 tiered attention 服务结果。
2. 逐层定位上述输出随历史变化的原因。
3. 与上游参考同权重质量对照；增槽与性能报告；已有模型回归（Qwen3 MoE、Qwen3.5/3.6 AR、SD/DFlash、冷前缀、共享池）。

## 代码量

总计固定在 `0abb433`，其后提交单列；生产统计包含Python及构建代码，排除测试、文档和生成物；保留重命名识别。

| 范围 | 增加 | 删除 | 净增 |
| --- | ---: | ---: | ---: |
| 相对功能开始前 `732f1ee`，含合入共享runtime，扣第三方原样行 | 6683 | 581 | 6102 |
| 相对共享runtime `d912bbe`，仅Flash-Next，扣第三方原样行 | 4272 | 269 | 4003 |
| I3a提交 | 132 | 22 | 110 |
| I3b提交 | 617 | 63 | 554 |
| 后续NVFP4状态上报修复 | 3 | 1 | 2 |
| I5 主机字节状态拆分 `2133e7e` | 40 | 3 | 37 |
| I5 tiered attention 按 token 选行 `c3799b1` | 10 | 16 | −6 |

第三方原样行单列1489：hc／ple／compress／expand／score共1143，attention保留346；改写的66行attention与FreeToken自有top-k全部计入生产。原始总生产相对 `732f1ee` 为+8172／−581，相对 `d912bbe` 为+5761／−269。测试代码在最终独立报告单独核算。
