# Flash-Next 独立服务黑盒交接

状态：按用户要求停止新增请求和测试。全部已启动测试客户端均已退出，模型服务由根协调者保留处理。本期仅 safetensors，FTW 暂缓；不能宣称一期完整验收通过。

## 已有结论

| 范围 | 结果与证据 |
| --- | --- |
| I3a，`c0ba02a`，GPU KV，R2/H4 | 88 条生成验证；真实暂停/重算各 2 次，完整答案正确。[汇总](/data2/servebig-envs/flash_next_i3_acceptance/i3a_blackbox/c0ba02a-i3a-summary.json) |
| I3b，`7139b31`，tiered，R2/H4 | 40 条前置验证通过；157309-token 单请求在公布上限内失败。[原始公开复现](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/7139b31-157309-public-repro.json) |
| 原失败精确复验，`0abb433`，R2/H4/2048 | 原请求 body 完全相同，157309 输入/1792 输出，128 条记录正确；未暂停且有实际主机读取，原失败关闭。[汇总](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-exact-repro-summary.json) |
| M1，`0abb433`，R2/H8/2048/CPU8 | 退出码 0，45 条验证通过；FP8 dense、INT8 KV、tiered、hybrid、layered、Graph4、Replay关、AR0，自动 context262144/C16。[汇总](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-m1-h8-summary.json) |

M1 三条输入为 157309/157310/157311 token，各输出 6656 token，正常停止；每条 512 记录逐字正确，内容帧实际交错，三请求段暂停/恢复/重算均未增加。长前缀重访复用 157248 token；共享前缀取消、busy/rejected 维护及同预算重建后的生成均通过。

M1 公开资源检查全程通过：runtime held 峰值 2,143,289,344 字节，host used 峰值 8,516,759,552 字节，均在原预算内；Graph decode 增量 28,985，选中主机 K/V+scale 逻辑读取增量 643,478,113,728 字节。最后 active requests/states 均为 0、host inflight 为 0；即时末次快照仍有 364,904,448 protected bytes，用户停止指令后没有追加排空测试。

## 证据边界与未运行项

- 冻结的 512 记录值是等差序列：证明长输出、跨请求状态和交错，不单独证明中尾信息不可由早段推导。
- 独立事实补充已冻结，**未运行**：[数据和答案](/data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/m1-independent-positions.fixtures.json)。输入仍为 157309/157310/157311；事实约位于 token `[38,50)`、`[78660,78671)`、`[157271,157283)`，三位置及请求之间无推导关系。
- I5 附着质量入口已准备，**未运行**：复用 `74888a0` 的 coding/research、多轮、GSM8K 32题和协议用例；既有冻结容差未改。参考候选由协调者另行提供。
- 其余 7 配置、长请求期间持续插入短请求/取消、队列/prefill/暂停/复制等待取消、非4/64暂停位置、host=0/naive/Replay压力、最终回收和共享副本唯一性、完整质量/容量/性能与已有模型回归仍有差集。
- 后续开启 Replay 前，应对齐“缺完整历史允许真实 miss”的公开契约：现有 warm-hit 强断言尚未适配该情况，不能把合法 miss 报成生产错误。

原始日志含准备期的契约路径或输入构造修正，先读取上述汇总；不要将这些准备中断混计为生产失败。后续测试仍应由未读取生产源码的独立作者接手。

## 测试提交与代码量

以下只计本代理新增/修改，不含原 `74888a0` 黑盒，也不计生成报告；生产代码均为 **+0/-0**。

| 阶段 | 提交 | 测试代码 + / - / 净增 | 文档新增 |
| --- | --- | --- | --- |
| I3a | `8caae70` | 454 / 0 / 454 | 69 |
| I3b | `716a1a8` | 279 / 5 / 274 | 45 |
| I4 独立事实，未运行 | `4cb06db` | 77 / 0 / 77 | 14 |
| I5 质量入口，未运行 | `e913c16` | 73 / 0 / 73 | 13 |

测试代码合计 **+883/-5，净增878**。I3b/I4 位于 `.worktrees/flash-next-i3b-blackbox` 的 `test/flash-next-i3b`；I5 位于 `.worktrees/flash-next-i5-blackbox` 的 `test/flash-next-i5`。本交接文档单独提交，不改变任何在跑用例。
