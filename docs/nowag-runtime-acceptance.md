# NoWAG 在线接入验收

状态：进行中，未达到可交付标准。生产 `0d20e8d`，独立黑盒 `4f96f55`；[PR #11](https://github.com/ServeBig-project/FreeToken/pull/11)。

## 当前结论

- 审计后的四批生产修复已提交，工作树与远程已同步。生产实现者没有读写测试；黑盒作者只收到公开契约、公开权重和运行环境。
- native C++、sm89 CUDA和wheel构建通过。wheel包含在线CUDA源、CPU头文件和实测profile，未带入已删除的实验kernel。运行目标为 `freetoken:555efd8` Docker、Torch 2.11.0+cu130、CUDA 13；该容器编译的扩展不能直接用于glibc较旧的宿主机。
- 首个GPU段23项通过：Qwen真权重的T=1/4/7/16、空间查询、层0，以及在该几何上的DSV4舍入／路由位置、SwiGLU-OAI与tanh-GELU数学。组件数学通过不代表对应真实模型服务通过。
- 第24项加载末层bank时资源OOM：测试进程占1.23 GiB，外部进程同时占22.16 GiB，剩余58.75 MiB不足分配66 MiB。已首错退出；本段不提供性能结论，未放宽容差或修改实现。
- 既有CPU定向回归35通过、3跳过；4个内部配置桩缺少真实配置字段，待独立公开启动检查通过后替代，不为旧桩增加生产兼容。
- 既有模型、CPU层配置与MoE基准入口回归45通过、8跳过；两项最初因CPU容器缺少驱动库失败，使用 `NVIDIA_VISIBLE_DEVICES=none` 挂载库但不暴露GPU后复验通过。没有因此修改生产代码。

## 尚未完成

| 验收部分 | 当前状态 |
| --- | --- |
| 单卡组件 | 首段23通过；末层、缓冲、并发流、Graph、D4／布局、真实DSV4及精确小样本待继续 |
| 真实服务与资源 | Qwen3.6／DSV4的驻留、offload、cpu／hybrid、缓存边界与维护待运行 |
| SD与Graph | self-SD／DFlash、实际回放与起草计数、尾批及控制项待运行 |
| FTW | 往返、缺数据、重命名、隔离两份源目录后的运行待完成 |
| TP2 | 真实双卡组件、合法小模型服务及FTW待资源安排；Qwen3.6／DSV4基座TP1限制保持 |
| 回归与性能 | 非NoWAG、三个交错配对block、显存／搬运证据待完成；最终公共runtime基座仍需同步 |
| GPT-OSS | 真实BASE缺失；已准备的组件数学不能替代bias与完整服务验收 |

黑盒已准备309项：116项CPU／独立参考、193项GPU门控；准备数量不是通过数量。具体选择与环境变量见独立测试树的 `blackbox_tests/nowag_runtime/README.md`。

## 运行材料

- 生产树：`.worktrees/nowag-runtime`；独立测试树：`.worktrees/nowag-runtime-blackbox`。
- 配对基线：`.worktrees/nowag-runtime-baseline`，固定 `e6f6d90`。基线所需外部在线插件只加入基线的PYTHONPATH，候选使用自己的在线模块。
- 原始日志、运行脚本、安装包：`/data2/servebig-envs/nowag_runtime_acceptance_20261009/`。首段为 `components-qwen-real.log`；构建为 `build-native.log`、`build-cuda.log`、`build-wheel.log`。
- 临时生成权重：`/dev/shm/nowag-runtime-acceptance-20261009`，逐组生成／验收，不同时保留多份大模型FTW。
- GPU使用与交接统一记录在根 `RESEARCH_PLAN.md`。发生过服务间空档误判，后续须明确交接后启动。

## 代码量

相对审计快照 `bf4acec`：生产+737／−1818，净−1081；生产agent测试改动0。按阶段提交为 `d2491e6`（+29／−21）、`d10266c`（+138／−60）、`384f525`（+530／−1724）、`d060783`（+55／−28）；累计端点不重复计算中间修改。

相对已含共享runtime、未含本轮NoWAG的 `e6f6d90`：生产+12407／−1332，净+11075，含项目自有在线kernel移入；实测profile JSON另计+1055。既有内部测试+336／−403；独立黑盒本次接手后另增+715／−63，尚未集成，最终交付时重新核算。
