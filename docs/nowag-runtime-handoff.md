# NoWAG 续修交接

2026-10-10。用户要求本会话收尾，交回恢复额度的原agent。**这是工作交接，不是功能完成或可合并结论。** 本文供协调／实现接手；独立黑盒作者只接收[公开契约](nowag-runtime-public-contract.md)、公开失败及自己的测试，不能读本文或实现审计。

## 从哪里继续

| 用途 | 根目录下的worktree／分支 | 当前提交 |
| --- | --- | --- |
| 生产与正式PR | `.worktrees/nowag-runtime` / `feat/nowag-runtime`，[PR #11](https://github.com/ServeBig-project/FreeToken/pull/11) | 最新生产 `fd59354`，交接文档随后提交 |
| 独立黑盒 | `.worktrees/nowag-runtime-blackbox` / `test/nowag-runtime` | `9f03cf4`；已集成进生产分支，最后两次采用cherry-pick |
| 同公共runtime基线 | `.worktrees/nowag-runtime-baseline` / `research/nowag-runtime-baseline` | `d912bbe` |
| 原始审计快照 | `.worktrees/nowag-runtime-audit` / `review/nowag-runtime` | 受审 `bf4acec`，报告 `5e892bf` |
| 主线入口 | `FreeToken/` / `main` | `732f1ee`，保持干净 |

范围延续原授权：独立在线专家组件及已有合法公共执行路径；离线量化／训练和GGUF不在本轮。模型数学由模型组件提供，压缩权重／kernel／空间需求由格式模块提供，缓存搬运、调度、Graph及通信继续走公共流程。不得新增模型名字名单。Qwen3.6／DSV4加载器TP1、SD TP1等原组件限制保持。Flash-Next的FTW暂缓决定不适用于本轮NoWAG FTW。

## 审计与修复

原始[审计报告](../../nowag-runtime-audit/docs/nowag-runtime-audit.md)及[诊断证据](../../nowag-runtime-audit/docs/nowag-runtime-audit-evidence.json)保留在独立树，是 `bf4acec` 的白盒结论，不能当作新头仍存在同样问题或独立黑盒通过的证明。

| 已确认的问题／边界 | 生产处理 | 验证边界 |
| --- | --- | --- |
| P1：GPT-OSS bias和TP分区hook未导出；NoWAG误改非专家FP8精度 | `d2491e6` 接通模型hook，隔离原非专家精度 | 随机GPT-OSS GPU bias组件5项通过；完整FTW／非NoWAG回归未完成 |
| P1：重建漏记常驻workspace／shared；P2：槽数计价过大、重复H2D、别名重复计量、FTW TP临时主机bank滞留 | `d10266c` 统一实际容量费用、生命周期与存储计量 | GPU输出／空间／槽数检查通过；维护和整模型内存证据仍待验 |
| clean slate：SD格式名单、CPU格式解析侵入公共执行器、GPT-OSS全驻与CPU TP限制、无调用CUDA实验路径 | `384f525` 按绑定能力检查，格式私有CPU解析，补执行路径，删除死代码 | 75组件和3后端服务有证据；完整SD／TP2／性能未验，不能宣称全部模块化交付 |
| 旧CPU扩展不能兑现新接口 | `d060783` 在载权重前明确要求重编译 | 独立公开启动3项通过；4项过时内部配置桩已由该覆盖替代 |
| 同次缩小runtime、增大专家缓存可能因中间分配OOM | `982d6aa` 先释放待替换旧runtime，再增专家／workspace；预算政策不变 | 独立联合维护用例已准备，未运行；总规划已通知共享runtime协调者 |
| FTW整组GPT-OSS bias缺失可能静默启动，rank1丢弃down bias前未查完整性 | `adc204f` TP分片前按模型必要bank检查 | 只完成生产语法／空白检查；独立FTW缺整组bias测试尚缺 |
| BASE是FTW时显式 `--nowag-expert-path SIDE` 被忽略 | `fd59354` 显式SIDE优先；BASE自身供非专家／必要bias；支持bank和普通weights布局，未知权重仍严格检查 | Python-only；需独立覆盖原生BASE、两类FTW BASE、有／无SIDE及GPT-OSS bias |

最后两项来自继续审查实际公开输入，不是原始7项的一部分。`fd59354` 新增主要为GPT-OSS真实FTW布局读取，未新增配置或兼容框架。最新改动未做完整独立运行验收，仍需P5整理和最终边界复核。

| 生产phase | 增加／删除／净增行 |
| --- | ---: |
| `d2491e6` | +29 / −21 / +8 |
| `d10266c` | +138 / −60 / +78 |
| `384f525` | +530 / −1724 / −1194 |
| `d060783` | +55 / −28 / +27 |
| `982d6aa` | +5 / −3 / +2 |
| `adc204f` | +14 / −3 / +11 |
| `fd59354` | +72 / −26 / +46 |

生产agent测试改动0。公共runtime `d912bbe` 经 `c3771c0` 合入，其+84/−53另计；独立套件经 `5b28b26` 集成，冷缓存／显式greedy后续分别为 `c42eb45`、`d46260a`。阶段数含中间重复修改，不相加冒充最终端点；完整代码量见[验收记录](nowag-runtime-acceptance.md#代码量)。本次交接提交仅改文档，生产／测试均+0/−0。

## 验收结果与第一步

[验收记录](nowag-runtime-acceptance.md)逐项列出版本和日志：75项GPU组件、3项旧扩展公开错误、3种真实服务后端通过；CPU模型45通过／8跳过、调度预算139通过。完整服务／SD／FTW／TP2／性能仍未完成。

当前最小继续点是 `test_service.py::test_http_semantics` 的显式greedy控制。之前流式／非流式不一致在候选和基线完全一致；原请求 `temperature=0` 继承 `top_p=0.95`，未进入当前greedy分支。`9f03cf4` 仅补该对请求 `top_p=1`，控制尚未跑。若通过，独立黑盒作者再修正其通用意图为greedy的helper，不覆盖用户显式top_p，不改温度采样或容差；随后复验受影响输出比较。若仍失败，保留公开请求／响应继续定位，不改生产采样政策来绕过验收。

黑盒作者确认的缺项：原生GPT-OSS有“所有层同时缺gate_up_proj_bias和down_proj_bias”检查；FTW仅有Qwen整shard缺失，**没有**逐组删除GPT-OSS gate_bias／up_bias／down_bias及TP切分后不可见缺失的检查；现有无bias Qwen TP2不能替代。显式SIDE覆盖FTW也需独立补齐。只向测试作者传递这些公开行为，不给生产diff。

后续补齐资源维护、服务、SD、FTW，TP2排最后，最后性能／清洁安装／P5。不要重复75项组件来替代未完成路径；最新FTW改动需针对覆盖。

## 环境和运行材料

根目录 `/home/nengneng/AIPrometheus/servebig/servebig-project`；结果目录 `/data2/servebig-envs/nowag_runtime_acceptance_20261009`。

- 结果目录含 `run-gpu1.sh`／`run-gpu2.sh`、`env-gpu1.sh`／`env-gpu2.sh`：Docker运行及输入变量；`run-cpu.sh` 不暴露GPU，但挂载驱动库供依赖导入。服务端口31891。
- 镜像 `freetoken:555efd8`，解释器 `/home/nengneng/miniconda3/envs/freetoken-dev/bin/python`，Torch2.11.0+cu130、Triton3.6、CUDA13、sm89。native已编译；此容器扩展要求的glibc高于宿主机，验收在Docker中运行。
- runner已固定USER／LOGNAME与Torch编译缓存目录，避免无系统用户名时Torch初始化中断。CPU容器使用NVIDIA_VISIBLE_DEVICES=none，不能去掉驱动库挂载。
- CUDA缓存 `/home/nengneng/.cache/torch_extensions/py312_cu130`；临时权重 `/dev/shm/nowag-runtime-acceptance-20261009`。已有D4／D6及layout输入可复用。FTW大文件逐个做，避免同时保存多份。
- 旧ABI环境在结果目录 `stale-python/` 和 `old_extensions/`，用于公开错误检查；不要覆盖生产扩展。`executor_interface_version`是整数1，不是函数。
- 结果目录 `installed/` 与wheel最后刷新于 `c2612eb`，**不含两项新FTW修复**。后续需用现有native更新build_py／wheel并解包；Python-only修复无需重编译native。
- `run-isolated-ftw-gpu1.sh`／gpu2版只挂安装包、测试、目标FTW、参考和结果，原项目／BASE／SIDE不挂入；用于真正自包含验收，尚未通过该项。刷新包后再跑。
- 配对基线PYTHONPATH必须包含 `.worktrees/nowag-runtime-baseline/python:servebig-nowag-plugin/src` 的绝对路径，环境脚本已设置；候选只使用自身在线模块。基线扩展已准备。旧基线另保留 `research/nowag-runtime-baseline-e6f6d90` 引用。
- 禁止直接跑 `tests/e2e`：它会自行使用GPU2。需要写测试时继续安排独立作者，当前实现和黑盒子agent均已冻结。

获明确单卡交接后可从此命令继续；这是待运行命令，不是本次通过记录：

```bash
RESULTS=/data2/servebig-envs/nowag_runtime_acceptance_20261009
source "$RESULTS/env-gpu1.sh"
"$RESULTS/run-gpu1.sh" \
  /home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/nowag-runtime-blackbox \
  python -m pytest -x -v --tb=short -p no:cacheprovider \
  blackbox_tests/nowag_runtime/test_service.py::test_http_semantics \
  > "$RESULTS/service-qwen-http-greedy-control-gpu1.log" 2>&1
```

真实Qwen／DSV4／DFlash路径在env脚本。随机GPT-OSS输入在 `/dev/shm/nowag-runtime-acceptance-20261009/gptoss-random-small`，对应BASE `base`、SIDE `scratch/synth-gptoss-d6-random-row_major`；跑其用例时按README设置NOWAG_GPTOSS_BASE、NOWAG_SCRATCH和NOWAG_GPTOSS_CACHE=8。H/I2880、2层、4专家、非零三组bias；不是训练模型质量证据。

## 资源交接

**本会话已停止发起GPU任务，无NoWAG容器或GPU进程待回收；没有停止其他会话。runtime-pool最高优先。** GPU2已于00:55 UTC明确归还Check。01:44观察GPU1仍有runtime服务（端口31922），GPU2仍有 `ft-flash-next-m1-gpu2`；PID和显存会变，以根RESEARCH_PLAN.md中的实际所有者交接为准，不能把服务间隙当授权。

| 卡 | UUID | CPU核 |
| --- | --- | --- |
| GPU1 | `GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd` | 8–15,24–31 |
| GPU2 | `GPU-847e9c75-56a9-1090-4f4f-7d70a71792dd` | 0–7,16–23 |

GPU0未授权。runner的空闲保护只能阻止已占用启动，不代替所有者确认；此前空闲检查后重叠启动已导致一次OOM。跨会话Check不能由本线程collaboration工具直接定位，双方已采用根 `RESEARCH_PLAN.md` 协调；返回agent可沿用该入口。没有停止／释放他人任务的授权。
