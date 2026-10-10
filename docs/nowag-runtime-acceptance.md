# NoWAG 在线接入验收

状态：**交接中，尚未达到可交付标准**。2026-10-10，最新生产 `fd59354`，独立黑盒 `9f03cf4`；[PR #11](https://github.com/ServeBig-project/FreeToken/pull/11)。受测版本按下表记录，旧结果不冒充最新提交的完整验收。[接手入口与审计处理](nowag-runtime-handoff.md)。

## 已有证据

日志目录为 `/data2/servebig-envs/nowag_runtime_acceptance_20261009/`，下表文件名均相对此目录。

| 范围 | 结果与版本 | 证据 |
| --- | --- | --- |
| 构建 | native C++、sm89 CUDA、wheel通过；两项最新FTW修复为Python-only，最终wheel尚需刷新 | `build-native.log`、`build-cuda.log`、`build-wheel-current.log` |
| Qwen真实权重组件 | 45通过：生产 `0d20e8d`／测试 `4f96f55` 首段23；生产 `c2612eb`／测试 `af773ad` 补齐22 | `components-qwen-real.log`、`components-qwen-remaining-gpu2.log` |
| GPT-OSS带bias组件 | 独立随机小模型5通过，生产 `c2612eb`／测试 `af773ad`；不代表训练模型质量 | `components-gptoss-bias-gpu2.log` |
| DSV4真实权重组件 | 5通过，同上版本 | `components-dsv4-real-gpu2.log` |
| D4／布局／精确样本 | 20通过，同上版本 | `components-layouts-exact-gpu2.log` |
| 旧CPU扩展公开错误 | 3通过；独立旧扩展环境验证cpu／hybrid／CPU层配置在载权重前明确要求重编译 | `stale-extension-errors-gpu1.log` |
| Qwen服务后端 | fused／cpu／hybrid对offload参考3通过；生产 `5b28b26`、测试 `af773ad`。采样注意事项见下节 | `service-execution-modes-gpu1.log` |
| 模型与CPU层配置回归 | 45通过、8跳过；模型、逐层CPU配置及MoE基准入口 | `cpu-public-regression.log`、`cpu-driver-recheck.log` |
| 公共调度与预算回归 | 合入 `d912bbe` 后139通过，4个过时内部桩排除；独立公开错误3项通过后已移除这4项 | `cpu-runtime-integration-env-fixed.log` |

GPU组件共 **75项通过**，包括D4/D6、两种assignment布局、数学族／bias、输出与workspace、缓存槽重映射／容量、空批与padding、独立流和实际CUDA Graph回放。不是完整服务／SD Graph通过。

首个GPU1段23项通过后因其他任务占22.16 GiB而OOM：本进程1.23 GiB，剩余58.75 MiB不足下一次66 MiB分配；缺项已在明确交接的GPU2窗口补齐。没有改变数值容差。CPU容器初始驱动库缺失、用户名缺失导致的Torch编译缓存错误分别通过环境修正解决，后一错误在共同基线同样复现，未因此改生产代码。

## 尚未关闭的HTTP对照

流式与非流式16-token结果不同；分离冷 `cache_group` 后仍复现。共同基线 `d912bbe`＋原外部NoWAG插件产生完全相同的两段差异文本。因此不能直接归为本轮NoWAG回归，也未宣称问题已解决。

原请求只有 `temperature=0`，继承模型 `top_p=0.95`，在当前公共采样语义下未进入greedy分支；BF16最大logit并列时仍可能随机选择。独立作者提交 `9f03cf4`，只给这对请求显式补 `top_p=1`，保留冷缓存、usage和严格文本断言；**该控制尚未运行**，启动被GPU占用保护拒绝。

证据：`service-qwen-http-gpu1.log`、`service-qwen-http-cold-control-gpu1.log`、`service-baseline-http-cold-control-gpu1.log`；公开请求与响应在 `logs/svc_http-stream-control.json`、`logs-baseline-http/svc_http-stream-control.json`。控制通过后，再由独立作者修正其他意图为greedy的默认请求并复验受影响对照；显式采样参数、温度采样与冻结容差保持不变。

## 剩余交付项

- Qwen3.6完整服务、缓存／状态／资源边界、CPU层分配、并发／取消、各batching路径、预取／D2D、固定池及共享池维护；DSV4真实服务。
- `982d6aa` 联合缩小runtime／增大专家缓存，独立 `test_rebuild_shared_exchange.py` 已编写但未跑。
- self-SD／DFlash及控制项、步数1/2/4/8、实际Graph／草稿计数和自然尾批。
- FTW往返、缺数据、重命名、原目录不可见的容器运行；最新 `adc204f` 缺整组bias拒绝和 `fd59354` 显式SIDE覆盖FTW均待独立验收。现有套件尚缺GPT-OSS整组FTW bias缺失／TP后不可见缺失及新覆盖优先级用例。
- 真实TP2组件、合法小模型服务与FTW；Qwen3.6／DSV4基座加载器仍TP1。真实训练GPT-OSS BASE仍缺，已有随机fixture只提供数学和加载证据。
- 非NoWAG回归、相同输出工作量的三个交错性能对照、显存／搬运记录、最终清洁安装和P5生产整理。当前没有可用的性能交付结论。

独立套件已集成：310项（116 CPU／reference、194 GPU门控），准备数量不是通过数量。测试选择与环境变量见 `blackbox_tests/nowag_runtime/README.md`。生产实现者没有读写测试；黑盒作者未读生产源码或实现笔记。

## 代码量

统计端点 `d912bbe..fd59354`：共同基线已含公共runtime修复、未含本轮NoWAG，避免把并行功能计入本实现。生成文件、第三方、文档和profile不计生产源码。

| 类别 | 增加 | 删除 | 净增 |
| --- | ---: | ---: | ---: |
| 生产源码 `python/`（排除JSON） | 12,473 | 1,339 | 11,134 |
| 其中项目自有NoWAG kernel移入 | 10,872 | 0 | 10,872 |
| 其余生产源码 | 1,601 | 1,339 | 262 |
| 既有内部测试 | 334 | 425 | −91 |
| 独立黑盒Python（含参考／输入生成器） | 3,272 | 0 | 3,272 |

实测profile另计+1,055；构建配置+1/−1；benchmark+64/−33；黑盒README+226。项目自有旧kernel移入仍计FreeToken源码增加，不称为全新算法，也不按第三方剔除。分阶段修复的提交和代码量见[交接文档](nowag-runtime-handoff.md)。
