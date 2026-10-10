# NoWAG 在线接入验收

状态：单卡与 TP2 验收全部通过。2026-10-10，生产 `570808b`（已合入 main `ea3b9df`），独立黑盒 `test/nowag-runtime@d009efe`；[PR #11](https://github.com/ServeBig-project/FreeToken/pull/11)。证据目录 `/data2/servebig-envs/nowag_runtime_acceptance_20261009/`，下表的批次目录都相对于它。

## 结果

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| GPU 组件（D4/D6、布局、数学族与 bias、workspace、缓存槽、Graph 回放） | 75 通过 | 根目录 `components-*.log` |
| 服务：后端、batching、HTTP、并发与取消、缓存容量边界、CPU 层、重建与维护、状态读数、公开错误 | 全部通过（最终头 `9c449e0` 起复跑） | `b13/` |
| self-SD、DFlash（步数 1/2/4/8、Graph 与 eager 计数）、分层 SD、DSV4 offload/cpu/hybrid | 通过 | `b6/`、`b8/`、`b13/svc-rest.log` |
| FTW：Qwen D6/D4/word-major、DSV4、gpt-oss 往返；缺分片；重命名；只挂安装包和 FTW 的隔离运行；gpt-oss 逐组、逐层缺 bias；显式 SIDE 覆盖 FTW 专家 | 通过 | `b7/`、`b8/`、`b9/` |
| gpt-oss 随机夹具：bind 数值、四种服务模式、缺 bias 拒绝 | 通过 | `b8/gptoss.log`、`b9/` |
| 联合维护（缩小 runtime 同时增大专家缓存）、旧 CPU 扩展拒绝 | 通过 | `b7/`、`b13/rebuild.log` |
| 非 NoWAG 回归：BF16／NVFP4 输出与基线一致；未给 NoWAG 时默认后端不变 | 通过 | `b10/perf.log` |
| 配对性能（与 `d912bbe`＋原外部插件对比） | DSV4 通过。Qwen 首次 TTFT 中位数 0.705 对 0.657 s，超过 5% 门限；当时主机内存近乎耗尽。复跑两次通过，比值 0.97–1.04 | `b10/`、`b11/` |
| 清洁安装 | 最终 wheel 装进全新 venv：kernel 源码和 profile 齐全，`ft serve`/`ft checkpoint` 都有 `--nowag-expert-path` | `wheel-final/` |
| TP2 组件（rank 求和、bind 数值，含 DSV4、gpt-oss） | 10 通过 | `b12/` |
| TP2 服务与 FTW | 原生 TP2 6 项通过（offload/cpu/hybrid × D4/D6，48 个提示一致率 0.65–0.85）；FTW 配 TP2 在 ready 前被拒绝，同一 FTW 在 TP1 下正常服务 | `b14/`、`b16/` |
| CPU 内部测试 | 1230 通过、0 失败 | — |

## 本轮修复

- `079d4f1`：没有共享 runtime 时，显式 `--cuda-graph-max-bs` 可以大于并发数，专家临时缓冲区按 Graph 实际录制的最大批次分配；关闭 SD Graph 时也计 eager 次数。
- `2452058`：关闭 Graph 时，DFlash 的 eager 草稿也计数。
- `570808b`：FTW 检查点配 TP>1 时在 ready 前拒绝（用户决定）。原因是 FTW 按 TP1 整块保存非专家权重，只有各模型的原生读取器会分片；删除了 FTW 路径上随之失效的逐 rank 切分代码。

## 公共组件的已知行为（非 NoWAG 引入）

- DSV4 的 cpu/hybrid 路径逐次不可复现：不用 NoWAG 时同一提示连续两次输出也不同。黑盒对这两条路径不要求逐字一致。
- DFlash 只接受纯 FlashInfer，因此不能与 layered-pipeline 或 `--speculative-phase all` 组合，已断言为公开错误。
- joint batching 按用户决定废弃，已从契约和黑盒删除；它在取消后会丢 1 个 KV 页，基线同样复现，已转告 runtime 负责人。

## 黑盒规则的修订

都由独立作者按公开事实修改，数值容差没有放宽：

- greedy 请求显式发 `top_p=1`；
- 非法组合改为断言拒绝；
- harness 改为等上一个服务的整个进程组退出；
- FTW 测试结束后删除自己产出的目录。

TP2 原生服务的比较规则在看到 GPU 结果后修订过三次：`cf2ef2e` 改为前 8 词一致，`9a27e88` 改为以换 backend 为噪声参照，`d009efe` 改为一致率不低于 0.35。最终依据是用公开 bind 接口测得的数值：NoWAG 两个 rank 之和对 TP1 的相对误差为 0.0034，BF16 专家同样对半切分为 0.0033，可见 TP2 只多一次跨卡 bf16 舍入。测试用的随机 sidecar 与基座无关，专家输出约为 BF16 的 32 倍，greedy 输出对舍入格外敏感：同一模型换回 BF16 专家时，TP2 与 TP1 24/24 一致。错误分片估计会让一致率接近 0.06。

## 代码量

相对 main `ea3b9df`（已含共享 runtime）统计。文档、profile 不计入生产源码。

| 类别 | 增加 | 删除 | 净增 |
| --- | ---: | ---: | ---: |
| 生产源码 `python/`（排除 JSON） | 12,453 | 1,345 | 11,108 |
| 其中项目自有 NoWAG kernel 移入 | 10,872 | 0 | 10,872 |
| 其余生产源码（接入本身） | 1,581 | 1,345 | 236 |
| 既有内部测试 | 334 | 425 | −91 |
| 独立黑盒 Python | 3,507 | 0 | 3,507 |

实测 profile 另计 +1,055，黑盒 README +226。生产实现者没有读写测试；黑盒作者没有读生产源码或实现笔记。
