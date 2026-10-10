# NoWAG 在线接入验收

状态：单卡验收全部通过；TP2 待最后一次复跑。2026-10-10，生产 `570808b`（已合入 main `ea3b9df`），独立黑盒 `test/nowag-runtime@12fb760`；[PR #11](https://github.com/ServeBig-project/FreeToken/pull/11)。证据目录 `/data2/servebig-envs/nowag_runtime_acceptance_20261009/`，下表的批次目录都相对于它。

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
| TP2 服务与 FTW | 原生 TP2 3 过 3 败，都是小模型逐字比较过严（见下）；FTW TP2 按用户决定改为拒绝，待复跑 | `b12/`；复跑 `b14/` |
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

`cf2ef2e` 把 TP2 原生服务的比较改为“前 8 个词一致”。这条规则是看到失败后才改的：小随机模型上，TP1 与 TP2 在第 13–18 个 token 后才分叉，TP1 只换 backend 也在相近位置分叉；数值正确性由组件级 TP2 测试保证。

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
