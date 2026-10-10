# Runtime 共享池收尾

受测生产头 `d912bbe`，基线 `main@732f1ee`，正式 [PR #9](https://github.com/ServeBig-project/FreeToken/pull/9)。已补 self-SD、自适应 DFlash／cap、hybrid／ReplaySSM 长历史服务验证，关闭本轮复现的运行期 prefill OOM。**同配置另有一次 Graph 捕获阶段启动 OOM，尚未归因；成功重放不代表启动问题已解决。** TP 和多模态验证按用户决定暂缓，不作为本轮补验前置，也不计为通过。

## 修复与边界

- Graph 捕获使用的临时页现在按普通持有／释放记账，不再留下永久 pinned 计数；真正的哨兵仍常驻。
- SD 先决定本轮 AR／草稿形状，只有实际选择 SD 才申请资源、必要时回收冷缓存。自适应计时排除资源申请耗时，实际裁剪后的执行形状参与成本记录。保留已确认的冷缓存回收规则，没有改成 SD 只能使用空闲内存。
- 启动 prefill 容量探针使用留给执行的真实显存，不再对已经包含 Graph／激活开销的总用量重复施加缓存比例限制。
- 共享模式 layered 波次中同时保留的全部 token 数受启动实测容量约束。旧逻辑把可执行的 1661-token tile 外推为 9966-token 波次，真实负载在后续 GDN 临时结果分配时 OOM；现在按实际容量分块。原分池路径保持原规则。

没有缩小专家容量、runtime 总预算、并发上限、输入或输出长度，也没有关闭 Graph 来通过长历史复验。紧显存下实际 prefill tile 小于配置上限是公开的容量裁剪：本次 8192 的配置上限生效为 1661。它可能增加 prefill 分块次数，因此用完整请求时间报告结果。

## 验收结果

独立测试作者未读生产源码、diff 或实现笔记。完整公开结果见 [独立补验报告](https://github.com/ServeBig-project/FreeToken/blob/test/runtime-pool/docs/runtime-pool-final-blackbox-report.md)；旧服务矩阵与其受测版本见同分支的 `docs/runtime-pool-blackbox-report.md`。每组按实际受测提交列出，未重跑的组不冒称最新头实测。

| 验证 | 版本 | 结果 |
| --- | --- | --- |
| 既有 CPU engine/scheduler/kvcache/server | `1d3009a` | 953 通过，6 跳过 |
| 既有独立 DFlash CLI／配置 CPU | `187dc9a` | 12 通过 |
| 执行预算／重建／维护 CPU | `271ce06` | 56 通过 |
| layered prefill 容量 CPU | `d912bbe` | 14 通过，1 跳过 |
| 固定资源短→长→短与重复前缀 | main `732f1ee` / `6556e13` | 各 14/14；专家 2048、C4、约 2.75 GiB，各输出 1408 tokens |
| self-SD，0.5 GiB，host 关闭 | `187dc9a` | 两轮各 8/8；长轮输出 14528 tokens，实际 4 次暂停／重算，压力后继续真实 SD |
| 自适应 DFlash，cap 128，0.75 GiB | `187dc9a` | 8/8；实际 AR 与 SD，3 次 CPU 恢复，保留冷前缀复用 |
| 原 58 请求 hybrid＋ReplaySSM＋DFlash | `d912bbe` | 58/58，0 错误；输入 4076–45475 tokens，输出上限 2–196，未缩放 |

58 请求包含 12 warmup、46 scored，全序列 214.585 s、1847 输出 tokens。scored 平均延迟 23.955 s、TTFT 22.401 s，p95 TTFT 38.507 s，最大流式输出间隔 0.921 s。实际执行 Replay AR 1703 tokens、verify 142 tokens；DFlash draft/accepted/verify 为 113/49/29；运行期间没有新增 Graph 捕获。

同一进程固定 runtime 9,263,120,384 B、专家 5000，无缓存重建。KV 峰值时持有 4600 MiB、GDN state 2220 MiB；结束时分别为 3280/3180 MiB，发生 2315 次解除映射，观测到物理用途转换。最大 held 9,258,926,080 B 未超过预算。该轮没有活跃请求暂停；暂停／恢复证据来自前两项独立压力测试。

AR 配对完整请求吞吐为分池 52.537、共享 51.529 token/s，约 2% 差异；本轮宿主机并非独占，不据此断言实现退化或持平。此前干净条件对照仅适用于当时的版本和负载。历史原始 OOM 发生在 DFlash compact 合入之前，也不能把本次成功全部归因于共享池。

## 剩余问题与证据

`hybrid-trace-v3` 在 `d912bbe`、同一资源配置下于 CUDA Graph 捕获末尾 OOM，尚未 ready；随后 `hybrid-trace-v4` 启动并完成原 58 请求。失败轮和成功轮均保留。前者初始化后可用显存 2.10 GiB，后者 2.50 GiB；range Graph 后分别为 1.17／2.04 GiB。失败轮日志没有分配器分项，不能据此编造根因或宣称零已知失败。

随后对同一 `d912bbe`、同参数做一次仓库外注入诊断，成功启动，生产代码未改动。range Graph 消耗 286 MiB，其中 PyTorch reserved 只增 20 MiB；86 个 SD verify Graph 再消耗 1356 MiB，其中 reserved 只增 68 MiB，其余 1288 MiB 属于分配器外占用。捕获阶段确有显著原生开销，单看 PyTorch reserved 会漏计；但失败轮额外的约 0.87 GiB 差异未复现，因此这不是启动故障的完整归因。未据此缩减 Graph 覆盖、调整用户预算或增加盲目清缓存代码。诊断文件：`startup-memory-injection/sitecustomize.py` 与 `hybrid-startup-diagnostic-server.log/json`；它是诊断，不计入独立验收通过数。

原始证据目录：`/data2/servebig-envs/runtime_pool_closeout_20261009/`。各 `*-server.json` 保存完整参数和受测源码提交；请求／SSE／usage／stats／资源采样与 CPU 日志分别归档。长历史采样在 `hybrid-trace-v4-status/cache-status.jsonl`，请求与汇总在 `hybrid-trace-v4/`。不以自然语言文本差异断言数值错误，也不把协议与 usage 通过等同于逐 token 数学等值证明。

## 代码量

相对功能前基线 `732f1ee`：生产（含构建入口）37 文件，**+2429／−330，净 +2099**；生成文件、第三方、文档与测试不计。生产分支测试改动为零。独立测试分支相对同一基线：22 个 Python 测试／辅助文件，**+2844／−0，净 +2844**；其中本轮新增 243 行，之前 2601 行（含算子与服务测试）。旧记录的 1531 行仅为服务阶段口径，不是整个测试分支。报告和 `.gitignore` 不计测试代码。

| 本轮提交 | 内容 | 生产增加／删除／净增 |
| --- | --- | --- |
| `a414c7b` | 临时捕获页记账 | +5／−1／+4 |
| `1d3009a` | 选择实际 SD 后申请资源 | +43／−47／−4 |
| `6556e13` | 状态与设计文档 | 0／0／0 |
| `187dc9a` | 决策计时排除申请 | +2／−0／+2 |
| `271ce06` | prefill 执行预算 | +6／−3／+3 |
| `d912bbe` | 波次保留量不超实测容量 | +8／−5／+3 |

本轮相对接手头 `a189310` 合计生产 +64／−56，净 +8；黑盒由独立作者提交 `ee6343f`（+242／−0）和 `15682e9`（+2／−1），报告提交 `7d44381`。
