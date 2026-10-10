# Runtime pool 合并记录

合并依据：用户于2026-10-10在知晓剩余验收尚未完成后明确要求“合并”。生产代码端点为 `17bbe02`，功能前基线为 `main@732f1ee`。**已通过的验证与尚未完成的验证分开记录；合入不代表完整验收矩阵全部通过。** TP、多模态验证按用户决定暂缓。

## 最终功能与修复

KV、GDN、ReplaySSM、DFlash历史／窗口和SD临时状态共用显式runtime总预算，专家池独立。共享模式支持legacy／layered-pipeline；joint在启动前明确拒绝。默认启用策略与分池模式保持现有公开约定。

本轮关闭联合申请重复扣页、过量淘汰、AR误淘汰冷前缀、捕获临时页记账、prefill执行容量与保留波次上限、后建token表及采样工作区计价、Graph关闭时的临时页申请、物理容量与组件元数据统计问题。

最后一项生产修复 `17bbe02`：联合维护同时缩小runtime并扩大专家缓存时，在Graph清理之后、专家扩容之前释放旧runtime，避免最终预算合法却在中间分配时OOM。生产+4／−2；既有维护／缓存CPU回归26项通过，其GPU联合维护验收仍未完成。

## 已完成的验证

| 验证 | 实际受测生产版本 | 结果 |
| --- | --- | --- |
| 最终审计补验 J1–J6 | `f4647b2` | 14/14，无跳过；小池Replay／DFlash、小上下文与长度错误、采样突发、runtime重建、Graph关闭下真实SD、joint／不足一页配置拒绝、公开统计 |
| 自动并发采样突发 | `f4647b2` | 6→7 GiB重建前后各提交98请求，每条64输出；生效C98保持，实际观测输出交叠4，不宣称同时执行98行 |
| 新维护顺序的CPU接口回归 | `17bbe02` | 26通过 |
| 联合准入／淘汰、DFlash、共享前缀Replay压力 | `7155bcf` | C9/9、E6/6、G4/4；实际暂停／恢复／重算，共享前缀压力输出与solo逐字一致 |
| self-SD／自适应DFlash与cap压力 | `187dc9a` | 各组8/8；self-SD有4次重算，DFlash自适应有3次CPU恢复 |
| 原始hybrid＋ReplaySSM＋DFlash长历史负载 | `d912bbe` | 58/58、零请求错误，输入和输出未缩短；全程214.585秒，1847输出tokens |

独立测试与报告已集成：`blackbox_tests/runtime_pool/`、[最终补验](runtime-pool-merge-acceptance.md)、[此前服务矩阵](runtime-pool-blackbox-report.md)、[连续负载与压力结果](runtime-pool-final-blackbox-report.md)。旧报告保留当时的版本与结论；后续证据更正和当前状态以本页为准。

全量CPU套件在f4647b2与基线具有同一组12失败／4错误，未记为全量通过。本轮A/B宿主机不独占，完整请求吞吐分池52.537、共享51.529 token/s；不能据约2%差异归因实现退化或证明持平。

## 未完成项及启动失败归类

- 最终生产端点17bbe02上的原58请求重放：两次尝试均遇同卡NoWAG任务重叠，未进入请求重放；第二次由协调者终止自己的重叠试跑。不能借d912bbe的通过记录宣称最终头通过。
- 联合维护 `R8.626953125GiB／专家5000 → R4.75GiB／专家7200`：独立驱动已提交，公开最终总预算合法且减少256,491,520字节，但GPU服务验收未执行。
- 旧d912bbe的Graph启动OOM已确认不是独占GPU实验：runtime在20:17:09失败，同卡NoWAG测试在20:17:10.772结束，其CUDA错误记录另一个测试进程占1.23GiB。该样本保留为同卡争用污染，不再作为共享池自身启动不稳定的证据。

证据根目录：`/data2/servebig-envs/runtime_pool_closeout_20261009/`。`final-j1-j6-pytest.log`记录14项通过；`hybrid-trace-v3-contention-evidence.json`记录旧失败的交叉证据；`merge-final-gpu-processes.jsonl`和`merge-final-isolated-gpu-processes.jsonl`记录最终两次尝试的重叠进程；`merge-final-trace-isolated-abort.json`说明第二次终止。失败原始日志未删除。

## 代码量

相对732f1ee，生产（含构建入口）39文件 **+2453／−322，净+2131**；独立黑盒29个Python文件 **+3326／−0，净+3326**。文档、生成文件、第三方不计。实现与黑盒作者分离；测试集成没有改变生产代码。
