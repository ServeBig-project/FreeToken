# I3b tiered K/V 独立服务验收

只读公开契约、HTTP 响应和模型 tokenizer；没有阅读生产源码、内部测试或实现 diff。复用同一独立作者的 I3a HTTP 用例，新增 host 预算、活跃驻留及长历史断言。

## 固定前提和运行

协调者启动服务。原失败候选为 `7139b31`；首个修复复验使用协调者指定的新候选，单 GPU，AR，layered-pipeline，dense FP8、INT8 KV、tiered，Graph 1–4，Replay 关闭；仍为 runtime 2GiB、host 4GiB、专家 2048、CPU 8 核。脚本不启动或重启服务，合法维护仅重建到原 runtime 容量。

首个复验原样发送保存的公开请求 JSON，不重建 prompt，也不改变采样字段或 cache_group；期望输出直接取原请求中要求复制的记录。此入口只复验该请求，不重复前置短用例或发后续请求：

```bash
python -B blackbox_tests/flash_next_i3b/accept.py \
  --url http://127.0.0.1:18230 --runtime-gib 2 --host-gib 4 \
  --pressure-tokens 16381 --output-limit 4096 --copy-lines 128 \
  --graph on --replay off --timeout 1800 \
  --original-request /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/7139b31-157309-public-repro.json \
  --report /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-exact-repro.jsonl
```

原请求的答案依赖早段 128 条记录，不将它计作早/中/尾位置覆盖。主矩阵新增多用户任务另行覆盖三个位置，保持原复验不变。本期只用 safetensors，FTW 暂缓。

原失败关闭后，协调者固定 H8 并通知主矩阵服务 ready，再运行完整路径：

```bash
python -B blackbox_tests/flash_next_i3b/accept.py \
  --url http://127.0.0.1:18230 --runtime-gib 2 --host-gib 8 \
  --pressure-tokens 16381 --output-limit 8192 --copy-lines 512 \
  --graph on --replay off --timeout 1800 \
  --report /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/delivery-full.jsonl
```

Python 环境需要 `httpx`、`transformers` 和源检查点 tokenizer。不得以 `python -O` 禁用断言。报告路径须不存在。`--scenario maintenance` 仅续跑取消/维护；`--scenario pause` 仅续跑长历史，保留已完成证据。

## 验收顺序和结论边界

1. 先执行短输入、4/64 边界、不同历史复用/隔离、共享前缀取消及同预算重建，核对完整答案与 usage。
2. 只发一条约公开 context 上限六成的新历史，取 `64n+61` 长度。本轮为 157309 token；固定复制任务有独立预期。开始前没有活跃请求。
3. 只有 paused 未增加、host payload 页数超过开始前全部历史页数、GPU payload 页数不足以放下该完整输入时，该唯一活跃请求的持续 SSE 才计作“活跃历史驻 host 时仍推进”。末尾仍须完整答案、usage 和 finish reason 正确。不能只凭冷缓存数量下结论。
4. 重访同一长前缀，再交错提交三条完整约157K、不同历史的请求。每份答案依赖早/中/尾三段，各请求值不同；记录实际源 token 区间。保持预算不变，核对完整输出、真实输出交错、冷缓存淘汰、准入/暂停和最终进度。主矩阵请求上限8192、512条记录在候选运行前冻结，原缺陷复验仍为4096/128。

每次状态采样都核对 runtime 物理加总、固定专家容量、`host_used_bytes <= host_budget_bytes`、host payload 字节不超过 host 使用量及实际 tiered 配置。

三个驻留字段不能单独证明 QSA 选中的数据实际从 host 读取。当前用 `prefix_cache.runtime.kv_host_read_bytes` 的前后增量补充实际读取证据；若完成后的第一次采样没有增量，只再采样一次，遵守公开的一次异步滞后语义。报告区分整个唯一请求期间的读取增量与开始在 host 驻留条件下输出之后的读取增量。汇总计数仍不能独立证明相同内容只有一份 host 副本。

输出保存原始请求、SSE、响应、状态和计数差值。`incomplete` 表示仅取得本脚本覆盖的阶段证据，不能据此宣布完整一期交付。
