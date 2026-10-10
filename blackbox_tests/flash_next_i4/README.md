# 157K 历史的位置语义补充

冻结的 512 条等差记录继续验证长输出、不同请求状态和输出交错；不把它当成中尾信息不可由早段推导的证明。

本补充将三条互不可推导的独立事实分别放在每份历史的早、中、尾段。三请求值不同，输入仍为 157309/157310/157311 token，输出上限128；只补位置检索，不重复长输出压力。数据、答案和实际 token 区间在发送前导出为 fixtures JSON，并保持不变。

```bash
python -B blackbox_tests/flash_next_i4/positions.py \
  --url http://127.0.0.1:18230 \
  --fixtures /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/m1-independent-positions.fixtures.json \
  --report /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-m1-independent-positions.jsonl
```

仅用于已冻结的 M1：R2/H8/专家2048、Graph开、Replay关、tiered。短答案不能替代原512记录长输出验收。任何错误立即结束，不改变输入或预算重试。
