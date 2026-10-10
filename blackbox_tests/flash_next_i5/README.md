# 已启动服务的独立质量用例

复用独立黑盒提交 `74888a0` 的原任务、检查器和冻结容差，不导入服务实现、不启动或停止服务。只在当前资源/压力验收没有失败后运行。

```bash
python -B blackbox_tests/flash_next_i5/quality.py \
  --url http://127.0.0.1:18230 --name M1 \
  --report /data2/servebig-envs/flash_next_i3_acceptance/i3b_blackbox/0abb433-m1-quality.jsonl
```

执行原用例 01/02/03/07/08：短语义与代码执行、coding/research 多轮、默认 thinking 的 chat 接口、固定 GSM8K 32 题、stream/非 stream、stop、EOS 和 ignore_eos。GSM8K 下限保持原先冻结的 26/32，不依据新候选结果调整。

每步保存请求、响应、结果及公开配置；失败立即结束。另存 `.records.json`，供后续相同任务的精度/路径/独立参考比较。原分池配额、专家增槽和固定16K错误断言不在本入口内；本次通过也不替代整个配置矩阵或最终质量对照。FTW 不在本期范围。
