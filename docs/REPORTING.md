# 更新评测档案与图表

报告构建只读取日志、结果和训练标量，不调用模型或仿真。

## 从已有运行导出快照

在能读取原始评测目录的机器上，用三份状态文件更新当前候选快照：

```bash
python scripts/export_evaluation_snapshot.py \
  --state v3_full100=/path/to/v3_full/status.json \
  --state v4_3k10=/path/to/v4_3k/status.json \
  --state v4_6k10=/path/to/v4_6k/status.json
python scripts/export_evaluation_history.py --eval-root /path/to/evals
python scripts/build_evaluation_report.py
python scripts/verify_results.py
python scripts/verify_evaluation_history.py
```

新运行逐集JSONL只读取以换行符结束的完整记录。测满任务交叉核对协议、专家通过seed、实际成功数、结果文件与日志末次计数；未测满任务保留实际分母。已停止的v4记录不会自动补跑。

历史档案扫描指定目录下的每个运行，包括没有产生完整任务结果的启动。优先用推理日志中的实际权重路径识别版本、步数和raw/EMA；没有加载记录时保留未知项。早期日志按最后一段连续回合计数导出，不将重试累加为独立样本。JSON保留来源文件哈希和历史累计成功数；CSV保留每个运行的完整/部分任务计数。

生成器更新README中的`EVALUATION_REPORT`区域、各版本明细、任务表、两个case和评测图。其余训练叙述由维护者按已执行事实更新。仓库中的时间表示捕获快照的时间，不是持续刷新的实时状态。

## 重绘训练曲线

```bash
python scripts/plot_results.py
```

v4的原始6000步指标来自TensorBoard，来源摘要见`results/training/v4_tensorboard.provenance.json`。训练曲线用于描述优化过程，不作为闭环提分证据。

## 演示材料

当前保留两条真实日志case，仿真视频补录已移出计划。已有真机Demo继续使用现有材料，见[演示说明](DEMO.md)。当前工作重点为下一版候选接入与01–04正式提交材料。
