# 数据、统计与资产溯源

## 演示来源

训练读取RoboTwin官方Aloha-AgileX 50任务×50条clean演示，LeRobot v3.0元数据为2500集、548985帧。randomized环境与unseen指令只用于评测，评测记录位于results/evaluation，不作为训练输入。

数据、模型和仿真资产由使用者从各自发布渠道获取，本仓库不重新分发。

## 历史归一化统计：明确的待核查项

已报告v2/v3配置使用上游`assets/norm_stats/robotwin.json`，当前文件的count为6062592，SHA256为：

`488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8`

该文件保存为`configs/robotwin_historical_norm_stats.json`用于追溯；它不是根据本项目548985帧clean数据重新生成的统计。当前加载路径从robot config获取norm_stats，不能只看训练YAML里字段就认定实际来源。

后续需要核实上游统计来源、历史运行所用版本及赛事适用边界。本仓库如实呈现这一未决项，不把现有模型描述成“全部训练监督已通过最终合规审核”。count本身不能代替数据来源证明，也不能据第三方转述直接宣告违规。

**不能只换推理统计继续沿用旧结果。**归一化/反归一化与权重共同决定动作语义；新的统计或映射必须重新验证，并保留完整谱系。本次发布没有修改模型、统计数值或确认成绩。

## 预训练表征教师

MoGe-2、LingBot-Depth和DINO-Video为官方配方的几何/视频监督链。模型来源、实际路径和辅助损失见configs；教师冻结，只在训练使用。它们不同于官方`lingbot-vla-v2-6b-robotwin`后训练策略，后者不作为本项目训练起点或教师。

## 结果资产

确认JSON保留真实attempts/successes；逐集JSONL保留task、setting、指令类型、seed、实际指令、policy成败和动作步数。内部机器路径、进程环境和异常堆栈未放入发布目录。W&B指标导出仅包含训练标量。
