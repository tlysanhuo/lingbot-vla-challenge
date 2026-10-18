# 数据、统计与资产溯源

## 演示来源

训练读取RoboTwin官方Aloha-AgileX 50任务×50条clean演示，LeRobot v3.0元数据为2500集、548985帧。randomized环境与unseen指令只用于评测，评测记录位于results/evaluation，不作为训练输入。

数据、模型和仿真资产由使用者从各自发布渠道获取，本仓库不重新分发。

## 实际配套归一化统计

已报告v2/v3/v4配置使用上游`assets/norm_stats/robotwin.json`，当前文件的count为6062592，SHA256为：

`488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8`

该文件保存为`configs/robotwin_historical_norm_stats.json`用于追溯；它不是根据本项目548985帧clean数据重新生成的统计。当前加载路径从robot config获取norm_stats，不能只看训练YAML里字段就认定实际来源。

2026-10-14的来源追查确认：该文件与上游首次提交`99370ae113efe4632fd7361ed32c0b45864731f5`中的统计逐字节一致；同版本的生成输入清单包含50项clean与50项randomized。[上游清单](https://github.com/Robbyant/lingbot-vla-v2/blob/99370ae113efe4632fd7361ed32c0b45864731f5/assets/training_data/robotwin.txt)与[协作者给出的生成命令](https://github.com/Robbyant/lingbot-vla-v2/issues/20#issuecomment-5175096550)建立了与混合示例配方的关联。

未保存该文件的原始生成执行日志。项目实际训练数据为clean2500，配套统计按上述原文件及哈希保留，便于重现历史权重。

另行重算的clean2500统计仅作为未启用附件，没有用于任何本项目训练或评测。已完成的百集评测使用原配套统计；v4也没有切换新统计，不将新附件写成已有成绩的来源。

**不能只换推理统计继续沿用旧结果。**归一化/反归一化与权重共同决定动作语义；新的统计或映射必须重新验证，并保留完整谱系。本次方案更新保留实际使用的模型和统计数值。

## 预训练表征教师

MoGe-2、LingBot-Depth和DINO-Video为官方配方的几何/视频监督链。模型来源、实际路径和辅助损失见configs；教师冻结，只在训练使用。它们不同于官方`lingbot-vla-v2-6b-robotwin`后训练策略，后者不作为本项目训练起点或教师。

## 结果资产

确认JSON保留真实attempts/successes；逐集JSONL保留task、setting、指令类型、seed、实际指令、policy成败和动作步数。内部机器路径、进程环境和异常堆栈未放入发布目录。W&B指标导出仅包含训练标量。
