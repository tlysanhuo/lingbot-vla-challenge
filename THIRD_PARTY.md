# 第三方来源

- **LingBot-VLA 2.0**：Robbyant，Apache-2.0。固定版本见configs/upstream-lock.json。overlays中的上游派生文件保留原许可；VLM、MoE动作专家、几何/视频教师均来自上游。
- **RoboTwin**：MIT，许可存于licenses/RoboTwin-MIT.txt。固定版本见版本锁，仿真场景和资产需按原渠道准备。
- **VeOmni / LeRobot / PyTorch / W&B**：使用其接口和运行环境，遵循各自许可；W&B图表由本项目自己的训练指标导出绘制。
- **PatchWAM**：c2r增强参数设计参考其clean-to-random实验；本项目在LingBot数据路径实现，收紧颜色扰动并进行时间一致性检查。参考不代表复现其论文分数。
- cuRobo sm_90运行时兼容处理参考RoboTwin issue #452。

本仓库不重新分发基础模型、官方演示数据或仿真大资产。新增工程代码采用仓库根目录的Apache-2.0许可。
