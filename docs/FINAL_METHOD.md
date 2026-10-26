# 最终方案：v7完整clean混合精修

本项目最终选择v7 step2000 raw。一份权重覆盖RoboTwin全部50任务，不按任务或setting切换模型。

## 完整本地成绩

| 版本 | randomized/unseen | clean/seen | 双场等权均分 |
| --- | --- | --- | --- |
| v3@4k raw | 2369/5000 = 47.38% | 3902/5000 = 78.04% | 62.71% |
| v5@3k raw | 2398/5000 = 47.96% | 3942/5000 = 78.84% | 63.40% |
| v7@2k raw | 2366/5000 = 47.32% | 3974/5000 = 79.48% | 63.40% |

每版都是50任务×2setting×100有效回合。v5/v7总分相同；v7比v5的clean多成功32集，randomized少成功32集。选择v7体现clean优先取舍，不声称它在总分上超过v5。与v3相比，v7clean +1.44pp、randomized −0.06pp、均分 +0.69pp。独立执行，不是严格逐seed配对；本地成绩不是官方排名。

## 权重谱系与预算

官方LingBot-VLA-v2-6b → v2@19000 raw → v3新增4000步raw → v7新增2000步raw。v5独立从v3继续3000步，不是v7的父权重。v6尝试停止并保留记录，没有可公布的完整10000集成绩，不能当作对照结论。

v7使用独立8卡节点，global1024=8×micro32×accum4；Muon、L1_fm、FP32、梯度检查点。200步warmup从2e-6到5e-6，cosine至1e-6，1000/2000步保存，提交raw。没有额外消融或多候选网格。

## 完整样本的50/50分支

每条训练样本50%概率保持完整clean当前/未来三相机图像，并关闭该样本state noise；另50%概率保留原C2R和原关节状态扰动（p=.5,std=.01）。动作监督保持官方演示真值。

完整clean分支覆盖全样本的当前/未来视角，与原C2R里逐相机20%直通不同。v7不启用局部擦除，冻结Depth/Video辅助监督、loss/padding和原统计保持配套。训练语言仅来自release seen绑定；没有randomized或unseen训练输入。

DP rank独立flow noise/time随机流写入恢复状态。它和样本混合共同进入本候选，没有另训归因对照，不把组合收益归因给某个单一改动。

## 评测与故障处理

两个阶段先randomized/unseen，再clean/seen；每任务100有效回合、描述规模100。FP32、chunk50、每次执行前30步后重观测、denoise10、torch.compile，一卡一个服务。

v5全量恢复保留所有已完成成败回合；v7发生Warp缓存冲突后关闭旧运行并重新评测。发布的v7只统计当前重启10000集，旧约624集排除。专家拒绝seed不计为模型失败；Warp缓存按进程隔离，规划器初始化失败立即中止，不持续空转跳seed。

## 配套统计和复现

训练均为官方clean2500演示、548985帧。原配套统计SHA256为488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8；未启用重算clean统计附件。其来源和历史限制如实保留在[DATA_PROVENANCE](DATA_PROVENANCE.md)。

[便携v7配置](../configs/v7.yaml) · [原始配置的脱敏档案](../configs/history/v7_original.json) · [复现步骤](REPRODUCIBILITY.md) · [完整机器可读对比](../results/evaluation/final_comparison.json) · [50任务表](FINAL_TASK_RESULTS.md)。代码Apache-2.0，第三方许可按THIRD_PARTY保留；仓库不包含比赛提交ZIP、大权重、官方训练数据或登录凭据。
