# LingBot-VLA 2.0 · RoboTwin 联合训练与泛化评测

基于 **LingBot-VLA 2.0（6B）** 的具身模型后训练项目：从数据接入、视觉与状态增强、分布式训练，到检查点选型、闭环评测和实验追踪，构建一条可审计的 RoboTwin 50 任务联合训练流程。

**Full-parameter post-training · Multi-view augmentation · Stateful checkpointing · Closed-loop evaluation**

[项目演进](#项目演进) · [实验结果](#实验结果) · [训练曲线](#训练曲线) · [方法实现](#方法实现) · [复现指南](docs/REPRODUCIBILITY.md) · [工程设计](docs/ENGINEERING.md)

![v3 confirmation](assets/figures/confirmation.png)

## 项目演进

项目按 **v0流程验证 → v1正式基线 → v2视觉泛化 → v3精修与确认** 连续迭代。每一阶段都保留训练配置、指标来源、观察到的问题和下一步决策。

| 阶段 | 核心工作 | 实际训练 | 形成的下一步依据 |
|---|---|---|---|
| v0 | 打通数据、监督、训练和保存链路 | 1000步，global256 | 从可运行流程进入正式训练 |
| **v1** | **无视觉增强的全50任务基线** | **4600步，global1024** | 历史clean60.8/randomized34.4，识别泛化缺口 |
| v2 | c2r多视角视觉增强 | 20k日程，选19k检查点 | 泛化改善后进入平台期，转向精修 |
| v3 | 低LR、seen绑定、状态噪声；raw/EMA选型 | 从v2@19k权重续训4000步 | 4k raw完成1000集确认，均值65.6 |

阶段编号是项目版本，基座均为LingBot-VLA 2.0。v0、v1、v2分别从官方基座初始化，v3才直接继承v2；研究过程连续，权重谱系按实际记录展示。历史v1两setting都用unseen指令、exec50，不能与当前确认分直接作同口径涨幅计算。

各阶段的训练过程和曲线在下文直接展开；[训练历程档案](docs/TRAINING_HISTORY.md)保留更详细的来源与复现记录。

## 实验结果

当前选定模型为 **v3 · step 4000 · raw**。2026-10-10 完成全部 **50任务 × 2 setting × 10集** 确认评测，成功数来自逐集策略执行记录，并与任务结果文件及日志交叉核验。

| Setting | 指令 | 成功 / 实际回合 | 成功率 |
|---|---|---:|---:|
| Clean | seen | 404 / 500 | **80.8%** |
| Randomized | unseen | 252 / 500 | **50.4%** |
| 两 setting 等权均值 | — | 656 / 1000 | **65.6%** |

推理配置：**FP32、动作块50、执行前30步后重新观测、10步去噪、torch.compile**。一份权重覆盖全部50任务。

这些是内部确认结果，不是官方排名或官方复测成绩。历史5集选型最高67.4%单独归档，不与本次10集结果混算。训练演示限定为官方clean2500条；历史归一化统计来源的待复核项见[数据与统计溯源](docs/DATA_PROVENANCE.md)。

- [机器可读结果](results/evaluation/v3_confirmation.json) · [1000集逐集记录](results/evaluation/v3_episodes.jsonl) · [评测协议](results/evaluation/protocol.json)
- [逐任务可视化](assets/figures/task_breakdown.png) · [检查点选型与负结果](docs/EXPERIMENTS.md)

## 训练曲线

项目围绕同一个问题逐步推进：先把完整训练链路跑通，再建立50任务基线，随后改善跨场景泛化，最后进行低学习率精修和扩大样本确认。下图按阶段分别展示动作loss与学习率；各阶段使用独立优化步横轴。

![Training evolution](assets/figures/training_evolution.png)

### v0 · 打通训练与保存链路

最初从官方预训练基座开始，使用clean演示完成**1000步**流程验证：micro8 × 2卡 × accum16 = **global256**，Muon，LR **5e-5**。这一阶段验证数据读取、原生辅助监督、分布式训练和checkpoint保存，为正式基线提供可运行的起点。

v0没有用于代表最终性能的完整确认成绩。[原始训练指标](results/training/v0_tensorboard.csv)与[配置档案](configs/history/v0_original.yaml)均保留。

### v1 · 4600步正式基线

在流程验证之后，v1重新从**LingBot-VLA 2.0官方基座**训练，建立覆盖全部50任务的无视觉增强基线。实际训练**4600步**，原配置为micro32 × 8卡 × accum4 = **global1024**，Muon，cosine学习率 **1e-4 → 5e-5**。

![v1 baseline training](assets/figures/training_v1.png)

训练loss持续下降，但历史全50任务×5集评测中，clean为 **152/250 = 60.8%**，randomized为 **86/250 = 34.4%**。跨场景落差促使下一阶段研究视觉泛化，而不只是继续增加基线训练步数。

这组历史评测两边均使用**unseen指令、exec50**；不能与当前clean/seen、exec30的确认分直接计算同口径增益。v1当时的W&B同步因NumPy兼容问题失败，图表从完整TensorBoard事件恢复；重启重复记录按每个(step, metric)保留最新观测，未补点。

[4600步指标CSV](results/training/v1_tensorboard.csv) · [曲线来源](results/training/v1_provenance.json) · [原始配置](configs/history/v1_original.yaml) · [4卡便携配置](configs/v1.yaml) · [逐任务结果](results/evaluation/v1_4600_exec50.json)

### v2 · 用c2r增强改善视觉泛化

针对v1观察到的跨场景差距，v2从同一官方基座**重新训练**，保留Muon、global1024、动作损失与原生几何/视频监督，引入c2r多视角增强：光度、通道统计和频域扰动；同一相机的当前帧与未来帧复用参数，并保留20%原图直通。

v2使用**20k步学习率日程**，选取**19k检查点**。它继承v1的实验经验，权重并不从v1继续。

![v2 visual generalization training](assets/figures/training_v2.png)

历史exec50、两边unseen协议下，全50任务×5集为clean **194/250 = 77.6%**、randomized **122/250 = 48.8%**。后续采用exec30并纠正clean指令口径后，v2@19k记录为 **78.4 / 47.2**。这些结果按各自协议保存。

早期面板支持保留视觉增强；中后期训练loss继续下降，而闭环收益趋于平台，促使项目转向低学习率精修。训练量和日程等条件并未完全控制，因此不把v1到v2的全部差值归因于c2r。

[19000步指标CSV](results/training/v2_tensorboard.csv) · [来源与重启处理](results/training/v2_provenance.json) · [W&B运行](https://wandb.ai/3120252125-/ant-lingbot-competition/runs/sft-v2-aug-c2r-v1-clean2500-20k) · [训练配置](configs/v2.yaml) · [历史逐任务结果](results/evaluation/v2_19000_exec50.json)

### v3 · 低学习率精修、选型与确认

v3以 **v2@19k raw权重**初始化，重新建立优化器与调度器，训练**4000步**，cosine学习率 **2e-5 → 2e-6**。保留c2r，加入seen语义绑定和关节状态噪声；同时维护EMA，在2k与4k保存raw/EMA候选。

[![v3 refinement training](assets/figures/training_v3.png)](https://wandb.ai/3120252125-/ant-lingbot-competition/runs/sft-v3-tail-v1-clean2500-parent19000-r2)

通过检查点选型后采用**4k raw**，再完成全部50任务×2setting×10集确认，得到 **clean/seen 80.8%、randomized/unseen 50.4%、等权均值65.6%**。2k EMA与4k EMA的表现不同，选型保留了负结果，没有将EMA简单判定为普遍有效或无效。

v3图表直接来自已结束的W&B运行，导出**3998条指标记录，覆盖step1–4000**。浅色为原始值，深色为75条记录滑动均值；缺失点未填补。W&B访问沿用原项目权限，仓库内图片与CSV可直接查看。

[W&B原始运行](https://wandb.ai/3120252125-/ant-lingbot-competition/runs/sft-v3-tail-v1-clean2500-parent19000-r2) · [指标CSV](results/training/v3_wandb.csv) · [导出来源](results/training/v3_wandb_provenance.json) · [训练配置](configs/v3.yaml) · [检查点选型记录](docs/EXPERIMENTS.md)

### 权重谱系与经验迭代

```mermaid
flowchart LR
    Base[LingBot-VLA 2.0<br/>官方预训练基座]
    Base --> V0[v0 · 流程验证]
    Base --> V1[v1 · 4600步基线]
    Base --> V2[v2 · c2r / 19k检查点]
    V2 -->|weights-only| V3[v3 · 4000步精修]
    V0 -.流程经验.-> V1
    V1 -.泛化问题与方法迭代.-> V2
    V3 --> Confirm[4k raw<br/>1000集确认]
```

项目经验按v0→v1→v2→v3积累，实际权重初始化关系按图中的实线记录。训练loss用于复盘优化过程，模型选择依据闭环成功率。

## 方法实现

```mermaid
flowchart LR
    A[官方 clean 演示<br/>50任务 × 50条] --> B[LeRobot 数据与统一动作映射]
    B --> C[多相机 c2r 视觉增强<br/>当前与未来帧一致]
    B --> D[seen 指令语义绑定<br/>关节状态输入噪声]
    C --> E[LingBot-VLA 2.0<br/>VLM + MoE Action Expert]
    D --> E
    C --> F[冻结的几何与视频表征教师]
    F --> G[当前 / 未来辅助监督]
    G --> E
    E --> H[Muon + FSDP2<br/>分阶段学习率日程]
    H --> I[Raw / EMA 保存与导出]
    I --> J[单权重闭环评测<br/>逐集日志与结果核验]
```

**视觉泛化。**训练期引入逐相机采样的光度、通道统计和频域扰动；同一相机的当前帧/未来帧复用参数。保留20%原图直通，将hue范围限制为±0.03，避免颜色语义任务的监督被破坏。实现：[augment.py](overlays/lingbot-vla-v2/lingbotvla/data/vla_data/augment.py)。

**语言与状态正则。**从官方seen模板恢复对象、手臂和空间关系，按episode绑定候选指令；歧义匹配保留原句。2159/2500条轨迹具备有效替换候选。状态噪声仅施加于有效连续关节，动作目标、夹爪和补齐维度保持原义。实现：[seen_bindings.py](overlays/lingbot-vla-v2/lingbotvla/data/vla_data/seen_bindings.py)、[数据变换](overlays/lingbot-vla-v2/lingbotvla/data/vla_data/utils.py)。

**阶段续训与恢复。**v3从v2@19k权重初始化，重建优化器和调度器，执行4000步cosine精修。阶段契约绑定父权重路径、配方与seen清单摘要；保存worker随机数、优化器与分片EMA状态，拒绝不兼容恢复。实现：[training_stage.py](overlays/lingbot-vla-v2/lingbotvla/checkpoint/training_stage.py)、[ema.py](overlays/lingbot-vla-v2/lingbotvla/checkpoint/ema.py)。

**评测可信度。**显式绑定clean→seen、randomized→unseen，缩短测试回合数时仍保持100条描述生成规模；记录专家筛选和策略成败，交叉检查实际回合、结果和日志。支持显式GPU映射、服务就绪检查及CPU线程控制。

完整设计与取舍见[方法说明](docs/METHOD.md)。

## 快速查看与复现

仅查看实验记录和重绘图表，无需GPU：

```bash
python -m pip install -r requirements-report.txt
python scripts/verify_results.py
python scripts/plot_results.py
```

准备固定版本上游源码并应用项目改动：

```bash
python scripts/setup_sources.py
# 按 docs/REPRODUCIBILITY.md 安装官方训练/仿真环境及模型、数据资产
python scripts/train.py --stage v3 --gpus 0,1,2,3 --dry-run
```

训练与评测入口分别为 `train.sh`、`eval.sh`；先查实际GPU占用，训练与评测串行。当前发布版完成了CPU与入口检查，完整硬件复现沿用已验证的工作区流程；跨机器从零复现仍需目标环境验证。详细步骤见[复现指南](docs/REPRODUCIBILITY.md)。

## 真机演示

附带一个独立的双臂字母块排列演示，用于展示语言条件化操作与多机位观测。它使用独立真机数据，不参与上面的RoboTwin训练或分数计算。

[![Dual-arm demo](assets/demo/preview.jpg)](assets/demo/demo.mp4)

[观看完整演示](assets/demo/demo.mp4) · [演示说明](docs/DEMO.md)

## 仓库结构

```text
configs/          可迁移训练配方、版本锁与历史统计
scripts/          训练、评测、结果验证、制图与W&B同步
train.sh eval.sh  运行入口
overlays/         叠加到固定上游版本的项目改动
results/          W&B标量、确认结果、逐集记录与来源
assets/           曲线图、任务图和演示
runtime_patches/  H100 cuRobo运行时兼容处理
tests/            增强、语义绑定、EMA与恢复检查
docs/             方法、实验、工程设计与复现记录
```

模型权重、演示数据、仿真大资产和运行环境按原始发布渠道准备，不随代码仓库分发。

## 上游与许可

基于 [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) 与 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)，参与[天池蚂蚁灵波具身大模型挑战赛](https://tianchi.aliyun.com/competition/entrance/532514)。原生模型结构与预训练能力归属于上游；本仓库记录后训练、数据正则、恢复、实验追踪及评测工程改动。

代码采用Apache-2.0；上游和RoboTwin相关许可见[第三方说明](THIRD_PARTY.md)。
