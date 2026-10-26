# 复现指南

## 1. 最轻量的复查

只需Python及requirements-report.txt，无需数据或GPU：

```bash
python -m pip install -r requirements-report.txt
python scripts/verify_results.py
python scripts/verify_evaluation_history.py
python scripts/plot_results.py
```

verify_results检查历史十集与训练记录；verify_evaluation_history检查最新10000集快照、任务计数和档案一致性。图表由仓库中的CSV/JSON生成。

## 2. 固定源码与环境

建议Linux，Python3.12。训练沿用上游PyTorch2.8/CUDA12.8与flash-attn2.8；仿真使用独立RoboTwin环境（Python3.10、SAPIEN3.0.0b1、cuRobo0.7.8）。这不是跨CUDA/ROCm通用锁文件；AMD环境应使用其官方适配方案并单独说明协议。

```bash
python scripts/setup_sources.py
```

该脚本串行限速读取固定版本源码包，并应用overlays。已存在的src目录会拒绝覆盖。已有本地原始Git仓库时可离线准备：

```bash
python scripts/setup_sources.py --from-workspace /path/to/workspace
# workspace/src/lingbot-vla-v2 和 workspace/src/RoboTwin 为原始Git仓库
```

随后按[官方LingBot安装说明](https://github.com/Robbyant/lingbot-vla-v2)建立训练环境，按[固定RoboTwin评测说明](https://github.com/Robbyant/lingbot-vla-v2/blob/bc643d74a0127fab8788da993b261d4d64101138/experiment/robotwin/README.md)准备仿真环境与资产。MoGe/LingBot-Depth源码和utils3d等依赖遵照上游安装要求补齐。源码包无Git元数据时，editable安装可设置`SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0`。

## 3. 模型和数据位置

设置环境变量：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/RoboTwin_lerobot_v30
export OUTPUT_ROOT=/path/to/experiment_outputs
```

模型位置约定：

```text
$MODEL_ROOT/
├── pretrained-v2-6b/hf_ckpt/          指定预训练基座的可加载目录
├── Qwen3-VL-4B-Instruct/              tokenizer/config依赖
├── moge-2-vitb-normal/model.pt
└── lingbot-vla-v2-6b/
    ├── depth/model.pt
    └── dino_video/{teacher_step_10000.pth,config.yaml}
```

`DATA_ROOT`必须是官方clean2500演示的LeRobot目录，包含data/meta/videos。仿真对象、机械臂与纹理资产按官方渠道放到src/RoboTwin要求的位置；randomized资产仅供评测。

实际配套归一化统计见[DATA_PROVENANCE.md](DATA_PROVENANCE.md)。重现历史结果必须保留对应映射；设计新统计方案时须建立新实验，不能悄悄替换后继续报告旧分数。

## 4. 历史训练阶段

v1基线已包含，先选择需要复现的单一阶段：

```bash
bash train.sh --stage v1 --gpus 0,1,2,3 --dry-run
```

`configs/v1.yaml`保留4600步日程、global1024和无增强设置，按4卡×micro32×accum8适配；原8卡配置单独保存在`configs/history/v1_original.yaml`。v0配置仅作为流程验证档案。v1、v2分别从基座训练，不能将三阶段命令自动串成连续权重续训。

在训练环境执行，先进行不占卡的参数检查：

```bash
bash train.sh --stage v2 --gpus 0,1,2,3 --dry-run
```

配置使用global1024=32×4×8。确认资产与当前GPU占用后运行同一命令去掉`--dry-run`。W&B使用自己的登录凭据，脚本不包含API key。

**v2训练日程原本是20k步，选用19k检查点后结束。**为了复现该检查点的学习率轨迹，不要把max_steps直接改为19000；可按原日程运行并保留19k检查点。保存参数和实际GPU配置以本仓库历史配置说明为准。

构建seen绑定并准备v3：

```bash
python scripts/build_seen_bindings.py --dataset "$DATA_ROOT" --output "$OUTPUT_ROOT/prepared/seen_bindings.json"
bash train.sh --stage v3 --gpus 0,1,2,3 --dry-run
# 确认父权重位于 OUTPUT_ROOT/runs/sft_v2_aug_c2r_lr1e4_gb1024/checkpoints/global_step_19000/hf_ckpt
```

v3从父权重建立新阶段，4000步，2k/4k保存raw/EMA。去掉`--dry-run`才实际训练。不要与评测并行；本机四卡总配额还需核对其他会话/容器实际占用，脚本的本命名空间检查不能识别所有外部任务。

训练器原生记录TensorBoard。同步指标到自己的W&B运行：

```bash
python scripts/sync_wandb.py --tb_dir "$OUTPUT_ROOT/runs/sft_v3_tail_v1_r2/runs" --project YOUR_PROJECT --name YOUR_RUN
```

## 5. 闭环评测

在能同时调用训练/仿真两个环境的终端执行：

```bash
bash eval.sh \
  --checkpoint "$OUTPUT_ROOT/runs/sft_v3_tail_v1_r2/checkpoints/global_step_4000/hf_ckpt" \
  --gpus 0,1 --episodes 10 \
  --conda-sh /path/to/miniforge3/etc/profile.d/conda.sh \
  --inference-env lingbotvla --sim-env RoboTwin --dry-run
```

去掉`--dry-run`运行。默认100集；内部确认显式传10。新运行按randomized/unseen→clean/seen，FP32、chunk50/exec30、10去噪步。历史2026-10-10确认按当时已启动顺序clean→randomized完成，该事实保留在protocol.json。

用唯一新的`--output`目录区分不同实验，避免把已有任务记录混入新模型。client保留每集描述生成数量100；不能通过缩短该数量改变指令分布。部分任务失败须查看原始记录，不伪造完成数或靠反复重跑挑选成功。

H100运行时可使用runtime_patches/sitecustomize.py中的cuRobo兼容处理。该处理只关闭有问题的fused LBFGS kernel；不修改任务成功条件。图形驱动与Vulkan由目标机器正确安装。

## 6. 组件测试

源码准备完成后，在训练环境执行CPU测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/test_c2r_augment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/test_v3_preparation.py
```

分布式分片EMA测试使用CPU/Gloo两个rank，参见test_v3_sharded_ema.py参数。测试与干运行能验证接口、保存恢复逻辑和发布完整性；新机器的完整模型训练和仿真仍需在安装后验证。

## 7. 最终v7与v5的独立八卡复现

先按上文完成固定源码与公开资产准备，并构建seen绑定。v5/v7均从复现出的v3@4k继续，是两条不同谱系。

```bash
bash train.sh --stage v7 --gpus 0,1,2,3,4,5,6,7 --dedicated-node --dry-run
# 或 stage v5，分别对应configs/v7.yaml与configs/v5.yaml
```

这两个八卡入口仅用于独立节点，本机默认四卡配额及训练/评测串行约束保持。dry-run不占卡，去掉该标记才运行，提交仓库更新不启动新训练。v7代码覆盖包含完整clean分支和DP rank独立随机流；两者关闭时保留历史v2/v3契约。

```bash
bash eval.sh --checkpoint "$OUTPUT_ROOT/runs/sft_v7_clean_final_2k_v1_8gpu/checkpoints/global_step_2000/hf_ckpt" --gpus 0,1 --episodes 100 --conda-sh /path/to/miniforge3/etc/profile.d/conda.sh --output "$OUTPUT_ROOT/new_v7_eval" --dry-run
```

支持独立八卡加`--dedicated-node`；原统计、FP32/compile/chunk50/exec30/denoise10，先randomized后clean。仅成功重启的逐进程Warp客户端用于新运行，规划器失败立即退出。最终数据核验可运行`python scripts/verify_final_results.py`。公开代码和本地提交包复现入口有所封装区别，核心源码和配方一致。
