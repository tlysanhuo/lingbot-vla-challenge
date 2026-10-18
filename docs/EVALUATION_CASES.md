# 两个真实评测case

选择同一个任务的首次成功和首次失败各一例，均来自v3 4k raw百集评测的randomized/unseen记录。它们用于展示逐集证据，不是任务成功率的代表性抽样，也不是同seed对照。

完整评测未保存仿真视频，当前保留两条原始日志case。仿真视频补录已移出当前计划；现有真机Demo见[演示说明](DEMO.md)。

以下为原始日志中已经存在的真实记录。无视频时不能进一步断言失败是抓空、碰撞、识别错误或放置偏差。

## 成功case

| 字段 | 记录 |
| --- | --- |
| 模型 | v3 4k raw |
| 任务 | `move_pillbottle_pad` |
| Setting / 指令 | randomized / unseen |
| 回合 / seed | 2 / 100001 |
| 实际指令 | Grab the white cylindrical bottle and set it on the pad |
| 动作步数 | 150 |
| 环境成败判定 | 成功 |

[原始逐集记录](../results/evaluation/v3_full100_episodes.jsonl#L1302)

## 失败case

| 字段 | 记录 |
| --- | --- |
| 模型 | v3 4k raw |
| 任务 | `move_pillbottle_pad` |
| Setting / 指令 | randomized / unseen |
| 回合 / seed | 1 / 100000 |
| 实际指令 | Use the left arm to grab the plastic orange bottle with white lid and place it |
| 动作步数 | 400 |
| 环境成败判定 | 失败 |

[原始逐集记录](../results/evaluation/v3_full100_episodes.jsonl#L1301)
