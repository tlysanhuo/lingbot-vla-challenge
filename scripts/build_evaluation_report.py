"""Build version archives, task tables and two real episode cases from committed data."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/evaluation'
START, END = '<!-- EVALUATION_REPORT:START -->', '<!-- EVALUATION_REPORT:END -->'
STAGES = {
    'v0': ('流程验证', '完成1000步训练，验证数据、监督、保存链路。原选点评测取消，现存评测目录中没有可核验的v0闭环成功率；不以零分代替缺失结果。'),
    'v1': ('正式基线与执行长度检查', '从官方基座训练4600步，建立无视觉增强基线。目录中的4k是截断命名；实际加载记录为global_step_4600。包含全50任务、15任务面板、exec30/50及启动失败记录。clean历史指令为unseen。'),
    'v2': ('视觉增强、检查点扫描与协议修正', '从基座重新训练，引入c2r增强，选用19k检查点。保留1k至19k面板/全任务结果、四检查点平均的负结果，以及exec30和clean/seen纠正后的独立评测。不同任务数或指令口径不合并。'),
    'v3': ('低学习率精修、raw/EMA选型与扩样', '承接v2@19k，新增4000步低LR精修、seen绑定和状态噪声。保留2k/4k raw/EMA矩阵、每任务10集确认，以及当前每任务100集评测。五集67.4与十集65.6分开记录；百集结果仍以快照状态为准。'),
    'v4': ('演示均衡续训与候选筛选', '承接v3@4k，实际新增6000步；50%帧均匀＋50%演示均匀采样，保留原增强与配套归一化。训练已完成，3k/6k raw筛选均有记录。3k randomized完整为49.0%；其clean和6k筛选提前停止，不报完整双场均分。'),
    'environment': ('环境验证记录', '环境/官方模型锚点单独保留，不划为我们训练出的v0至v4模型成绩。'),
}
STATUS = {'complete': '完成', 'in_progress': '进行中', 'stopped': '已停止',
          'not_started': '未开始', 'complete_50_tasks': '50任务完成',
          'subset_or_interrupted': '面板或部分完成', 'no_completed_task': '无完整任务结果'}


def load(path):
    return json.loads((ROOT / path).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def score(row, annotate=False):
    if not row or not row['attempts']:
        return '—'
    text = f"{100 * row['successes'] / row['attempts']:.1f}% ({row['successes']}/{row['attempts']})"
    if annotate and row.get('status', 'complete') != 'complete':
        text += ' †'
    return text


def model(run):
    steps = '/'.join(map(str, run['checkpoint_steps'])) or '?'
    return '16k–19k四点平均' if run['variant'].startswith('average') else f"{steps} · {run['variant']}"


def archives(index):
    rows, csv_rows = [], []
    for version, meta in index['versions'].items():
        data = load(meta['file'])
        runs = data['runs']
        title, intro = STAGES[version]
        lines = [f'# {version} · {title}', '', intro, '', f"快照时间：{index['snapshot_at']}。", '',
                 '[返回迭代总览](../EVALUATION_HISTORY.md) · [机器可读档案](../../' + meta['file'] + ')', '',
                 '成功率仅汇总有结果文件且日志计数一致的任务。未测任务不记零分；未满50任务的行是子集成绩。每次运行单独保留，重复/重启不拼接成更大样本。', '']
        overview = []
        for run in runs:
            row = [f"[{run['id']}](#{run['id']})", model(run), run['setting'],
                   '/'.join(run['instruction_types']) or '未知', str(run['execute_steps']),
                   run['completed_tasks'], score(run), STATUS[run['status']]]
            overview.append(row)
            for partial, values in [(False, run['per_task']), (True, run['partial_per_task'])]:
                for task, value in values.items():
                    csv_rows.append({'version': version, 'run_id': run['id'], 'checkpoint': model(run),
                                     'setting': run['setting'], 'task': task,
                                     'instruction_type': value['instruction_type'],
                                     'execute_steps': value['execute_steps'], 'successes': value['successes'],
                                     'attempts': value['attempts'],
                                     'success_rate': round(100 * value['successes'] / value['attempts'], 6),
                                     'partial_task': partial, 'source_run': run['source_run']})
        lines += [table(['运行', '检查点', 'Setting', '指令', '执行步数', '完整任务数', '完整任务子集成功率', '状态'], overview) if runs else '暂无可核验评测记录。', '']
        for run in runs:
            lines += [f"## {run['id']}", '', f"**{model(run)} · {run['setting']} · {STATUS[run['status']]}**", '',
                      f"来源运行：`{run['source_run']}`。", '',
                      '加载权重：' + ('；'.join('`' + p + '`' for p in run['checkpoint_paths']) or '日志未记录') + '。', '',
                      f"逐任务完整样本数分布：`{run['episodes_per_completed_task']}`；历史精度证据：`{run['precision']}`。", '']
            if run['per_task']:
                lines += ['<details>', '<summary>展开全部任务成功率</summary>', '',
                          table(['任务', '成功 / 回合', '成功率', '指令', '执行步数'],
                                [[f'`{task}`', f"{r['successes']}/{r['attempts']}", f"{100*r['successes']/r['attempts']:.1f}%",
                                  r['instruction_type'], r['execute_steps'] if r['execute_steps'] is not None else '未知']
                                 for task, r in sorted(run['per_task'].items())]), '', '</details>', '']
            else:
                lines += ['无可核验的完整任务结果，不报告成功率。', '']
            if run['partial_per_task']:
                lines += ['未完成任务中已有的策略回合（不并入上方完整任务子集成绩）：', '',
                          table(['任务', '已记录成功率'], [[f'`{t}`', score(r)] for t, r in sorted(run['partial_per_task'].items())]), '']
        out = ROOT / 'docs/evaluation' / (version + '.md')
        out.parent.mkdir(exist_ok=True)
        out.write_text('\n'.join(lines).rstrip() + '\n')
        rows.append([f'[{version} · {title}](evaluation/{version}.md)', meta['run_count'], meta['runs_with_results'], intro])
    with (DATA / 'all_task_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    text = '\n'.join(['# 全版本评测与迭代档案', '', f"快照时间：**{index['snapshot_at']}**。", '',
                      f"完整扫描现存 **{index['directory_count']}个评测目录**，其中 **{index['runs_with_completed_tasks']}个有可核验的完整任务结果**。零结果启动、部分完成、负结果与正式结果均保留。", '',
                      table(['版本', '运行目录数', '有完整任务结果的运行数', '本阶段做了什么'], rows), '',
                      '## 如何阅读版本间变化', '',
                      'v0、v1、v2分别从官方基座初始化；v3承接v2@19k，v4承接v3@4k。v4的3k/6k是该阶段新增步数，累计训练量分别约26k/29k。', '',
                      '早期clean/unseen、后期clean/seen，以及5集、10集、100集分开呈现。面板任务选择、指令抽样和执行长度变化都会影响成绩，不能把所有点连成一条同条件涨分曲线。', '',
                      '![同口径检查点迭代](../assets/figures/checkpoint_evolution.png)', '',
                      'v2图仅取全50任务、exec50、两setting均unseen的检查点扫描；v3图为全50任务、exec30、clean/seen与randomized/unseen的五集选型。图中的连线只展示该组观测，不表示某一改动的因果贡献。', '',
                      '## 任务级数据与最新评测', '',
                      '[全版本逐任务CSV](../results/evaluation/all_task_results.csv) · [档案索引JSON](../results/evaluation/history/index.json) · [当前候选50任务表](TASK_RESULTS.md) · [两个逐集case](EVALUATION_CASES.md)', '',
                      '目录档案覆盖本次能读取的原始评测产物。v0选点评测取消，没有数值；不根据训练loss或旧文档中的粗略估计补造任务成绩。运行目录名中的时间戳按原样保留，不猜测早期生成脚本的时区。', ''])
    (ROOT / 'docs/EVALUATION_HISTORY.md').write_text(text)


def latest(snapshot, baseline):
    tasks = snapshot['tasks']
    rows = []
    for key, run in snapshot['runs'].items():
        for setting, half in run['settings'].items():
            completed = {'successes': half['complete_task_successes'], 'attempts': half['complete_task_attempts']}
            rows.append([run['label'], setting + '/' + half['instruction_type'], STATUS[half['status']],
                         f"{half['tasks_complete']}/50", f"{half['episodes_recorded']}/{50*run['episodes_per_task']}", score(completed)])
    overview = table(['评测', 'Setting / 指令', '状态', '测满任务', '已记录回合', '完整任务子集成功率'], rows)
    current = snapshot['runs']['v3_full100']['settings']
    task_rows = []
    for task in tasks:
        r = current['randomized']['per_task'][task]
        delta = f"{100*r['successes']/r['attempts'] - 10*baseline['randomized'][task]['successes']:+.1f} pp" if r['status'] == 'complete' else '—'
        task_rows.append([f'`{task}`', score(baseline['clean'][task]), score(baseline['randomized'][task]),
                          score(r, True), score(current['clean']['per_task'][task], True), delta])
    tasks_table = table(['任务', 'v3 clean/seen · 10集', 'v3 randomized/unseen · 10集',
                         'v3 randomized/unseen · 100集目标', 'v3 clean/seen · 100集目标', '随机半场扩样差值'], task_rows)
    v4 = snapshot['runs']
    v4_table = table(['任务', 'v4 3k randomized · 10集', 'v4 3k clean · 10集', 'v4 6k randomized · 10集'],
                     [[f'`{task}`', score(v4['v4_3k10']['settings']['randomized']['per_task'][task], True),
                       score(v4['v4_3k10']['settings']['clean']['per_task'][task], True),
                       score(v4['v4_6k10']['settings']['randomized']['per_task'][task], True)] for task in tasks])
    report = ['# 当前候选分任务成功率', '', f"快照时间：**{snapshot['snapshot_at']}**。这是手动导出的静态快照，不是实时看板。", '',
              overview, '', '汇总列只计算测满目标回合的任务；已记录回合数包含尚未测满的任务。未开始显示“—”，不是0%；†表示不足目标回合的任务，分母为实际已完成回合。未完成两setting时不计算双场均值。', '',
              '## v3 · 十集确认与百集扩样', '', tasks_table, '',
              '扩样差值仅对同一任务测满100集后计算。新旧评测并非完全相同的指令/随机数重放；不将两轮拼接为110集。', '',
              '## v4 · 已训练候选的筛选记录', '', v4_table, '',
              'v4 3k的randomized已完成；clean及6k筛选提前停止，保留原始计数。v4 6k的clean未开始。', '',
              '[按版本查看全部历史评测](EVALUATION_HISTORY.md) · [快照JSON](../results/evaluation/latest_snapshot.json)', '']
    (ROOT / 'docs/TASK_RESULTS.md').write_text('\n'.join(report))
    return overview, tasks_table


def cases():
    # Two cases only: the first observed success and first observed failure of the
    # same task. These are actual records, not a claimed seed-paired intervention.
    source = 'results/evaluation/v3_full100_episodes.jsonl'
    events = [json.loads(s) for s in (ROOT / source).read_text().splitlines()]
    selected = []
    for outcome in [True, False]:
        line, event = next((i, e) for i, e in enumerate(events, 1)
                           if e['task'] == 'move_pillbottle_pad' and e['setting'] == 'demo_randomized' and e['success'] is outcome)
        selected.append({'case_id': 'success' if outcome else 'failure', 'checkpoint': 'v3 4k raw',
                         'episode_file': source, 'line': line, 'episode': event, 'video': None})
    (DATA / 'cases.json').write_text(json.dumps({'selection': 'First recorded success and first recorded failure for move_pillbottle_pad in the frozen v3 full100 randomized snapshot.',
                                               'video_status': 'deferred_until_current_full_evaluation_completes',
                                               'cases': selected}, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 两个真实评测case', '',
             '选择同一个任务的首次成功和首次失败各一例，均来自v3 4k raw百集评测的randomized/unseen记录。它们用于展示逐集证据，不是任务成功率的代表性抽样，也不是同seed对照。', '',
             '**视频待补录。**历史及当前全量运行没有保存仿真视频。按项目安排，当前全量评测结束后再单独补录1—2个case；届时标记为新的演示评测，不将补录成败改写为下面旧回合的结果，也不并入原评测分数。', '',
             '以下为原始日志中已经存在的真实记录。无视频时不能进一步断言失败是抓空、碰撞、识别错误或放置偏差。', '']
    compact = []
    for case in selected:
        e = case['episode']; label = '成功' if e['success'] else '失败'
        url = f"../{source}#L{case['line']}"
        lines += [f"## {label}case", '', table(['字段', '记录'], [
            ['模型', case['checkpoint']], ['任务', '`' + e['task'] + '`'], ['Setting / 指令', 'randomized / unseen'],
            ['回合 / seed', f"{e['episode']} / {e['seed']}"], ['实际指令', e['instruction'].replace('|', '\\|')],
            ['动作步数', e['action_steps']], ['环境成败判定', label]]), '', f'[原始逐集记录]({url})', '']
        compact.append([label, '`' + e['task'] + '`', 'v3 · randomized/unseen',
                        f"episode {e['episode']} / seed {e['seed']}", e['action_steps']])
    (ROOT / 'docs/EVALUATION_CASES.md').write_text('\n'.join(lines))
    return table(['Case', '任务', '模型 / Setting', '原始回合', '动作步数'], compact)


def figures(snapshot, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = ROOT / 'assets/figures'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.facecolor': '#f8fafc',
                         'savefig.facecolor': '#f8fafc', 'axes.spines.top': False, 'axes.spines.right': False})
    old1 = load('results/evaluation/v1_4600_exec50.json')['results']
    old2 = load('results/evaluation/v2_19000_exec50.json')['results']
    fig, axes = plt.subplots(1, 2, figsize=(16, 15.5))
    fig.subplots_adjust(left=.16, right=.985, top=.88, bottom=.07, wspace=.70)
    fig.text(.04, .965, '50 tasks across model iterations', fontsize=25, weight='bold', color='#17233b')
    fig.text(.04, .935, 'Success rate (%)  |  actual counts and all checkpoints in the version archives', fontsize=12, color='#526078')
    cmap = plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#e5e7eb')
    for ax, setting in zip(axes, ['clean', 'randomized']):
        columns = [old1[setting], old2[setting], baseline[setting], snapshot['runs']['v4_3k10']['settings'][setting]['per_task']]
        labels = ['v1 4.6k\n5 ep', 'v2 19k\n5 ep', 'v3 4k\n10 ep', 'v4 +3k\n10 ep']
        if setting == 'randomized':
            columns.append(snapshot['runs']['v3_full100']['settings'][setting]['per_task']); labels.append('v3 4k\n100 ep*')
        arr = np.array([[100 * col[t]['successes'] / col[t]['attempts'] if col[t]['attempts'] else np.nan for col in columns] for t in snapshot['tasks']])
        ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=0, vmax=100, aspect='auto')
        ax.set_yticks(range(50), [t.replace('_', ' ') for t in snapshot['tasks']], fontsize=7.5)
        ax.set_xticks(range(len(labels)), labels, fontsize=9); ax.xaxis.tick_top(); ax.tick_params(length=0, pad=6)
        ax.set_title('Clean  /  unseen (v1,v2) → seen (v3,v4)' if setting == 'clean' else 'Randomized  /  unseen', pad=40, fontsize=12, weight='bold')
        for i, task in enumerate(snapshot['tasks']):
            for j, col in enumerate(columns):
                value = arr[i, j]
                partial = col[task].get('status', 'complete') not in ('complete', 'not_started')
                text = '—' if np.isnan(value) else f'{value:.0f}' + ('*' if partial else '')
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if value >= 65 else '#17233b')
    fig.text(.04, .035, 'v1/v2: exec50, 5 episodes/task. v3/v4: exec30, 10 episodes/task. Protocols differ; this is not a controlled gain comparison.', fontsize=10, color='#526078')
    fig.text(.04, .015, f"Gray = not evaluated. * = partial task / ongoing 100-episode expansion. Snapshot: {snapshot['snapshot_at']}", fontsize=10, color='#526078')
    fig.savefig(out / 'iteration_task_rates.png', dpi=140); plt.close(fig)
    history = load('results/evaluation/history/v2.json')['runs']
    steps = [10000, 12000, 14000, 15000, 16000, 17000, 18000, 19000]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.24, wspace=.25)
    fig.text(.06, .94, 'Checkpoint selection within each evaluation protocol', fontsize=21, weight='bold', color='#17233b')
    for setting, color in [('clean', '#008b83'), ('randomized', '#4263eb')]:
        values = []
        for step in steps:
            matches = [r for r in history if r['checkpoint_steps'] == [step] and r['setting'] == setting
                       and r['completed_tasks'] == 50 and r['variant'] == 'raw'
                       and r['instruction_types'] == ['unseen'] and r['execute_steps'] == [50]]
            assert len(matches) == 1, (step, setting)
            values.append(matches[0]['success_rate'])
        axes[0].plot([s / 1000 for s in steps], values, 'o-', color=color, label=setting + ' / unseen', lw=2)
    axes[0].set_title('v2  /  full 50 tasks × 5 episodes · exec50', loc='left')
    axes[0].set_xlabel('Checkpoint (thousand steps)'); axes[0].legend(frameon=False, fontsize=9)
    selection = list(csv.DictReader((DATA / 'checkpoint_selection_5eps.csv').open()))
    positions = np.arange(len(selection)); labels = [r['checkpoint'].replace(' raw', '\nraw').replace(' EMA', '\nEMA') for r in selection]
    for offset, setting, color in [(-.18, 'clean', '#008b83'), (.18, 'randomized', '#4263eb')]:
        values = [float(r[setting]) for r in selection]
        bars = axes[1].bar(positions + offset, values, width=.34, label=setting, color=color)
        axes[1].bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    axes[1].set_xticks(positions, labels, fontsize=8); axes[1].set_title('v3 selection  /  full 50 tasks × 5 episodes · exec30', loc='left')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, 100); ax.set_ylabel('Success rate (%)'); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.text(.06, .07, 'Separate protocols: v2 clean uses unseen instructions; v3 clean uses seen. Historical 5-episode selection is not the 10/100-episode confirmation.', fontsize=9, color='#526078')
    fig.savefig(out / 'checkpoint_evolution.png', dpi=150); plt.close(fig)


def main():
    snapshot = load('results/evaluation/latest_snapshot.json')
    baseline = load('results/evaluation/v3_confirmation.json')['results']
    index = load('results/evaluation/history/index.json')
    archives(index)
    overview, tasks = latest(snapshot, baseline)
    samples = cases()
    figures(snapshot, baseline)
    block = '\n'.join([START, '', '### 从v0到v4：全部评测记录', '',
                      '[v0 流程验证](docs/evaluation/v0.md) · [v1 基线](docs/evaluation/v1.md) · [v2 增强与检查点扫描](docs/evaluation/v2.md) · [v3 精修与扩样](docs/evaluation/v3.md) · [v4 均衡续训](docs/evaluation/v4.md)', '',
                      f"已归档现存 **{index['directory_count']}个评测目录**，包括零结果启动、面板、未完成评测、raw/EMA选型与负结果。每个版本页面均可展开各次运行的全部任务成绩。[迭代总览](docs/EVALUATION_HISTORY.md) · [所有版本逐任务CSV](results/evaluation/all_task_results.csv)", '',
                      '![检查点迭代与选型](assets/figures/checkpoint_evolution.png)', '',
                      '### 50任务成功率：跨版本展示', '',
                      '图中每格为成功率，灰色表示未测；星号表示尚未测满目标回合。v1/v2采用exec50且clean也为unseen，v3/v4采用exec30和clean/seen，因此跨版本图用于展示实验过程，不代表严格同条件增益。', '',
                      '![全版本任务成功率](assets/figures/iteration_task_rates.png)', '',
                      '### 最新评测快照', '', f"**更新时间：{snapshot['snapshot_at']}**。这是静态记录；百集全量仍以实际完成状态为准，不将未完成任务补零或报告为最终均分。", '',
                      overview, '', '“已记录回合”包含未满任务；最右列只汇总已测满的任务，部分任务集合的成功率不能当作完整50任务成绩。', '',
                      '<details>', '<summary>展开v3全部50任务：十集确认与百集进度</summary>', '', tasks, '',
                      '†：任务未满目标回合，显示实际分母。扩样差值只对已测满100集的任务计算。', '', '</details>', '',
                      '[v3/v4完整分任务表](docs/TASK_RESULTS.md) · [快照JSON与逐集记录索引](results/evaluation/latest_snapshot.json)', '',
                      '### 两个case：成功与失败', '', samples, '',
                      '两个case来自同一任务的真实逐集记录，实际指令与原始记录链接见[案例页](docs/EVALUATION_CASES.md)。**评测视频待当前全量结束后补录1—2例**；补录会单独标注，不计入原分数。', '', END])
    path = ROOT / 'README.md'; readme = path.read_text()
    if START in readme:
        readme = readme[:readme.index(START)] + block + readme[readme.index(END) + len(END):]
    else:
        readme = readme.replace('\n## 训练曲线\n', '\n' + block + '\n\n## 训练曲线\n')
    path.write_text(readme)
    print('Built six version archives, task CSV, latest task tables, two cases, and two figures.')


if __name__ == '__main__':
    main()
