"""Bridge: stream TensorBoard scalars from the LingBot training run into wandb.

The upstream trainer writes metrics only via AsyncTBWriter (no wandb.log calls),
so its wandb run shows System panels only. This script reads the tfevents file,
logs all scalars into a companion wandb run, and tracks progress in a state file
so repeated invocations upload only new points.
"""
import argparse
import glob
import json
import os
import re

# numpy 2.x compat shim: the installed tensorboard still references np.string_/
# np.unicode_ (removed in numpy 2.0) at import time of its plugin machinery.
import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--project", default="ant-lingbot-competition")
    ap.add_argument("--name", required=True)
    ap.add_argument("--state", default=None, help="json file tracking last logged step per tag")
    args = ap.parse_args()

    state_path = args.state or os.path.join(os.path.dirname(args.tb_dir.rstrip("/")), ".tb_wandb_state.json")
    state = {}
    if os.path.exists(state_path):
        state = json.load(open(state_path))

    event_files = sorted(glob.glob(os.path.join(args.tb_dir, "events.out.tfevents.*")))
    if not event_files:
        print("no event files yet")
        return

    acc = EventAccumulator(args.tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    new_points = 0
    payload = []
    for tag in tags:
        last = state.get(tag, -1)
        for ev in acc.Scalars(tag):
            if ev.step > last:
                payload.append((ev.step, tag, ev.value))
                state[tag] = ev.step
                new_points += 1

    if new_points == 0:
        print(f"up to date ({len(tags)} tags)")
        return

    import wandb

    # Pin the run id (derived from --name) so repeated 20-min invocations append
    # to ONE run. resume="allow" without a fixed id would create a new run each
    # invocation and silently shard the curve across runs.
    run_id = re.sub(r"[^a-zA-Z0-9_\-]", "", args.name.replace(" ", "_"))[-128:] or "tb-bridge"
    run = wandb.init(project=args.project, id=run_id, name=args.name, resume="allow")
    # group by step to keep one wandb.log call per training step
    by_step = {}
    for step, tag, value in payload:
        by_step.setdefault(step, {})[tag.replace("/", "_")] = value
    for step in sorted(by_step):
        run.log(by_step[step], step=step)
    run.finish()

    json.dump(state, open(state_path, "w"))
    print(f"logged {new_points} points across {len(tags)} tags -> {run.url}")


if __name__ == "__main__":
    main()
