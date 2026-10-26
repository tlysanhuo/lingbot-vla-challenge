"""Portable paths, configuration rendering and shared GPU resource checks."""
import json
import os
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]

def paths():
    return {"PROJECT_ROOT": str(ROOT),
            "MODEL_ROOT": os.environ.get("MODEL_ROOT", str(ROOT / "models")),
            "DATA_ROOT": os.environ.get("DATA_ROOT", str(ROOT / "data/RoboTwin_lerobot_v30")),
            "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", str(ROOT / "outputs"))}

def render_config(path):
    text = Path(path).read_text()
    for key, value in paths().items():
        text = text.replace("${" + key + "}", value)
    if "${" in text:
        raise ValueError("Unresolved configuration variable")
    return yaml.safe_load(text)

def gpu_gate(value, max_cards=4):
    gpus = value.split(",")
    if not gpus or len(gpus) > max_cards or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError("Select 1–4 distinct GPU indices")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {a.strip(): int(b) for a, b in (row.split(",") for row in raw.splitlines())}
    if any(g not in memory or memory[g] >= 1000 for g in gpus):
        raise RuntimeError("A requested GPU is unavailable or already occupied")
    # A shared host must not run our training and evaluation simultaneously.
    patterns = ("train_lingbotvla.py", "deploy.lingbot_vla_v2_policy", "eval_policy_client_lingbotvla.py")
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(arg == p or arg.endswith("/" + p) for arg in cmd for p in patterns):
                active.append(int(proc.name))
        except (OSError, ProcessLookupError):
            pass
    if active:
        raise RuntimeError(f"Existing training/evaluation workers detected: {active}")
    return gpus

def base_env():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env
