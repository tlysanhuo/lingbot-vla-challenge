#!/usr/bin/env python3
"""c2r_v1 增强栈单元断言 + CPU 吞吐基准。

用法: <lingbotvla-env-python> ant/scripts/test_c2r_augment.py
断言内容（对应验收第 1 层）:
  1. 采样确定性: 同 seed 同参数
  2. 中性参数 = 逐位恒等（uint8 往返无损）
  3. 直通比例 ≈ 20%
  4. 跨相机独立采样
  5. 无 NaN / 值域 / dtype / 形状保持 ((C,H,W) 与 (T,C,H,W))
  6. prepare_images 集成: recipe 分支返回 {相机:参数}；未来帧复用同参数；
     recipe=None 时 legacy 路径行为不变
  7. 吞吐基准: 最坏情况（全算子开）与典型采样，480x640 与 256x256
"""
import sys
import time

import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1] / "src/lingbot-vla-v2")
sys.path.insert(0, ROOT)

from lingbotvla.data.vla_data.augment import (  # noqa: E402
    PASSTHROUGH_P,
    apply_c2r_augmentation,
    sample_c2r_params,
)
from lingbotvla.data.vla_data.transform import prepare_images  # noqa: E402

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")


NEUTRAL = {
    "brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0,
    "gamma": 1.0, "exposure_stops": 0.0, "color_temp": 0.0,
    "noise_std": 0.0, "blur_sigma": None,
}
WORST = {
    "brightness": 1.2, "contrast": 1.3, "saturation": 1.4, "hue": 0.1,
    "gamma": 1.3, "exposure_stops": 0.5, "color_temp": 0.15,
    "noise_std": 0.03, "blur_sigma": 1.5,
    "adain": {"mean": [0.8, 0.2, 0.5], "std": [0.3, 0.3, 0.3], "blend": 1.0},
    "fft_amp": [0.5, 0.8, 1.2, 1.5, 0.7],
}

print("== 1. 采样确定性 ==")
torch.manual_seed(42); p1 = sample_c2r_params()
torch.manual_seed(42); p2 = sample_c2r_params()
check("same seed -> identical params", p1 == p2)
torch.manual_seed(7); p3 = sample_c2r_params()
check("diff seed -> params differ (or both None)", p1 != p3 or p1 is None)

print("== 2. 中性参数逐位恒等 ==")
img = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
out = apply_c2r_augmentation(img, dict(NEUTRAL))
check("uint8 bitwise identity under neutral params", torch.equal(out, img))

print("== 3. 直通比例 ==")
torch.manual_seed(0)
n_none = sum(1 for _ in range(10000) if sample_c2r_params() is None)
check(f"passthrough ~20% (got {n_none/10000:.3f})", abs(n_none/10000 - PASSTHROUGH_P) < 0.02)

print("== 4. 跨相机独立 ==")
torch.manual_seed(1)
diff = same = both_none = 0
for _ in range(2000):
    a, b = sample_c2r_params(), sample_c2r_params()
    if a is None and b is None: both_none += 1
    elif a != b: diff += 1
    else: same += 1
check(f"independent sampling (same-params {same}/2000)", same < 20)

print("== 5. 值域/形状/dtype ==")
for shape in [(3, 480, 640), (2, 3, 256, 256)]:
    x = torch.randint(0, 256, shape, dtype=torch.uint8)
    y = apply_c2r_augmentation(x, dict(WORST))
    check(f"shape {shape} preserved", tuple(y.shape) == shape)
    check(f"shape {shape} dtype uint8 + finite + in range",
          y.dtype == torch.uint8 and bool(y.float().isfinite().all()) and int(y.max()) <= 255)
clip = torch.randint(0, 256, (4, 3, 120, 160), dtype=torch.uint8)
yc = apply_c2r_augmentation(clip, dict(WORST))
check("(T,C,H,W) clip handled", tuple(yc.shape) == (4, 3, 120, 160))

print("== 6. prepare_images 集成 ==")
obs = {
    "state": torch.zeros(14, dtype=torch.float32),
    "image": {
        "observation.images.cam_high": torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8),
        "observation.images.cam_left_wrist": torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8),
        "observation.images.cam_right_wrist": torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8),
    },
}
keys = list(obs["image"].keys())

def call_pi(seed, **kw):
    torch.manual_seed(seed)
    return prepare_images(None, obs, image_keys=keys, train=True,
                          return_augment_params=True, **kw)

imgs, _, _, _, params = call_pi(123, image_augment_recipe="c2r_v1")
check("c2r returns per-camera param dict", isinstance(params, dict)
      and set(params.keys()) == set(keys)
      and all(v is None or isinstance(v, dict) for v in params.values()))

torch.manual_seed(123); params_ref = {k: sample_c2r_params() for k in keys}
# 复用同一 params 再跑一次（未来帧语义）: 同 seed 下输出应逐位一致
torch.manual_seed(999)
imgs_a, *_ = (lambda r: (prepare_images(None, obs, image_keys=keys, train=True,
      image_augment_recipe="c2r_v1", augment_params=r)))(params)
torch.manual_seed(999)
imgs_b, *_ = (lambda r: (prepare_images(None, obs, image_keys=keys, train=True,
      image_augment_recipe="c2r_v1", augment_params=r)))(params)
check("future-frame replay: same params + seed -> identical output",
      all(torch.equal(a, b) for a, b in zip(imgs_a, imgs_b)))

imgs_leg, _, _, _, params_leg = call_pi(123, image_augment_recipe=None)
check("legacy path returns flat param dict (not per-camera)",
      params_leg is None or not set(params_leg.keys()) & set(keys))

print("== 7. 吞吐基准 (per camera-frame) ==")
def bench(shape, params, n=100):
    x = torch.randint(0, 256, shape, dtype=torch.uint8)
    apply_c2r_augmentation(x, params)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        apply_c2r_augmentation(x, params)
    return (time.perf_counter() - t0) / n * 1000

for shape in [(3, 480, 640), (3, 256, 256)]:
    w = bench(shape, dict(WORST))
    print(f"  {shape}: worst-case {w:6.1f} ms/frame")
for shape in [(3, 480, 640), (3, 256, 256)]:
    torch.manual_seed(3)
    ts = []
    for _ in range(100):
        p = sample_c2r_params()
        if p is None:
            continue
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        t0 = time.perf_counter(); apply_c2r_augmentation(x, p); ts.append(time.perf_counter() - t0)
    typical = 1000 * sum(ts) / len(ts)
    print(f"  {shape}: typical {typical:6.1f} ms/frame (n={len(ts)})")

print()
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
