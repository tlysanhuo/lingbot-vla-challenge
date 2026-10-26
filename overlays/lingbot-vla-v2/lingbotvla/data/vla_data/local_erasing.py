"""Replayable, input-only local erasing for one camera in one training sample."""
import math

import torch


CAMERAS = ('camera_top', 'camera_wrist_left', 'camera_wrist_right')
REPLAY_KEY = '__local_erasing__'


def validate_local_erasing(probability, area_min, area_max, camera):
    if not 0 <= probability <= 1:
        raise ValueError('local_erasing_probability must be in [0, 1]')
    if not 0 < area_min <= area_max < 1:
        raise ValueError('local erasing area must satisfy 0 < min <= max < 1')
    if camera not in CAMERAS:
        raise ValueError('Unsupported local erasing camera: ' + str(camera))


def sample_local_erasing(images, probability, area_min, area_max, camera):
    """Sample once with the worker's checkpointed CPU torch RNG.

    None is an explicit no-erasure decision and must also be replayed for the
    future frame. Disabled augmentation consumes no random numbers.
    """
    validate_local_erasing(probability, area_min, area_max, camera)
    if probability == 0:
        return None
    # FeatureTransform uses fully qualified observation keys; direct helpers
    # also accept the short camera names used by the public configuration.
    camera = camera if camera in images else 'observation.images.' + camera
    if camera not in images:
        return None
    if torch.rand(()).item() >= probability:
        return None
    image = images[camera]
    if image.ndim != 3:
        raise ValueError('Local erasing expects a (C,H,W) camera image')
    height, width = map(int, image.shape[-2:])
    area = area_min + (area_max - area_min) * torch.rand(()).item()
    side = max(1, min(height, width, round(math.sqrt(area * height * width))))
    y = int(torch.randint(height - side + 1, ()).item())
    x = int(torch.randint(width - side + 1, ()).item())
    return {'camera': camera, 'height': height, 'width': width,
            'y': y, 'x': x, 'side': side}


def apply_local_erasing(image, params):
    """Return a new student image; never mutate input/teacher image storage."""
    if params is None:
        return image
    if image.ndim != 3 or tuple(image.shape[-2:]) != (params['height'], params['width']):
        raise ValueError('Current/future images must share local-erasing geometry')
    y, x, side = params['y'], params['x'], params['side']
    if not (0 <= y <= image.shape[-2] - side and 0 <= x <= image.shape[-1] - side and side > 0):
        raise ValueError('Invalid local-erasing rectangle')
    fill = image.float().mean(dim=(-2, -1), keepdim=True)
    if not image.is_floating_point():
        fill = fill.round()
    result = image.clone()
    result[:, y:y + side, x:x + side] = fill.to(dtype=image.dtype)
    return result
