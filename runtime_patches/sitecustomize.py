"""RoboTwin-env sitecustomize: opt-in cuRobo sm_90 safety patch (RoboTwin issue #452).

Set ANT_CUROBO_SAFE=1 to enable. When enabled, curobo's fused LBFGS CUDA kernel and
CUDA graphs are disabled at import time, working around the
`CUDA error: an illegal instruction was encountered` crash in lbfgs_step_cu.forward
on H100/H800 (sm_90) during MotionGen.warmup().

This is the community-documented workaround from
https://github.com/RoboTwin-Platform/RoboTwin/issues/452 — no upstream source files
are modified; off by default preserves official behavior.
"""
import os

if os.environ.get("ANT_CUROBO_SAFE") == "1":
    import importlib.abc
    import importlib.machinery

    class _CuroboPatchFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Intercept 'curobo.wrap.reacher.motion_gen' import and patch after load."""

        _target = "curobo.wrap.reacher.motion_gen"

        def find_spec(self, fullname, path=None, target=None):
            if fullname != self._target:
                return None
            # Let the normal machinery load it, then patch via exec_module wrapper.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    from curobo.opt.newton.lbfgs import LBFGSOpt

                    _orig_init = LBFGSOpt.__init__

                    def _patched_init(self, config=None):
                        if config is not None and hasattr(config, "use_cuda_kernel"):
                            config.use_cuda_kernel = False
                        _orig_init(self, config)
                        self.use_cuda_kernel = False

                    LBFGSOpt.__init__ = _patched_init
                    print("[sitecustomize] cuRobo sm_90 patch applied (use_cuda_kernel=False)", flush=True)
                except Exception as e:  # pragma: no cover
                    print(f"[sitecustomize] cuRobo patch failed: {e}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    import sys

    sys.meta_path.insert(0, _CuroboPatchFinder())
    print("[sitecustomize] ANT_CUROBO_SAFE=1: will patch curobo LBFGS on import", flush=True)
