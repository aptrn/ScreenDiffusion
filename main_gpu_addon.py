import importlib.util, os, sys
from pathlib import Path
import tempfile, time, queue, random, threading, pathlib, subprocess, shutil, re
from collections import deque
from multiprocessing import get_context, Queue
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Mapping, NamedTuple, Optional, Deque, Any, Sequence, Tuple, Union
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import PIL.Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
import tkinter as tk
import venv

import ctypes
import ctypes.wintypes as wint

# The Render Plan (issue #6, spec 6). Stdlib only, no torch and no GUI, so both this
# process and the worker can import it and a plan can be validated on either side.
from render_plan import (
    CROP as PLAN_CROP,
    MASKED as PLAN_MASKED,
    DEFAULT_MAX_INSTANCES,
    INITIAL_PLAN_VERSION,
    ActivePlan,
    RenderPlan,
    global_plan,
    plan_from_fields,
    priority_case_plan,
    t_index_for_denoise,
    t_index_ladder,
    validate_plan,
)

# Which TensorRT engine a configuration needs, and whether it is already built
# (issue #38). Stdlib, and shared with `bench.cli`'s build guard rather than
# mirrored here: a build the window allows and the harness refuses is two rules.
import engine_cache

# The tracker and the worker's detector (issue #7, spec 5.1 C3/C4). Neither imports
# torch or ultralytics at module scope - the weights are loaded on the detector's
# own thread, inside the worker process - so this import costs the GUI nothing.
from detection import fps_payload, is_detect_frame
from detector_worker import BackgroundDetector, UltralyticsDetector, frame_to_array

# The selective render path (issue #8, spec 5.1 C5/C7): which regions this frame
# renders, and how they are blended back onto the capture. Stdlib and numpy - the
# blend itself runs on the device (issue #31), but `device_compositor` imports torch
# inside the functions that need it, so this import costs the GUI process nothing.
from region_scheduler import RegionScheduler
from compositor import CROP, MASKED
from device_compositor import DeviceCompositor, crop_to_canvas, to_canvas
# Temporal stability (issue #32, spec 8.5): the plan's `seed_policy` applied to the
# engine's latent noise. Stdlib here too - torch lives inside the methods that write
# a tensor, so the GUI process pays nothing for this import either.
from seeding import CanvasGeometry, NoiseField

# The engine's canvas, and the capture geometry that is no longer the same thing
# (issue #39, spec 8.2). Every TensorRT engine this app builds is 512x512 whatever
# its directory name claims (spec 7.2), so the canvas is fixed and it is the
# *capture* that moves: a 512x512 capture window is too small to get an object
# into frame at all, and under the masked primitive a bigger one would only give
# each object *fewer* diffusion pixels - which is why the capture size and the
# plan's `primitive` are two halves of one change.
DIFFUSION_CANVAS = 512

# What the capture-size box offers. The default is the canvas, so the app out of
# the box is the app every committed measurement was taken on.
DEFAULT_CAPTURE = "512 x 512 (canvas)"
CAPTURE_PRESETS: Dict[str, Tuple[int, int]] = {
    DEFAULT_CAPTURE: (DIFFUSION_CANVAS, DIFFUSION_CANVAS),
    "960 x 540": (960, 540),
    "1280 x 720": (1280, 720),
    "1920 x 1080": (1920, 1080),
}


def capture_size(label: str) -> Tuple[int, int]:
    """The capture geometry a preset label names, or the canvas.

    Unknown falls back rather than raising: the label reaches here from a widget
    whose value a stale preference could have set, and starting the worker on the
    size it has always used is a better answer than not starting it.
    """
    return CAPTURE_PRESETS.get(label, CAPTURE_PRESETS[DEFAULT_CAPTURE])

APP_ROOT = (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent)
INTERNAL_DIR = APP_ROOT / "_internal"
try:
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

RUNTIME_DIR = os.path.join(APP_ROOT, "venv")
PY_ENV = RUNTIME_DIR
_PIP_DL_PERCENT_RE = re.compile(r"Downloading\s+[^\n]*\s(\d{1,3})%")
_PIP_LEGACY_BAR_RE = re.compile(r"\[\s*(\d{1,3})%\]")
GPU_DIAGNOSTIC = True

APP_DIR = APP_ROOT 
RUNTIME = APP_ROOT / "venv"
PY_EXE = RUNTIME / "Scripts" / "python.exe"
PIP_EXE = RUNTIME / "Scripts" / "pip.exe"
SITE_PACKAGES = RUNTIME / "Lib" / "site-packages"
BOOT_FLAG = "SD_RUNTIME_BOOTSTRAPPED" 

_user32 = ctypes.windll.user32
_gdi32  = ctypes.windll.gdi32

CreateRectRgn = _gdi32.CreateRectRgn
CombineRgn    = _gdi32.CombineRgn
DeleteObject  = _gdi32.DeleteObject
SetWindowRgn  = _user32.SetWindowRgn
RGN_OR        = 2

CUSTOM_COLORS = {
    "success": "#10B981",
    "error": "#EF4444", 
    "surface": "#374151"
}

GWL_EXSTYLE       = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TOOLWINDOW  = 0x00000080
GetWindowLongW    = _user32.GetWindowLongW
SetWindowLongW    = _user32.SetWindowLongW
SetLayeredWindowAttributes = _user32.SetLayeredWindowAttributes
LWA_ALPHA         = 0x00000002

def _prime_dll_search(site_packages_path, internal_dir_path):
    if os.name != "nt":
        return
    dll_packages = ["tensorrt", "xformers", "torch"]
    for package in dll_packages:
        pkg_dir = site_packages_path / package
        if not pkg_dir.exists():
            continue
        for target_dir in [pkg_dir / "lib", pkg_dir]:
            if target_dir.exists():
                try:
                    os.add_dll_directory(str(target_dir))
                except Exception:
                    pass
                current_path = os.environ.get("PATH", "")
                if str(target_dir) not in current_path:
                    os.environ["PATH"] = str(target_dir) + os.pathsep + current_path
    
    possible_dll_dirs = [site_packages_path / "bin", site_packages_path / "Library" / "bin", internal_dir_path / "bin", internal_dir_path / "Library" / "bin"]
    for dll_dir in possible_dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass

if getattr(sys, "frozen", False):
    APP_ROOT_WORKER = Path(sys.executable).parent
    SITE_PACKAGES_WORKER = SITE_PACKAGES
else:
    APP_ROOT_WORKER = Path(__file__).resolve().parent
    APP_DIR_WORKER = Path(os.getenv("LOCALAPPDATA", os.getcwd())) / "ScreenDiffusion"
    SITE_PACKAGES_WORKER = APP_DIR_WORKER / "runtime" / "Lib" / "site-packages"

internal_str = str(SITE_PACKAGES_WORKER)
if internal_str not in sys.path:
    sys.path.insert(0, internal_str)

_prime_dll_search(SITE_PACKAGES_WORKER, INTERNAL_DIR)
_prime_dll_search(SITE_PACKAGES, INTERNAL_DIR)

def try_import_dependency(name: str):
    try:
        import importlib
        return importlib.import_module("torch")
    except Exception:
        return None

class PreloadedDependencies:
    _torch = None
    _streamdiffusion = None
    _stream_wrapper = None
    _cached_models = {}
    _numpy = None
    _pil = None
    
    @classmethod
    def get_torch(cls):
        if cls._torch is None:
            try:
                cls._torch = try_import_dependency("torch")
            except Exception:
                cls._torch = None
        return cls._torch
    
    @classmethod
    def get_streamdiffusion(cls):
        if cls._streamdiffusion is None:
            cls._streamdiffusion = try_import_dependency("streamdiffusion")
        return cls._streamdiffusion
    
    @classmethod
    def get_stream_wrapper(cls):
        if cls._stream_wrapper is None:
            cls._stream_wrapper = _load_stream_wrapper()
        return cls._stream_wrapper
    
    @classmethod
    def get_numpy(cls):
        if cls._numpy is None:
            import numpy as np
            cls._numpy = np
        return cls._numpy
    
    @classmethod
    def get_pil(cls):
        if cls._pil is None:
            from PIL import Image, ImageDraw
            cls._pil = Image
            cls._pil_draw = ImageDraw
        return cls._pil
    
    @classmethod
    def preload_all(cls):
        def background_preload():
            cls.get_torch()
            cls.get_streamdiffusion()
            cls.get_numpy()
            cls.get_pil()
        thread = threading.Thread(target=background_preload, daemon=True)
        thread.start()
        return thread

if not getattr(sys, "frozen", False):
    torch = PreloadedDependencies.get_torch()
else:
    torch = None

def _patch_torch_imports():
    import sys
    import types
    if 'unittest.mock' not in sys.modules:
        try:
            import unittest.mock as mock
            sys.modules['unittest.mock'] = mock
        except ImportError:
            class MinimalMock:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, *args, **kwargs): return self
                def __getattr__(self, name): return self
            mock_module = types.ModuleType('unittest.mock')
            mock_module.MagicMock = MinimalMock
            mock_module.Mock = MinimalMock
            mock_module.patch = MinimalMock
            mock_module.PropertyMock = MinimalMock
            mock_module.ANY = object()
            sys.modules['unittest.mock'] = mock_module

def resource_path(name: str) -> str:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        exe_dir = os.path.dirname(sys.executable)
        candidates = [os.path.join(base, name), os.path.join(exe_dir, name), os.path.join(base, "_internal", name)]
        for p in candidates:
            if os.path.exists(p):
                return p
        return os.path.join(exe_dir, name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

def _patch_missing_modules():
    import sys
    import types
    if 'modulefinder' not in sys.modules:
        try:
            import modulefinder
            sys.modules['modulefinder'] = modulefinder
        except ImportError:
            modulefinder_module = types.ModuleType('modulefinder')
            modulefinder_module.ModuleFinder = type('ModuleFinder', (), {})
            sys.modules['modulefinder'] = modulefinder_module

_patch_missing_modules()
_patch_torch_imports()

def check_xformers_availability():
    try:
        import xformers
        import xformers.ops
        try:
            if hasattr(xformers.ops, 'memory_efficient_attention'):
                return True, "xformers is available"
            else:
                return False, "xformers installed but memory_efficient_attention not available"
        except Exception as e:
            return False, f"xformers installed but has compatibility issues: {e}"
    except ImportError:
        return False, "xformers not installed"
    except Exception as e:
        return False, f"xformers error: {e}"

def verify_streamdiffusion_dependencies():
    dependencies = ["streamdiffusion", "PIL", "numpy"]
    optional_deps = ["torch", "torchvision", "torchaudio", "diffusers", "transformers", "cv2", "xformers"]
    missing = []
    warnings = []
    for dep in dependencies:
        try:
            if dep == "PIL":
                import PIL.Image
                PIL.Image.new('RGB', (10, 10))
            elif dep == "cv2":
                import cv2
                cv2.__version__
            else:
                module = __import__(dep)
        except ImportError as e:
            missing.append(f"{dep}: {e}")
        except Exception as e:
            warnings.append(f"{dep}: loaded but has issues - {e}")
    for dep in optional_deps:
        try:
            if dep == "PIL" or dep == "cv2":
                continue
            elif dep == "xformers":
                xformers_available, xformers_msg = check_xformers_availability()
                if not xformers_available:
                    warnings.append(f"xformers: {xformers_msg} (optional)")
            else:
                module = __import__(dep)
        except ImportError:
            warnings.append(f"{dep}: not installed (optional)")
        except Exception as e:
            warnings.append(f"{dep}: loaded but has issues - {e}")
    return missing, warnings

def maybe_bootstrap_gpu(on_progress=None, progress_callback=None):
    if on_progress: on_progress("Checking for PyTorch 2.1.0+cu118...")
    torch_available = False
    try:
        import torch
        import torchvision
        torch_version = torch.__version__
        if "+cu124" in torch_version:
            if torch.cuda.is_available():
                if on_progress: on_progress(f"PyTorch {torch_version} with CUDA already available!")
                torch_available = True
                return True
            else:
                if on_progress: on_progress(f"PyTorch {torch_version} found but CUDA not available")
        else:
            if on_progress: on_progress(f"Found PyTorch {torch_version}, but need 2.1.0+cu118. Reinstalling...")
    except ImportError:
        if on_progress: on_progress("PyTorch not found, starting installation...")
    except Exception as e:
        if on_progress: on_progress(f"PyTorch check error: {e}")
    
    if not torch_available:
        if on_progress: on_progress("Installing PyTorch 2.1.0+cu118 and TorchVision 0.16.0+cu118...")
        try:
            py_cmd = _python_cmd_for_pip()
            torch_cmd = [*py_cmd, "-m", "pip", "install", "--upgrade", "--no-deps", "-vv", "torch==2.5.1+cu124", "torchvision==0.20.1+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"]
            if on_progress: on_progress("Downloading PyTorch 2.1.0 and TorchVision 0.16.0 packages...")
            for line in _gpu_run(torch_cmd, on_progress=on_progress, no_window=not GPU_DIAGNOSTIC, progress_callback=progress_callback):
                pass
            if on_progress: on_progress("PyTorch 2.1.0+cu118 installation complete!")
            if on_progress: on_progress("Verifying PyTorch installation...")
            try:
                import sys
                if 'torch' in sys.modules: del sys.modules['torch']
                if 'torchvision' in sys.modules: del sys.modules['torchvision']
                import torch
                import torchvision
                torch_version = torch.__version__
                torchvision_version = torchvision.__version__
                cuda_available = torch.cuda.is_available()
                if on_progress: on_progress(f"Installed: PyTorch {torch_version}, TorchVision {torchvision_version}")
                if cuda_available:
                    if on_progress: on_progress(f"Verification successful! CUDA is available")
                    return True
                else:
                    if on_progress: on_progress("PyTorch installed but CUDA not available (may need GPU drivers)")
                    return True
            except Exception as e:
                if on_progress: on_progress(f"Verification failed: {e}")
                return False
        except Exception as e:
            if on_progress: on_progress(f"PyTorch installation failed: {e}")
            return False
    return torch_available

def _startup_probe(log_fn=None, logfile_name="ScreenDiffusion_probe.txt"):
    import sys, os
    from pathlib import Path
    log_path = Path(os.environ.get("LOCALAPPDATA", ".")) / logfile_name
    def p(msg: str):
        if callable(log_fn):
            try: log_fn(msg)
            except Exception: pass
        try:
            with open(log_path, "a", encoding="utf-8") as f: f.write(msg + "\n")
        except Exception: pass
    p("=== STARTUP PROBE ===")
    try:
        p(f"frozen? {getattr(sys, 'frozen', False)}")
        p(f"exe: {sys.executable}")
        p(f"argv0: {sys.argv[:2]}")
        p(f"PYTHONPATH: {os.environ.get('PYTHONPATH','')}")
        p(f"PATH[0:3]: {os.environ.get('PATH','').split(os.pathsep)[:3]}")
        p(f"sys.path[0:5]: {sys.path[:5]}")
        sp1 = RUNTIME / "Lib" / "site-packages"
        p(f"RUNTIME: {RUNTIME}  exists={RUNTIME.exists()}")
        p(f"site-packages: {sp1}  exists={sp1.exists()}  on_sys_path={str(sp1) in sys.path}")
        from pathlib import Path as _P
        if getattr(sys, "frozen", False):
            sp2 = _P(sys.executable).parent / "_internal" / "Lib" / "site-packages"
            p(f"_internal site-packages: {sp2}  exists={sp2.exists()}  on_sys_path={str(sp2) in sys.path}")
        try:
            import importlib
            t = importlib.import_module("torch")
            p(f"import torch (exe): OK v{t.__version__}")
        except Exception as e:
            p(f"import torch (exe): FAIL -> {e!r}")
        try:
            for line in _gpu_run([str(PY_EXE), "-c", "import torch,sys;print('VENV_TORCH',torch.__version__)"]):
                p(f"[venv] {line}")
        except Exception as e:
            p(f"[venv] import torch: FAIL -> {e!r}")
    finally:
        p("=== END PROBE ===")

try:
    import os
    if os.environ.get(BOOT_FLAG) == "1": os.environ.pop(BOOT_FLAG, None)
except Exception:
    pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
GD_DETERMINISTIC = bool(int(os.getenv('GD_DETERMINISTIC', '0')))
# An override, and only that. Left empty the app looks for a model under its own
# models root - see `resolve_local_model_path`, which is what the field starts on.
LOCAL_MODEL_PATH = r""

# `models/` (multi-GB downloads) and `engines/` (compiled TensorRT engines) are
# gitignored, so a fresh git worktree has neither. These variables point both at one
# shared location; unset, they sit next to the app exactly as before.
SD_MODELS_DIR_ENV = "SD_MODELS_DIR"
SD_ENGINES_DIR_ENV = "SD_ENGINES_DIR"


# A path argument: a string, a Path, or nothing given.
_PathArg = Optional[Union[str, Path]]


def _unquoted_path(value: _PathArg) -> str:
    """`value` as a bare path string, empty when there is nothing usable.

    A path pasted into a Windows env var often keeps its surrounding quotes.
    """
    return "" if value is None else str(value).strip().strip('"').strip()


def _resolve_cache_dir(env_var: str, default_name: str, explicit: _PathArg = None,
                       base_dir: _PathArg = None,
                       environ: Optional[Mapping[str, str]] = None) -> Path:
    """Absolute cache root: `explicit`, else $env_var, else `<app root>/<default_name>`.

    Absolute at the resolution point on purpose - a relative path handed to the
    wrapper re-anchors to the worker's cwd, which is the bug this exists to fix.
    A relative *value* is anchored to the app root too, so the answer never
    depends on where the process was started from.

    `_resolve_engine_dir()` in wrapper.py mirrors this rule. The two cannot share
    an implementation: wrapper.py imports torch, and the GUI process must not.
    """
    environ = os.environ if environ is None else environ
    base = Path(APP_ROOT if base_dir is None else base_dir)
    raw = _unquoted_path(explicit) or _unquoted_path(environ.get(env_var))
    candidate = Path(raw).expanduser() if raw else base / default_name
    if not candidate.is_absolute():
        candidate = base / candidate
    # normpath, not resolve(): collapse `..` and settle on one slash direction
    # without touching the filesystem or following symlinks.
    return Path(os.path.normpath(candidate))


def resolve_models_dir(explicit: _PathArg = None, base_dir: _PathArg = None,
                       environ: Optional[Mapping[str, str]] = None) -> Path:
    """Where downloaded models live. See `SD_MODELS_DIR` in CLAUDE.md."""
    return _resolve_cache_dir(SD_MODELS_DIR_ENV, "models", explicit, base_dir, environ)


def resolve_engines_dir(explicit: _PathArg = None, base_dir: _PathArg = None,
                        environ: Optional[Mapping[str, str]] = None) -> Path:
    """Where compiled TensorRT engines live. See `SD_ENGINES_DIR` in CLAUDE.md."""
    return _resolve_cache_dir(SD_ENGINES_DIR_ENV, "engines", explicit, base_dir, environ)


def _cache_paths_banner(models_root: Path, engines_root: Path) -> str:
    """The line the worker logs at startup, so a wrong root is visible immediately."""
    return f"Cache roots: models={models_root} | engines={engines_root}"


# --- what the app starts on (issue #40, step 1) ------------------------------
#
# Two settings the app could not run correctly without, both wrong by default and
# both found by starting it rather than by reading it.

# A diffusers folder is one with a `model_index.json`. A directory a cancelled
# download left behind is not something the worker can load, and starting on one
# fails minutes later inside another process.
MODEL_INDEX = "model_index.json"

# Preferred, in this order, before anything else under the models root: the
# Download button writes `sd-turbo-fp16` there and `docs/second-machine.md` names
# the same folder, so a machine set up either way is found without a guess.
LOCAL_MODEL_NAMES = ("sd-turbo-fp16", "sd-turbo")


def is_diffusers_dir(path: _PathArg) -> bool:
    """Could the worker actually load this directory?"""
    return bool(path) and (Path(path) / MODEL_INDEX).is_file()


def resolve_local_model_path(explicit: _PathArg = None, models_root: _PathArg = None,
                             environ: Optional[Mapping[str, str]] = None) -> str:
    """The model folder the field starts on: `explicit`, else a local one, else "".

    The app could not start until someone pasted an absolute path, and nothing said
    so or said where to point it - while on most machines here a complete model
    already sits under the models root. Returning "" when there is genuinely
    nothing local is the old behaviour, which is what Browse and Download are for.
    """
    chosen = _unquoted_path(explicit)
    if chosen:
        return chosen
    root = resolve_models_dir(environ=environ) if models_root is None else Path(models_root)
    named = [root / name for name in LOCAL_MODEL_NAMES]
    try:
        rest = sorted(p for p in root.iterdir() if p.is_dir() and p not in named)
    except OSError:
        # No models root at all - a fresh worktree, or a machine before setup.
        rest = []
    for candidate in named + rest:
        if is_diffusers_dir(candidate):
            return str(candidate)
    return ""


# --- choosing the base model (issue #38, step 6) -----------------------------
#
# The model was a path in an entry box, and changing it meant knowing three other
# settings had to change with it. SD-Turbo is distilled to one step and fuses no
# LCM-LoRA; SD 1.5 is not and does, and rendered at one step with no LCM-LoRA it
# produces noise rather than a weaker restyle. So the model is picked from what is
# on disk, and picking one applies what it needs.

# What issue #38's step-count sweep measured SD 1.5 needs to run at all, and what
# SD-Turbo is distilled for. Not preferences: 4 steps is a batch-4 UNet engine and
# 1 step is a batch-1 one, so this number keys the build the next section warns about.
MODEL_STEPS_SD15 = 4
MODEL_STEPS_TURBO = 1

# The one-step schedule the window has always opened on, and what SD-Turbo wants.
# Index 30 is timestep 399: a higher index is *less* denoise, not more.
DEFAULT_T_INDEX_LIST: Tuple[int, ...] = (30,)


class ModelCompanions(NamedTuple):
    """The settings a base model cannot render without, and cannot choose itself."""

    use_lcm_lora: bool
    t_index_list: List[int]


def local_model_paths(models_root: _PathArg = None,
                      environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Every loadable model under the models root, the preferred names first.

    The same two rules `resolve_local_model_path` applies, for the same reasons: a
    directory without a `model_index.json` is a cancelled download the worker
    cannot load, and the folder the Download button writes is what a machine set up
    either way already has, so it opens the list.
    """
    root = resolve_models_dir(environ=environ) if models_root is None else Path(models_root)
    named = [root / name for name in LOCAL_MODEL_NAMES]
    try:
        rest = sorted(p for p in root.iterdir() if p.is_dir() and p not in named)
    except OSError:
        rest = []
    return [str(path) for path in named + rest if is_diffusers_dir(path)]


def model_label(path: _PathArg) -> str:
    """What a model is called in the picker: its folder name, not its whole path."""
    return Path(str(path)).name if path else ""


def model_companions(path: _PathArg) -> ModelCompanions:
    """LCM-LoRA and the step ladder this base model needs.

    `engine_cache.is_turbo_model` is `wrapper.py`'s own rule - `"turbo" in the
    path` - and it is the rule that decides whether LCM-LoRA is fused at all, so
    the window has to answer it the same way or offer a setting the worker ignores.
    The ladder is `render_plan.t_index_ladder`, so what the window builds and what
    a plan asks for are one schedule.
    """
    opening = DEFAULT_T_INDEX_LIST[0]
    if not path or engine_cache.is_turbo_model(path):
        return ModelCompanions(
            use_lcm_lora=False,
            t_index_list=t_index_ladder(opening, MODEL_STEPS_TURBO))
    return ModelCompanions(
        use_lcm_lora=True,
        t_index_list=t_index_ladder(opening, MODEL_STEPS_SD15))


# --- choosing a style LoRA (issue #44, steps 3-5) ----------------------------
#
# The same shape as the model picker above, for the same reason: the style LoRAs
# live in one known directory, there are three of them, and a file browser for
# three files in a known place is the wrong control. It is also how the wrong path
# spelling reached the engine key - Tk's dialog returns forward slashes - though
# what fixed *that* is `engine_cache.normalize_lora_key`, not this list.

# Where a staged LoRA lives, under the models root. `bench.models.LORAS_SUBDIR`
# spells the same directory, so a machine set up for the harness is one set up
# for the window.
LORAS_SUBDIR = "loras"

# What `_add_lora`'s dialog already accepts, as suffixes rather than as a glob -
# the dropdown and the browser must offer the same set of files.
LORA_SUFFIXES = (".safetensors", ".bin", ".pt")

# In the same directory and *not* a style: it is fused by the LCM-LoRA switch,
# which is a separate setting on the worker. Offering it here invites fusing it
# twice (issue #44's third trap).
LCM_LORA_FILENAME = "lcm-lora-sdv1-5.safetensors"

# The picker's resting value, and what it shows when the models root holds no
# style LoRA at all. Neither is a filename: a LoRA is *added* by choosing it, so a
# menu resting on a name would read as a selection that is not one. Both fall
# through `_on_lora_chosen`'s search for a listed file and add nothing.
ADD_LORA_PROMPT = "+ Add a style LoRA"
NO_LOCAL_LORAS = "no LoRAs in the models root"

# What a newly added LoRA is fused at until the slider is moved. It is also what
# `bench.scenarios.ScenarioConfig.lora_scale` defaults to, which is the scale the
# committed style engines were built at - so a listed LoRA reports `cached`
# without the user having to guess the number back (issue #44's fourth trap).
DEFAULT_LORA_SCALE = 1.0


def local_lora_paths(models_root: _PathArg = None,
                     environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Every style LoRA staged under the models root, in name order."""
    root = resolve_models_dir(environ=environ) if models_root is None else Path(models_root)
    try:
        entries = sorted((root / LORAS_SUBDIR).iterdir())
    except OSError:
        # No models root, or no `loras` under it - a machine before setup.
        return []
    return [str(path) for path in entries
            if path.is_file() and path.suffix.lower() in LORA_SUFFIXES
            and path.name != LCM_LORA_FILENAME]


def lora_label(path: _PathArg) -> str:
    """What a LoRA is called in the picker: its filename, not its whole path."""
    return Path(str(path)).name if path else ""


def _lora_phrase(lora_dict: Optional[Dict[str, float]]) -> str:
    """` with style-x.safetensors @ 0.90`, or nothing at all when none is fused.

    The scale is in it because the scale keys the engine: a sentence naming only
    the file would say `cached` about a build at another strength (issue #44's
    fourth trap). Empty rather than "no LoRAs", so the sentence a user has always
    read about the base model alone is unchanged.
    """
    if not lora_dict:
        return ""
    fused = ", ".join(f"{lora_label(path)} @ {scale:.2f}"
                      for path, scale in sorted(lora_dict.items()))
    return f" with {fused}"


# StreamDiffusion's three acceleration paths. `tensorrt` is the default because it
# is the only one this repo has ever measured: every figure in spec 7, every
# committed benchmark and the ~5 GB engine cache under `engines/` belong to that
# path. Starting on `xformers` silently selected something slower than anything
# anyone had benchmarked, and left the built engine unused.
NO_ACCELERATION = "none"
XFORMERS = "xformers"
TENSORRT = "tensorrt"
ACCELERATIONS = (NO_ACCELERATION, XFORMERS, TENSORRT)
DEFAULT_ACCELERATION = TENSORRT

# The batch size every committed benchmark and every cached engine was built at.
DEFAULT_FRAME_BUFFER_SIZE = 1


# --- what a change costs, said before it happens (issue #40, step 4) ---------

# Measured at 512x512 on both machines: ~5.0 GB on disk, and 15-25 minutes on the
# RTX 3080 laptop against ~5 on the 4090 (CLAUDE.md, spec 7.2). A user who changes
# one of these and then watches the app go quiet has been told nothing. One
# spelling, shared with the harness that refuses the same build.
ENGINE_BUILD_SIZE = engine_cache.ENGINE_BUILD_SIZE
ENGINE_BUILD_TIME = engine_cache.ENGINE_BUILD_TIME

# The settings that key a distinct engine *and* have a control in this window.
# Resolution keys one too and is not here: `wrapper.py` never forwards a resolution
# to the builder, so every engine this app builds is 512x512 (spec 7.2) and no
# widget changes it. A warning nothing can reach is a warning nobody ever sees.
ENGINE_KEYED_SETTINGS = {
    "step count": "The number of denoising steps is compiled in. Moving a step's "
                  "value is a runtime update and costs nothing.",
    "batch size": "The frame buffer size is compiled in.",
    "LoRA set": "TensorRT fuses LoRA weights into the UNet before it compiles it, "
                "so each combination of LoRAs and scales needs its own engine.",
}

# The standing version of the same fact, shown in the advanced section whether or
# not anything is being changed.
ENGINE_REBUILD_HINT = (
    "⚠  Step count, batch size and LoRAs each key their own TensorRT engine: "
    f"{ENGINE_BUILD_TIME} to build and {ENGINE_BUILD_SIZE} on disk, per combination."
)


def engine_rebuild_needed(acceleration: str) -> bool:
    """Only the TensorRT path compiles an engine; the others merely run slower."""
    return acceleration == TENSORRT


class EngineConfiguration(NamedTuple):
    """Which engine the settings in the window need, and whether it exists yet.

    `builds` is the whole question a user wants answered before Start: this
    configuration has no compiled engine, so pressing Start spends minutes making
    one. Before issue #38 nothing looked - `_confirm_engine_rebuild` fired on a
    non-default batch size or any listed LoRA and never on a model change, which
    is the one setting that guarantees a different engine.
    """

    model_path: str
    engine_dir: str
    engines_root: str
    cached: bool
    builds: bool
    steps: int
    free_bytes: int
    enough_disk: bool
    # The fused set, already in words - `_lora_phrase` of the `lora_dict` this
    # configuration was named from, so every sentence about the engine names the
    # LoRA *and* the scale that keyed it rather than re-deriving them (issue #44).
    loras: str = ""


def engine_configuration(model_path: str, acceleration: str, use_lcm_lora: bool,
                         steps: int, frame_buffer_size: int,
                         lora_dict: Optional[Dict[str, float]] = None,
                         engines_root: _PathArg = None,
                         use_tiny_vae: bool = True) -> EngineConfiguration:
    """What `wrapper.py` would look for, and what it would do if it were not there.

    Answered through `engine_cache`, which is a mirror of `create_prefix` held to
    it by a test - so a "no engine yet" here is the same directory the worker will
    miss a moment later, and not an approximation of it.
    """
    root = resolve_engines_dir() if engines_root is None else Path(engines_root)
    directory = engine_cache.engine_dir_name(
        model_path, use_lcm_lora=use_lcm_lora, use_tiny_vae=use_tiny_vae,
        unet_batch=engine_cache.unet_batch_size(frame_buffer_size, steps),
        width=DIFFUSION_CANVAS, height=DIFFUSION_CANVAS, lora_dict=lora_dict)
    cached = engine_cache.engine_is_cached(root, directory)
    free = engine_cache.free_bytes(root)
    return EngineConfiguration(
        model_path=str(model_path), engine_dir=directory, engines_root=str(root),
        cached=cached, builds=engine_rebuild_needed(acceleration) and not cached,
        steps=int(steps), free_bytes=free,
        enough_disk=free >= engine_cache.MIN_FREE_BYTES_FOR_ENGINE_BUILD,
        loras=_lora_phrase(lora_dict))


def _steps_phrase(steps: int, noun: str = "step") -> str:
    """`1 step` / `4 steps`, so three messages cannot pluralise it three ways."""
    return f"{steps} {noun}{'' if steps == 1 else 's'}"


def _engine_state_line(configuration: EngineConfiguration) -> str:
    """The standing sentence under the model: which engine this needs, and if it
    exists. One builder, so the LoRA set cannot be named in one of the two."""
    model = model_label(configuration.model_path)
    steps = _steps_phrase(configuration.steps)
    if configuration.cached:
        return f"Engine: cached for {model} at {steps}{configuration.loras}."
    return (f"⚠  No engine yet for {model} at {steps}{configuration.loras} - "
            f"Start will build one ({ENGINE_BUILD_TIME}, {ENGINE_BUILD_SIZE}).")


def _engine_missing_warning(configuration: EngineConfiguration) -> str:
    """What Start says before it spends the minutes, naming what made it different."""
    return (
        f"There is no compiled TensorRT engine for this configuration yet:\n\n"
        f"    {model_label(configuration.model_path) or configuration.model_path}, "
        f"{_steps_phrase(configuration.steps, 'denoising step')}, "
        f"{DIFFUSION_CANVAS}x{DIFFUSION_CANVAS}"
        f"{configuration.loras}\n"
        f"    {configuration.engine_dir}\n\n"
        f"Starting will build one first: about {ENGINE_BUILD_TIME}, and "
        f"{ENGINE_BUILD_SIZE} under {configuration.engines_root}. The window will "
        f"look frozen while it does.\n\n"
        "Continue?")


def _not_enough_disk_message(engines_root: _PathArg, free_bytes: int) -> str:
    """The refusal. Loud, before the build, rather than a half-written engine."""
    gib = engine_cache.BYTES_PER_GIB
    return (
        f"{free_bytes / gib:.1f} GB free on the volume holding {engines_root}, and "
        f"building a TensorRT engine needs "
        f"{engine_cache.MIN_FREE_BYTES_FOR_ENGINE_BUILD / gib:.1f} GB.\n\n"
        f"Free some space, or point SD_ENGINES_DIR at a volume that has it. "
        f"Stopping before the build rather than partway through it.")


def _engine_rebuild_warning(setting: str) -> str:
    """What the user is asked before a change that keys a different engine."""
    return (f"Changing the {setting} keys a different TensorRT engine.\n\n"
            f"{ENGINE_KEYED_SETTINGS.get(setting, '')}\n\n"
            f"There is no cached engine for the new {setting}, so one has to be "
            f"built: about {ENGINE_BUILD_TIME}, and {ENGINE_BUILD_SIZE} on disk.\n\n"
            "Continue?")


# Offline mode blocks diffusers' repo-id lookup, so prefer a local copy of the
# LCM-LoRA when one has been staged in the models root.
LOCAL_LCM_LORA = str(resolve_models_dir() / LORAS_SUBDIR / LCM_LORA_FILENAME)
PREVIEW_GAIN = 1.15

def enforce_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def verify_local_model_path_dir(path: str):
    if not path:
        # Blank used to fail naming nothing, which is exactly the case where the
        # user has no idea what to type. Say where a model would be.
        raise FileNotFoundError(
            "No model folder is set. Point it at a diffusers folder - "
            f"'⬇ Download SD-Turbo' puts one under {resolve_models_dir()}.")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Local model directory not found: {path}")

def _frame_to_rgb(frame: np.ndarray, force_swap_rb: Optional[bool] = False) -> np.ndarray:
    if frame is None or frame.size == 0: return frame
    if frame.shape[2] == 4: frame = frame[:, :, :3]
    if force_swap_rb: frame = frame[:, :, ::-1]
    return frame

# The scheduler's usable t_index range. Anything outside it either does nothing
# or corrupts the step schedule, so every control message is clamped into it.
T_INDEX_MIN = 2
T_INDEX_MAX = 49

def _clamp_t_index(value: Any) -> int:
    return max(T_INDEX_MIN, min(T_INDEX_MAX, int(value)))

# The hardcoded priority-case plan (issue #8, step 4) is behind an environment
# variable, not a widget: the selective path has to be drivable end to end before
# anything wires the GUI up, and the app as shipped still starts `global`. The
# two text fields that will replace this are a later issue.
DEMO_PLAN_ENV = "SD_DEMO_PLAN"
DEMO_PLAN_VALUES = ("1", "true", "yes", "on")

def demo_plan_requested(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Did someone ask the worker to start on the hardcoded priority-case plan?"""
    environ = os.environ if environ is None else environ
    return str(environ.get(DEMO_PLAN_ENV, "")).strip().lower() in DEMO_PLAN_VALUES

def _control_transition(msg: Any, t_index_list: List[int],
                        plan: Optional[RenderPlan] = None) -> Dict[str, Any]:
    """Pure half of the worker's control_queue drain: a message in, a state delta out.

    Returns {} for anything the worker should ignore, otherwise exactly one of
    "region", "t_index_list" (always paired with "engine_swap"), "prompt",
    "negative_prompt", "plan" (paired with "plan_notes") or "plan_error". Every side
    effect stays with the caller - notably the engine swap that "engine_swap" asks
    for, which costs minutes, and the plan swap, which happens at a frame boundary.

    `plan` is the newest plan the worker holds - the one this message replaces. The
    validator counts the new version up from it, which is what makes plan versions
    monotonic in the worker rather than in whichever producer happened to send one.
    """
    if not isinstance(msg, dict):
        return {}
    mtype = msg.get("type")

    if mtype == "set_region":
        r = msg.get("region")
        keys = ("left", "top", "width", "height")
        if isinstance(r, dict) and all(k in r for k in keys):
            return {"region": {k: int(r[k]) for k in keys}}
        return {}

    if mtype == "set_t_at":
        i = int(msg.get("index", -1))
        if not 0 <= i < len(t_index_list):
            return {}
        new_t = list(t_index_list)
        new_t[i] = _clamp_t_index(msg.get("value", 30))
        return {"t_index_list": new_t, "engine_swap": False}

    if mtype == "set_t_index_list":
        new_t = msg.get("t_index_list")
        if not isinstance(new_t, (list, tuple)) or len(new_t) == 0:
            return {}
        clean = [_clamp_t_index(x) for x in new_t]
        # The step *count* keys a distinct TensorRT engine; the values do not.
        return {"t_index_list": clean, "engine_swap": len(clean) != len(t_index_list)}

    if mtype == "set_prompt" and "prompt" in msg:
        return {"prompt": str(msg["prompt"])}

    if mtype == "set_negative_prompt" and "negative_prompt" in msg:
        return {"negative_prompt": str(msg["negative_prompt"])}

    if mtype == "set_plan" and "plan" in msg:
        previous = INITIAL_PLAN_VERSION if plan is None else plan.plan_version
        result = validate_plan(msg["plan"], previous_version=previous)
        if result.plan is None:
            # Spec 8.7: never a black screen. The plan in force keeps rendering and
            # the user is told why the new one did not take.
            return {"plan_error": result.reason}
        return {"plan": result.plan, "plan_notes": result.notes}

    return {}

def _format_fps(payload: Any) -> str:
    """The status line for whatever the worker put on the fps queue.

    Two shapes, and both have to read: the channel carried a bare number for as
    long as this app has existed, and a run whose plan names no target still sends
    one. A mapping may also carry what detection cost (issue #7, step 5) - the
    count, the detect, and what that detect amortises to at the plan's cadence,
    which is the figure spec 7.1 budgets and the only one worth watching.
    """
    fields = payload if isinstance(payload, dict) else {"fps": payload}
    fps = fields.get("fps")
    if not isinstance(fps, (int, float)):
        return "FPS: --"
    line = f"FPS: {int(round(float(fps)))}"
    if "detections" in fields:
        line += (f"  |  {fields['detections']} obj  "
                 f"{float(fields.get('detector_ms', 0.0)):.1f} ms/detect "
                 f"({float(fields.get('amortised_ms', 0.0)):.1f} ms/frame "
                 f"every {fields.get('detect_every_n')})")
    if "regions" in fields:
        # Issue #8: how many of K slots this frame used, how many objects are
        # waiting their turn, and how many were too small to render at all.
        line += (f"  |  {fields['regions']}/{fields.get('slots')} regions  "
                 f"{fields.get('deferred', 0)} waiting  "
                 f"{fields.get('skipped_small', 0)} too small")
    return line

# --- the GUI's end of the Render Plan (issue #22) ----------------------------
#
# The two fields the user types into *are* the control plane in v1: target text goes
# to the open-vocabulary detector, style text goes to StreamDiffusion. The mapping
# from what was typed to the message that crosses the queue is kept pure and at
# module scope, so it is testable without a Tk root - which is what the issue's Gate
# asks for.

# What a prompt or negative-prompt edit waits before it re-encodes on the live
# engine. Cheap, so it fires nearly as fast as the user types.
PROMPT_DEBOUNCE_MS = 150

# A target edit changes the detector's vocabulary, and `YOLOWorld.set_classes` drops
# the predictor: the next detect then costs ~108 ms more than a steady one (spec
# 8.1). Per keystroke that is three frames' budget per character, so the plan waits
# a good deal longer than a prompt edit does before it fires.
PLAN_DEBOUNCE_MS = 400

class PlanUpdate(NamedTuple):
    """What the two fields came to: a line to read, and a plan to send or not.

    Mirrors `render_plan.PlanValidation` one step further out - `message` is the
    `set_plan` to put on `control_queue`, and it is None exactly when `reason`
    holds the validator's stated refusal. `status` is filled either way, because
    the user gets told what happened either way.

    `notes` is what the validator changed on the way through, kept apart from the
    rendered `status` line so the plan area can draw it on its own (issue #40).
    """

    status: str
    message: Optional[Dict[str, Any]] = None
    reason: str = ""
    notes: Sequence[str] = ()

# The Detail box beside the two plan fields (issue #39, spec 8.2). One choice
# rather than two, because `crop` only makes sense at K=1 - above one slot the
# compositor falls back to `masked` on every frame - and a user should not be able
# to set the two inconsistently. The default is what the app has always done and
# what every committed measurement in this repo was taken under.
DETAIL_ALL_OBJECTS = "All objects, one frame (masked)"
DETAIL_ONE_OBJECT = "One object, full canvas (crop)"
DETAIL_PRESETS: Dict[str, Tuple[str, int]] = {
    DETAIL_ALL_OBJECTS: (PLAN_MASKED, DEFAULT_MAX_INSTANCES),
    DETAIL_ONE_OBJECT: (PLAN_CROP, 1),
}


def detail_plan(label: str) -> Tuple[str, int]:
    """The primitive and the slot count a Detail label asks for.

    Unknown falls back to today's behaviour rather than raising: the label reaches
    here from a widget whose value a stale preference could have set, and the plan
    this app has always rendered is a better answer than no plan at all.
    """
    return DETAIL_PRESETS.get(label, DETAIL_PRESETS[DETAIL_ALL_OBJECTS])


def _plan_status_line(plan: RenderPlan, notes: Sequence[str] = ()) -> str:
    """The line the status bar carries for a plan the GUI just built.

    Names one concept and one region because that is all a plan built from two
    fields can hold. It says "every" or "one at a time" from the plan's own slot
    count: under `crop` the frame renders a single object and the scheduler's
    round-robin moves on to the next one, which is a different promise from
    restyling all of them and should read as one.
    """
    target = plan.honoured_target
    if target is None:
        line = "Plan: whole frame, no target"
    elif target.max_instances == 1:
        line = (f"Plan: restyle one {target.concept} ({target.region}) at a time, "
                f"at the full canvas")
    else:
        line = f"Plan: restyle every {target.concept} ({target.region})"
    if notes:
        line += "  |  " + "; ".join(notes)
    return line

def _plan_update_from_fields(target: str, style: str, prompt: str,
                             negative_prompt: str,
                             detail: str = DETAIL_ALL_OBJECTS) -> PlanUpdate:
    """The GUI's fields as a control message, or the reason there is not one.

    The producer validates so a refusal is visible where it was typed instead of
    silent in the worker; the worker validates the same plan again, because it owns
    the version and because a plan can arrive from any hand.

    An empty target is not "restyle nothing": it is `mode: "global"`, the whole
    frame under one prompt, which is what this app has always done. An empty style
    falls back to the prompt box for the same reason - naming a target must not
    hand the engine an empty embedding and call it a style. The boxes arrive with
    the newline Tk's `get` appends, so everything is stripped on the way in.
    """
    primitive, max_instances = detail_plan(detail)
    result = plan_from_fields(target.strip(), style.strip() or prompt.strip(),
                              negative_prompt=negative_prompt.strip(),
                              primitive=primitive, max_instances=max_instances)
    if result.plan is None:
        return PlanUpdate(status=f"Plan rejected: {result.reason}", reason=result.reason)
    return PlanUpdate(status=_plan_status_line(result.plan, result.notes),
                      message={"type": "set_plan", "plan": result.plan.to_dict()},
                      notes=tuple(result.notes))


# --- what the window says about the plan (issue #40, steps 3 and 5) ----------

# Blank target is `mode: "global"` - the whole frame under one prompt, which is
# what this app has always done. Saying only "detection off" would read as "nothing
# is happening", so the line says what *is* happening first.
GLOBAL_STATE = "Plan: global — the whole frame.  Detection: off (no target)."

# A note is not an error. A refusal is drawn in `CUSTOM_COLORS["error"]`.
PLAN_NOTE_COLOR = "gray70"


def _plan_state_line(target: str, running: bool, payload: Any = None) -> str:
    """Whether detection is running, on what concept, and how much it is holding.

    Step 3 of issue #40. Until now the only sign the object-aware path was alive at
    all was the dense tail of the FPS line; this says it in words, beside the field
    that turns it on. `payload` is whatever the worker last put on the fps queue -
    the same mapping `_format_fps` renders - so a bare number, or nothing yet,
    reads as "there is nothing to report", not as an error.

    The concept named is the detector's own, not the field's: a target edit is
    debounced and then costs a vocabulary re-encode, so for a moment the two
    disagree and the one worth showing is the one actually being detected.
    """
    fields = payload if isinstance(payload, dict) else {}
    concept = target.strip()
    if "detections" in fields:
        detected = ", ".join(fields.get("concepts") or ()) or concept
        held = fields["detections"]
        plural = "" if held == 1 else "s"
        return (f"Plan: selective — every {detected}."
                f"  Detection: on, {held} object{plural} held, "
                f"every {fields.get('detect_every_n')} frames.")
    if not concept:
        return GLOBAL_STATE
    if not running:
        return f"Plan: selective — every {concept}.  Detection: starts with generation."
    return f"Plan: selective — every {concept}.  Detection: starting..."


def _plan_note(update: PlanUpdate) -> Tuple[str, str]:
    """The line under the two fields, and the colour to draw it in.

    Step 5 of issue #40: the validator's refusal and its notes belong where the
    user is looking. The status bar still gets them too, but it is shared with the
    worker's own messages and the next one replaces whatever was there.
    """
    if update.message is None:
        return f"Rejected: {update.reason}", CUSTOM_COLORS["error"]
    if update.notes:
        return "Adjusted: " + "; ".join(update.notes), PLAN_NOTE_COLOR
    return "", PLAN_NOTE_COLOR

SHOW = {
    "model_path": True, "prompt": True, "negative_prompt": True, "seed": True,
    "frame_buffer_size": True, "acceleration": True, "use_denoising_batch": False,
    "cfg_type": False, "guidance_scale": False, "delta": False, "similar_image_filter": False,
    "offline": False, "lora": True, "use_lcm_lora": True, "step_count": False,
}

# The engine knobs (issue #40, step 2). Each is a property of how the app runs
# rather than of what it makes, and three of them key a distinct TensorRT engine.
# They are built inside the collapsed "Advanced" section instead of beside the two
# fields that *are* the product's interface. `SHOW` still decides whether one
# exists at all, which is why `step_count` is False: changing the number of
# denoising steps rebuilds the engine, while the sliders that set their values -
# the live strength control - stay in the primary panel.
ADVANCED = ("seed", "frame_buffer_size", "acceleration", "use_lcm_lora",
            "use_denoising_batch", "step_count")

ADVANCED_CLOSED = "▸  Advanced  —  engine settings"
ADVANCED_OPEN = "▾  Advanced  —  engine settings"

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

MONITOR_DEFAULTTONEAREST = 2

def _monitor_rc_from_point(x: int, y: int):
    hmon = _user32.MonitorFromPoint(POINT(x, y), MONITOR_DEFAULTTONEAREST)
    mi = MONITORINFO(); mi.cbSize = ctypes.sizeof(MONITORINFO)
    _user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    r = mi.rcMonitor
    return r.left, r.top, r.right, r.bottom

class FloatingCaptureWindow:
    """The on-screen rectangle the worker captures.

    Two sides rather than one since issue #39: the capture geometry is no longer
    the engine's square canvas, and a 512x512 window is exactly the problem that
    issue was opened about - too small to get a whole object into frame.
    """

    def __init__(self, master: tk.Tk, inner_w=512, inner_h=512, border_px=8, handle_h=28):
        self.master = master
        self.inner_w = int(inner_w)
        self.inner_h = int(inner_h)
        self.border_px = int(border_px)
        self.handle_h = int(handle_h)
        total_w = self.inner_w + self.border_px * 2
        total_h = self.handle_h + self.inner_h + self.border_px
        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            x0 = int(master.winfo_rootx()) + 50
            y0 = int(master.winfo_rooty()) + 50
        except Exception:
            x0, y0 = 200, 200
        self.win.geometry(f"{total_w}x{total_h}+{x0}+{y0}")
        self.canvas = tk.Canvas(self.win, width=total_w, height=total_h, highlightthickness=0, bd=0, relief="flat")
        self.canvas.pack(fill="both", expand=True)
        self.border_color = "#ef4444"
        self.handle_color = "#7f1d1d"
        self.canvas.create_rectangle(0, 0, total_w, self.handle_h, fill=self.handle_color, outline=self.handle_color)
        self.canvas.create_text(10, self.handle_h // 2, anchor="w", text=f"Capture {self.inner_w}×{self.inner_h}", fill="#ffffff", font=("Segoe UI", 9, "bold"))
        self.canvas.create_rectangle(0, self.handle_h, self.border_px, self.handle_h + self.inner_h + self.border_px, fill=self.border_color, outline=self.border_color)
        self.canvas.create_rectangle(self.border_px + self.inner_w, self.handle_h, total_w, self.handle_h + self.inner_h + self.border_px, fill=self.border_color, outline=self.border_color)
        self.canvas.create_rectangle(0, self.handle_h + self.inner_h, total_w, self.handle_h + self.inner_h + self.border_px, fill=self.border_color, outline=self.border_color)
        self._apply_window_region(total_w, total_h)
        self._drag_start = None
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        try:
            hwnd = int(self.win.winfo_id())
            ex = GetWindowLongW(hwnd, GWL_EXSTYLE)
            SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TOOLWINDOW)
            SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
        except Exception:
            pass

    def _apply_window_region(self, total_w: int, total_h: int):
        try:
            region = CreateRectRgn(0, 0, total_w, self.handle_h)
            left = CreateRectRgn(0, self.handle_h, self.border_px, self.handle_h + self.inner_h + self.border_px)
            CombineRgn(region, region, left, RGN_OR); DeleteObject(left)
            right = CreateRectRgn(self.border_px + self.inner_w, self.handle_h, total_w, self.handle_h + self.inner_h + self.border_px)
            CombineRgn(region, region, right, RGN_OR); DeleteObject(right)
            bottom = CreateRectRgn(0, self.handle_h + self.inner_h, total_w, self.handle_h + self.inner_h + self.border_px)
            CombineRgn(region, region, bottom, RGN_OR); DeleteObject(bottom)
            SetWindowRgn(self.win.winfo_id(), region, True)
        except Exception:
            pass

    def _on_mouse_down(self, evt):
        if 0 <= evt.y <= self.handle_h:
            self._drag_start = (evt.x_root, evt.y_root)
            self._start_geom = (int(self.win.winfo_x()), int(self.win.winfo_y()))

    def _on_mouse_drag(self, evt):
        if not self._drag_start: return
        dx = evt.x_root - self._drag_start[0]
        dy = evt.y_root - self._drag_start[1]
        x, y = self._start_geom
        self.win.geometry(f"+{x+dx}+{y+dy}")
        try:
            if hasattr(self.master, "_on_capture_window_moved"):
                self.master._on_capture_window_moved()
        except Exception:
            pass

    def destroy(self):
        try: self.win.destroy()
        except Exception: pass

    def inner_rect_screen(self):
        self.win.update_idletasks()
        x = int(self.win.winfo_rootx()) + self.border_px
        y = int(self.win.winfo_rooty()) + self.handle_h
        return {"left": x, "top": y, "width": self.inner_w, "height": self.inner_h}

def _screen_capture_loop_dx(stop_evt: threading.Event, height: int, width: int, region_ref: Dict[str, Dict[str, int]], max_buffer: int, inputs_list: List[Any]):
    torch = PreloadedDependencies.get_torch()
    if torch is None: return

    def _append_shed(q, item, maxlen):
        try:
            if hasattr(q, "maxlen") and q.maxlen is not None:
                if len(q) >= q.maxlen: q.popleft()
                q.append(item)
            else:
                while len(q) >= maxlen: q.pop(0)
                q.append(item)
        except Exception: pass

    try:
        import dxcam
        camera = dxcam.create(output_idx=0, output_color="RGB")
    except Exception:
        _screen_capture_loop_mss(stop_evt, height, width, region_ref, max_buffer, inputs_list)
        return

    device = torch.device("cuda")
    H_t, W_t = int(height), int(width)
    C = 3
    dtype_cpu = torch.uint8
    staging_cpu = None
    need_resize = None
    frame_count = 0
    last_log_time = time.time()

    try:
        while not stop_evt.is_set():
            rect = region_ref["rect"]
            L, T = int(rect["left"]), int(rect["top"])
            W, H = int(rect["width"]), int(rect["height"])
            R, B = L + W, T + H
            try:
                frame = camera.grab(region=(L, T, R, B), new_frame_only=False)
            except TypeError:
                frame = camera.grab(region=(L, T, R, B))
            if frame is None:
                time.sleep(0.001)
                continue
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if staging_cpu is None:
                H_c, W_c = int(frame.shape[0]), int(frame.shape[1])
                staging_cpu = torch.empty((H_c, W_c, C), dtype=dtype_cpu, pin_memory=True)
                need_resize = (H_c != H_t) or (W_c != W_t)
            staging_cpu.numpy()[...] = frame
            img = staging_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            img = img.permute(2, 0, 1).unsqueeze(0)
            img.mul_(1.0 / 255.0)
            if need_resize:
                img = torch.nn.functional.interpolate(img, size=(H_t, W_t), mode="bilinear", align_corners=False)
            _append_shed(inputs_list, img, max_buffer)
            frame_count += 1
            now = time.time()
            if now - last_log_time > 5.0:
                fps = frame_count / (now - last_log_time)
                try: buf_len = len(inputs_list)
                except Exception: buf_len = -1
                frame_count = 0
                last_log_time = now
    except Exception:
        try: camera.stop()
        except Exception: pass
        _screen_capture_loop_mss(stop_evt, height, width, region_ref, max_buffer, inputs_list)
    finally:
        try: camera.stop()
        except Exception: pass

def _screen_capture_loop_mss(stop_evt: threading.Event, height: int, width: int, region_ref: Dict[str, Dict[str, int]], max_buffer: int, inputs_list: List[Any]):
    torch = PreloadedDependencies.get_torch()
    if torch is None: return
    try: import mss
    except Exception: return
    with mss.mss() as sct:
        while not stop_evt.is_set():
            rect = region_ref["rect"]
            monitor = {k: int(rect[k]) for k in ("left","top","width","height")}
            raw = sct.grab(monitor)
            bgra  = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
            frame = bgra[:, :, :3][:, :, ::-1]
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = np.array(PIL.Image.fromarray(frame, "RGB").resize((width, height), resample=PIL.Image.BICUBIC))
            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
            inputs_list.append(tensor)
            if isinstance(inputs_list, list) and len(inputs_list) > max_buffer:
                del inputs_list[:-max_buffer]

def _load_stream_wrapper():
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    wrapper_path = os.path.join(base_dir, "wrapper.py")
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wrapper"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "StreamDiffusionWrapper")

def image_generation_process(out_queue: Queue, fps_queue: Queue, close_queue: Queue, status_queue: Queue, control_queue: Queue, debug_queue: Queue, t_index_list: List[int], model_path_dir: str, controlnet_paths: List[str], controlnet_scales: List[float], lora_dict: Optional[Dict[str, float]], use_lcm_lora: bool, lcm_lora_id: Optional[str], prompt: str, negative_prompt: str, frame_buffer_size: int, width: int, height: int, acceleration: Literal["none", "xformers", "tensorrt"], use_denoising_batch: bool, seed: int, cfg_type: Literal["none", "full", "self", "initialize"], guidance_scale: float, delta: float, do_add_noise: bool, enable_similar_image_filter: bool, similar_image_filter_threshold: float, similar_image_filter_max_skip_frame: float, monitor_receiver: Connection, offline: bool = True, engine_dir: Optional[str] = None, canvas_width: int = DIFFUSION_CANVAS, canvas_height: int = DIFFUSION_CANVAS) -> None:
    import sys, os
    from pathlib import Path
    
    if getattr(sys, "frozen", False):
        APP_ROOT_WORKER = Path(sys.executable).parent
        SITE_PACKAGES_WORKER = APP_ROOT_WORKER / "venv" / "Lib" / "site-packages"
    else:
        APP_ROOT_WORKER = Path(__file__).resolve().parent
        SITE_PACKAGES_WORKER = APP_ROOT_WORKER / "venv" / "Lib" / "site-packages"
        
    internal_str = str(SITE_PACKAGES_WORKER)
    if internal_str not in sys.path:
        sys.path.insert(0, internal_str)
        
    _prime_dll_search(SITE_PACKAGES_WORKER, INTERNAL_DIR)
    import_torch = PreloadedDependencies.get_torch()
    
    def _status(msg: str):
        try: status_queue.put_nowait(msg)
        except Exception: pass

    # The GUI passes an already-absolute root; going through the same rule anyway
    # is what stops a relative one from re-anchoring to this process's cwd.
    models_root = resolve_models_dir()
    engines_root = resolve_engines_dir(engine_dir)
    banner = _cache_paths_banner(models_root, engines_root)
    print(banner, flush=True)
    _status(banner)

    try:
        if import_torch is None:
            _prime_dll_search(SITE_PACKAGES_WORKER, INTERNAL_DIR)
            import_torch = PreloadedDependencies.get_torch()
            if import_torch is None: return

        StreamDiffusionWrapper = _load_stream_wrapper()
        if offline: enforce_offline_mode()
        verify_local_model_path_dir(model_path_dir)

        def _build_stream(steps: List[int]):
            """Build and prepare a wrapper for `steps`, on the current prompts.

            TensorRT keys an engine on the step *count*, so calling this for a count
            with no cached engine compiles one - minutes, not milliseconds.
            """
            wrapper = StreamDiffusionWrapper(
                model_id_or_path=model_path_dir, t_index_list=list(steps),
                frame_buffer_size=frame_buffer_size,
                width=canvas_width, height=canvas_height,
                warmup=2, acceleration=acceleration, do_add_noise=do_add_noise,
                enable_similar_image_filter=enable_similar_image_filter,
                similar_image_filter_threshold=similar_image_filter_threshold,
                similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
                mode="img2img", use_denoising_batch=use_denoising_batch,
                cfg_type=cfg_type, seed=seed, lora_dict=lora_dict,
                use_lcm_lora=use_lcm_lora, lcm_lora_id=lcm_lora_id,
                engine_dir=str(engines_root)
            )
            wrapper.prepare(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=guidance_scale, delta=delta)
            return wrapper

        stream = _build_stream(t_index_list)

        def _apply_prompt(text: str) -> None:
            """Re-encode `text` on the live engine, or carry on without it.

            Every prompt change goes through here - the prompt box, the negative
            box, and a plan swap. A prompt the encoder chokes on must not take the
            frame loop down with it, so the failure is swallowed and the engine
            keeps rendering the embedding it already has.
            """
            try:
                stream.stream.update_prompt(text)
            except Exception:
                pass

        first_rect = monitor_receiver.recv()
        region_ref = {"rect": dict(first_rect)}
        inputs = deque(maxlen=frame_buffer_size * 2)
        cap_stop = threading.Event()
        cap_thr = threading.Thread(target=_screen_capture_loop_dx, args=(cap_stop, height, width, region_ref, frame_buffer_size, inputs), daemon=True)
        cap_thr.start()
        current_t_index_list = list(t_index_list)
        frame_count = 0
        frame_index = 0
        # The plan in force. It starts as today's behaviour expressed as a plan -
        # one prompt over the whole frame - so "no plan yet" is never a state the
        # frame loop has to handle. Only `set_plan` moves it, and only at a frame
        # boundary (see `begin_frame` below).
        active_plan = ActivePlan(global_plan(prompt, negative_prompt,
                                             previous_version=INITIAL_PLAN_VERSION))
        # Detection (issue #7). Built after the engine, because the engine is the
        # tenant that must get its VRAM first, and loaded later still: the weights
        # are only read once a plan names a concept, so a `global` plan pays
        # nothing. Every call the frame loop makes on it returns immediately - the
        # detect itself happens on this thread, never on the frame path.
        detection = BackgroundDetector(UltralyticsDetector(models_root), log=_status)
        detection.start()
        tracks = detection.tracks
        # The selective render path (issue #8). The scheduler holds the round-robin
        # cursor across frames and the compositor holds the last alpha map - and the
        # copy of it it uploaded (issue #31) - so a frame between two detector ticks
        # builds neither and copies nothing.
        scheduler = RegionScheduler()
        compositor = DeviceCompositor()
        # The plan's `seed_policy`, applied to the engine's noise field before the
        # call that reads it (issue #32). It holds one noise realisation per live
        # track, so the field an object renders under follows the object rather
        # than the screen; under `fixed` - the default, and what this app has
        # always done - it writes nothing at all.
        noise = NoiseField(base_seed=seed)
        if demo_plan_requested():
            # Step 4: the priority case, hardcoded, so the whole path can be driven
            # before a widget exists to drive it. Submitted rather than installed,
            # so it lands at a frame boundary like any other plan.
            demo = priority_case_plan(previous_version=active_plan.latest.plan_version)
            active_plan.submit(demo)
            _status(f"Demo plan ({DEMO_PLAN_ENV}): restyle the "
                    f"{demo.targets[0].region} of every {demo.targets[0].concept}")

        while close_queue.empty():
            try:
                while True:
                    update = _control_transition(control_queue.get_nowait(),
                                                 current_t_index_list, active_plan.latest)
                    if "region" in update:
                        region_ref["rect"] = update["region"]
                    elif "t_index_list" in update:
                        current_t_index_list = update["t_index_list"]
                        if update["engine_swap"]:
                            _status(f"Swapping engine for {len(current_t_index_list)} steps...")

                            # Flush the current engine from VRAM before the next one loads
                            del stream
                            import gc
                            gc.collect()
                            import_torch.cuda.empty_cache()

                            stream = _build_stream(current_t_index_list)
                            # A new engine is a new noise field and a render that
                            # has nothing to do with the last one's.
                            noise.reset()
                            compositor.reset_ema()
                            _status("Engine swap complete!")
                        else:
                            # Same step count (slider was dragged). Update values instantly!
                            stream.set_t_index_list(current_t_index_list)
                    elif "prompt" in update:
                        prompt = update["prompt"]
                        _apply_prompt(prompt)
                    elif "negative_prompt" in update:
                        negative_prompt = update["negative_prompt"]
                        # Inherited: this refreshes the stream with the *positive*
                        # prompt. The negative one only lands on the next prepare().
                        _apply_prompt(prompt)
                    elif "plan" in update:
                        active_plan.submit(update["plan"])
                        for note in update["plan_notes"]:
                            _status(f"Plan note: {note}")
                    elif "plan_error" in update:
                        _status(f"Plan rejected: {update['plan_error']}")
            except Exception:
                pass

            try:
                # One read of the active plan per frame. Everything downstream uses
                # `frame_plan`, so a plan submitted mid-frame lands on the next
                # frame and never on half of this one.
                frame_plan = active_plan.begin_frame()
                if frame_plan.changed:
                    prompt = frame_plan.plan.effective_prompt
                    negative_prompt = frame_plan.plan.effective_negative_prompt
                    _apply_prompt(prompt)
                    # Cold path: a changed vocabulary is re-encoded and re-warmed
                    # on the detector's thread, so the frame after a prompt edit
                    # does not pay the ~108 ms `set_classes` costs (spec 8.1).
                    detection.follow(frame_plan.plan)
                    # The two temporal-stability levers (issue #32, spec 8.5).
                    # Both are runtime writes - the noise is a plain tensor
                    # `add_noise` reads in Python and the EMA is the
                    # compositor's - so neither can cost a TensorRT rebuild.
                    noise.follow(frame_plan.plan)
                    compositor.set_output_ema(frame_plan.plan.settings.output_ema)
                    # The plan's denoise, as a value on the live schedule. Only the
                    # values move, never the step *count*, so this is a runtime
                    # update and not an engine rebuild. A plan with no target
                    # carries the schema's default rather than a strength anyone
                    # typed, so it leaves the t_index slider where the user put it.
                    honoured = frame_plan.plan.honoured_target
                    if honoured is not None and len(current_t_index_list) == 1:
                        wanted = _clamp_t_index(
                            t_index_for_denoise(frame_plan.plan.effective_denoise))
                        if wanted != current_t_index_list[0]:
                            current_t_index_list = [wanted]
                            try: stream.set_t_index_list(current_t_index_list)
                            except Exception: pass
                detect_every_n = frame_plan.plan.settings.detect_every_n

                t0 = time.time()
                if frame_buffer_size == 1:
                    batch = inputs[-1]
                else:
                    sampled = []
                    for i in range(frame_buffer_size):
                        idx = max(0, len(inputs) - frame_buffer_size + i)
                        sampled.append(inputs[idx])
                    batch = import_torch.cat(sampled)

                if isinstance(inputs, list) and len(inputs) > frame_buffer_size * 2:
                    del inputs[:-frame_buffer_size]

                if isinstance(batch, import_torch.Tensor):
                    batch = batch.to(device=stream.device, dtype=stream.dtype)

                # Every `detect_every_n`th frame the newest capture is handed to
                # the detector, which may or may not get to it - it drops what it
                # cannot keep up with, exactly as the capture deque does. The
                # detect then runs alongside this frame's diffusion rather than
                # in front of it, and `tracks` is one reference read.
                if is_detect_frame(frame_index, detect_every_n):
                    detection.offer(batch, frame_index)
                tracks = detection.tracks
                # Frames the pipeline took, not frames the GUI accepted:
                # `frame_count` stalls whenever the preview queue is full, and a
                # cadence counted on a stalled number is every frame or no frame.
                frame_index += 1

                # C5 and C7 (issue #8). One selection per frame, off the plan this
                # frame bound, and the compositor turns it into what the frame is:
                # the capture as it stands, a full-frame render, or a full-frame
                # render composited through a feathered mask.
                selection = scheduler.select(tracks, frame_plan.plan, width, height)
                render = compositor.frame(selection, width, height,
                                          frame_plan.plan.settings.primitive)

                images = []
                if render.diffuses:
                    if render.action in (MASKED, CROP):
                        # Onto the capture the engine was given, never onto the
                        # previous output: outside the regions the frame has to be
                        # the captured pixels, byte for byte.
                        #
                        # And on the device (issue #31). `output_type="pt"` leaves
                        # the render where it was made, the blend runs beside it,
                        # and the frame makes one host copy - of uint8, after the
                        # mask, instead of float, before it.
                        #
                        # What the engine is handed is the *canvas* (issue #39):
                        # the whole capture squeezed onto it under `masked`, and
                        # one region blown up to fill it under `crop` - one
                        # diffusion call either way, spent in two different
                        # places. The geometry is what maps the selection's
                        # captured-pixel boxes onto that canvas for the noise.
                        geometry = CanvasGeometry(width, height, canvas_width,
                                                  canvas_height, render.crop)
                        frame_canvas = (
                            to_canvas(batch, canvas_width, canvas_height)
                            if render.crop is None else
                            crop_to_canvas(batch, render.crop,
                                           canvas_width, canvas_height))
                        noise.apply(stream, selection, geometry)
                        rendered = stream.img2img(frame_canvas, output_type="pt")
                        images = [Image.fromarray(frame) for frame
                                  in compositor.blend_device(batch, rendered,
                                                             render.alpha,
                                                             render.crop)]
                    else:
                        res = stream.img2img(
                            to_canvas(batch, canvas_width, canvas_height))
                        if isinstance(res, Image.Image): images = [res]
                        elif isinstance(res, list): images = res
                else:
                    # A selective plan that found nothing to restyle. There is no
                    # pixel anyone asked to change, so the frame costs no diffusion
                    # call - and the loop still produces a frame.
                    #
                    # It also ends the output EMA's history: the next render is on
                    # the other side of a gap, and averaging across one blends a
                    # frame with one from before the object left the screen.
                    compositor.reset_ema()
                    images = [Image.fromarray(frame_to_array(batch))]

                for im in images:
                    try: 
                        out_queue.put(im, block=False)
                        frame_count += 1
                    except queue.Full: pass

                elapsed = time.time() - t0
                fps = (1.0/elapsed) if elapsed > 0 else 0.0
                while not fps_queue.empty():
                    try: fps_queue.get_nowait()
                    except Exception: break
                payload = fps_payload(int(round(fps)), tracks, detect_every_n)
                payload.update(scheduler.status(selection))
                fps_queue.put(payload)
            except Exception:
                time.sleep(0.01)

        detection.stop()
        cap_stop.set()
        cap_thr.join(timeout=2.0)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass

def _python_cmd_for_pip():
    import sys, shutil
    if getattr(sys, "frozen", False):
        v = f"-{sys.version_info.major}.{sys.version_info.minor}"
        if shutil.which("py"): return ["py", v]
        p = shutil.which("python")
        if p: return [p]
        raise RuntimeError("No external Python found in PATH.")
    return [sys.executable]

def _gpu_run(cmd, env=None, on_progress=None, cwd=None, shell=False, no_window=True, progress_callback=None):
    import os, subprocess, tempfile, time, re, sys
    from pathlib import Path
    log_path = Path(tempfile.gettempdir()) / "screen_diffusion_pip_debug.log"

    def tee(line: str):
        if callable(on_progress):
            try: on_progress(line)
            except Exception: pass
        try:
            with open(log_path, "a", encoding="utf-8") as f: f.write(line + "\n")
        except Exception: pass

    env2 = dict(os.environ)
    if env: env2.update(env)
    env2["PYTHONUNBUFFERED"] = "1"
    env2["PYTHONIOENCODING"] = "utf-8"
    env2["PIP_PROGRESS_BAR"] = "on"
    env2["PIP_NO_COLOR"] = "1"
    env2["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    creationflags = 0
    startupinfo = None
    if os.name == "nt" and no_window:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        try: startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        except Exception: pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False, bufsize=0, env=env2, cwd=cwd, shell=shell, creationflags=creationflags, startupinfo=startupinfo)
    try:
        assert proc.stdout is not None
        buf = bytearray()
        last_yield_time = time.time()
        while True:
            chunk = proc.stdout.read(64)
            if not chunk:
                if proc.poll() is not None: break
                time.sleep(0.01)
                continue
            buf.extend(chunk)
            try:
                text = buf.decode("utf-8", errors="replace")
                lines = text.split('\n')
                if not text.endswith('\n'):
                    incomplete = lines[-1]
                    buf = bytearray(incomplete.encode("utf-8"))
                    lines = lines[:-1]
                else:
                    buf = bytearray()
                for line in lines:
                    if '\r' in line: line = line.split('\r')[-1]
                    line = line.strip()
                    if line:
                        tee(line)
                        yield line
                now = time.time()
                if buf and (now - last_yield_time) > 0.5:
                    partial = buf.decode("utf-8", errors="replace")
                    if '\r' in partial: partial = partial.split('\r')[-1]
                    partial = partial.strip()
                    if partial and len(partial) > 20:
                        tee(partial)
                        yield partial
                        last_yield_time = now
            except Exception: pass
        if buf:
            try:
                line = buf.decode("utf-8", errors="replace").strip()
                if line:
                    tee(line)
                    yield line
            except Exception: pass
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode})")

def _parse_pip_progress(line: str, progress_callback):
    import re
    line = line.strip()
    download_patterns = [r"Downloading\s+[^\n]*\s+(\d{1,3})%", r"[\|#]\s*(\d{1,3})%", r"(\d{1,3})%\s*[\|#]"]
    install_patterns = [r"Building wheel for.*\((\d{1,3})%\)", r"Installing.*\((\d{1,3})%\)"]
    for pattern in download_patterns:
        match = re.search(pattern, line)
        if match:
            percent = int(match.group(1))
            progress_callback(percent / 100.0, f"Downloading: {percent}%")
            return
    for pattern in install_patterns:
        match = re.search(pattern, line)
        if match:
            percent = int(match.group(1))
            progress_callback(percent / 100.0, f"Installing: {percent}%")
            return
    if "Successfully installed" in line: progress_callback(1.0, "Installation complete!")
    elif "Building wheel" in line and "complete" in line.lower(): progress_callback(1.0, "Build complete!")
    elif "Downloaded" in line and "MB" in line: progress_callback(1.0, "Download complete!")

def _search_caches_for_wheel(pattern: str):
    import os, re
    from pathlib import Path
    cache_dirs = []
    for env_var in ["LOCALAPPDATA", "APPDATA", "TEMP"]:
        val = os.environ.get(env_var)
        if val: cache_dirs.append(Path(val) / "pip" / "cache")
    cache_dirs.append(Path.home() / ".cache" / "pip")
    cache_dirs.append(INTERNAL_DIR / ".pip-cache")
    
    best_match = None
    best_pattern_idx = 999
    patterns = pattern if isinstance(pattern, list) else [pattern]
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists(): continue
        for root, _, files in os.walk(cache_dir):
            for filename in files:
                if not filename.endswith('.whl'): continue
                for idx, pat in enumerate(patterns):
                    if re.match(pat, filename, re.IGNORECASE):
                        wheel_path = Path(root) / filename
                        if not (wheel_path.is_file() and os.access(wheel_path, os.R_OK)): continue
                        if idx < best_pattern_idx:
                            best_match = wheel_path
                            best_pattern_idx = idx
                        if idx == 0: return wheel_path
    return best_match

def _find_wheel_in_cache(package_name: str, version: str, cuda_version: str = "cu118"):
    import re
    version_escaped = re.escape(version)
    cuda_escaped = re.escape(cuda_version)
    patterns = [
        rf"^{package_name}-{version_escaped}[\+%2B]{cuda_escaped}-cp\d+-cp\d+-.*\.whl$",
        rf"^{package_name}-{version_escaped}\.post\d+[\+%2B]{cuda_escaped}-cp\d+-cp\d+-.*\.whl$",
        rf"^{package_name}-{version_escaped}[\+%2B]{cuda_escaped}\.whl$",
        rf"^{package_name}-{version_escaped}.*{cuda_escaped}.*\.whl$",
    ]
    return _search_caches_for_wheel(patterns)

def _find_streamdiffusion_in_cache(package_name: str):
    return _search_caches_for_wheel(rf"^{package_name}-.*\.whl$")

def install_streamdiffusion(on_progress=None, skip_torch=True, progress_callback=None):
    py_cmd = _python_cmd_for_pip()
    streamdiffusion_package = "streamdiffusion[tensorrt]"
    if on_progress: on_progress("Installing StreamDiffusion...")
    try:
        cmd = [*py_cmd, "-m", "pip", "install", "--no-deps", streamdiffusion_package]
        for line in _gpu_run(cmd, on_progress=on_progress, no_window=not GPU_DIAGNOSTIC, progress_callback=progress_callback): pass
        if on_progress: on_progress("StreamDiffusion installed!")
    except Exception: return False
    try:
        test_cmd = [*py_cmd, "-c", "import sys; sys.path.insert(0, r'{}'); import streamdiffusion;".format(str(INTERNAL_DIR))]
        subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        return True
    except Exception: return False

def _gpu_relaunch_into_runtime():
    import sys, os
    os.environ[BOOT_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

class StreamGUI(ctk.CTk):
    def __init__(self):
        import multiprocessing as mp
        if mp.current_process().name != 'MainProcess': return
        super().__init__()
        self.title("Screen Diffusion")
        self.geometry("1180x840")
        self.minsize(1060, 760)
        self._set_window_icon()
        self.collapse_var = ctk.BooleanVar(value=False)
        # The engine knobs start folded away (issue #40, step 2).
        self.advanced_var = ctk.BooleanVar(value=False)
        self.collapse_btn = None
        self.left_panel = None
        self.prompts_row = None
        self.status_bar = None
        self.header_frame = None
        self._cache_mon_running = False
        self.preview_dim = 512
        self.proc_ctx = None
        self.proc_worker = None
        self.out_q = self.fps_q = self.status_q = self.control_q = self.debug_q = self.close_q = None
        self.monitor_sender = self.monitor_receiver = None
        self.running = False
        # Blank unless someone set the override: the field starts on whatever
        # complete model is under the models root, so a machine that has one can
        # start without pasting a path (issue #40, step 1).
        self.model_var = ctk.StringVar(value=resolve_local_model_path(LOCAL_MODEL_PATH))
        # What the picker shows (a folder name) against what the worker is given (a
        # path), and the standing sentence about the engine that pair needs.
        self.model_choice_var = ctk.StringVar(value=model_label(self.model_var.get()))
        self.engine_state_var = ctk.StringVar(value="")
        # Each entry: {"path": str, "scale": float}. Converted to StreamDiffusion's
        # lora_dict ({path: scale}) at start time. `lora_choice_var` is the picker
        # above them, which is a verb rather than a state: it rests on its prompt
        # and choosing a name adds a row (issue #44).
        self.lora_items: List[Dict[str, Any]] = []
        self.lora_choice_var = ctk.StringVar(value=ADD_LORA_PROMPT)
        self._lora_widgets: List[Any] = []
        self.prompt_var = ctk.StringVar(value="flip book animation, black and white rough sketch, rough drawing")
        self.neg_prompt_var = ctk.StringVar(value="low quality, bad quality, blurry, low resolution")
        # The Render Plan's two fields (issue #22). Both start blank, which is
        # `mode: "global"` - the whole frame under the prompt box, as it always was.
        # What is new (issue #40) is that the window now says so: `plan_state_var`
        # carries whether detection is running and on what, and `plan_note_var`
        # carries the validator's own words under the field that caused them.
        self.target_var = ctk.StringVar(value="")
        self.style_var = ctk.StringVar(value="")
        self.plan_state_var = ctk.StringVar(value=GLOBAL_STATE)
        self.plan_note_var = ctk.StringVar(value="")
        # The newest fps payload, which is where detection's own state comes from.
        self._fps_payload: Any = None
        # Where the frame's one diffusion call is spent (issue #39).
        self.detail_var = ctk.StringVar(value=DETAIL_ALL_OBJECTS)
        self.seed_var = ctk.StringVar(value="1")
        # The capture geometry, which since issue #39 is not the engine's canvas.
        # The label is the state; `capture_size` reads the two numbers off it at
        # start, so there is one place a capture size can come from.
        self.capture_var = ctk.StringVar(value=DEFAULT_CAPTURE)
        self.buffer_var = ctk.StringVar(value=str(DEFAULT_FRAME_BUFFER_SIZE))
        self.accel_var = ctk.StringVar(value=DEFAULT_ACCELERATION)
        # Ignored for sd-turbo (already 1-step). For SD1.5 this pulls
        # latent-consistency/lcm-lora-sdv1-5, which needs ~4 steps.
        self.use_lcm_lora_var = ctk.BooleanVar(value=False)
        self.denoise_batch_var = ctk.BooleanVar(value=True)
        self.cfg_type_var = ctk.StringVar(value="none")
        self.guidance_var = ctk.StringVar(value="0.0")
        self.delta_var = ctk.StringVar(value="0.0")
        self.sim_filter_var = ctk.BooleanVar(value=False)
        self.sim_thresh_var = ctk.StringVar(value="0.99")
        self.sim_maxskip_var = ctk.StringVar(value="10.0")
        self.offline_var = ctk.BooleanVar(value=True)
        self._debounce_prompt = self._debounce_neg = self._debounce_region = self._debounce_plan = None
        self.t_index_list: List[int] = list(DEFAULT_T_INDEX_LIST)
        self._lockables: List[ctk.CTkBaseClass] = []
        self._step_sliders: List[ctk.CTkSlider] = []
        self.capwin: Optional[FloatingCaptureWindow] = None
        self.bind("<h>", lambda e: self._toggle_capture_window())
        self.bind("<H>", lambda e: self._toggle_capture_window())
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            img = Image.open(logo_path)
            self.logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(28, 28))
        except Exception:
            self.logo_img = None
        self._build_ui()
        # What the window opens on has an engine or it does not, and that is worth
        # knowing before Start rather than after it (issue #38, step 6).
        self._refresh_engine_state()
        self._setup_input_validation()
        self.bind("<Configure>", self._on_window_configure)
        self._poll_queues()
        self.protocol("WM_DELETE_WINDOW", self.do_quit)

    def _toggle_capture_window(self):
        if not self.running or self.capwin is None: return
        try:
            if self.capwin.win.state() == "normal":
                self.capwin.win.withdraw()
                self.hide_capture_btn.configure(text="👁 Show (H)")
                self.status_var.set("Capture window hidden")
            else:
                self.capwin.win.deiconify()
                self.hide_capture_btn.configure(text="👁 Hide (H)")
                self.status_var.set("Capture window visible")
                self._send_region_update()
        except Exception: pass

    def _set_window_icon(self):
        try:
            icon_paths = [resource_path("icon2.ico"), os.path.join(os.path.dirname(__file__), "icon2.ico")]
            for icon_path in icon_paths:
                if os.path.exists(icon_path):
                    self.iconbitmap(icon_path)
                    return True
        except Exception: pass
        return False

    def _setup_input_validation(self):
        def validate_numeric_input(P):
            if P == "" or P == "-": return True
            try: float(P); return True
            except ValueError: return False
        vcmd = (self.register(validate_numeric_input), '%P')
        numeric_entries = [
            getattr(self, '_w_seed_entry', None),
            getattr(self, '_w_guidance_entry', None),
            getattr(self, '_w_delta_entry', None),
            getattr(self, '_w_sim_thresh', None),
            getattr(self, '_w_sim_maxskip', None),
        ]
        for entry in numeric_entries:
            if entry: entry.configure(validate="key", validatecommand=vcmd)

    def _download_sd_turbo(self):
        try:
            download_dir = filedialog.askdirectory(title="Select directory to download sd-turbo model",
                                                   initialdir=str(resolve_models_dir()))
            if not download_dir: return
            model_path = os.path.join(download_dir, "sd-turbo-fp16")
            self._show_download_dialog(model_path)
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to start download: {e}")

    def _show_download_dialog(self, model_path):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Downloading sd-turbo fp16")
        dialog.geometry("600x400")
        dialog.transient(self)
        dialog.grab_set()
        self._download_process = None
        self._download_cancelled = False
        self._download_thread = None
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        ctk.CTkLabel(dialog, text="Downloading sd-turbo fp16 model...", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 10))
        ctk.CTkLabel(dialog, text=f"Destination: {model_path}", wraplength=500).pack(pady=(0, 10))
        status_label = ctk.CTkLabel(dialog, text="Starting download...")
        status_label.pack(side="top", pady=(0, 10))
        cancel_btn = ctk.CTkButton(dialog, text="Cancel", command=lambda: self._cancel_download(dialog, model_path))
        cancel_btn.pack(side="bottom", pady=(10, 20))
        scroll_frame = ctk.CTkScrollableFrame(dialog, width=500, height=150)
        scroll_frame.pack(side="top", fill="both", expand=True, padx=20, pady=(0, 10))
        file_uis = {}
        def ui_updater(action, filename=None, progress=0.0, text=""):
            if self._download_cancelled: return
            if action == "global": status_label.configure(text=text)
            elif action == "update" and filename:
                if filename not in file_uis:
                    row_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
                    row_frame.pack(fill="x", pady=2)
                    display_name = filename if len(filename) < 25 else filename[:22] + "..."
                    lbl = ctk.CTkLabel(row_frame, text=f"{display_name}: 0%", width=220, anchor="w")
                    lbl.pack(side="left", padx=(0, 10))
                    bar = ctk.CTkProgressBar(row_frame, width=200)
                    bar.pack(side="right", fill="x", expand=True)
                    bar.set(0)
                    file_uis[filename] = {'label': lbl, 'bar': bar, 'name': display_name}
                display_name = file_uis[filename]['name']
                file_uis[filename]['label'].configure(text=f"{display_name}: {int(progress*100)}%")
                file_uis[filename]['bar'].set(progress)
        self._current_download_dialog = dialog
        self._current_download_path = model_path
        self._download_thread = threading.Thread(target=self._run_hf_download, args=(model_path, ui_updater, dialog), daemon=True)
        self._download_thread.start()

    def _cleanup_partial_download(self, model_path):
        try:
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
        except Exception: pass

    def _cancel_download(self, dialog, model_path):
        self._download_cancelled = True
        if hasattr(self, '_download_process') and self._download_process:
            try:
                self._download_process.terminate()
                import time
                time.sleep(1)
                if self._download_process.poll() is None:
                    self._download_process.kill()
            except Exception: pass
        self._cleanup_partial_download(model_path)
        dialog.destroy()
        if hasattr(self, '_current_download_dialog'): del self._current_download_dialog
        if hasattr(self, '_current_download_path'): del self._current_download_path

    def _run_hf_download(self, model_path, ui_updater, dialog):
        try:
            model_repo = "stabilityai/sd-turbo"
            def safe_ui_update(action, filename=None, progress=0.0, text=""):
                if self._download_cancelled: return
                try: self.after(0, lambda: ui_updater(action, filename, progress, text))
                except Exception: pass
            def check_cancelled(): return self._download_cancelled
            safe_ui_update("global", text="Checking Hugging Face CLI...")
            if check_cancelled(): return
            try:
                subprocess.run(["huggingface-cli", "--version"], capture_output=True, check=True, timeout=10)
                use_cli = True
            except: use_cli = False
            if not use_cli:
                safe_ui_update("global", text="Installing huggingface_hub...")
                try: subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True, capture_output=True, timeout=120)
                except Exception as e:
                    safe_ui_update("global", text=f"Installation failed: {e}")
                    return
            safe_ui_update("global", text="Starting download...")
            if check_cancelled(): return
            os.makedirs(model_path, exist_ok=True)
            cmd = ["huggingface-cli", "download", model_repo, "--local-dir", model_path, "--local-dir-use-symlinks", "False", "--resume-download", "--exclude", "sd_turbo.safetensors", "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"]
            self._download_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False, bufsize=0)
            try:
                buffer = bytearray()
                tracked_urls = set()
                while True:
                    if check_cancelled():
                        self._download_process.terminate()
                        import time
                        time.sleep(2)
                        if self._download_process.poll() is None: self._download_process.kill()
                        break
                    chunk = self._download_process.stdout.read(64)
                    if not chunk:
                        if self._download_process.poll() is not None: break
                        import time
                        time.sleep(0.01)
                        continue
                    buffer.extend(chunk)
                    while b'\r' in buffer or b'\n' in buffer:
                        r_idx = buffer.find(b'\r')
                        n_idx = buffer.find(b'\n')
                        idx = r_idx if r_idx != -1 and (n_idx == -1 or r_idx < n_idx) else n_idx
                        line_bytes = buffer[:idx]
                        buffer = buffer[idx + 1:]
                        try:
                            line = line_bytes.decode('utf-8', errors='replace').strip()
                            if line:
                                if "Fetching" in line and "%" in line:
                                    import re
                                    match = re.search(r'(\d+)%', line)
                                    if match:
                                        pct = int(match.group(1))
                                        safe_ui_update("global", text=f"Fetching files... {pct}%")
                                if "downloading https://" in line and ".incomplete" in line:
                                    import re
                                    match = re.search(r'downloading (https://\S+) to (.*\.incomplete)', line)
                                    if match:
                                        url = match.group(1)
                                        filepath = match.group(2)
                                        if url not in tracked_urls:
                                            tracked_urls.add(url)
                                            url_parts = url.split('/')
                                            filename = f"{url_parts[-2]}/{url_parts[-1]}" if len(url_parts) >= 2 else url_parts[-1]
                                            def track_progress(dl_url, path, fname):
                                                import time, os, urllib.request
                                                try:
                                                    req = urllib.request.Request(dl_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
                                                    with urllib.request.urlopen(req) as response:
                                                        total_size = int(response.headers.get('Content-Length', 0))
                                                    if total_size > 0:
                                                        while not check_cancelled():
                                                            if os.path.exists(path):
                                                                current = os.path.getsize(path)
                                                                percent = current / total_size
                                                                safe_ui_update("update", filename=fname, progress=percent)
                                                            else:
                                                                safe_ui_update("update", filename=fname, progress=1.0)
                                                                break
                                                            time.sleep(0.5)
                                                except Exception: pass
                                            import threading
                                            threading.Thread(target=track_progress, args=(url, filepath, filename), daemon=True).start()
                        except Exception: pass
                returncode = self._download_process.wait(timeout=5)
                if check_cancelled():
                    self._cleanup_partial_download(model_path)
                    return
                if returncode == 0:
                    safe_ui_update("global", text="Download completed successfully!")
                    self.after(0, lambda: self.model_var.set(model_path))
                    self.after(2000, dialog.destroy)
                    self.after(0, lambda: messagebox.showinfo("Download Complete", f"sd-turbo model downloaded successfully to:\n{model_path}"))
                else:
                    if not check_cancelled():
                        error_msg = f"Download failed with exit code {returncode}"
                        safe_ui_update("global", text=error_msg)
            except Exception: pass
            finally:
                if self._download_process and self._download_process.poll() is None:
                    try:
                        self._download_process.terminate()
                        self._download_process.wait(timeout=2)
                    except:
                        try: self._download_process.kill()
                        except: pass
                self._download_process = None
        except Exception: pass

    def _toggle_advanced(self):
        """Fold the engine knobs in or out. Folded at start - issue #40, step 2."""
        opening = not self.advanced_var.get()
        self.advanced_var.set(opening)
        self._w_advanced_toggle.configure(text=ADVANCED_OPEN if opening else ADVANCED_CLOSED)
        if opening: self._advanced_body.grid()
        else: self._advanced_body.grid_remove()

    def _toggle_collapse(self):
        if self.collapse_var.get():
            self.collapse_var.set(False)
            self.collapse_btn.configure(text="📱 Compact")
            if self.header_frame: self.header_frame.grid(row=0, column=0, sticky="nw", padx=(12, 0), pady=(10, 0))
            if self.left_panel: self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(6, 6))
            if self.prompts_row: self.prompts_row.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
            if self.status_bar: self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
            self.grid_rowconfigure(0, weight=0); self.grid_rowconfigure(1, weight=1); self.grid_rowconfigure(2, weight=0); self.grid_rowconfigure(3, weight=0); self.grid_columnconfigure(0, weight=0); self.grid_columnconfigure(1, weight=1)
            self.geometry("1180x840")
            self.minsize(1060, 760)
        else:
            self.collapse_var.set(True)
            self.collapse_btn.configure(text="📖 Expand")
            if self.header_frame: self.header_frame.grid_remove()
            if self.left_panel: self.left_panel.grid_remove()
            if self.prompts_row: self.prompts_row.grid_remove()
            if self.status_bar: self.status_bar.grid_remove()
            self.grid_rowconfigure(0, weight=0); self.grid_rowconfigure(1, weight=1); self.grid_rowconfigure(2, weight=0); self.grid_rowconfigure(3, weight=0); self.grid_columnconfigure(0, weight=0); self.grid_columnconfigure(1, weight=1)
            preview_size = self.preview_dim + 150
            self.geometry(f"{preview_size + 50}x{preview_size + 100}")
            self.minsize(preview_size, preview_size)

    def _update_collapse_state(self):
        collapsed = self.collapse_var.get()
        left_panel = None
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkScrollableFrame) and child._width == 440:
                left_panel = child
                break
        prompts_row = None
        for child in self.winfo_children():
            if hasattr(child, 'winfo_children') and child.winfo_children():
                for subchild in child.winfo_children():
                    if hasattr(subchild, 'winfo_children') and any('textbox' in str(widget).lower() for widget in subchild.winfo_children()):
                        prompts_row = child
                        break
                if prompts_row: break
        status_bar = None
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame) and child._height == 36:
                status_bar = child
                break
        header = None
        for child in self.winfo_children():
            if child == self.title_label.master:
                header = child
                break
        elements_to_toggle = [left_panel, prompts_row, status_bar, header]
        for element in elements_to_toggle:
            if element:
                if collapsed: element.grid_remove()
                else: element.grid()
        if hasattr(self, 'gpu_frame') and self.gpu_frame:
            if collapsed: self.gpu_frame.grid_remove()
            else: self.gpu_frame.grid()

    def _check_and_hide_gpu_frame(self):
        torch_available = False
        try:
            import sys, os
            torch_dir = SITE_PACKAGES / "torch" 
            if not torch_dir.exists(): return
            site_packages_str = str(SITE_PACKAGES)
            if site_packages_str not in sys.path: sys.path.insert(0, site_packages_str)
            torch_lib_dir = torch_dir / "lib"
            if torch_lib_dir.exists():
                try: os.add_dll_directory(str(torch_lib_dir))
                except: pass
                current_path = os.environ.get("PATH", "")
                torch_lib_str = str(torch_lib_dir)
                if torch_lib_str not in current_path: os.environ["PATH"] = torch_lib_str + os.pathsep + current_path
            _prime_dll_search(SITE_PACKAGES, INTERNAL_DIR)
            import torch
            try: cuda_available = torch.cuda.is_available()
            except Exception: cuda_available = False
            torch_available = True
        except Exception: pass
        if torch_available:
            try:
                if hasattr(self, 'gpu_frame') and self.gpu_frame.winfo_ismapped():
                    self.gpu_frame.grid_remove()
                if hasattr(self, 'status_var'):
                    import torch
                    self.status_var.set(f"GPU runtime ready ✅ (PyTorch {torch.__version__})")
            except Exception: pass

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0); self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0); self.grid_rowconfigure(1, weight=1); self.grid_rowconfigure(2, weight=0); self.grid_rowconfigure(3, weight=0)
        left_header = ctk.CTkFrame(self, fg_color="transparent")
        left_header.grid(row=0, column=0, sticky="nw", padx=(12, 0), pady=(10, 0))
        left_header.grid_columnconfigure(0, weight=0); left_header.grid_columnconfigure(1, weight=0); left_header.grid_columnconfigure(2, weight=1); left_header.grid_columnconfigure(3, weight=0)
        if getattr(self, "logo_img", None):
            ctk.CTkLabel(left_header, image=self.logo_img, text="").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.title_label = ctk.CTkLabel(left_header, text="Screen Diffusion", font=ctk.CTkFont(size=26, weight="bold"))
        self.title_label.grid(row=0, column=1, sticky="w")
        self.version_label = ctk.CTkLabel(left_header, text="Version 0.2 ", font=ctk.CTkFont(size=14, slant="italic"), text_color="gray70")
        self.version_label.grid(row=0, column=3, sticky="e", padx=(0, 4))
        self.header_frame = left_header
        left = ctk.CTkScrollableFrame(self, width=440, corner_radius=12)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(6, 6))
        left.grid_columnconfigure(0, weight=1)
        self.left_panel = left
        row = 0
        # --- what to restyle: the Render Plan's producer (issue #22), leading ---
        #
        # These two fields are the product's interface in v1 - there is no LLM
        # producer coming (spec 6) - and they used to sit below the preview among
        # the engine knobs, with nothing anywhere saying whether detection was even
        # running. Target text goes to the open-vocabulary detector, style text
        # goes to StreamDiffusion. Both stay editable while generation runs:
        # changing what is restyled must not mean stopping the run.
        plan_frame = ctk.CTkFrame(left, corner_radius=10)
        plan_frame.grid(row=row, column=0, sticky="ew", pady=(4, 8))
        plan_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(plan_frame, text="Restyle", font=ctk.CTkFont(size=16, weight="bold"), anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        ctk.CTkLabel(plan_frame, text="Target  —  what to restyle (blank: the whole frame)", anchor="w", text_color="gray70").grid(row=1, column=0, sticky="ew", padx=10)
        self._w_target_entry = ctk.CTkEntry(plan_frame, textvariable=self.target_var)
        self._w_target_entry.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 6))
        self._w_target_entry.bind("<KeyRelease>", self._on_plan_field_changed)
        # One style, not one per object: issue #5 chose the full-frame masked
        # primitive, which is one prompt embedding per frame however many matched.
        ctk.CTkLabel(plan_frame, text="Style  —  one style for every match (blank: the prompt below)", anchor="w", text_color="gray70").grid(row=3, column=0, sticky="ew", padx=10)
        self._w_style_entry = ctk.CTkEntry(plan_frame, textvariable=self.style_var)
        self._w_style_entry.grid(row=4, column=0, sticky="ew", padx=10, pady=(2, 6))
        self._w_style_entry.bind("<KeyRelease>", self._on_plan_field_changed)
        # Detail: where the frame's one diffusion call is spent (issue #39). Not a
        # lockable - like the two fields above it, changing what is restyled must
        # not mean stopping the run - and it is a plan field, so it costs no
        # engine rebuild. It leads the left panel with them (issue #40, step 2).
        ctk.CTkLabel(plan_frame, text="Detail  —  where the frame's one diffusion call goes", anchor="w", text_color="gray70").grid(row=5, column=0, sticky="ew", padx=10)
        self._w_detail_combo = ctk.CTkComboBox(
            plan_frame, values=list(DETAIL_PRESETS), variable=self.detail_var,
            command=self._on_plan_field_changed)
        self._w_detail_combo.grid(row=6, column=0, sticky="ew", padx=10, pady=(2, 6))
        # Step 3: the plan's state in words, and step 5: the validator's, right
        # under the field that produced them.
        self._w_plan_state = ctk.CTkLabel(plan_frame, textvariable=self.plan_state_var, anchor="w", justify="left", wraplength=400)
        self._w_plan_state.grid(row=7, column=0, sticky="ew", padx=10, pady=(0, 2))
        self._w_plan_note = ctk.CTkLabel(plan_frame, textvariable=self.plan_note_var, anchor="w", justify="left", wraplength=400, text_color=PLAN_NOTE_COLOR)
        self._w_plan_note.grid(row=8, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._w_plan_note.grid_remove()
        row += 1
        if SHOW.get("model_path", True):
            ctk.CTkLabel(left, text="Base model:", anchor="w").grid(row=row, column=0, sticky="ew", pady=(4, 0))
            mp = ctk.CTkFrame(left); mp.grid(row=row+1, column=0, sticky="ew")
            mp.grid_columnconfigure(0, weight=1); mp.grid_columnconfigure(1, weight=0); mp.grid_columnconfigure(2, weight=0)
            # Issue #38 step 6: pick from what is on disk. The entry stays, because
            # a model outside the models root is still a legal answer and Browse is
            # how it gets typed - but choosing one should not require knowing a path.
            self._w_model_combo = ctk.CTkOptionMenu(
                mp, values=self._model_choices(),
                variable=self.model_choice_var, command=self._on_model_chosen)
            self._w_model_combo.grid(row=0, column=0, sticky="ew", padx=(0,6), pady=6)
            self._w_model_browse = ctk.CTkButton(mp, text="Browse", command=self._browse_model, width=70)
            self._w_model_browse.grid(row=0, column=1, padx=(0,6), pady=6)
            self._w_model_download = ctk.CTkButton(mp, text="⬇ Download SD-Turbo", command=self._download_sd_turbo, width=140, fg_color="#10B981", hover_color="#059669")
            self._w_model_download.grid(row=0, column=2, pady=6)
            self._w_model_entry = ctk.CTkEntry(mp, textvariable=self.model_var)
            self._w_model_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 6))
            # What that model needs compiling, and whether it is compiled. The one
            # thing the window could not say before Start went quiet for minutes.
            self._w_engine_state = ctk.CTkLabel(mp, textvariable=self.engine_state_var, anchor="w", justify="left", wraplength=400)
            self._w_engine_state.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 6))
            self._register_lockables(self._w_model_entry, self._w_model_browse, self._w_model_download, self._w_model_combo)
            row += 2
        if SHOW.get("lora", True):
            lf = ctk.CTkFrame(left); lf.grid(row=row, column=0, sticky="ew", pady=(4, 6))
            lf.grid_columnconfigure(0, weight=1)
            lh = ctk.CTkFrame(lf, fg_color="transparent")
            lh.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
            ctk.CTkLabel(lh, text="Style LoRAs", font=ctk.CTkFont(weight="bold")).pack(side="left")
            # Issue #44 step 4: Browse is kept. It is no longer the only way in, so
            # it is labelled for what it is rather than for what it adds.
            self._w_lora_add = ctk.CTkButton(lh, text="Browse", width=70, command=self._add_lora)
            self._w_lora_add.pack(side="right")
            # Step 3: the styles are three files in one known directory, so they are
            # picked from a list. Choosing one adds it - the same one door Browse
            # goes through (`_add_lora_path`).
            self._w_lora_combo = ctk.CTkOptionMenu(
                lh, values=self._lora_choices(), variable=self.lora_choice_var,
                command=self._on_lora_chosen)
            self._w_lora_combo.pack(side="right", padx=(0, 6))
            self._register_lockables(self._w_lora_add, self._w_lora_combo)
            self._loras_holder = ctk.CTkFrame(lf, fg_color="transparent")
            self._loras_holder.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 6))
            self._loras_holder.grid_columnconfigure(0, weight=1)
            self._build_loras_ui()
            row += 1
        # --- denoising strength: the live control, and it stays primary --------
        #
        # Moving a step's *value* is a runtime update (`set_t_index_list`); it is
        # the step *count* that keys a new engine, and those two buttons are in
        # the advanced section below with the rest of the engine knobs.
        steps_frame = ctk.CTkFrame(left, fg_color="transparent")
        steps_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=(4, 6))
        step_header = ctk.CTkFrame(steps_frame, fg_color="transparent")
        step_header.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(step_header, text="Denoising Steps", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkLabel(step_header, text="higher index = less denoise", text_color="gray70").pack(side="right")
        self._steps_holder = ctk.CTkFrame(steps_frame)
        self._steps_holder.pack(fill="both", expand=True, pady=5)
        self._build_steps_ui()
        row += 1
        # --- advanced: the engine knobs, folded away (issue #40, step 2) -------
        #
        # Everything in `ADVANCED` lives in here. None of it is a choice about
        # what the app makes, and three of them cost a TensorRT rebuild, which is
        # what the standing hint says before anything is touched.
        adv = ctk.CTkFrame(left, corner_radius=10)
        adv.grid(row=row, column=0, sticky="ew", pady=(4, 6))
        adv.grid_columnconfigure(0, weight=1)
        self._w_advanced_toggle = ctk.CTkButton(adv, text=ADVANCED_CLOSED, command=self._toggle_advanced, anchor="w", fg_color="transparent", hover_color=CUSTOM_COLORS["surface"])
        self._w_advanced_toggle.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        adv_body = ctk.CTkFrame(adv, fg_color="transparent")
        adv_body.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        adv_body.grid_columnconfigure(0, weight=1)
        self._advanced_body = adv_body
        self._advanced_body.grid_remove()
        ctk.CTkLabel(adv_body, text=ENGINE_REBUILD_HINT, anchor="w", justify="left", wraplength=390, text_color="gray70").grid(row=0, column=0, sticky="ew", pady=(0, 6))
        g2 = ctk.CTkFrame(adv_body); g2.grid(row=1, column=0, sticky="ew", pady=(0,6))
        for i in range(6): g2.grid_columnconfigure(i, weight=1)
        col = 0
        if SHOW.get("seed", True):
            ctk.CTkLabel(g2, text="Seed").grid(row=0, column=col, sticky="w")
            self._w_seed_entry = ctk.CTkEntry(g2, textvariable=self.seed_var, width=80)
            self._w_seed_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_seed_entry)
        if SHOW.get("frame_buffer_size", True):
            ctk.CTkLabel(g2, text="Buffer").grid(row=0, column=col, sticky="w")
            self._w_buffer_entry = ctk.CTkEntry(g2, textvariable=self.buffer_var, width=70)
            self._w_buffer_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_buffer_entry)
        # Capture size. Not the engine's canvas: TensorRT builds every engine at
        # 512x512 (spec 7.2), so this widens what is *grabbed* and the frame loop
        # resizes onto the canvas. Locked while a run is live - the capture thread
        # and the capture window are both sized at start.
        ctk.CTkLabel(g2, text="Capture").grid(row=0, column=col, sticky="w")
        self._w_capture_combo = ctk.CTkComboBox(
            g2, values=list(CAPTURE_PRESETS), variable=self.capture_var, width=150)
        self._w_capture_combo.grid(row=1, column=col, sticky="ew"); col += 1
        self._register_lockables(self._w_capture_combo)
        if SHOW.get("acceleration", True):
            ctk.CTkLabel(g2, text="Acceleration").grid(row=0, column=col, sticky="w")
            self._w_accel_combo = ctk.CTkComboBox(g2, values=list(ACCELERATIONS), variable=self.accel_var, width=120)
            self._w_accel_combo.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_accel_combo)
        if SHOW.get("use_lcm_lora", True):
            self._w_lcm_switch = ctk.CTkSwitch(g2, text="LCM-LoRA", variable=self.use_lcm_lora_var)
            self._w_lcm_switch.grid(row=1, column=col, sticky="w", padx=(6,0)); col += 1
            self._register_lockables(self._w_lcm_switch)
        if SHOW.get("use_denoising_batch", True):
            self._w_denoise_switch = ctk.CTkSwitch(g2, text="Denoising batch", variable=self.denoise_batch_var)
            self._w_denoise_switch.grid(row=1, column=col, sticky="w"); col += 1
            self._register_lockables(self._w_denoise_switch)
        if SHOW.get("step_count", False):
            steps_row = ctk.CTkFrame(adv_body, fg_color="transparent")
            steps_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
            ctk.CTkLabel(steps_row, text="Denoising step count").pack(side="left")
            self._w_step_add = ctk.CTkButton(steps_row, text="+ Add", width=50, command=self._add_step)
            self._w_step_add.pack(side="right", padx=(5,0))
            self._w_step_remove = ctk.CTkButton(steps_row, text="- Remove", width=60, command=self._remove_step)
            self._w_step_remove.pack(side="right")
            self._register_lockables(self._w_step_add, self._w_step_remove)
        g3 = ctk.CTkFrame(adv_body); g3.grid(row=3, column=0, sticky="ew", pady=(0,6)); g3.grid_remove()
        for i in range(6): g3.grid_columnconfigure(i, weight=1)
        col = 0
        if SHOW.get("cfg_type", False):
            ctk.CTkLabel(g3, text="CFG type").grid(row=0, column=col, sticky="w")
            self._w_cfg_combo = ctk.CTkComboBox(g3, values=["none","full","self","initialize"], variable=self.cfg_type_var, width=120)
            self._w_cfg_combo.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_cfg_combo)
        if SHOW.get("guidance_scale", False):
            ctk.CTkLabel(g3, text="Guidance").grid(row=0, column=col, sticky="w")
            self._w_guidance_entry = ctk.CTkEntry(g3, textvariable=self.guidance_var, width=70)
            self._w_guidance_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_guidance_entry)
        if SHOW.get("delta", False):
            ctk.CTkLabel(g3, text="Delta").grid(row=0, column=col, sticky="w")
            self._w_delta_entry = ctk.CTkEntry(g3, textvariable=self.delta_var, width=70)
            self._w_delta_entry.grid(row=1, column=col, sticky="ew"); col += 1
            self._register_lockables(self._w_delta_entry)
        if SHOW.get("offline", False):
            self._w_offline_switch = ctk.CTkSwitch(g3, text="Offline", variable=self.offline_var)
            self._w_offline_switch.grid(row=1, column=col, sticky="w"); col += 1
            self._register_lockables(self._w_offline_switch)
        if SHOW.get("similar_image_filter", False):
            sim = ctk.CTkFrame(adv_body); sim.grid(row=4, column=0, sticky="ew", pady=(0,6))
            sim.grid_columnconfigure(0, weight=0); sim.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(sim, text="Similar Image Filter").grid(row=0, column=0, sticky="w", pady=(0,4))
            self._w_sim_switch  = ctk.CTkSwitch(sim, text="Enable", variable=self.sim_filter_var); self._w_sim_switch.grid(row=1, column=0, sticky="w")
            self._w_sim_thresh  = ctk.CTkEntry(sim, textvariable=self.sim_thresh_var, width=80); self._w_sim_thresh.grid(row=2, column=1, sticky="w", padx=(6,0))
            self._w_sim_maxskip = ctk.CTkEntry(sim, textvariable=self.sim_maxskip_var, width=80); self._w_sim_maxskip.grid(row=3, column=1, sticky="w", padx=(6,0))
            self._register_lockables(self._w_sim_switch, self._w_sim_thresh, self._w_sim_maxskip)
        row += 1
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(6, 6))
        right.grid_rowconfigure(0, weight=0); right.grid_rowconfigure(1, weight=1); right.grid_rowconfigure(2, weight=0); right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Preview").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))
        self.preview_container = ctk.CTkFrame(right, corner_radius=8, width=self.preview_dim, height=self.preview_dim)
        self.preview_container.grid(row=1, column=0, sticky="n", padx=12, pady=(8, 12))
        self.preview_container.grid_propagate(False)
        self._photo = CTkImage(light_image=Image.new("RGB", (self.preview_dim, self.preview_dim)),size=(self.preview_dim, self.preview_dim))
        self.preview_panel = ctk.CTkLabel(self.preview_container, image=self._photo, text="")
        self.preview_panel.place(x=0, y=0)
        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        actions.grid_columnconfigure(0, weight=1); actions.grid_columnconfigure(1, weight=0); actions.grid_columnconfigure(2, weight=1)
        button_container = ctk.CTkFrame(actions, fg_color="transparent")
        button_container.grid(row=0, column=1, sticky="ew")
        self.start_btn = ctk.CTkButton(button_container, text="🚀 Start Generation", command=self._on_start, width=140, height=36, fg_color=CUSTOM_COLORS["success"], hover_color="#059669", corner_radius=8, font=ctk.CTkFont(weight="bold"))
        self.start_btn.grid(row=0, column=0, padx=(0, 8))
        self.stop_btn = ctk.CTkButton(button_container, text="⏹ Stop", command=self._on_stop, width=100, height=36, fg_color=CUSTOM_COLORS["error"], hover_color="#DC2626", corner_radius=8, font=ctk.CTkFont(weight="bold"))
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))
        self.stop_btn.configure(state="disabled")
        self.hide_capture_btn = ctk.CTkButton(button_container, text="👁 Hide (H)", command=self._toggle_capture_window, width=100, height=36, fg_color="#6B7280", hover_color="#4B5563", corner_radius=8, font=ctk.CTkFont(weight="bold"))
        self.hide_capture_btn.grid(row=0, column=2, padx=(0, 8))
        self.hide_capture_btn.configure(state="disabled")
        self.collapse_btn = ctk.CTkButton(button_container, text="📱 Compact", command=self._toggle_collapse, width=100, height=36, fg_color="#6B7280", hover_color="#4B5563", corner_radius=8, font=ctk.CTkFont(weight="bold"))
        self.collapse_btn.grid(row=0, column=3, padx=(0, 8)) 
        ctk.CTkButton(button_container, text="❌ Quit", command=self.do_quit, width=100, height=36, fg_color=CUSTOM_COLORS["surface"], hover_color=CUSTOM_COLORS["error"], corner_radius=8, font=ctk.CTkFont(weight="bold")).grid(row=0, column=4)
        prompts_row = ctk.CTkFrame(self)
        prompts_row.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
        prompts_row.grid_columnconfigure(0, weight=1); prompts_row.grid_columnconfigure(1, weight=1)
        prompts_row.grid_rowconfigure(0, weight=1)
        # Target and Style used to sit here, in a row of their own above these two
        # boxes. They lead the left panel now (issue #40, step 2); what is left
        # here is the global prompt, which is what a blank target restyles the
        # whole frame with and what a blank style falls back to.
        prompt_frame = ctk.CTkFrame(prompts_row, corner_radius=10); prompt_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        prompt_frame.grid_rowconfigure(1, weight=1); prompt_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(prompt_frame, text="Prompt", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.prompt_txt = ctk.CTkTextbox(prompt_frame, height=120); self.prompt_txt.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.prompt_txt.delete("1.0", "end"); self.prompt_txt.insert("1.0", self.prompt_var.get()); self.prompt_txt.bind("<KeyRelease>", self._on_prompt_changed)
        neg_frame = ctk.CTkFrame(prompts_row, corner_radius=10); neg_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        neg_frame.grid_rowconfigure(1, weight=1); neg_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(neg_frame, text="Negative Prompt", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.neg_prompt_txt = ctk.CTkTextbox(neg_frame, height=120); self.neg_prompt_txt.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.neg_prompt_txt.delete("1.0", "end"); self.neg_prompt_txt.insert("1.0", self.neg_prompt_var.get()); self.neg_prompt_txt.bind("<KeyRelease>", self._on_neg_prompt_changed)
        self.prompts_row = prompts_row
        status = ctk.CTkFrame(self, height=36, corner_radius=12)
        status.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        status.grid_columnconfigure(0, weight=0); status.grid_columnconfigure(1, weight=1)
        self.fps_var = ctk.StringVar(value="FPS: --"); self.status_var = ctk.StringVar(value="idle")
        ctk.CTkLabel(status, textvariable=self.fps_var).grid(row=0, column=0, sticky="w", padx=12)
        ctk.CTkLabel(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=12)
        self.status_bar = status

    def _validate_numeric_parameters(self):
        try:
            seed_str = self.seed_var.get().strip()
            if seed_str == "": seed = 1
            else:
                seed = int(seed_str)
                if seed < 0: seed = 0
            self.seed_var.set(str(seed))
            buffer_str = self.buffer_var.get().strip()
            if buffer_str == "": buffer_size = 1
            else:
                buffer_size = int(buffer_str)
                if buffer_size < 1: buffer_size = 1
            self.buffer_var.set(str(buffer_size))
            guidance_str = self.guidance_var.get().strip()
            if guidance_str == "": guidance = 0.0
            else:
                guidance = float(guidance_str)
                if guidance < 0: guidance = 0.0
            self.guidance_var.set(str(guidance))
            delta_str = self.delta_var.get().strip()
            if delta_str == "": delta = 0.0
            else:
                delta = float(delta_str)
                if delta < 0: delta = 0.0
            self.delta_var.set(str(delta))
            sim_thresh_str = self.sim_thresh_var.get().strip()
            if sim_thresh_str == "": sim_thresh = 0.99
            else:
                sim_thresh = float(sim_thresh_str)
                if not (0.0 <= sim_thresh <= 1.0): sim_thresh = 0.99
            self.sim_thresh_var.set(str(sim_thresh))
            sim_maxskip_str = self.sim_maxskip_var.get().strip()
            if sim_maxskip_str == "": sim_maxskip = 10.0
            else:
                sim_maxskip = float(sim_maxskip_str)
                if sim_maxskip < 0: sim_maxskip = 10.0
            self.sim_maxskip_var.set(str(sim_maxskip))
            return True
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid numeric value: {e}")
            return False

    def _register_lockables(self, *ws):
        for w in ws:
            if w is not None: self._lockables.append(w)

    def _set_state(self, widgets, state: str):
        for w in widgets:
            try: w.configure(state=state)
            except Exception: pass

    def _apply_running_state(self):
        if self.running:
            self._set_state(self._lockables, "disabled")
            try: self.prompt_txt.configure(state="normal")
            except Exception: pass
            try: self.neg_prompt_txt.configure(state="normal")
            except Exception: pass
            for s in self._step_sliders:
                try: s.configure(state="normal")
                except Exception: pass
        else:
            self._set_state(self._lockables, "normal")

    def _confirm_engine_rebuild(self, setting: str) -> bool:
        """Ask before a change that keys a different TensorRT engine (issue #40).

        True when the change may go ahead - including on the acceleration paths
        that compile nothing at all, where there is no build to warn about.
        """
        if not engine_rebuild_needed(self.accel_var.get()): return True
        return bool(messagebox.askokcancel("TensorRT engine build required",
                                           _engine_rebuild_warning(setting)))

    def _add_step(self):
        if self.running: return
        if not self._confirm_engine_rebuild("step count"): return
        # Automatically make the new step 10 less than the last one to prevent duplicates
        last_val = self.t_index_list[-1] if self.t_index_list else 40
        new_val = max(2, last_val - 10)
        
        self.t_index_list.append(new_val)
        self._build_steps_ui()
        if self.running: 
            try: self.control_q.put_nowait({"type": "set_t_index_list", "t_index_list": list(self.t_index_list)})
            except Exception: pass

    def _remove_step(self):
        if len(self.t_index_list) > 1:
            if not self._confirm_engine_rebuild("step count"): return
            self.t_index_list.pop()
            self._build_steps_ui()
            if self.running: 
                try: self.control_q.put_nowait({"type": "set_t_index_list", "t_index_list": list(self.t_index_list)})
                except Exception: pass

    def _build_steps_ui(self):
        for child in self._steps_holder.winfo_children(): child.destroy()
        self._step_sliders = []
        for idx, val in enumerate(self.t_index_list):
            row = idx
            ctk.CTkLabel(self._steps_holder, text=f"Step {idx+1}").grid(row=row, column=0, sticky="w", padx=(0,6), pady=4)
            disp = ctk.StringVar(value=str(int(val)))
            ctk.CTkLabel(self._steps_holder, textvariable=disp, width=28).grid(row=row, column=1, sticky="w", padx=(0,6))
            slider = ctk.CTkSlider(self._steps_holder, from_=2, to=49, number_of_steps=48, command=lambda v, i=idx, d=disp: self._on_step_changed(i, v, d))
            slider.set(int(val))
            slider.grid(row=row, column=2, sticky="ew", padx=(0,6))
            self._step_sliders.append(slider)
            self._steps_holder.grid_columnconfigure(2, weight=1)
        self._apply_running_state()

    def _on_step_changed(self, index: int, v, disp_var):
        try:
            ival = max(2, min(49, int(float(v))))
        except Exception:
            return
    
        disp_var.set(str(ival))
    
        # Update just the value that changed — no sorting, no reordering
        if 0 <= index < len(self.t_index_list):
            self.t_index_list[index] = ival

        if self.running and getattr(self, 'control_q', None):
            try:
                self.control_q.put_nowait({
                    "type": "set_t_index_list",
                    "t_index_list": list(self.t_index_list)
                })
            except Exception:
                pass

    def _on_prompt_changed(self, _evt=None):
        if not self.running: return
        if self._debounce_prompt is not None:
            try: self.after_cancel(self._debounce_prompt)
            except Exception: pass
        self._debounce_prompt = self.after(PROMPT_DEBOUNCE_MS, self._push_prompt_runtime)

    def _on_neg_prompt_changed(self, _evt=None):
        if not self.running: return
        if self._debounce_neg is not None:
            try: self.after_cancel(self._debounce_neg)
            except Exception: pass
        self._debounce_neg = self.after(PROMPT_DEBOUNCE_MS, self._push_neg_prompt_runtime)

    def _push_prompt_runtime(self):
        if not getattr(self, 'control_q', None): return
        try:
            txt = self.prompt_txt.get("1.0", "end").strip()
            self.control_q.put_nowait({"type": "set_prompt", "prompt": txt})
        except Exception: pass

    def _push_neg_prompt_runtime(self):
        if not getattr(self, 'control_q', None): return
        try:
            txt = self.neg_prompt_txt.get("1.0", "end").strip()
            self.control_q.put_nowait({"type": "set_negative_prompt", "negative_prompt": txt})
        except Exception: pass

    def _show_plan_note(self, note: str, colour: str):
        """The validator's own words, under the field that caused them.

        The row exists only while there is something to read in it - an empty
        label would leave a hole under the two fields on every plan that went
        through cleanly, which is nearly all of them.
        """
        self.plan_note_var.set(note)
        try:
            self._w_plan_note.configure(text_color=colour)
            if note: self._w_plan_note.grid()
            else: self._w_plan_note.grid_remove()
        except Exception: pass

    def _refresh_plan_state(self):
        """Redraw the plan line from the fields and the newest fps payload.

        Called from the poll loop, so it must not write a variable that has not
        changed - a Tk write per 10 ms poll is a redraw per 10 ms poll.
        """
        line = _plan_state_line(self.target_var.get(), self.running, self._fps_payload)
        if line != self.plan_state_var.get(): self.plan_state_var.set(line)

    def _on_plan_field_changed(self, _evt=None):
        """A target or style edit. Debounced hard - see `PLAN_DEBOUNCE_MS`."""
        # Ahead of the running guard: what the plan *will* be is readable whether
        # or not anything is generating, and it should not wait out the debounce.
        self._refresh_plan_state()
        if not self.running: return
        if self._debounce_plan is not None:
            try: self.after_cancel(self._debounce_plan)
            except Exception: pass
        self._debounce_plan = self.after(PLAN_DEBOUNCE_MS, self._push_plan_runtime)

    def _push_plan_runtime(self):
        """Send the plan the two fields describe, or say why there is not one.

        The outcome always reaches the status area: the validator's notes when the
        plan went through, its stated reason when it did not. A refused plan is
        never put on the queue - the worker would validate it to the same refusal,
        and the user would read it nowhere near the field they typed it into.
        """
        update = _plan_update_from_fields(
            self.target_var.get(), self.style_var.get(),
            self.prompt_txt.get("1.0", "end"), self.neg_prompt_txt.get("1.0", "end"),
            self.detail_var.get(),
        )
        self.status_var.set(update.status)
        note, colour = _plan_note(update)
        self._show_plan_note(note, colour)
        if update.message is not None and getattr(self, "control_q", None):
            try: self.control_q.put_nowait(update.message)
            except Exception: pass
        self._refresh_plan_state()

    def _overlay_screen_rect(self) -> Dict[str, int]:
        if self.capwin is not None: return self.capwin.inner_rect_screen()
        self.update_idletasks()
        x = int(self.preview_panel.winfo_rootx()); y = int(self.preview_panel.winfo_rooty())
        return {"left": x, "top": y, "width": self.preview_dim, "height": self.preview_dim}

    def _send_region_update(self):
        if self.running and getattr(self, "control_q", None):
            try: self.control_q.put_nowait({"type": "set_region", "region": self._overlay_screen_rect()})
            except Exception: pass

    def _on_window_configure(self, _evt=None):
        if self.running and self.capwin is not None:
            try:
                if self._debounce_region is not None: self.after_cancel(self._debounce_region)
            except Exception: pass
            self._debounce_region = self.after(50, self._send_region_update)

    def _append_gpu_log(self, line: str):
        try:
            self.gpu_log.configure(state="normal")
            self.gpu_log.insert("end", line + "\n")
            self.gpu_log.see("end")
        finally:
            self.gpu_log.configure(state="disabled")

    def _gpu_prog_set(self, frac: float):
        try:
            f = max(0.0, min(1.0, float(frac)))
            self.gpu_prog.set(f)
            self.gpu_pct_var.set(f"{int(round(f*100))}%")
            try: self.update_idletasks()
            except Exception: pass
        except Exception: pass

    def _on_enable_gpu(self):
        if getattr(self, "_gpu_install_in_progress", False):
            try: self.gpu_btn.configure(state="disabled")
            except Exception: pass
            return
        try: self.gpu_btn.configure(state="disabled", text="Installing PyTorch...")
        except Exception: pass
        self._gpu_install_in_progress = True
        self._append_gpu_log("Starting PyTorch installation...")
        self._gpu_prog_set(0.05)
        def progress_callback(progress, status=""):
            self._gpu_prog_set(progress)
            if status: self._append_gpu_log(status)
        def worker():
            try:
                def on_progress(line: str): self._append_gpu_log(line)
                self._append_gpu_log("Installing PyTorch with CUDA support...")
                self._gpu_prog_set(0.05)
                success = maybe_bootstrap_gpu(on_progress=on_progress, progress_callback=progress_callback)
                if success:
                    self._append_gpu_log("PyTorch installation completed successfully!")
                    self._append_gpu_log("Restarting application in 3 seconds...")
                    self.after(3000, self._perform_automatic_restart)
                else:
                    self._append_gpu_log("PyTorch installation failed")
                    self._gpu_prog_set(0.0)
                    try: self.gpu_btn.configure(state="normal", text="Install PyTorch - RETRY")
                    except Exception: pass
            except Exception as e:
                self._append_gpu_log(f"Installation failed: {e}")
                self._gpu_prog_set(0.0)
                try: self.gpu_btn.configure(state="normal", text="Install PyTorch - RETRY")
                except Exception: pass
            finally:
                self._gpu_install_in_progress = False
        import threading
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _perform_automatic_restart(self):
        try:
            self._append_gpu_log("Restarting application now...")
            self.update_idletasks()
            import sys
            internal_str = str(INTERNAL_DIR)
            if internal_str not in sys.path: sys.path.insert(0, internal_str)
            import time
            time.sleep(1)
            _gpu_relaunch_into_runtime()
        except Exception as e:
            self._append_gpu_log(f"Automatic restart failed: {e}")
            messagebox.showinfo("Restart Required", "PyTorch installed successfully!\n\nPlease manually restart the application to use GPU features.")

    # ---------------- LoRA management ----------------

    def _lora_choices(self) -> List[str]:
        """What the picker offers: every style LoRA staged under the models root.

        Three files in one known directory, which is what issue #44 replaced a file
        browser with. Browse is still beside it, for a LoRA that lives elsewhere.
        """
        labels = [lora_label(path) for path in local_lora_paths()]
        return [ADD_LORA_PROMPT] + labels if labels else [NO_LOCAL_LORAS]

    def _on_lora_chosen(self, label: str):
        """A LoRA picked from the list of what is on disk (issue #44, step 3).

        The menu snaps back to its prompt: what it did was add a row below, and a
        menu left showing a filename would claim to be the fused set - which it is
        not, since more than one LoRA can be listed.
        """
        self.lora_choice_var.set(self._lora_choices()[0])
        for path in local_lora_paths():
            if lora_label(path) == label:
                self._add_lora_path(path)
                return

    def _add_lora_path(self, path: str):
        """Add one LoRA, unless that file is already fused. The one door.

        Duplicates are matched on `engine_cache.normalize_lora_key`, not on the raw
        string: two spellings of one file key a single engine (issue #44) but would
        be two `lora_dict` entries, and the wrapper fuses every entry - so the same
        weights would go in twice at the same scale.
        """
        if self.running: return
        key = engine_cache.normalize_lora_key(path)
        if any(engine_cache.normalize_lora_key(item["path"]) == key
               for item in self.lora_items):
            return
        self.lora_items.append({"path": path, "scale": DEFAULT_LORA_SCALE})
        self._build_loras_ui()
        self._refresh_engine_state()

    def _add_lora(self):
        """Browse - kept, because a LoRA outside the models root is a legal answer."""
        if self.running: return
        paths = filedialog.askopenfilenames(
            title="Select LoRA file(s)",
            initialdir=str(resolve_models_dir() / LORAS_SUBDIR),
            filetypes=[("LoRA weights", " ".join(f"*{s}" for s in LORA_SUFFIXES)),
                       ("All files", "*.*")],
        )
        for path in paths or ():
            self._add_lora_path(path)

    def _remove_lora(self, index: int):
        if self.running: return
        if 0 <= index < len(self.lora_items):
            self.lora_items.pop(index)
            self._build_loras_ui()
            self._refresh_engine_state()

    def _on_lora_scale_changed(self, index: int, value, disp_var):
        try:
            scale = round(float(value), 2)
        except Exception:
            return
        if 0 <= index < len(self.lora_items):
            self.lora_items[index]["scale"] = scale
        disp_var.set(f"{scale:.2f}")
        # The scale keys the engine as much as the file does, so the standing
        # sentence has to follow the slider (issue #44's fourth trap).
        self._refresh_engine_state()

    def _build_loras_ui(self):
        # Rows are rebuilt wholesale, so keep their widgets out of self._lockables
        # (which never prunes) and track them here instead, like _step_sliders.
        for child in self._loras_holder.winfo_children(): child.destroy()
        self._lora_widgets = []
        if not self.lora_items:
            ctk.CTkLabel(self._loras_holder, text="No LoRAs loaded (optional)",
                         text_color="gray60", anchor="w").grid(row=0, column=0, sticky="ew", pady=2)
            self._apply_running_state()
            return
        for idx, item in enumerate(self.lora_items):
            r = ctk.CTkFrame(self._loras_holder)
            r.grid(row=idx, column=0, sticky="ew", pady=2)
            r.grid_columnconfigure(0, weight=1)
            name = os.path.basename(item["path"])
            if len(name) > 34: name = name[:31] + "..."
            ctk.CTkLabel(r, text=name, anchor="w").grid(row=0, column=0, sticky="ew", padx=(6, 4), pady=(4, 0))
            btn_rm = ctk.CTkButton(r, text="✕", width=28, fg_color="#B91C1C", hover_color="#991B1B",
                                   command=lambda i=idx: self._remove_lora(i))
            btn_rm.grid(row=0, column=1, rowspan=2, padx=(4, 6))
            disp = ctk.StringVar(value=f"{item['scale']:.2f}")
            ctk.CTkLabel(r, textvariable=disp, width=40).grid(row=1, column=0, sticky="e", padx=(0, 4))
            sl = ctk.CTkSlider(r, from_=0.0, to=2.0, number_of_steps=200,
                               command=lambda v, i=idx, d=disp: self._on_lora_scale_changed(i, v, d))
            sl.set(item["scale"])
            sl.grid(row=2, column=0, sticky="ew", padx=(6, 4), pady=(0, 4))
            self._lora_widgets += [btn_rm, sl]
        self._set_state(self._lora_widgets, "disabled" if self.running else "normal")
        self._apply_running_state()

    def _lora_dict(self) -> Optional[Dict[str, float]]:
        """StreamDiffusion expects {path: scale}, or None when no LoRAs are used."""
        if not self.lora_items: return None
        return {item["path"]: float(item["scale"]) for item in self.lora_items}

    def _browse_model(self):
        d = filedialog.askdirectory(title="Select diffusers model folder",
                                    initialdir=str(resolve_models_dir()))
        if d: self._apply_model(d)

    def _model_choices(self) -> List[str]:
        """What the picker offers: every loadable model under the models root.

        Whatever is currently set comes first even when it is outside that root -
        a Browse-d path is a legal answer and the list must not silently drop it.
        """
        labels = [model_label(path) for path in local_model_paths()]
        current = model_label(self.model_var.get())
        if current and current not in labels:
            labels.insert(0, current)
        return labels or [""]

    def _on_model_chosen(self, label: str):
        """A model picked from the list of what is on disk (issue #38, step 6)."""
        for path in local_model_paths():
            if model_label(path) == label:
                self._apply_model(path)
                return

    def _apply_model(self, path: str):
        """Set the model, and with it the two settings it cannot render without.

        SD-Turbo is distilled to one step and fuses no LCM-LoRA; SD 1.5 is not and
        does. Choosing the model and leaving those behind is how a user gets noise
        with nothing in the window saying why, so `model_companions` moves them
        together - and the step *count* keys a TensorRT engine, which is what the
        check at Start is for.
        """
        self.model_var.set(path)
        self.model_choice_var.set(model_label(path))
        companions = model_companions(path)
        self.use_lcm_lora_var.set(companions.use_lcm_lora)
        if list(self.t_index_list) != list(companions.t_index_list):
            self.t_index_list = list(companions.t_index_list)
            self._build_steps_ui()
        self._refresh_engine_state()

    def _refresh_engine_state(self):
        """Draw that sentence for whatever the window currently adds up to.

        Called by every control that keys an engine - the model, and since issue
        #44 the LoRA set and each LoRA's scale - so what it says is the question
        Start is about to ask on disk.
        """
        configuration = self._engine_configuration()
        self.engine_state_var.set(
            "" if configuration is None else _engine_state_line(configuration))

    def _engine_configuration(self) -> Optional[EngineConfiguration]:
        """What the settings in the window add up to, or None on a path that builds
        nothing at all."""
        if not engine_rebuild_needed(self.accel_var.get()):
            return None
        try:
            buffer_size = int(self.buffer_var.get())
        except (TypeError, ValueError):
            buffer_size = DEFAULT_FRAME_BUFFER_SIZE
        return engine_configuration(
            model_path=self.model_var.get(), acceleration=self.accel_var.get(),
            use_lcm_lora=bool(self.use_lcm_lora_var.get()),
            steps=len(self.t_index_list), frame_buffer_size=buffer_size,
            lora_dict=self._lora_dict() or None)

    def _confirm_engine_available(self) -> bool:
        """Say what Start is about to spend, and refuse a volume that cannot hold it.

        The last moment before minutes are spent inside another process, and the
        only check here that looks on disk rather than guessing from a setting.
        """
        configuration = self._engine_configuration()
        if configuration is None or configuration.cached:
            return True
        if not configuration.enough_disk:
            messagebox.showerror(
                "Not enough disk space",
                _not_enough_disk_message(configuration.engines_root,
                                         configuration.free_bytes))
            return False
        return bool(messagebox.askokcancel("TensorRT engine build required",
                                           _engine_missing_warning(configuration)))

    def _on_start(self):
        if self.running: return
        if not self._validate_numeric_parameters(): return
        try: verify_local_model_path_dir(self.model_var.get())
        except Exception as e: messagebox.showerror("Paths error", str(e)); return
        lora_dict = self._lora_dict()
        if lora_dict:
            missing = [p for p in lora_dict if not os.path.isfile(p)]
            if missing:
                messagebox.showerror("LoRA error",
                    "These LoRA files no longer exist:\n\n" + "\n".join(missing))
                return
            # The build itself is not asked about here: LoRA weights are fused
            # into the UNet before the engine is compiled, so the fused set is
            # part of the directory the check below looks for.
        # Whether *this* configuration has an engine, looked up on disk rather than
        # guessed from whether a setting is at its default (issue #38, step 6). It
        # covers the batch size, the step count, the model and the fused LoRA set
        # in one reading, because those are exactly what key the directory - and it
        # is the last moment before minutes are spent inside another process.
        if not self._confirm_engine_available(): return
        ctx = get_context("spawn")
        self.proc_ctx = ctx
        self.out_q = ctx.Queue(maxsize=2); self.fps_q = ctx.Queue(); self.status_q = ctx.Queue()
        self.control_q = ctx.Queue(); self.debug_q = ctx.Queue(); self.close_q = ctx.Queue()
        self.monitor_sender, self.monitor_receiver = ctx.Pipe()
        controlnet_paths: List[str] = []; controlnet_scales: List[float] = []
        # One reading of what was picked, for the worker, the capture thread and
        # the capture window alike - three sizes that have to agree or the mask
        # lands somewhere other than the object.
        capture_w, capture_h = capture_size(self.capture_var.get())
        self.proc_worker = ctx.Process(
            target=image_generation_process,
            args=(
                self.out_q, self.fps_q, self.close_q, self.status_q, self.control_q, self.debug_q, 
                list(map(int, self.t_index_list)), self.model_var.get(), controlnet_paths, controlnet_scales, 
                lora_dict, bool(self.use_lcm_lora_var.get()),
                (LOCAL_LCM_LORA if os.path.isfile(LOCAL_LCM_LORA) else None),
                self.prompt_txt.get("1.0", "end").strip(), self.neg_prompt_txt.get("1.0", "end").strip(), 
                int(self.buffer_var.get()), capture_w, capture_h, 
                self.accel_var.get(), True, int(self.seed_var.get()), 
                "none", 0.0, 0.5, True, False, 0.99, 10.0,  # <-- Changed the first 'False' to 'True' here!
                self.monitor_receiver, True, str(resolve_engines_dir()),
                DIFFUSION_CANVAS, DIFFUSION_CANVAS,
            )
        )
        self.proc_worker.start()
        self.capwin = FloatingCaptureWindow(self, inner_w=capture_w, inner_h=capture_h, border_px=8, handle_h=28)
        try: self.monitor_sender.send(self._overlay_screen_rect())
        except Exception: pass
        self.running = True
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.hide_capture_btn.configure(state="normal")
        self._apply_running_state()
        # A target typed before Start is applied as soon as the worker drains its
        # queue. A blank one sends nothing: the worker already starts on today's
        # behaviour as a plan, and SD_DEMO_PLAN's priority case has to survive a
        # headless run rather than be overwritten by an empty field.
        if self.target_var.get().strip():
            self._push_plan_runtime()
        self._refresh_plan_state()

    def _on_stop(self):
        if not self.running: return
        try:
            if self.close_q is not None:
                try: self.close_q.put_nowait(True)
                except Exception: pass
            if self.proc_worker is not None:
                self.proc_worker.join(timeout=5)
                if self.proc_worker.is_alive(): self.proc_worker.terminate()
            self.proc_worker = None
        finally:
            try:
                if self.capwin is not None:
                    try:
                        if self.capwin.win.state() != "normal": self.capwin.win.deiconify()
                    except: pass
                    self.capwin.destroy()
            finally:
                self.capwin = None
            self.running = False
            self._fps_payload = None
            self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
            self._apply_running_state()
            self._refresh_plan_state()

    def _on_capture_window_moved(self):
        if self.running: self._send_region_update()

    def _poll_queues(self):
        if self.out_q is not None:
            try:
                while True:
                    pil = self.out_q.get_nowait()
                    if isinstance(pil, Image.Image): self._update_preview(pil)
            except Exception: pass
        if self.fps_q is not None:
            try:
                while True:
                    self._fps_payload = self.fps_q.get_nowait()
                    self.fps_var.set(_format_fps(self._fps_payload))
            except Exception: pass
        self._refresh_plan_state()
        if self.status_q is not None:
            try:
                while True:
                    msg = self.status_q.get_nowait()
                    self.status_var.set(msg)
            except Exception: pass
            
        poll_delay = 10 if self.accel_var.get() == "tensorrt" else 33
        self.after(poll_delay, self._poll_queues)

    def _update_preview(self, pil_img: Image):
        dim = self.preview_dim
        canvas = Image.new("RGB", (dim, dim), (30, 30, 30))
        img = pil_img.copy()
        img.thumbnail((dim, dim), PIL.Image.BICUBIC)
        x = (dim - img.width) // 2
        y = (dim - img.height) // 2
        canvas.paste(img, (x, y))
        self._ctk_img = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(dim, dim))
        self.preview_panel.configure(image=self._ctk_img)

    def do_quit(self):
        if not messagebox.askyesno("Quit", "Are you sure you want to quit and stop all workers?"): return
        if self.running: self._on_stop()
        self.destroy()

def preload_critical_components():
    if not getattr(sys, "frozen", False):
        _prime_dll_search(SITE_PACKAGES, INTERNAL_DIR)
        PreloadedDependencies.preload_all()

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.freeze_support()
        preload_critical_components()
        app = StreamGUI()
        
        def set_window_icon_backup():
            try:
                icon_paths = [resource_path("icon2.ico")]
                for icon_path in icon_paths:
                    if os.path.exists(icon_path):
                        if icon_path.lower().endswith('.ico'):
                            app.iconbitmap(icon_path)
                            break
                        else:
                            try:
                                img = Image.open(icon_path)
                                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                                temp_ico = os.path.join(tempfile.gettempdir(), "temp_app_icon.ico")
                                img.save(temp_ico, format='ICO')
                                app.iconbitmap(temp_ico)
                                break
                            except Exception: continue
            except Exception: pass
        
        app.after(500, set_window_icon_backup)
        app.mainloop()
        
    except Exception:
        pass