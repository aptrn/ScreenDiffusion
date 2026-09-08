import functools
import gc
import os
import sys
from pathlib import Path
import traceback
from typing import List, Literal, Mapping, Optional, Union, Dict

# `engine_cache` is stdlib and sits beside this file. It holds the one spelling of
# the engine-directory rule `create_prefix` below renders, shared with the window
# and the bench guard - so it is imported rather than copied (issue #44). This
# module is loaded by path (`spec_from_file_location`), which does not put its own
# directory on the import path, hence the insert.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Aliased because `create_prefix`'s enclosing scope binds a local of the same name.
from engine_cache import lora_fingerprint as engine_lora_fingerprint

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Add this flag


REPO_ROOT = Path(__file__).resolve().parent
SD_ENGINES_DIR_ENV = "SD_ENGINES_DIR"

# A path argument: a string, a Path, or nothing given.
_PathArg = Optional[Union[str, Path]]


def _unquoted_path(value: _PathArg) -> str:
    """`value` as a bare path string, empty when there is nothing usable.

    A path pasted into a Windows env var often keeps its surrounding quotes.
    """
    return "" if value is None else str(value).strip().strip('"').strip()


def _resolve_engine_dir(engine_dir: _PathArg = None, base_dir: _PathArg = None,
                        environ: Optional[Mapping[str, str]] = None) -> Path:
    """Absolute engines root: `engine_dir`, else $SD_ENGINES_DIR, else `<repo>/engines`.

    Compiled engines are ~5.1 GB each and gitignored, so a relative path - which
    re-anchors to the process cwd - makes a worktree rebuild caches it already has.
    Relative values are anchored to the repo root instead. See CLAUDE.md.

    Mirrors `_resolve_cache_dir()` in main_gpu_addon.py: the GUI process must not
    import this module, so the rule is spelled out in both places.
    """
    environ = os.environ if environ is None else environ
    base = Path(REPO_ROOT if base_dir is None else base_dir)
    raw = _unquoted_path(engine_dir) or _unquoted_path(environ.get(SD_ENGINES_DIR_ENV))
    candidate = Path(raw).expanduser() if raw else base / "engines"
    if not candidate.is_absolute():
        candidate = base / candidate
    # normpath, not resolve(): collapse `..` and settle on one slash direction
    # without touching the filesystem or following symlinks.
    return Path(os.path.normpath(candidate))


def _broadcast_list(vals: Optional[List[float]], n: int, default: float = 1.0) -> List[float]:
    if n <= 0:
        return []
    if not vals:
        return [default] * n
    if len(vals) >= n:
        return vals[:n]
    return [vals[0]] * n


# --- the unbatched denoising route (issue #46) --------------------------------
#
# `use_denoising_batch` decides whether the denoising steps go through the UNet as
# one batch of N or as N calls of one. It matters far past the milliseconds,
# because with it *off* the step count stops keying the engine: `trt_unet_batch_size`
# is `frame_buffer_size` whatever the count, so every rung of a quality ladder runs
# on the one engine that is already built. That is the second of the two ways issue
# #46 can pay for a runtime step count, and it could not be measured, because
# `__init__` refused it outright for img2img.
#
# The refusal was right about upstream's implementation and wrong about the idea.
# `StreamDiffusion.predict_x0_batch`'s unbatched branch does `self.init_noise =
# x_t_latent` - it overwrites the prepared noise field with the *current frame's*
# noised latent, and `encode_image` reads `init_noise[0]` on the next call. That is
# harmless for txt2img, which the branch was written for and which never encodes an
# image; for img2img it means every frame after the first is noised with the last
# frame's latent. It also re-draws `torch.randn_like` at every intermediate rung,
# which is a fresh noise field per step per frame - the opposite of what this app's
# temporal-stability story rests on (spec 8.5: the shipped field is drawn once and
# pinned to the canvas).
#
# So the route is implemented here rather than declared impossible, as the batched
# path's own analogue: the prepared field is left alone and read at every rung, the
# way the batched path reads `init_noise[i]` at rung i.


def unbatched_predict_x0(stream, x_t_latent: torch.Tensor) -> torch.Tensor:
    """Denoise one frame through `stream`'s whole ladder, one UNet call per rung.

    The batched path pipelines: it puts every rung in one batch, so a frame's later
    rungs are spent on *earlier* frames' latents and the answer comes out N-1 frames
    late. This one spends every rung on the frame it was given, which is why it is
    worth measuring against and not only worth costing.
    """
    rungs = len(stream.sub_timesteps_tensor)
    x_0_pred = x_t_latent
    for idx in range(rungs):
        t_list = stream.sub_timesteps_tensor[idx].view(1).repeat(stream.frame_bff_size)
        x_0_pred, _ = stream.unet_step(x_t_latent, t_list, idx)
        if idx + 1 < rungs:
            x_t_latent = stream.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            if stream.do_add_noise:
                # `init_noise[0:1]`, not a fresh `randn_like`: one field, drawn once
                # in `prepare` and pinned to the canvas, is what `seeding.py` writes
                # into and what every flicker figure in this repo was measured on.
                x_t_latent = (x_t_latent + stream.beta_prod_t_sqrt[idx + 1]
                              * stream.init_noise[0:1])
    return x_0_pred


def enable_unbatched_img2img(stream) -> None:
    """Bind `unbatched_predict_x0` over the pipeline's own, on this instance only.

    An instance attribute rather than a subclass or a patched module: the pipeline
    is constructed inside `_load_model` and shared with the accelerated paths, and
    the shipped configuration must reach exactly the code it reached before.
    """
    stream.predict_x0_batch = functools.partial(unbatched_predict_x0, stream)


def rebuild_tensors_for_tlist(self, new_t_list: List[int]):
    """
    Rebuild tensors for new t_list (compatibility with original script)
    """
    if hasattr(self, 'stream'):
        # Call the internal method that handles t_list updates
        self._update_t_list_attributes(self.stream, new_t_list)
    else:
        # Fallback: update the t_list and hope for the best
        self.t_list = new_t_list

class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        self.sd_turbo = "turbo" in model_id_or_path

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        # img2img unbatched used to raise here. It is supported now, by
        # `enable_unbatched_img2img` below - see the note beside it for what
        # upstream's own branch does to `init_noise` and why this one does not.
        self.use_unbatched_img2img = mode == "img2img" and not use_denoising_batch

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        # ===== LIVE STEPS =====
        # Keep a mutable, live copy of steps that can be changed at runtime.
        self.t_index_list = list(t_index_list)

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if self.use_unbatched_img2img:
            enable_unbatched_img2img(self.stream)

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    # ===== LIVE STEPS API =====
    @torch.no_grad()
    def update_t_index_at(self, index: int, value: int) -> None:
        """Update a single step at runtime without full rebuild"""
        if not (0 <= index < len(self.t_index_list)):
            return
    
        new_value = max(1, min(49, int(value)))
        if self.t_index_list[index] == new_value:
            return
        
        self.t_index_list[index] = new_value
    
        # Directly update the underlying stream's t_list
        if hasattr(self.stream, 't_list'):
            self.stream.t_list[index] = new_value
        
        # Force tensor rebuild by calling the full update
        self.set_t_index_list(self.t_index_list)

    def set_t_index_list(self, new_steps: List[int]) -> None:
        """
        Update the entire t_index_list at runtime. If the backend exposes
        a native setter, use it; otherwise patch step-dependent caches.
        """
        new_steps = [int(max(1, min(49, int(v)))) for v in list(new_steps)]
        
        # Stop redundant rebuilds from slider spam!
        if hasattr(self, 't_index_list') and self.t_index_list == new_steps:
            return 
            
        self.t_index_list = list(new_steps)
    
        # Try native methods first
        for name in ("set_t_index_list", "set_t_indexes", "set_timesteps", "set_timestep_indexes"):
            if hasattr(self.stream, name):
                try:
                    getattr(self.stream, name)(self.t_index_list)
                    print(f"[Wrapper] Used native {name} to update t_list: {self.t_index_list}")
                    return
                except Exception as e:
                    print(f"[Wrapper] Native {name} failed: {e}")
    
        # Fallback: rebuild step caches manually
        print(f"[Wrapper] Using fallback _rebuild_step_caches for t_list: {self.t_index_list}")
        self._rebuild_timestep_tensors()

    def _rebuild_timestep_tensors(self):
        """
        Best-effort refresh of any cached tensors that depend on t_index_list.
        Safe no-op if attributes are missing in this build.
        """
        s = self.stream
    
        try:
            if s is None or not (hasattr(s, "scheduler") and hasattr(s, "timesteps")):
                print("[Wrapper] Cannot rebuild - missing scheduler or timesteps")
                return
        
            # Map indices to scheduler timesteps
            sub = [s.timesteps[int(t)] for t in self.t_index_list]
            print(f"[Wrapper] Mapped t_list {self.t_index_list} to timesteps: {sub}")
        
            repeats = getattr(s, "frame_buffer_size", 1)
            s.sub_timesteps = sub
            s.sub_timesteps_tensor = torch.repeat_interleave(
                torch.tensor(sub, dtype=torch.long, device=s.device),
                repeats=repeats, 
                dim=0
            )
            print(f"[Wrapper] Updated sub_timesteps_tensor: {s.sub_timesteps_tensor}")
        
            # Update c_skip / c_out (required for boundary conditions)
            try:
                c_skip_list, c_out_list = [], []
                for t in sub:
                    c_skip, c_out = s.scheduler.get_scalings_for_boundary_condition_discrete(t)
                    c_skip_list.append(c_skip)
                    c_out_list.append(c_out)
            
                # Fix: correct .to() syntax
                s.c_skip = torch.stack(c_skip_list).view(len(sub), 1, 1, 1).to(
                    device=s.device, dtype=s.dtype
                )
                s.c_out = torch.stack(c_out_list).view(len(sub), 1, 1, 1).to(
                    device=s.device, dtype=s.dtype
                )
                print(f"[Wrapper] Updated c_skip and c_out")
            except Exception as e:
                print(f"[Wrapper] Could not update c_skip/c_out: {e}")
        
            # Update alpha/beta terms (CRITICAL for proper denoising)
            try:
                a_sqrt, b_sqrt = [], []
                for t in sub:
                    a_sqrt.append(s.scheduler.alphas_cumprod[t].sqrt())
                    b_sqrt.append((1 - s.scheduler.alphas_cumprod[t]).sqrt())
            
                a_sqrt_tensor = torch.stack(a_sqrt).view(len(sub), 1, 1, 1).to(
                    device=s.device, dtype=s.dtype
                )
                b_sqrt_tensor = torch.stack(b_sqrt).view(len(sub), 1, 1, 1).to(
                    device=s.device, dtype=s.dtype
                )
            
                s.alpha_prod_t_sqrt = torch.repeat_interleave(a_sqrt_tensor, repeats=repeats, dim=0)
                s.beta_prod_t_sqrt = torch.repeat_interleave(b_sqrt_tensor, repeats=repeats, dim=0)
                print(f"[Wrapper] Updated alpha_prod_t_sqrt and beta_prod_t_sqrt")
            except Exception as e:
                print(f"[Wrapper] Could not update alpha/beta: {e}")
        
            print(f"[Wrapper] ✅ Step caches rebuilt successfully")
        
        except Exception as e:
            print(f"[Wrapper] ❌ Error in _rebuild_step_caches: {e}")
            import traceback
            traceback.print_exc()

    def set_t_index_at(self, i: int, value: int) -> None:
        """Update a single step (clamped to 1..49)."""
        if not (0 <= i < len(self.t_index_list)):
            return
        v = max(1, min(49, int(value)))
        self.t_index_list[i] = v
        self.set_t_index_list(self.t_index_list)

    # Optional: allow worker to inject a deterministic generator
    def set_generator(self, gen: torch.Generator):
        if hasattr(self.stream, "generator"):
            try:
                self.stream.generator = gen
            except Exception:
                pass

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None,
        output_type: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.
        output_type : Optional[str]
            This call's output type, overriding the wrapper's own. The selective
            render path asks for "pt" per call (issue #31): a masked frame is
            blended on the device and comes home once, afterwards, while every
            other frame still wants the PIL image the caller displays.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(
            image_tensor, output_type=output_type or self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        # `pt` is the device path (issue #31): the caller blends the frame where it
        # already is and pays one host copy after the mask rather than before it, so
        # moving the tensor here would be the round trip that change exists to
        # remove. Every other output type is a host object and has to come home.
        tensor = image_tensor if output_type == "pt" else image_tensor.cpu()
        if self.frame_buffer_size > 1:
            return postprocess_image(tensor, output_type=output_type)
        else:
            return postprocess_image(tensor, output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = None,
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """

        try:  # Load from local directory
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                model_id_or_path,
                variant="fp16",
                use_safetensors=True,
            ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        # LCM-LoRA is a scheduler trick that sd-turbo does not need, so it stays
        # gated on sd_turbo. User-supplied LoRAs are a different thing entirely and
        # must load for every model - keeping them inside this guard made the UI's
        # LoRA list silently do nothing on sd-turbo.
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

        if lora_dict is not None:
            for lora_name, lora_scale in lora_dict.items():
                try:
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                except Exception as e:
                    # Surface the reason instead of continuing with a model that
                    # silently lacks the LoRA the user asked for.
                    raise RuntimeError(
                        f"Failed to load LoRA '{os.path.basename(str(lora_name))}': {e}\n\n"
                        f"The LoRA must match the base model's architecture "
                        f"(sd-turbo is SD 2.1-based), and LoRAs containing "
                        f"convolution layers (LoCon/LyCORIS) are not supported by "
                        f"diffusers {__import__('diffusers').__version__}."
                    ) from e
                print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration != "tensorrt":
                # NHWC lets cuDNN/tensor cores pick faster fp16 conv kernels for the
                # conv-heavy UNet. Measured ~+11% here. TensorRT builds its own graph,
                # so the torch-side memory format is irrelevant on that path.
                try:
                    stream.unet.to(memory_format=torch.channels_last)
                    print("[INFO] UNet set to channels_last (NHWC) memory format.")
                except Exception as e:
                    print(f"[WARN] channels_last not applied: {e}")
            if acceleration == "xformers":
                print("[INFO] Bypassing broken xformers... Using PyTorch Native Flash Attention (SDPA) instead!")
                # Modern diffusers natively default to SDPA on PyTorch 2.0+, so no extra code is needed here.
            if acceleration == "tensorrt":
                from polygraphy import cuda
                # ---- Polygraphy compatibility shim for StreamDiffusion TensorRT engines ----
                # Some Polygraphy versions (0.48+ / 0.49+ / newer NVIDIA wheels)
                # do not expose trt_util.get_bindings_per_profile(), but StreamDiffusion's
                # TensorRT Engine classes still call it during inference.
                from polygraphy.backend.trt import util as trt_util
                if not hasattr(trt_util, "get_bindings_per_profile"):
                    def _sd_get_bindings_per_profile(engine):
                        profiles = int(getattr(engine, "num_optimization_profiles", 1) or 1)

                        # TensorRT 8.x binding API
                        if hasattr(engine, "num_bindings"):
                            return int(engine.num_bindings) // profiles

                        # TensorRT 10.x named I/O tensor API fallback.
                        # This only fixes the missing Polygraphy helper. If the installed
                        # TensorRT build has removed other binding APIs used by StreamDiffusion,
                        # pin TensorRT to an 8.x/9.x build or port StreamDiffusion's engine.py.
                        if hasattr(engine, "num_io_tensors"):
                            return int(engine.num_io_tensors) // profiles

                        raise AttributeError(
                            "Could not determine TensorRT bindings per profile: "
                            "engine has neither num_bindings nor num_io_tensors"
                        )

                    trt_util.get_bindings_per_profile = _sd_get_bindings_per_profile
                    print("[TRT Compat] Patched polygraphy.backend.trt.util.get_bindings_per_profile")
                # ---- TensorRT 10.x API Compatibility Shim ----
                import tensorrt as trt
                
                # 1. Patch ICudaEngine
                if not hasattr(trt.ICudaEngine, "get_binding_dtype"):
                    print("[TRT Compat] Patching TensorRT 10.x ICudaEngine API...")
                    
                    def _get_name(self, idx):
                        return idx if isinstance(idx, str) else self.get_tensor_name(idx)
                        
                    def _get_shape(self, idx):
                        name = idx if isinstance(idx, str) else self.get_tensor_name(idx)
                        return self.get_tensor_shape(name)
                        
                    def _get_dtype(self, idx):
                        name = idx if isinstance(idx, str) else self.get_tensor_name(idx)
                        return self.get_tensor_dtype(name)
                        
                    def _is_input(self, idx):
                        name = idx if isinstance(idx, str) else self.get_tensor_name(idx)
                        return self.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    
                    trt.ICudaEngine.num_bindings = property(lambda self: getattr(self, "num_io_tensors", 0))
                    trt.ICudaEngine.get_binding_name = _get_name
                    trt.ICudaEngine.get_binding_shape = _get_shape
                    trt.ICudaEngine.get_binding_dtype = _get_dtype
                    trt.ICudaEngine.binding_is_input = _is_input
                    
                    # Handle index brackets and legacy attributes
                    trt.ICudaEngine.has_implicit_batch_dimension = False
                    trt.ICudaEngine.__getitem__ = lambda self, idx: idx if isinstance(idx, str) else self.get_tensor_name(idx)
                    trt.ICudaEngine.__len__ = lambda self: getattr(self, "num_io_tensors", 0)

                # 2. Patch IExecutionContext
                if not hasattr(trt.IExecutionContext, "set_binding_shape"):
                    print("[TRT Compat] Patching TensorRT 10.x IExecutionContext API...")
                    
                    def _set_shape(self, idx, shape):
                        name = idx if isinstance(idx, str) else self.engine.get_tensor_name(idx)
                        return self.set_input_shape(name, shape)
                        
                    def _get_ctx_shape(self, idx):
                        name = idx if isinstance(idx, str) else self.engine.get_tensor_name(idx)
                        return self.get_tensor_shape(name)
                    
                    # Translate Legacy V2 compute calls into Modern V3 compute calls
                    def _execute_async_v2(self, bindings, stream_handle):
                        for i, ptr in enumerate(bindings):
                            if ptr: # Safely map memory addresses natively
                                name = self.engine.get_tensor_name(i)
                                self.set_tensor_address(name, ptr)
                        return self.execute_async_v3(stream_handle)
                        
                    trt.IExecutionContext.set_binding_shape = _set_shape
                    trt.IExecutionContext.get_binding_shape = _get_ctx_shape
                    
                    if not hasattr(trt.IExecutionContext, "execute_async_v2"):
                        trt.IExecutionContext.execute_async_v2 = _execute_async_v2
                # ----------------------------------------------
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                # A built engine is only valid for the exact resolution and the
                # exact fused-LoRA weights it was compiled against. Both must be
                # part of the cache key, otherwise a stale engine gets loaded and
                # fed mismatched tensors -> CUDA illegal memory access.
                #
                # The fingerprint comes from `engine_cache`, which is what the
                # window and the bench guard ask too. A second copy here was the
                # two-rules bug that module exists to prevent - and it was the copy
                # that hashed the raw path string, so two spellings of one file were
                # two engines and neither side could find the other's (issue #44).
                lora_fingerprint = engine_lora_fingerprint(lora_dict)

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    base = maybe_path.stem if maybe_path.exists() else model_id_or_path
                    return (
                        f"{base}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}"
                        f"--max_batch-{max_batch_size}--min_batch-{min_batch_size}"
                        f"--res-{self.width}x{self.height}--lora-{lora_fingerprint}"
                        f"--mode-{self.mode}"
                    )

                engine_dir = _resolve_engine_dir(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_stream, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0: # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream