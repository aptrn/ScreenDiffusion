Screen Diffusion V0.2
A real-time AI transformation tool.

![ScreenDiffusion demo](media/SD-1.gif)
![ScreenDiffusion demo](media/SD-3.gif)

What Is Screen Diffusion?

Screen Diffusion is a live image-to-image AI renderer built around StreamDiffusion that transforms your computer screen into living art — in real time.
Whatever you display — a game, a 3D scene, a photo, a design , a webcam— can be instantly reimagined through SD-Turbo model to reveal new styles, moods, or worlds.

This project is completely free for everyone to use and explore.

🛠️ Prerequisites:

  - Windows 11 / Windows 10
  - NVIDIA GPU with 8GB+ VRAM (RTX 3080 or higher recommended)
  - Supports RTX 30, 40, and 50 series
  - TensorRT support
  - **No manual Python install needed.** Setup uses [uv](https://docs.astral.sh/uv/),
    which downloads its own Python 3.11 and builds a fully isolated environment
    inside the project folder. Your system Python is never touched.
  - Use Screen Diffusion to download the Sd-Turbo model
     Or
     Download SD-Turbo Model and store it locally - (download link) https://huggingface.co/stabilityai/sd-turbo/tree/main

📦 Installation

  - Clone the repo or Download as ZIP file.
  - Run **setup.bat**. It installs uv if you don't have it, fetches Python 3.11,
    and creates the isolated environment in `.venv`. The first run downloads
    several GB of CUDA/PyTorch wheels, so give it time. If a download is
    interrupted, just re-run setup.bat - it resumes.
  - Run **run.bat** to open the app.
  - Download the SD-Turbo model if you haven't already.
  - Click Start Generation.
  - Choose your capture region, adjust prompts, and enjoy the transformation.
  - Click Hide or press H on your keyboard to draw inside the capture region.

🎨 LoRAs

  Use the **LoRAs** panel (under the model path) to stack one or more LoRA files.
  Click **+ Add LoRA**, pick `.safetensors` / `.bin` / `.pt` files, and set each
  one's strength with its slider (0.00-2.00, default 1.00). The red button removes
  an entry. LoRAs can only be changed while generation is stopped.

  Two things to know:

  - The LoRA must match the base model's architecture. SD-Turbo is **SD 2.1**-based,
    so SD 1.5 and SDXL LoRAs will not load. LoRAs containing convolution layers
    (LoCon / LyCORIS) are also unsupported by this diffusers version. In both cases
    you get an explicit error rather than a silently ignored LoRA.
  - With **TensorRT**, LoRA weights are fused into the compiled engine, so each
    distinct LoRA + strength combination needs its own engine build (several
    minutes, cached afterwards). The app warns you before starting such a build.
    On the `none` accelerator LoRAs apply immediately with no build step.

🔁 Reproducibility & isolation

  Dependencies are declared in `pyproject.toml` and pinned exactly in `uv.lock`,
  so every machine gets an identical environment. Nothing is installed globally
  and no conda is involved - the whole environment is the `.venv` folder, and
  deleting it fully uninstalls the app's dependencies.

  Useful commands (run from the project folder):

  ```
  uv sync                  # create / repair the environment from the lockfile
  uv run python main_gpu_addon.py   # launch without run.bat
  uv lock --upgrade        # deliberately refresh the pinned versions
  ```

Pull requests are welcome!  

Please open an issue first to discuss major changes.

