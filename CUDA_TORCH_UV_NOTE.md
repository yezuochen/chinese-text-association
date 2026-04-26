# PyTorch CUDA + uv on Windows — Installation Note

## The Problem

`uv sync` resolves `torch` from PyPI by default, which gives you the **CPU-only** wheel — even when you configure a CUDA index URL in `pyproject.toml`.

Specifically:
- `torch>=2.5.0,<2.11` resolves to `2.10.0` (CPU-only on Windows)
- Adding `[[tool.uv.index]] url = "https://download.pytorch.org/whl/cu121"` in `pyproject.toml` does **not** help — uv's lock mechanism still records only the PyPI CPU wheel
- `uv pip install` of a CUDA wheel is immediately overwritten by the next `uv sync`

This is a known uv behavior: the lock file controls what's installed, not the index URL. Override-dependencies affect resolution, not the installed wheel.

---

## The Fix — Run After Every `uv sync`

### Option 1: Use the helper script (recommended)

```bash
# After any uv sync:
bash setup_cuda_torch.sh
```

First run downloads the ~2.3 GB wheel and caches it at:
```
%LOCALAPPDATA%\torch-cu121\torch-2.5.1+cu121-cp311-cp311-win_amd64.whl
```
Subsequent runs use the cached file and are fast.

### Option 2: Manual one-liner

```bash
python -c "
import zipfile, os, shutil
wheel = os.path.join(os.environ['LOCALAPPDATA'], 'torch-cu121',
                     'torch-2.5.1+cu121-cp311-cp311-win_amd64.whl')
site = '.venv/Lib/site-packages'
shutil.rmtree(os.path.join(site, 'torch'), ignore_errors=True)
for d in os.listdir(site):
    if d.startswith('torch-'):
        shutil.rmtree(os.path.join(site, d), ignore_errors=True)
with zipfile.ZipFile(wheel) as z:
    z.extractall(site)
"
```

---

## Verify

```bash
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output:
```
2.5.1+cu121 True
```

To test GPU compute:
```bash
.venv\Scripts\python.exe -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
print('GPU OK:', (x @ y).shape, '|', torch.cuda.get_device_name(0))
"
```

---

## Why `uv run python` Doesn't Work

`uv run python` uses the `.venv` virtual environment, but `uv pip install` inside that venv (or any `uv sync`) will re-resolve `torch` from PyPI and overwrite the CUDA wheel back to CPU.

Workaround: always run scripts directly with the venv Python path — `C:\path\to\project\.venv\Scripts\python.exe` — not `uv run python`. The CUDA wheel extracted into `site-packages` persists through Python invocations; only `uv sync` resets it.

---

## Make it Automatic

If you want this to run automatically after `uv sync`, add it as a task in your pipeline Makefile or task chain:

```makefile
sync-and-cuda:
    uv sync
    bash setup_cuda_torch.sh
```

Or use the uv-based task scheduler after each `uv sync` event.

---

## Tested Environment

- OS: Windows 11 Home
- CPU: Intel Core Ultra 7 (platform_machine reports `AMD64`)
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver 566.14)
- Python: 3.11.12 via uv
- PyTorch wheel used: `torch-2.5.1+cu121-cp311-cp311-win_amd64.whl`
- URL: `https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl`

---

## Quick-Reference (Copy-Paste for New Projects)

```bash
# 1. Create project, add torch to pyproject.toml
uv sync

# 2. Immediately run CUDA setup
bash /path/to/setup_cuda_torch.sh   # or the one-liner above

# 3. Verify
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The `setup_cuda_torch.sh` script lives at the project root of `company/UC/hw/` in this workspace. Copy it to any new project that needs GPU torch.