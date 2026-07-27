# Bootcamp setup — shared `bootcamp_utils` library

All session notebooks (PYNQ 101, 201, 301, 501, 503, …) share **one** helper
library: `bootcamp_sessions/bootcamp_utils/`. It holds the common / complex code
(FPGA + DPU setup, model running, Grove LED bar & OLED, webcam capture, image
plotting, pose helpers) so the notebooks stay simple for 6th–12th graders.

Notebooks import it directly with **no `sys.path` lines**:

```python
from bootcamp_utils import load_dpu, setup_dpu_buffers, run_model, classify
```

## Make it work on a freshly flashed KV260 (scalable, zero-touch)

Pick ONE of these — both make `import bootcamp_utils` work from every notebook:

### Option A — automatic on first boot (recommended for fleets)

Bake this into the image build so every flashed board self-installs on first boot:

```bash
cd /home/root/jupyter_notebooks/PYNQ_Bootcamp/bootcamp_sessions
sudo cp bootcamp-utils.service /etc/systemd/system/
sudo systemctl enable bootcamp-utils.service
```

On first boot the service runs `install.sh`, which:
1. `pip install -e .` — installs `bootcamp_utils` as an editable package.
2. Drops an IPython startup fallback so imports work even if pip couldn't run
   (e.g. no network).

Because it's editable + idempotent, this scales to any number of boards:
**flash → boot → notebooks just work.**

### Option B — run once manually

```bash
cd /home/root/jupyter_notebooks/PYNQ_Bootcamp/bootcamp_sessions
bash install.sh
```

## Editing the library

Because the install is *editable*, any change you make to files under
`bootcamp_utils/` takes effect immediately — just restart the notebook kernel.
No reinstall needed.

## Files

- `bootcamp_utils/`        — the shared library (see its own README)
- `pyproject.toml`         — makes it a pip-installable package
- `install.sh`             — installs it (editable) + IPython fallback
- `bootcamp-utils.service` — systemd one-shot to run install.sh on first boot
