# openWakeWord Retraining

Self-contained training pipeline for the four wake words used by the HRI
speech stack: `frida`, `yes`, `no`, `stop`. Produces the `*.onnx` files that
`kws_oww.py` (in the `home2` repo, at
`hri/packages/speech/scripts/kws_oww.py`) loads at runtime from
`<home2>/hri/packages/speech/assets/oww/`.

## One-command quickstart

From the repo root, on a fresh clone (ideally a CUDA-capable Linux box):

```bash
export OWW_ASSETS_DIR=<home2>/hri/packages/speech/assets/oww
./hri/kws/run.sh
```

That runs:

1. `setup.sh` — creates an isolated venv under `.venv/`, installs pinned
   training deps, clones `piper-sample-generator` under `.third_party/`, and
   downloads every dataset into `.data/`.
2. `train_all.sh` — runs `train.py` once per wake word using the YAML configs
   in `configs/`, producing `.data/models/<word>.onnx`.
3. `integrate.sh` — copies the four trained `.onnx` files into
   `$OWW_ASSETS_DIR`, backing up the existing files to
   `<word>.onnx.bak` first.

Re-running after a crash is cheap: every step is idempotent, and `train.py`
itself skips already-generated positive/negative clips when it sees them on
disk.

## Requirements

- `python3.10` or `python3.11` on `PATH`. The repo-level venv at
  `/Desktop/home2/venv` runs Python 3.13, which is too new for several
  training deps, so `setup.sh` builds its own venv here under `.venv/`.
- `git` for cloning `piper-sample-generator`.
- ~10 GB of free disk under this directory: ~4.1 GB of datasets (see below),
  plus room for the venv, `.third_party/`, and per-word training outputs.
- Strongly recommended: an NVIDIA GPU. On macOS or another CPU-only host the
  pipeline still runs but takes many hours per word.

## What gets downloaded

All of these land under `hri/kws/.data/` (ignored by git — see `.gitignore`
below if you want to add one):

| Asset                                                       | Size     | Used as                         |
| ------------------------------------------------------------ | -------- | ------------------------------- |
| `mit_rirs/`                                                  | ~0.05 GB | Room-impulse-response augment.  |
| `audioset/bal_train_09.parquet` + extracted `audioset_16k/`  | ~1.7 GB  | Background music/speech/noise.  |
| `negative_features_large.npy` (ACAV100M)                     | ~2 GB    | Adversarial negative features.  |
| `validation_set_features.npy`                                | ~0.3 GB  | False-positive validation set.  |
| OWW feature extractors (mel + embedding)                     | <50 MB   | Cached by openwakeword itself.  |

Total: ~4.1 GB. We rely on AudioSet alone for background audio rather than
adding FMA — the `rudraml/fma` HuggingFace loader requires `trust_remote_code`
and can't be streamed, and AudioSet already covers music, speech, and noise.
`download_data.py` prints an estimate and prompts before downloading (the
prompt is auto-accepted when invoked from `run.sh`).

## Iterating on a single word

```bash
source hri/kws/.venv/bin/activate
cd hri/kws
python train.py \
  --training_config configs/frida.yaml \
  --generate_clips --augment_clips --train_model
```

Tweak the YAML (phrase variants, `custom_negative_phrases`,
`max_negative_weight`, `steps`) and re-run. To force regeneration of the
synthetic audio, delete the per-word output directory under `.data/models/`.

### Per-word tuning notes

- **`no`** — the trickiest word. It collides with `know`, `now`, `note`,
  `november`, etc. Its config uses `max_negative_weight: 800` (highest of
  the four words) and explicitly lists those collisions as negative phrases.
- **`yes`** — penalize `yesterday`, `yet`, `yellow` so substring hits don't
  fire the wake word.
- **`stop`** — penalize `stopped`, `stopping`, `shop`, `top`, `sop`.
- **`frida`** — includes phonetic spellings `freeda` / `freida` as positive
  variants for Piper synthesis and penalizes `free the`, `freedom`, `fridge`,
  `afraid` as negatives.

## Verifying the trained models

After `integrate.sh` copies the new `.onnx` files into place:

1. **Sanity load** (from the training venv):
   ```python
   from openwakeword.model import Model
   m = Model(
       wakeword_models=[
           "<home2>/hri/packages/speech/assets/oww/frida.onnx",
           "<home2>/hri/packages/speech/assets/oww/yes.onnx",
           "<home2>/hri/packages/speech/assets/oww/no.onnx",
           "<home2>/hri/packages/speech/assets/oww/stop.onnx",
       ],
       inference_framework="onnx",
   )
   print(sorted(m.models.keys()))  # -> ['frida', 'no', 'stop', 'yes']
   ```
   This mirrors exactly what `kws_oww.py` does at startup, in the `home2`
   repo.

2. **Offline smoke test** — feed
   `<home2>/hri/packages/speech/assets/listening_chime.wav` plus a couple of
   Piper-generated "frida" WAVs through `model.predict()` and print per-key
   scores. Expect ≥0.5 on the true word, ≤0.1 on the chime.

3. **ROS node check** (requires the ROS environment, not the training venv):
   ```bash
   ros2 run speech kws_oww.py
   # publish a pre-recorded wav to /hri/processedAudioChunk and watch /speech/oww
   ```

## Rolling back

`integrate.sh` leaves the previous models at
`$OWW_ASSETS_DIR/<word>.onnx.bak`. To revert:

```bash
cd <home2>/hri/packages/speech/assets/oww
for w in frida yes no stop; do mv "$w.onnx.bak" "$w.onnx"; done
```

## Directory layout

```
hri/kws/
├── README.md          # this file
├── train.py           # openWakeWord training driver, vendored from upstream
│                       # openwakeword's train.py (Apache-2.0)
├── requirements.txt   # pinned training deps
├── setup.sh           # venv + deps + piper-sample-generator + datasets
├── download_data.py   # idempotent dataset downloader (invoked by setup.sh)
├── train_all.sh       # runs train.py for every YAML in configs/
├── integrate.sh       # copies .onnx outputs into assets/oww/ (with .bak)
├── run.sh             # top-level setup + train + integrate
├── configs/
│   ├── frida.yaml
│   ├── yes.yaml
│   ├── no.yaml
│   └── stop.yaml
├── .venv/             # created by setup.sh (gitignored)
├── .third_party/      # created by setup.sh (gitignored)
│   └── piper-sample-generator/
└── .data/             # created by setup.sh (gitignored)
    ├── mit_rirs/
    ├── audioset_16k/
    ├── negative_features_large.npy
    ├── validation_set_features.npy
    └── models/<word>.onnx        # training outputs
```
