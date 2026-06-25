# MedicalNet CT-Brain Classifier

Fine-tunes a [MedicalNet / Med3D](https://github.com/Tencent/MedicalNet) 3D-ResNet
backbone (pretrained on 23 medical datasets) into a **3-class CT-brain
classifier**: `normal`, `near_normal`, `abnormal`.

Each **series** (`data/{class}/{study}/{series}/*.dcm`) is one training sample,
labelled by its study's class. Splits are made at the **study** level so series
from the same study never leak across train/val/test.

## Layout
```
medicalnet_ctbrain/
├── setup.sh                 # create venv + install (CUDA torch by default)
├── requirements.txt
├── configs/default.yaml     # all hyperparameters
├── scripts/build_index.py   # data/ -> manifest.csv (series rows + splits)
└── src/
    ├── models/              # Med3D ResNet + head + pretrained weight loader
    ├── data/                # DICOM->volume loader, MONAI transforms, Dataset
    ├── train.py  evaluate.py  predict.py
    ├── engine.py  utils.py
```

## 1. Setup (on the GCP GPU box)
```bash
cd medicalnet_ctbrain
bash setup.sh                 # CUDA 12.1 wheels; CUDA=cu118 / CUDA=cpu to change
source .venv/bin/activate
```

## 2. Build the data manifest
**Preferred — reuse the exact same splits as `ct_brain_classifier`** so both
models train/evaluate on identical patient partitions (directly comparable, no
re-leakage):
```bash
python scripts/build_index.py \
    --from-splits /root/ritikkumar/train_data/csvs/splits \
    --out data/manifest.csv
```
This copies the `train/val/test` assignment from ct_brain_classifier's
`{train,val,test}.csv` verbatim (18360 series / 11082 studies:
normal/near_normal/abnormal). Data lives at
`/root/ritikkumar/train_data/{normal,near_normal,abnormal}/{study}/{series}/*.dcm`.

Fallback — scan a data root and make a fresh study-grouped, class-stratified
split (only if you don't have the ct_brain splits):
```bash
python scripts/build_index.py --data-root /root/ritikkumar/train_data --out data/manifest.csv
```
Both print per-split counts and skip thin series (scouts/localizers) via `--min-slices`.

## 3. Train
```bash
python src/train.py --config configs/default.yaml
# overrides:
python src/train.py --config configs/default.yaml --train.epochs 80 --train.lr 1e-4
```
- MedicalNet `resnet34` weights are **auto-downloaded** on first run from the
  official HF mirror
  [`TencentMedicalNet/MedicalNet-Resnet34`](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34)
  (verified: public, ~253MB; all 198 backbone tensors load cleanly). Cached to
  `~/.cache/medicalnet/`. If offline, drop `resnet_34_23dataset.pth` there or
  pass `--model.pretrained_path /path/file.pth`.
- Handles class imbalance via a weighted sampler **and** balanced-CE loss.
- Head-only warmup for the first few epochs, then unfreezes the backbone.
- Checkpoints `best.pth` (by val macro-F1) and `last.pth` to `runs/...`.
- Logs to TensorBoard (`runs/<run>/tb/`) **and** `history.json`. Watch live:
  ```bash
  tensorboard --logdir runs/
  ```
  Scalars include loss/train, loss/val, lr, and every metric
  (accuracy, macro_f1, per-class sensitivity/specificity, AUC, ...).

## 4. Evaluate / Predict
```bash
python src/evaluate.py --ckpt runs/medicalnet_r34/best.pth --split test
python src/predict.py  --ckpt runs/medicalnet_r34/best.pth --study /path/to/study
```

## Key config knobs (`configs/default.yaml`)
| field | meaning |
|---|---|
| `data.spatial_size` | `[D,H,W]` every volume is resized to (default `48×224×224`) |
| `data.windows` | CT windows → input channels. `["brain"]`, or e.g. `["brain","subdural","bone"]` for 3-channel |
| `model.depth` | MedicalNet depth: 10/18/34/50 |
| `train.sampler` / `train.class_weighting` | imbalance handling |
| `train.freeze_backbone_epochs` | head-only warmup length |
| `train.monitor` | `macro_f1` or `balanced_acc` (checkpoint + early-stop metric) |

## Notes
- `abnormal` may be empty/partial while data downloads — the pipeline tolerates
  missing classes; just re-run `build_index.py` and retrain when complete.
- For larger backbones (resnet50) reduce `batch_size` or `spatial_size` if you
  hit GPU OOM.
```
