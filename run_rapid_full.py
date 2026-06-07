import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')

DATA_YAML = Path('data') / 'cctv_person_car' / 'dataset.yaml'
PROJECT = Path(r"C:/r_yolo")
RUN_NAME = 'cctv_person_car_yolov8'
PRETRAINED = 'yolov8n.pt'
IMG_SIZE = 640
BATCH = 4
EPOCHS_FROZEN = 1
EPOCHS_FINETUNE = 2
FREEZE_LAYERS = 10

if not DATA_YAML.exists():
    print('dataset.yaml not found at', DATA_YAML)
    raise SystemExit(1)

print('Running rapid end-to-end: Phase A (frozen) then Phase B (finetune)')
print('Project dir:', PROJECT)

from ultralytics import YOLO

# Phase A: frozen
print('\n== Phase A: frozen (1 epoch) ==')
model = YOLO(PRETRAINED)
res_a = model.train(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    epochs=EPOCHS_FROZEN,
    batch=BATCH,
    freeze=FREEZE_LAYERS,
    project=str(PROJECT),
    name=f"{RUN_NAME}_frozen",
)
print('Phase A done.')

# Find best.pt from Phase A
best_a = None
cand = PROJECT.rglob('best.pt')
for p in cand:
    if 'frozen' in str(p.parent.parent).lower() or 'frozen' in p.parts:
        best_a = p
        break
if not best_a:
    # fallback: first best.pt
    try:
        best_a = next(PROJECT.rglob('best.pt'))
    except StopIteration:
        best_a = None

if not best_a or not best_a.exists():
    print('ERROR: best.pt from Phase A not found. Aborting Phase B.')
    raise SystemExit(1)

print('Found Phase A best:', best_a)

# Phase B: finetune
print('\n== Phase B: finetune (2 epochs) ==')
model_ft = YOLO(str(best_a))
res_b = model_ft.train(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    epochs=EPOCHS_FINETUNE,
    batch=BATCH,
    freeze=0,
    lr0=1e-4,
    project=str(PROJECT),
    name=f"{RUN_NAME}_finetune_quick",
)
print('Phase B done.')

print('\nRapid end-to-end finished. Checkpoints saved under', PROJECT)
