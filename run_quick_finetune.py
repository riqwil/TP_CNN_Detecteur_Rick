import os
from pathlib import Path

# Use non-interactive matplotlib backend in case imports touch it
os.environ.setdefault('MPLBACKEND', 'Agg')

BEST = Path(r"C:/r_yolo/cctv_person_car_yolov8_frozen/weights/best.pt")
DATA_YAML = Path('data') / 'cctv_person_car' / 'dataset.yaml'
PROJECT = Path(r"C:/r_yolo")
RUN_NAME = 'cctv_person_car_yolov8_finetune_quick'
IMG_SIZE = 640
BATCH = 4
EPOCHS = 2

if not BEST.exists():
    print('best.pt not found at', BEST)
    raise SystemExit(1)
if not DATA_YAML.exists():
    print('dataset.yaml not found at', DATA_YAML)
    raise SystemExit(1)

print('Starting quick finetune using', BEST)
print('Data YAML:', DATA_YAML)

from ultralytics import YOLO

model = YOLO(str(BEST))
results = model.train(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    freeze=0,
    lr0=1e-4,
    project=str(PROJECT),
    name=RUN_NAME,
)

print('Finetune finished. Results object:', type(results))
