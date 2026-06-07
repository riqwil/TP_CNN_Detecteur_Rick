import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')

BEST = Path(r"C:/r_yolo/cctv_person_car_yolov8_finetune_quick/weights/best.pt")
DATA_YAML = Path('data') / 'cctv_person_car' / 'dataset.yaml'
IMG_SIZE = 640
BATCH = 4

if not BEST.exists():
    print('best.pt not found at', BEST)
    raise SystemExit(1)
if not DATA_YAML.exists():
    print('dataset.yaml not found at', DATA_YAML)
    raise SystemExit(1)

print('Evaluating', BEST)
print('Using dataset yaml:', DATA_YAML)

from ultralytics import YOLO

model = YOLO(str(BEST))
metrics = model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, batch=BATCH)

print('Evaluation finished. Metrics object type:', type(metrics))
try:
    # ultralytics metrics object contains nested attributes
    print('mAP50-95:', float(metrics.box.map))
    print('mAP50   :', float(metrics.box.map50))
    print('Precision:', float(metrics.box.mp))
    print('Recall   :', float(metrics.box.mr))
except Exception:
    print(metrics)
