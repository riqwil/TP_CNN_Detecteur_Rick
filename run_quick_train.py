from pathlib import Path
import sys
import os

# Force non-interactive matplotlib backend to avoid backend registry errors
os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    from ultralytics import YOLO
except Exception as e:
    print('Erreur import ultralytics:', e)
    sys.exit(1)

# Config (do not change unless you know what you do)
DATASET_DIR = Path('data') / 'cctv_person_car'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATA_YAML = str((DATASET_DIR / 'dataset.yaml').resolve())
IMG_SIZE = 640
BATCH = 4
EPOCHS = 1
FREEZE = 10
PRETRAINED_WEIGHTS = 'yolov8n.pt'
PROJECT_DIR = Path('C:/r_yolo')
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME = 'cctv_person_car_yolov8'

print('DATA_YAML =', DATA_YAML)
print('PROJECT_DIR =', PROJECT_DIR.resolve())
print('IMG_SIZE, BATCH, EPOCHS, FREEZE =', IMG_SIZE, BATCH, EPOCHS, FREEZE)

model = YOLO(PRETRAINED_WEIGHTS)
print('Modèle chargé:', PRETRAINED_WEIGHTS)

try:
    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        freeze=FREEZE,
        project=str(PROJECT_DIR),
        name=f"{RUN_NAME}_frozen",
    )
    print('Training finished.')
    print(results)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
