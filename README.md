# ID-OCR-POC

POC ระบบ OCR อ่านข้อมูลจากบัตรประชาชน Mockup  
Pipeline: **Detector (YOLOv8) → Recognizer (CRNN+CTC) → Post-process → API**

##  Quickstart
```bash
conda env create -f env.yml
conda activate idocr

# 1) Generate mockup dataset
python src/gen_mock_dataset.py

# 2) Train detector
python src/train_detector.py --data configs/det_data.yaml --epochs 30

# 3) Train recognizer
python src/train_recognizer.py \
  --train_tsv data/rec/labels/train.tsv \
  --val_tsv data/rec/labels/val.tsv \
  --charset configs/charset.txt \
  --epochs 30 --batch 64 --out runs/rec

# 4) Run inference pipeline
python src/infer_pipeline.py \
  --det_w runs/det/train/weights/best.pt \
  --rec_w runs/rec/best.pt \
  --images data/det/images/test \
  --out runs/vis

# 5) Run API
export JWT_SECRET="supersecret"
export AES_KEY="32bytessecret32bytessecret!!"
uvicorn api.app:app --reload --port 8000
>>>>>>> 0bbc95d (WIP: local changes before rebase)
