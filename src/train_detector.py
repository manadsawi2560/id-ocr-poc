# src/train_detector.py
from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/det_data.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                project="runs/det")

if __name__ == "__main__":
    main()
