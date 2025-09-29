import argparse, os, cv2, json, torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.crnn_model import CRNN
from src.post_rules import map_fields

def load_recognizer(rec_w, num_classes):
    model = CRNN(num_classes=num_classes)
    state = torch.load(rec_w, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model

def ctc_decode(logits, idx2char):
    probs = logits.softmax(-1)
    seq = probs.argmax(-1)
    out=[]
    for row in seq:
        prev=None; s=""
        for k in row.tolist():
            if k!=0 and k!=prev: s += idx2char[k]
            prev=k
        out.append(s.strip())
    return out

def load_charset(path):
    s = Path(path).read_text(encoding="utf-8")
    chars = list(s.strip("\n"))
    if chars[0] != " ":
        chars = [" "] + chars
    table=["<BLANK>"]; seen=set()
    for ch in chars:
        if ch==" " and len(table)>1: continue
        if ch not in seen: table.append(ch); seen.add(ch)
    idx2char = {i:c for i,c in enumerate(table)}
    return table, idx2char

def preprocess_crop(img, height=48, maxw=256):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = g.shape
    neww = int(w*height/h)
    g = cv2.resize(g, (min(maxw,neww), height))
    if g.shape[1] < maxw:
        pad = np.full((height, maxw-g.shape[1]), 255, np.uint8)
        g = np.concatenate([g, pad], axis=1)
    g = (255 - g).astype(np.float32)/255.0
    ten = torch.from_numpy(g).unsqueeze(0).unsqueeze(0)
    return ten

def load_models(det_w, rec_w, device="cuda", charset="configs/charset.txt"):
    det = YOLO(det_w)
    chars, idx2 = load_charset(charset)
    rec = load_recognizer(rec_w, num_classes=len(chars)).to(device)
    rec.eval()
    return det, rec, idx2

def run_ocr(img_bgr, det, rec, idx2char, device="cuda"):
    H,W = img_bgr.shape[:2]
    res = det.predict(img_bgr, imgsz=max(H,W), conf=0.25, verbose=False)
    boxes = []
    for r in res:
        if r.boxes is None: continue
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            crop = img_bgr[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
            ten = preprocess_crop(crop).to(device)
            with torch.no_grad():
                logits = rec(ten)
            text = ctc_decode(logits, idx2char)[0]
            boxes.append((x1,y1,x2,y2,text,conf, (y1+y2)//2))
    boxes.sort(key=lambda x: x[-1])
    lines = [(b[4], b[-1], b[5]) for b in boxes]
    fields = map_fields(lines)
    return {"fields": fields, "boxes": [(b[0],b[1],b[2],b[3],b[4],b[5]) for b in boxes]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_w", required=True)
    ap.add_argument("--rec_w", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="runs/vis")
    ap.add_argument("--charset", default="configs/charset.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(args.det_w)
    chars, idx2 = load_charset(args.charset)
    rec = load_recognizer(args.rec_w, num_classes=len(chars)).to(device).eval()

    os.makedirs(args.out, exist_ok=True)
    preds = {}
    for p in Path(args.images).glob("*.jpg"):
        img = cv2.imread(str(p))
        result = run_ocr(img, det, rec, idx2, device=device)
        preds[p.name] = result
        for (x1,y1,x2,y2,text,conf) in result["boxes"]:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, text, (x1, max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.imwrite(str(Path(args.out)/p.name), img)
    Path(Path(args.out)/"preds.json").write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved visualizations and preds.json to", args.out)

if __name__=="__main__":
    main()
