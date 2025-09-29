import argparse, json, cv2
from pathlib import Path

def draw(image_path, boxes_texts, out_path):
    img = cv2.imread(str(image_path))
    for (x1,y1,x2,y2,text,conf) in boxes_texts:
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, f"{text} ({conf:.2f})", (x1, max(20,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    for rel, info in pred.items():
        boxes = info.get("boxes", [])
        outp = Path(args.out)/Path(rel).name
        draw(Path(args.images)/Path(rel).name, boxes, outp)
    print("Saved visuals to", args.out)

if __name__=="__main__":
    main()
