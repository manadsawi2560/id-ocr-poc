import cv2, random, string
import numpy as np
from pathlib import Path

RNG = random.Random(1234)

# ---------- Helper ----------
def thai_id13_checksum(d12: str) -> str:
    s = sum(int(d12[i])*(13-i) for i in range(12))
    return str((11 - (s % 11)) % 10)

def make_id13() -> str:
    """สร้างเลขบัตร 13 หลักถูกต้องตาม checksum"""
    base = ''.join(RNG.choice(string.digits) for _ in range(12))
    return base + thai_id13_checksum(base)

def rand_name():
    def rword(lo, hi):
        n = RNG.randint(lo, hi)
        s = ''.join(RNG.choice(string.ascii_letters) for _ in range(n))
        return s.capitalize()
    return rword(4,8), rword(5,10)

# ---------- Render Card ----------
def render_card(w=1000, h=630):
    canvas = np.full((h,w,3), 240, np.uint8)
    cv2.rectangle(canvas, (0,0), (w,80), (220,230,240), -1)

    # gen ข้อมูล
    id13 = make_id13()              # ได้ string ยาว 13 หลัก
    fn, ln = rand_name()

    # ลำดับบรรทัด: ID13 → FirstName → LastName
    lines = [id13, fn, ln]
    orgs  = [(80, 200), (80, 300), (80, 400)]
    boxes = []

    for i,(txt,(x,y)) in enumerate(zip(lines,orgs)):
        fs = 2.6 if i==0 else 2.2   # ID13 ตัวใหญ่กว่านิดหน่อย
        th = 3
        (tw,thx), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.putText(canvas, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (20,20,20), th, cv2.LINE_AA)
        boxes.append([x, y-thx-10, x+tw, y+10])

    return canvas, lines, boxes

# ---------- Save YOLO format ----------
def to_yolo(x1,y1,x2,y2,W,H):
    cx, cy = (x1+x2)/2/W, (y1+y2)/2/H
    w,  h  = (x2-x1)/W, (y2-y1)/H
    return cx,cy,w,h

def main(n=1200, out="data"):
    out = Path(out)
    idxs = list(range(n)); RNG.shuffle(idxs)
    split = {"train":idxs[:int(.8*n)], "val":idxs[int(.8*n):int(.9*n)], "test":idxs[int(.9*n):]}
    for split_name, ids in split.items():
        (out/f"det/images/{split_name}").mkdir(parents=True, exist_ok=True)
        (out/f"det/labels/{split_name}").mkdir(parents=True, exist_ok=True)
        (out/f"rec/images/{split_name}").mkdir(parents=True, exist_ok=True)
        (out/f"rec/labels").mkdir(parents=True, exist_ok=True)
        tsv = (out/f"rec/labels/{split_name}.tsv").open("w", encoding="utf-8")

        for i in ids:
            img, lines, boxes = render_card()
            H,W = img.shape[:2]
            img_path = out/f"det/images/{split_name}/card_{i:05d}.jpg"
            cv2.imwrite(str(img_path), img)

            # detector labels
            with open(out/f"det/labels/{split_name}/card_{i:05d}.txt","w") as lf:
                for b in boxes:
                    cx,cy,w,h = to_yolo(*b,W,H)
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # recognizer crops
            for j,(b,txt) in enumerate(zip(boxes,lines)):
                x1,y1,x2,y2 = map(int,b)
                crop = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                rel = f"images/{split_name}/card_{i:05d}_{j}.png"
                cv2.imwrite(str(out/"rec"/rel), crop)
                tsv.write(f"{rel}\t{txt}\n")

        tsv.close()
    print("Mock data generated under", out)

if __name__=="__main__":
    main()
