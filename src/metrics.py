import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    args = ap.parse_args()

    preds = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    n=len(preds); filled=0
    for _,info in preds.items():
        f = info["fields"]
        if f.get("id13"): filled+=1
    print(f"Samples: {n}, ID13 non-empty: {filled/n:.3f}")

if __name__=="__main__":
    main()
