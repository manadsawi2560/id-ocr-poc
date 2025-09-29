import argparse, os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from jiwer import wer
from editdistance import eval as edit_distance
from pathlib import Path
from crnn_model import CRNN

def read_charset(path):
    s = Path(path).read_text(encoding="utf-8").strip("\n")
    chars = list(s)
    if chars[0] != " ":
        chars = [" "] + chars
    table=["<BLANK>"]
    seen=set()
    for ch in chars:
        if ch==" " and len(table)>1: 
            continue
        if ch not in seen:
            table.append(ch); seen.add(ch)
    idx = {c:i for i,c in enumerate(table)}
    return table, idx

def normalize_text(t):
    return t.strip()

class LineDataset(Dataset):
    def __init__(self, tsv, charset_index, height=48, maxw=256):
        self.items = []
        self.height=height; self.maxw=maxw
        self.c2i=charset_index
        base = Path(tsv).parent.parent
        for line in Path(tsv).read_text(encoding="utf-8").splitlines():
            rel, text = line.split("\t", 1)
            self.items.append((base/rel, normalize_text(text)))
    def __len__(self): return len(self.items)
    def encode(self, text):
        ids=[]
        for ch in text:
            ids.append(self.c2i.get(ch, self.c2i.get("-", 1)))
        return torch.tensor(ids, dtype=torch.long)
    def __getitem__(self, i):
        p, text = self.items[i]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((self.height, self.maxw), np.uint8)
        h,w = img.shape
        neww = int(w*self.height/h)
        img = cv2.resize(img, (min(self.maxw,neww), self.height))
        if img.shape[1] < self.maxw:
            pad = np.full((self.height, self.maxw - img.shape[1]), 255, np.uint8)
            img = np.concatenate([img, pad], axis=1)
        img = (255 - img).astype(np.float32)/255.0
        img = torch.from_numpy(img).unsqueeze(0)
        target = self.encode(text)
        return img, target, text

def collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    tgts = [b[1] for b in batch]
    flat = torch.cat([t for t in tgts])
    tgt_lengths = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    input_lengths = torch.full((imgs.size(0),), fill_value=imgs.size(-1)//4, dtype=torch.long)
    texts = [b[2] for b in batch]
    return imgs, flat, tgt_lengths, input_lengths, texts

def ctc_greedy_decode(logits, idx2char):
    probs = logits.softmax(-1)
    seq = probs.argmax(-1)
    out_texts=[]
    for row in seq:
        prev = None; s=""
        for k in row.tolist():
            if k!=0 and k!=prev:
                s += idx2char[k]
            prev = k
        out_texts.append(s.strip())
    return out_texts

def evaluate(model, loader, device, idx2char):
    model.eval()
    cer_n=0; cer_d=0; wer_acc=0; n=0
    with torch.no_grad():
        for imgs, _, _, _, gt_texts in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            pred_texts = ctc_greedy_decode(logits, idx2char)
            for p,g in zip(pred_texts, gt_texts):
                cer_n += edit_distance(p, g)
                cer_d += max(1, len(g))
                wer_acc += wer(g, p)
                n += 1
    return cer_n/cer_d, wer_acc/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--charset", default="configs/charset.txt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--maxw", type=int, default=256)
    ap.add_argument("--out", default="runs/rec")
    args = ap.parse_args()

    chars, c2i = read_charset(args.charset)
    idx2char = {i:c for i,c in enumerate(chars)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = LineDataset(args.train_tsv, c2i, args.height, args.maxw)
    val_ds   = LineDataset(args.val_tsv, c2i, args.height, args.maxw)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    model = CRNN(num_classes=len(chars)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best=1e9; os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train()
        total=0; steps=0
        for imgs, flat, tgt_lengths, input_lengths, _ in train_ld:
            imgs = imgs.to(device)
            logits = model(imgs)
            logp = logits.log_softmax(-1).permute(1,0,2)
            loss = ctc(logp, flat.to(device), input_lengths.to(device), tgt_lengths.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); steps+=1
        cer, wr = evaluate(model, val_ld, device, idx2char)
        print(f"[Ep {ep}] loss={total/max(1,steps):.4f}  CER={cer:.4f}  WER={wr:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if cer < best:
            best = cer
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

if __name__=="__main__":
    main()
