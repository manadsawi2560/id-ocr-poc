import os, time, cv2, jwt, torch, numpy as np, re, easyocr
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------- ของเดิมสำหรับ OCR ID ----------
from src.infer_pipeline import load_models, run_ocr
from src.security import aesgcm_encrypt

# ---------- CONFIG ----------
SECRET = os.getenv("JWT_SECRET", "manadsawi")

_default_key = "thisisademoAESGCMkey32bytes!!"
AES_KEY = os.getenv("AES_KEY", _default_key)[:32].encode()

RETURN_PLAINTEXT = os.getenv("RETURN_PLAINTEXT", "true").lower() in ("1", "true", "yes")
SKIP_AUTH = os.getenv("SKIP_AUTH", "false").lower() in ("1", "true", "yes")

DET_W   = os.getenv("DET_W", "runs/det/train2/weights/best.pt")
REC_W   = os.getenv("REC_W", "runs/rec/best.pt")
CHARSET = os.getenv("CHARSET", "configs/charset.txt")

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()

# ---------- APP ----------
app = FastAPI(title="OCR Service (ID + INCOME)")

det = rec = idx2 = None
reader_en = None  # EasyOCR สำหรับ slip รายได้

# ---------- STARTUP ----------
@app.on_event("startup")
def _startup_load():
    global det, rec, idx2, reader_en

    print("[startup] Loading EasyOCR English reader ...")
    reader_en = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("[startup] EasyOCR ready ✅")

    print("[startup] Loading ID OCR models ...")
    det, rec, idx2 = load_models(
        det_w=DET_W,
        rec_w=REC_W,
        device=DEVICE,
        charset=CHARSET
    )
    print("[startup] All models loaded ✅")

# ---------- Helper ----------
def ensure_ready():
    if det is None or rec is None or idx2 is None:
        raise HTTPException(503, "ID OCR model not ready")

def check_auth(authorization: Optional[str]):
    if SKIP_AUTH:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        jwt.decode(token, SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid token")

def read_image_to_bgr(upload: UploadFile) -> np.ndarray:
    img_bytes = upload.file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img

# ---------- Token ----------
class TokenRequest(BaseModel):
    sub: str = "tester"
    ttl_seconds: int = 3600

@app.post("/token")
def issue_token(req: TokenRequest):
    now = int(time.time())
    payload = {"sub": req.sub, "iat": now, "exp": now + req.ttl_seconds}
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    return {"token": token}

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "id_ocr_ready": det is not None,
        "easyocr_ready": reader_en is not None,
        "return_plaintext": RETURN_PLAINTEXT,
        "skip_auth": SKIP_AUTH,
    }

# ---------- OCR: ID CARD ----------
@app.post("/ocr/id")
async def ocr_id(
    authorization: Optional[str] = Header(default=None),
    image: UploadFile = File(...)
):
    check_auth(authorization)
    ensure_ready()

    img = read_image_to_bgr(image)
    result = run_ocr(img, det, rec, idx2, device=DEVICE)

    if RETURN_PLAINTEXT:
        return JSONResponse(result)

    cipher = aesgcm_encrypt(result, AES_KEY)
    return {"ciphertext": cipher}

# ---------- OCR: INCOME SLIP (EasyOCR) ----------
@app.post("/ocr/income")
async def ocr_income(
    authorization: Optional[str] = Header(default=None),
    image: UploadFile = File(...)
):
    check_auth(authorization)
    if reader_en is None:
        raise HTTPException(503, "EasyOCR not ready")

    img = read_image_to_bgr(image)

    try:
        result = reader_en.readtext(img, detail=1, paragraph=True)
        full_text = " ".join([r[1] for r in result if len(r) >= 2])
    except Exception as e:
        raise HTTPException(500, f"EasyOCR error: {e}")

    text = full_text.lower().replace(",", "")
    match = re.search(r"net\s*pay[^0-9]{0,10}(\d{4,7})", text)
    amount = int(match.group(1)) if match else None

    # fallback หาเลขมากสุด
    if not amount:
        nums = [int(x) for x in re.findall(r"\b\d{4,7}\b", text)]
        if nums:
            amount = max(nums)

    payload = {
        "amount": amount,
        "text": full_text[:1000],
        "count_words": len(full_text.split()),
    }
    return JSONResponse(payload)
