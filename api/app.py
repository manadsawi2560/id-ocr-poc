import os, cv2, numpy as np, torch, jwt
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from src.infer_pipeline import load_models, run_ocr
from src.security import aesgcm_encrypt


SECRET = os.getenv("JWT_SECRET", "manadsawi")


_default_key = "thisisademoAESGCMkey32bytes!!"
AES_KEY = os.getenv("AES_KEY", _default_key)[:32].encode()
app = FastAPI(title="ID OCR POC")

det, rec, idx2 = None, None, None

@app.on_event("startup")
def _load():
    global det, rec, idx2
    det, rec, idx2 = load_models(
        det_w="runs/det/train2/weights/best.pt",
        rec_w="runs/rec/best.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        charset="configs/charset.txt"
    )

def check_auth(authorization:str|None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ",1)[1]
    try:
        jwt.decode(token, SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid token")

@app.post("/ocr")
async def ocr(authorization: str | None = Header(default=None),
              image: UploadFile = File(...)):
    check_auth(authorization)
    img_bytes = await image.read()
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    result = run_ocr(img, det, rec, idx2, device="cuda" if torch.cuda.is_available() else "cpu")
    cipher = aesgcm_encrypt(result, AES_KEY)
    return {"ciphertext": cipher}
