import base64, os, json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def aesgcm_encrypt(plaintext:dict, key:bytes):
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, json.dumps(plaintext).encode(), None)
    return base64.b64encode(nonce+ct).decode()

def aesgcm_decrypt(s:str, key:bytes):
    raw = base64.b64decode(s)
    nonce, ct = raw[:12], raw[12:]
    aes = AESGCM(key)
    return json.loads(aes.decrypt(nonce, ct, None).decode())
