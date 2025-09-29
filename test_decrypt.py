import os, json, sys
from src.security import aesgcm_decrypt

cipher = sys.stdin.read().strip()
key = os.getenv("AES_KEY","")[:32].encode()

print(json.dumps(aesgcm_decrypt(cipher, key), ensure_ascii=False, indent=2))
