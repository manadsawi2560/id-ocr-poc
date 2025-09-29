def id13_checksum_ok(s: str) -> bool:
    if len(s)!=13 or not s.isdigit(): return False
    total = sum(int(s[i])*(13-i) for i in range(12))
    check = (11 - (total % 11)) % 10
    return check == int(s[-1])

def map_fields(lines):
    texts = [t for t,_,_ in lines]
    id13="" 
    for t in texts:
        cand = t.replace(" ", "")
        if cand.isdigit() and len(cand)==13 and id13_checksum_ok(cand):
            id13=cand; break
    others = [t for t in texts if t.replace(" ","")!=id13]
    return {
        "first_name": others[0] if len(others)>0 else "",
        "last_name":  others[1] if len(others)>1 else "",
        "id13": id13
    }
