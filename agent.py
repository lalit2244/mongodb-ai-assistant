"""
MongoDB AI Agent — Invock ERP  v4.3  (production-grade, fully direct-query)
Every common question answered without LLM pipeline generation.
LLM used ONLY for analysis text, never for building MongoDB filters.

Critical fix in v4.3:
  IQT iCompanyId format unknown at query time — use $in:[ObjectId,string]
  to match REGARDLESS of how iCompanyId is stored. This is the only
  guaranteed-correct approach without a separate format-detection probe.
"""
AGENT_VERSION = "4.3"
print(f"[Agent] Loading agent.py version {AGENT_VERSION}")

import os, json, re, calendar
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI",
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster")
DB_NAME = "dev-cluster"

# ═══════════════════════════ DoH DNS Resolver ═════════════════════════════════

def resolve_mongodb_srv_via_doh(uri: str) -> str:
    if not uri.startswith("mongodb+srv://"):
        return uri
    try:
        import urllib.request, json as _json
        m = re.match(r"mongodb\+srv://([^:@]+):([^@]+)@([^/?]+)/?([^?]*)\??(.*)", uri)
        if not m:
            print("[DoH] Could not parse URI — using original")
            return uri
        user, pwd, host, db_part, extra_params = m.groups()
        db_part = db_part or DB_NAME
        print(f"[DoH] Resolving SRV for {host} via Cloudflare DoH...")
        srv_url = f"https://cloudflare-dns.com/dns-query?name=_mongodb._tcp.{host}&type=SRV"
        req = urllib.request.Request(srv_url,
              headers={"accept": "application/dns-json", "User-Agent": "Python/3"})
        resp = _json.loads(urllib.request.urlopen(req, timeout=15).read())
        if not resp.get("Answer"):
            print(f"[DoH] Trying Google DoH...")
            req2 = urllib.request.Request(
                f"https://dns.google/resolve?name=_mongodb._tcp.{host}&type=SRV",
                headers={"User-Agent": "Python/3"})
            resp = _json.loads(urllib.request.urlopen(req2, timeout=15).read())
        if not resp.get("Answer"):
            return uri
        hosts = []
        for ans in resp["Answer"]:
            if ans.get("type") == 33:
                parts = str(ans["data"]).split()
                if len(parts) == 4:
                    hosts.append(f"{parts[3].rstrip('.')}:{parts[2]}")
        if not hosts:
            return uri
        print(f"[DoH] Found {len(hosts)} shard(s): {hosts}")
        rs_name = None
        try:
            req_txt = urllib.request.Request(
                f"https://cloudflare-dns.com/dns-query?name={host}&type=TXT",
                headers={"accept": "application/dns-json", "User-Agent": "Python/3"})
            txt_resp = _json.loads(urllib.request.urlopen(req_txt, timeout=10).read())
            for ans in (txt_resp.get("Answer") or []):
                if ans.get("type") == 16:
                    txt = str(ans["data"]).strip('"').strip("'")
                    if "replicaSet=" in txt:
                        rs_name = txt.split("replicaSet=")[1].split("&")[0].strip()
                        print(f"[DoH] replicaSet = {rs_name}")
                        break
        except Exception as e:
            print(f"[DoH] TXT lookup failed (non-fatal): {e}")
        params = "tls=true&authSource=admin&tlsAllowInvalidCertificates=true"
        if rs_name:
            params += f"&replicaSet={rs_name}"
        direct_uri = f"mongodb://{user}:{pwd}@{','.join(hosts)}/{db_part}?{params}"
        print(f"[DoH] Direct URI ready — {len(hosts)} host(s)")
        return direct_uri
    except Exception as e:
        print(f"[DoH] Resolution failed: {e} — falling back to original URI")
        return uri


# ═══════════════════════════ MongoDB Helpers ══════════════════════════════════

def get_mongo_client():
    try:
        uri = resolve_mongodb_srv_via_doh(MONGODB_URI)
        c = MongoClient(uri, serverSelectionTimeoutMS=10000)
        c.admin.command("ping")
        return c
    except Exception as e:
        print(f"[MongoDB] {e}")
        return None

def get_db(client): return client[DB_NAME]

def deep_sanitize(obj):
    if isinstance(obj, dict):  return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [deep_sanitize(i) for i in obj]
    if isinstance(obj, ObjectId): return str(obj)
    if isinstance(obj, datetime): return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, float) and obj != obj: return None
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    return str(obj)

def agg(db, col, pipe, limit=None):
    if limit: pipe = pipe + [{"$limit": limit}]
    return [deep_sanitize(d) for d in db[col].aggregate(pipe, allowDiskUse=True)]

def find(db, col, q, proj=None, sort=None, limit=100):
    cur = db[col].find(q, proj or {"_id": 0})
    if sort: cur = cur.sort(sort)
    return [deep_sanitize(d) for d in cur.limit(limit)]

def detect_date_type(client) -> str:
    try:
        s = get_db(client)["Voucher"].find_one({"type":"sales"}, {"_id":0,"issueDate":1})
        if s: return "date_object" if isinstance(s.get("issueDate"), datetime) else "string"
    except: pass
    return "date_object"

def get_stats(client):
    db   = get_db(client)
    cols = {"Voucher":"Voucher","Item":"Item","Business":"Business",
            "ItemQuantityTracker":"ItemQuantityTracker","Contact":"Contact",
            "Account":"Account","IBranch":"IBranch","IUser":"IUser","ICompany":"ICompany"}
    stats = {}
    for key, col in cols.items():
        try:    stats[key] = db[col].estimated_document_count()
        except: stats[key] = 0
    return stats

# ═══════════════════════════ Date Helpers ═════════════════════════════════════

def get_dates():
    now = datetime.utcnow()
    fm  = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    lme = fm - timedelta(seconds=1)
    lms = lme.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ys  = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return {"now":now,"year_start":ys,"last_month_start":lms,"last_month_end":lme,
            "this_month_start":fm,
            "today_start":now.replace(hour=0,minute=0,second=0,microsecond=0),
            "lm_num":lme.month,"lm_year":lme.year,"tm_num":now.month,"ty":now.year}

def dt_conv(obj):
    if isinstance(obj, dict): return {k: dt_conv(v) for k, v in obj.items()}
    if isinstance(obj, list): return [dt_conv(i) for i in obj]
    if isinstance(obj, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S","%Y-%m-%d"]:
            try: return datetime.strptime(obj, fmt)
            except: pass
    return obj

# ═══════════════════════════ IQT Company Filter Helper ═══════════════════════

def iqt_cid_filter(obj_id, str_id) -> dict:
    """
    THE DEFINITIVE FIX for IQT iCompanyId format ambiguity.

    ItemQuantityTracker.iCompanyId may be stored as ObjectId OR string
    depending on when the record was created. We CANNOT know which without
    a per-company probe. Instead, use $in:[ObjectId, string] which matches
    BOTH formats guaranteed — zero false negatives, zero performance penalty
    (MongoDB evaluates $in with an index scan on both values).
    """
    if obj_id is None and str_id is None:
        return {}
    vals = []
    if obj_id is not None:
        vals.append(obj_id)        # ObjectId format
    if str_id is not None and str_id not in vals:
        vals.append(str_id)        # string format
    return {"iCompanyId": {"$in": vals}}


# ═══════════════════════════ Company Fuzzy Resolver ═══════════════════════════

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())).strip()

def clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def tri(s: str):
    return set(s[i:i+3] for i in range(len(s)-2))

def fuzzy(query: str, candidate: str) -> float:
    q, c = norm(query), norm(candidate)
    if not q or not c: return 0.0
    if q == c: return 1.0
    if q in c or c in q: return 0.93
    qc, cc = clean(query), clean(candidate)
    if qc == cc: return 0.98
    if qc in cc or cc in qc: return 0.91
    tq, tc = tri(qc), tri(cc)
    tri_sc = len(tq & tc) / max(len(tq | tc), 1) if tq and tc else 0.0
    qt = {t for t in q.split() if len(t) > 1}
    ct = {t for t in c.split() if len(t) > 1}
    tok_sc = len(qt & ct) / max(len(qt), len(ct)) if qt and ct else 0.0
    pre_sc = sum(1 for qw in qt if any(cw.startswith(qw) or qw.startswith(cw)
                                        for cw in ct)) / max(len(qt), 1) if qt else 0.0
    penalty = min(len(ct - qt) * 0.06, 0.25) if qt and ct else 0.0
    return max(0.0, min(tri_sc*0.50 + tok_sc*0.30 + pre_sc*0.20 - penalty, 1.0))

def resolve_company(client, name: str) -> Optional[Dict]:
    db = get_db(client)
    all_cos = list(db["ICompany"].find({}, {"_id":1,"name":1}))
    if not all_cos: return None
    scored = sorted([(fuzzy(name, d.get("name","")), d) for d in all_cos], key=lambda x: -x[0])
    best_sc, best = scored[0]
    print(f"[Fuzzy] '{name}' → top3: {[(round(s,3),d['name']) for s,d in scored[:3]]}")
    if best_sc < 0.15: return None
    obj_id = best["_id"]
    str_id = str(obj_id)
    n_obj  = db["Voucher"].count_documents({"iCompanyId": obj_id}, maxTimeMS=4000)
    n_str  = db["Voucher"].count_documents({"iCompanyId": str_id}, maxTimeMS=4000)
    real_id = obj_id if n_obj >= n_str else str_id
    total   = max(n_obj, n_str)
    print(f"[Company] '{best['name']}' obj={n_obj} str={n_str}")
    return {"real_id":real_id,"_id_obj":obj_id,"_id_str":str_id,
            "name":best["name"],"total_vouchers":total,"score":best_sc}

def resolve_company_by_hex(client, hex_id: str) -> Optional[Dict]:
    """Resolve a company directly from a 24-char hex ObjectId string."""
    try:
        db     = get_db(client)
        obj_id = ObjectId(hex_id)
        str_id = hex_id
        doc    = db["ICompany"].find_one({"_id": obj_id}, {"_id":0,"name":1})
        name   = doc.get("name", f"Company {hex_id[:8]}...") if doc else f"Company {hex_id[:8]}..."
        n_obj  = db["Voucher"].count_documents({"iCompanyId": obj_id}, maxTimeMS=4000)
        n_str  = db["Voucher"].count_documents({"iCompanyId": str_id}, maxTimeMS=4000)
        real_id = obj_id if n_obj >= n_str else str_id
        total   = max(n_obj, n_str)
        print(f"[HexID] Resolved '{name}' obj={n_obj} str={n_str}")
        return {"real_id":real_id,"_id_obj":obj_id,"_id_str":str_id,
                "name":name,"total_vouchers":total,"score":1.0}
    except Exception as e:
        print(f"[HexID] Failed to resolve {hex_id}: {e}")
        return None

def extract_company_name(question: str) -> Optional[str]:
    q = question.strip()
    if re.search(r'\b[0-9a-fA-F]{24}\b', question):
        return None
    patterns = [
        r"company\s+(?:with|named?|called?|of|id|having|like)?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s*(?:\?|$|\.|,)",
        r"(?:in|for|of|from)\s+(?:the\s+)?company\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s*(?:\?|$|\.|,)",
        r"(?:in|with|for|from)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s+company\b",
        r"(?:vouchers?|sales?|purchases?|records?|revenue|invoices?)\s+(?:in|of|for|from)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,40}?)['\"]?\s*(?:\?|$|\.|,)",
        r"([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,40}?)(?:'s)\s+(?:vouchers?|sales?|data|revenue|customers?)",
        r"(?:of|for)\s+([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)\s*$",
    ]
    stopwords = {"company","the","a","an","in","for","of","with","has","have",
                 "me","my","all","this","that","these","those","its","their",
                 "collection","icompany","ibranch","iuser","voucher","item",
                 "id","search","find","get","show","list","fetch","what","which"}
    generic   = {"sales","purchase","voucher","revenue","data","record","item","companies",
                 "trend","customer","invoice","monthly","total","how","many","what","most",
                 "collection","icompany","ibranch","search","find","created","highest",
                 "top","list","show","all","ranked","best","number","count","maximum"}
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'\b(' + '|'.join(re.escape(w) for w in stopwords) + r')\b\s*$',
                          '', name, flags=re.I).strip()
            words = name.lower().split()
            if (len(name) >= 3
                    and name.lower() not in generic
                    and not re.match(r'^[0-9a-fA-F]{24}$', name)
                    and not all(w in generic for w in words)):
                return name
    return None

# ═══════════════════════════ LLM ═════════════════════════════════════════════

def get_llm():
    k = os.getenv("GROQ_API_KEY")
    if not k: raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=k, temperature=0)

# ═══════════════════════════ Direct Query Library ════════════════════════════

class Q:
    """
    Query builder.

    For Voucher / Business / Item:
        iCompanyId stored as ObjectId  → use self.cid  (ObjectId or string per Voucher probe)

    For ItemQuantityTracker:
        iCompanyId format UNKNOWN → use iqt_cid_filter(obj_id, str_id)
        which generates $in:[ObjectId, string] — matches both formats guaranteed.
    """

    def __init__(self, db, company: Optional[Dict] = None):
        self.db      = db
        self.company = company
        # For Voucher/Business/Item (probed format)
        self.cid     = company["real_id"]  if company else None
        # For IQT — both forms, filter handles $in
        self.obj_id  = company["_id_obj"]  if company else None
        self.str_id  = company["_id_str"]  if company else None

    def _mf(self, base: Dict) -> Dict:
        """Voucher / Business / Item filter — uses probed real_id format."""
        if self.cid is not None:
            base["iCompanyId"] = self.cid
        return base

    def _iqt(self, base: Dict) -> Dict:
        """
        ItemQuantityTracker filter — uses $in:[ObjectId,string].
        Guaranteed to match regardless of how iCompanyId was stored.
        """
        if self.obj_id is not None or self.str_id is not None:
            f = iqt_cid_filter(self.obj_id, self.str_id)
            base.update(f)
        return base

    # ── Voucher queries ───────────────────────────────────────────────────────

    def voucher_count(self, vtype=None, name="Company"):
        mf = self._mf({"type": vtype} if vtype else {})
        rows = agg(self.db, "Voucher", [
            {"$match": mf},
            {"$group": {"_id":None,"total_vouchers":{"$sum":1},
                        "total_amount":{"$sum":"$billFinalAmount"}}},
            {"$project": {"_id":0,"total_vouchers":1,"total_amount":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"total_vouchers",
                      "title":f"{name} — {(vtype or 'all').title()} Vouchers"}

    def vouchers_by_type(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({})},
            {"$group": {"_id":"$type","count":{"$sum":1},"amount":{"$sum":"$billFinalAmount"}}},
            {"$sort":{"count":-1}},
            {"$project":{"_id":0,"type":"$_id","count":1,"amount":1}}
        ])
        return rows, {"type":"bar","x_field":"type","y_field":"count",
                      "title":f"{name} — Vouchers by Type"}

    def voucher_by_status(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({})},
            {"$group": {"_id":"$status","count":{"$sum":1},"amount":{"$sum":"$billFinalAmount"}}},
            {"$sort":{"count":-1}},
            {"$project":{"_id":0,"status":"$_id","count":1,"amount":1}}
        ])
        return rows, {"type":"bar","x_field":"status","y_field":"amount",
                      "title":f"{name} — Vouchers by Status"}

    def top_customers(self, limit=15, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":"sales","party.name":{"$ne":None}})},
            {"$group": {"_id":"$party.name","revenue":{"$sum":"$billFinalAmount"},"invoices":{"$sum":1}}},
            {"$sort":{"revenue":-1}},{"$limit":limit},
            {"$project":{"_id":0,"customer":"$_id","revenue":1,"invoices":1}}
        ])
        return rows, {"type":"bar","x_field":"customer","y_field":"revenue",
                      "title":f"{name} — Top {limit} Customers"}

    def top_suppliers(self, limit=15, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":"purchase","party.name":{"$ne":None}})},
            {"$group": {"_id":"$party.name","amount":{"$sum":"$billFinalAmount"},"invoices":{"$sum":1}}},
            {"$sort":{"amount":-1}},{"$limit":limit},
            {"$project":{"_id":0,"supplier":"$_id","amount":1,"invoices":1}}
        ])
        return rows, {"type":"bar","x_field":"supplier","y_field":"amount",
                      "title":f"{name} — Top {limit} Suppliers"}

    def unpaid_invoices(self, name="Company"):
        rows = find(self.db,"Voucher",self._mf({"status":"unpaid"}),
            {"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,"issueDate":1},
            sort=[("dueAmount",-1)],limit=50)
        return rows, {"type":"table","x_field":"voucherNo","y_field":"dueAmount",
                      "title":f"{name} — Unpaid Invoices"}

    def sales_vs_purchases(self, name="Company"):
        rows = agg(self.db,"Voucher",[
            {"$match": self._mf({"type":{"$in":["sales","purchase"]}})},
            {"$group": {"_id":"$type","total":{"$sum":"$billFinalAmount"},"count":{"$sum":1}}},
            {"$project":{"_id":0,"type":"$_id","total":1,"count":1}}
        ])
        return rows, {"type":"bar","x_field":"type","y_field":"total",
                      "title":f"{name} — Sales vs Purchases"}

    def avg_order_value(self, name="Company"):
        rows = agg(self.db,"Voucher",[
            {"$match": self._mf({"type":"sales"})},
            {"$group": {"_id":None,"avg_order_value":{"$avg":"$billFinalAmount"},"total_orders":{"$sum":1}}},
            {"$project":{"_id":0,"avg_order_value":1,"total_orders":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"avg_order_value",
                      "title":f"{name} — Average Order Value"}

    # ── ItemQuantityTracker queries — ALL use self._iqt() ─────────────────────

    def monthly_trend(self, years=None, name="Company"):
        d   = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._iqt({"voucherType":"sales","year":{"$in":yrs}})},
            {"$group": {"_id":{"year":"$year","month":"$month"},
                        "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
            {"$sort":{"_id.year":1,"_id.month":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1,"qty":1}}
        ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Sales Trend"}

    def total_revenue(self, year=None, name="Company"):
        d    = get_dates()
        year = year or d["ty"]   # always default to current year
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._iqt({"voucherType":"sales","year":year})},
            {"$group": {"_id":None,"total_revenue":{"$sum":"$amount"},"total_qty":{"$sum":"$qty"}}},
            {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"total_revenue",
                      "title":f"{name} — Revenue ({year})"}

    def top_products(self, by="amount", limit=15, name="Company"):
        d  = get_dates()
        yf = "amount" if by == "amount" else "qty"
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._iqt({"voucherType":"sales","year":d["ty"]})},
            {"$group": {"_id":"$itemId","amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
            {"$sort":{yf:-1}},{"$limit":limit},
            {"$project":{"_id":0,"item":"$_id","amount":1,"qty":1}}
        ])
        return rows, {"type":"bar","x_field":"item","y_field":yf,
                      "title":f"{name} — Top {limit} Products ({d['ty']})"}

    def purchase_trend(self, years=None, name="Company"):
        d   = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._iqt({"voucherType":"purchase","year":{"$in":yrs}})},
            {"$group": {"_id":{"year":"$year","month":"$month"},"amount":{"$sum":"$amount"}}},
            {"$sort":{"_id.year":1,"_id.month":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1}}
        ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Purchase Trend"}

    # ── Item / Stock / Business ───────────────────────────────────────────────

    def stock(self, name="Company"):
        q = {"isHidden":False,"availableQty":{"$gt":0}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db,"Item",q,
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",-1)],limit=100)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Stock / Inventory"}

    def low_stock(self, threshold=10, name="Company"):
        q = {"isHidden":False,"availableQty":{"$gt":0,"$lte":threshold}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db,"Item",q,
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",1)],limit=50)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Low Stock (≤{threshold})"}

    def customer_list(self, name="Company"):
        q = {"relationType":{"$in":["customer","both"]}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db,"Business",q,
            {"_id":0,"name":1,"city":1,"state":1,"relationType":1},
            sort=[("name",1)],limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Customer List"}

    def supplier_list(self, name="Company"):
        q = {"relationType":{"$in":["supplier","both"]}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db,"Business",q,
            {"_id":0,"name":1,"city":1,"state":1},
            sort=[("name",1)],limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Supplier List"}


# ═══════════════════════════ Intent Router ════════════════════════════════════

def route(question: str, company: Optional[Dict], db) -> Optional[Tuple]:
    q   = question.lower().strip()
    n   = company["name"] if company else "All Companies"
    qb  = Q(db, company)   # pass full company dict — Q extracts what it needs
    d   = get_dates()
    has  = lambda *ws: any(w in q for w in ws)
    miss = lambda *ws: not any(w in q for w in ws)

    # ── Company ranking ───────────────────────────────────────────────────────
    if re.search(r"(companies|company).*(most|top|highest|ranked?|maximum|max|list).*(voucher|sales|invoice)|"
                 r"(most|top|highest).*(voucher|sales).*(compan)|"
                 r"which compan.*(most|highest|top).*(voucher|sales)|"
                 r"rank.*compan.*voucher|list compan.*voucher|compan.*most.*created|"
                 r"top\s+\d*\s*compan.*(sales|voucher)|compan.*by.*(sales|voucher)|"
                 r"list.*compan.*by.*(most|sales|voucher)|compan.*sales.*rank|"
                 r"list.*compan.*(most|sales|voucher)|compan.*(created|having).*(sales|voucher)|"
                 r"(sales|voucher).*(compan).*(list|rank|top|most)|"
                 r"show.*compan.*(sales|voucher)|compan.*with.*most.*(sales|voucher)", q):
        vtype = "purchase" if "purchase" in q else "sales"
        lim   = 20
        nm = re.search(r"top\s+(\d+)", q)
        if nm: lim = min(int(nm.group(1)), 50)
        return companies_by_voucher_count(db, vtype, lim)

    # ── Voucher counts ────────────────────────────────────────────────────────
    if re.search(r"how many.*(sales|purchase|receipt|payment).*(voucher|invoice|bill|record)|"
                 r"(voucher|invoice|bill).*(count|how many|total number|number of)", q):
        vtype = ("sales" if "sales" in q else "purchase" if "purchase" in q else
                 "receipt" if "receipt" in q else "payment" if "payment" in q else None)
        return qb.voucher_count(vtype, n)

    if re.search(r"how many voucher|voucher count|number of voucher|count.*voucher|total.*voucher", q):
        vtype = "sales" if "sales" in q else "purchase" if "purchase" in q else None
        return qb.voucher_count(vtype, n)

    # ── Revenue / Sales total ─────────────────────────────────────────────────
    if re.search(r"total.*(revenue|sales|amount)|revenue.*total|(sales|revenue).*this year|ytd|year.*to.*date", q):
        yr = d["ty"] if has("this year","ytd","year to date",str(d["ty"])) else d["ty"]
        return qb.total_revenue(yr, n)

    # ── Purchase total ────────────────────────────────────────────────────────
    if re.search(r"total.*purchase|purchase.*total|purchase.*this year", q) and miss("voucher","count"):
        yr = d["ty"] if has("this year",str(d["ty"])) else d["ty"]
        match = qb._iqt({"voucherType":"purchase","year":yr})
        rows  = agg(db,"ItemQuantityTracker",[
            {"$match": match},
            {"$group": {"_id":None,"total_purchases":{"$sum":"$amount"}}},
            {"$project":{"_id":0,"total_purchases":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"total_purchases",
                      "title":f"{n} — Total Purchases ({yr})"}

    # ── Monthly / time trend ──────────────────────────────────────────────────
    if re.search(r"monthly.*trend|trend.*month|month.*sales|sales.*trend|"
                 r"last 12 month|12 month|month.*wise|monthly.*sales|sales.*monthly", q):
        return qb.monthly_trend(name=n)

    if re.search(r"monthly.*purchase|purchase.*trend|purchase.*month", q):
        return qb.purchase_trend(name=n)

    # ── Sales vs purchases ────────────────────────────────────────────────────
    if re.search(r"sales.*vs.*purchase|purchase.*vs.*sales|compare.*sale|sale.*comparison|"
                 r"sales.*and.*purchase|purchase.*and.*sales", q):
        return qb.sales_vs_purchases(n)

    # ── Top customers ─────────────────────────────────────────────────────────
    if re.search(r"top.*customer|best.*customer|customer.*revenue|customer.*sales|"
                 r"biggest.*customer|largest.*customer|customer.*list|list.*customer|"
                 r"show.*customer|all.*customer|customer.*ranking", q):
        return qb.top_customers(name=n)

    # ── Top suppliers ─────────────────────────────────────────────────────────
    if re.search(r"top.*supplier|best.*supplier|supplier.*list|list.*supplier|"
                 r"show.*supplier|all.*supplier|vendor|purchase.*from", q):
        return qb.top_suppliers(name=n)

    # ── Top products ──────────────────────────────────────────────────────────
    if re.search(r"top.*product|best.*product|most.*sold|product.*revenue|"
                 r"item.*sold|which.*product|item.*ranking|popular.*item|"
                 r"top.*item|best.*item|fast.*moving", q):
        by = "qty" if has("qty","quantity","units","pieces") else "amount"
        return qb.top_products(by, name=n)

    # ── Unpaid / outstanding ──────────────────────────────────────────────────
    if re.search(r"unpaid|outstanding|overdue|due.*amount|pending.*payment|"
                 r"receivable|not.*paid|dues", q):
        return qb.unpaid_invoices(n)

    # ── Average order value ───────────────────────────────────────────────────
    if re.search(r"avg.*order|average.*order|order.*value|aov|avg.*invoice|"
                 r"average.*invoice|per.*order", q):
        return qb.avg_order_value(n)

    # ── Vouchers by status ────────────────────────────────────────────────────
    if re.search(r"paid.*voucher|payment.*status|voucher.*status|status.*voucher|"
                 r"partial.*payment|how many.*paid|how many.*unpaid", q):
        return qb.voucher_by_status(n)

    # ── Vouchers by type ─────────────────────────────────────────────────────
    if re.search(r"voucher.*type|type.*voucher|voucher.*breakdown|breakdown.*voucher|"
                 r"all.*type.*voucher|what type", q):
        return qb.vouchers_by_type(n)

    # ── Stock / inventory ─────────────────────────────────────────────────────
    if re.search(r"stock|inventory|available.*qty|items.*in.*stock|current.*stock|"
                 r"how many.*item|item.*available|product.*stock", q):
        if re.search(r"low|less|below|shortage|running out", q):
            return qb.low_stock(name=n)
        return qb.stock(n)

    # ── Customer list / count ─────────────────────────────────────────────────
    if re.search(r"list.*customer|show.*customer|all.*customer|customer.*list|"
                 r"how many.*customer|count.*customer|number.*customer", q):
        cid = qb.cid
        if re.search(r"how many|count|number", q):
            rows = agg(db,"Business",[
                {"$match": ({"iCompanyId":cid,"relationType":{"$in":["customer","both"]}}
                            if cid else {"relationType":{"$in":["customer","both"]}})},
                {"$count":"total_customers"}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_customers",
                          "title":f"{n} — Total Customers"}
        return qb.customer_list(n)

    # ── Supplier list / count ─────────────────────────────────────────────────
    if re.search(r"list.*supplier|show.*supplier|all.*supplier|supplier.*list|"
                 r"how many.*supplier|count.*supplier", q):
        cid = qb.cid
        if re.search(r"how many|count|number", q):
            rows = agg(db,"Business",[
                {"$match": ({"iCompanyId":cid,"relationType":{"$in":["supplier","both"]}}
                            if cid else {"relationType":{"$in":["supplier","both"]}})},
                {"$count":"total_suppliers"}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_suppliers",
                          "title":f"{n} — Total Suppliers"}
        return qb.supplier_list(n)

    return None


# ═══════════════════════════ Schema-level shortcuts ════════════════════════════

def companies_by_voucher_count(db, vtype="sales", limit=20) -> Tuple[List, Dict]:
    pipe = [
        {"$match": {"type": vtype}},
        {"$group": {"_id":"$iCompanyId","voucher_count":{"$sum":1},
                    "total_amount":{"$sum":"$billFinalAmount"}}},
        {"$sort": {"voucher_count":-1}},
        {"$limit": limit}
    ]
    rows      = list(db["Voucher"].aggregate(pipe, allowDiskUse=True))
    all_cos   = list(db["ICompany"].find({},{"_id":1,"name":1}))
    id_to_name = {str(c["_id"]): c.get("name","Unknown") for c in all_cos}
    result = []
    for r in rows:
        cid = str(r["_id"]) if r["_id"] else None
        result.append({
            "company":       id_to_name.get(cid, f"Unknown ({cid[:8] if cid else '?'}...)"),
            "voucher_count": int(r.get("voucher_count",0)),
            "total_amount":  round(float(r.get("total_amount",0) or 0), 2)
        })
    return result, {"type":"bar","x_field":"company","y_field":"voucher_count",
                    "title":f"Companies by {vtype.title()} Voucher Count"}


def schema_shortcut(q: str) -> Optional[Dict]:
    d    = get_dates()
    miss = lambda *ws: not any(w in q for w in ws)

    def plan(qt, col, pipe=None, fq=None, proj=None, sort=None, limit=100,
             tmpl="", ct="table", x=None, y=None, title=""):
        return {"query_type":qt,"collection":col,"pipeline":pipe,
                "find_query":fq,"projection":proj,"sort":sort,"limit":limit,
                "answer_template":tmpl,
                "chart_suggestion":{"type":ct,"x_field":x,"y_field":y,"title":title},
                "clarification_needed":False}

    if re.search(r"branch|branches|location", q) and miss("sales","revenue","voucher","customer","company with","in company"):
        return plan("find","IBranch",fq={},proj={"_id":0,"name":1,"city":1,"state":1,"code":1},
            limit=300,tmpl="All branches.",ct="table",title="All Branches")

    if re.search(r"\buser|users|staff|employee\b", q) and miss("company with","in company","sales","voucher"):
        return plan("find","IUser",fq={},proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},
            limit=500,tmpl="All users.",ct="table",title="All Users")

    if re.search(r"^(list |show |get |how many )?(all )?compan", q) and miss("with","in company","for","sales","voucher","most","top","rank"):
        return plan("find","ICompany",fq={},proj={"_id":0,"name":1,"industry":1,"financialYear":1},
            limit=200,tmpl="All companies.",ct="table",title="All Companies")

    if re.search(r"monthly.*trend|trend.*month|last 12 month|12 month|month.*wise", q) and miss("company","with","in","for","purchase"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[{"$match":{"voucherType":"sales","year":{"$in":[d["ty"]-1,d["ty"]]}}},
                  {"$group":{"_id":{"year":"$year","month":"$month"},
                             "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                  {"$sort":{"_id.year":1,"_id.month":1}},
                  {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1,"qty":1}}],
            tmpl="Monthly sales trend.",ct="line",x="month",y="amount",
            title="Monthly Sales Trend (Last 12 Months)")

    if re.search(r"total.*(revenue|sales).*year|revenue.*this year|sales.*this year|ytd", q) and miss("company","with","in"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[{"$match":{"voucherType":"sales","year":d["ty"]}},
                  {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"},"total_qty":{"$sum":"$qty"}}},
                  {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}],
            tmpl=f"Total sales revenue {d['ty']}.",ct="metric",y="total_revenue",
            title=f"Total Revenue {d['ty']}")

    if re.search(r"sales.*last month|last month.*sales|revenue.*last month", q) and miss("company","with"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[{"$match":{"voucherType":"sales","year":d["lm_year"],"month":d["lm_num"]}},
                  {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"}}},
                  {"$project":{"_id":0,"total_revenue":1}}],
            tmpl="Last month sales.",ct="metric",y="total_revenue",
            title=f"Sales — {calendar.month_abbr[d['lm_num']]} {d['lm_year']}")

    if re.search(r"sales.*vs.*purchase|purchase.*vs.*sales", q) and miss("company","with","in"):
        return plan("aggregate","Voucher",
            pipe=[{"$match":{"type":{"$in":["sales","purchase"]},"iCompanyId":{"$ne":None}}},
                  {"$group":{"_id":"$type","total":{"$sum":"$billFinalAmount"},"count":{"$sum":1}}},
                  {"$project":{"_id":0,"type":"$_id","total":1,"count":1}}],
            tmpl="Sales vs purchases.",ct="bar",x="type",y="total",
            title="Sales vs Purchases (All Companies)")

    if re.search(r"top.*customer|best.*customer|customer.*revenue", q) and miss("company","with","in","list","show"):
        return plan("aggregate","Voucher",
            pipe=[{"$match":{"type":"sales","iCompanyId":{"$ne":None},"party.name":{"$ne":None}}},
                  {"$group":{"_id":"$party.name","revenue":{"$sum":"$billFinalAmount"},"invoices":{"$sum":1}}},
                  {"$sort":{"revenue":-1}},{"$limit":15},
                  {"$project":{"_id":0,"customer":"$_id","revenue":1,"invoices":1}}],
            tmpl="Top 15 customers by revenue.",ct="bar",x="customer",y="revenue",
            title="Top 15 Customers by Revenue")

    if re.search(r"top.*product|best.*product|most.*sold|which.*product.*sold", q) and miss("company","with","in"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[{"$match":{"voucherType":"sales","year":d["ty"]}},
                  {"$group":{"_id":"$itemId","revenue":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                  {"$sort":{"revenue":-1}},{"$limit":15},
                  {"$project":{"_id":0,"item":"$_id","revenue":1,"qty":1}}],
            tmpl=f"Top 15 products by revenue ({d['ty']}).",ct="bar",x="item",y="revenue",
            title=f"Top Products by Revenue ({d['ty']})")

    if re.search(r"unpaid|outstanding|overdue", q) and miss("company","with","in"):
        return plan("find","Voucher",
            fq={"status":"unpaid","iCompanyId":{"$ne":None}},
            proj={"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,"issueDate":1},
            sort={"dueAmount":-1},limit=50,tmpl="Unpaid invoices.",
            ct="table",title="Unpaid Invoices")

    if re.search(r"stock|inventory|available.*qty", q) and miss("company","with","in"):
        return plan("find","Item",
            fq={"isHidden":False,"availableQty":{"$gt":0}},
            proj={"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort={"availableQty":-1},limit=100,tmpl="Current stock.",
            ct="table",title="Stock / Inventory")

    if re.search(r"avg.*order|average.*order|aov", q) and miss("company","with","in"):
        return plan("aggregate","Voucher",
            pipe=[{"$match":{"type":"sales","iCompanyId":{"$ne":None}}},
                  {"$group":{"_id":None,"avg_order_value":{"$avg":"$billFinalAmount"},"orders":{"$sum":1}}},
                  {"$project":{"_id":0,"avg_order_value":1,"orders":1}}],
            tmpl="Average order value.",ct="metric",y="avg_order_value",
            title="Average Order Value (All Companies)")

    if re.search(r"how many (customer|supplier|client)", q) and miss("company","with","in"):
        rel   = "customer" if "customer" in q or "client" in q else "supplier"
        field = f"total_{rel}s"
        return plan("aggregate","Business",
            pipe=[{"$match":{"relationType":rel}},{"$count":field}],
            tmpl=f"Total {rel}s.",ct="metric",y=field,title=f"Total {rel.title()}s")

    return None


# ═══════════════════════════ LLM Fallback ════════════════════════════════════

SCHEMA_TEXT = """
DATABASE: dev-cluster — Invock ERP (Jewellery, India, ₹ INR)
Voucher(1.3M): type,billFinalAmount,dueAmount,paidAmount,status,iCompanyId(ObjectId),issueDate(Date),party.name,voucherNo
ItemQuantityTracker(2.1M): voucherType,month(int),year(int),itemId,qty,amount,iCompanyId(string or ObjectId)
Item(450K): name,skuBarcode,availableQty,unit,iCompanyId,isHidden
Business(45K): name,relationType,city,state,iCompanyId
IBranch(264),IUser(399),ICompany(135): name,industry/financialYear
Account(51K): name,accountGroupName,balance,iCompanyId
"""

def llm_prompt(dates):
    d = dates
    return f"""You are a MongoDB expert for Invock ERP. Return ONLY valid JSON, no markdown.
{{
  "query_type":"aggregate"|"find"|"none",
  "collection":"<exact name>",
  "pipeline":[...]|null,
  "find_query":{{...}}|null,
  "projection":{{"_id":0,...}}|null,
  "sort":{{...}}|null,
  "limit":50,
  "answer_template":"<one sentence>",
  "chart_suggestion":{{"type":"bar"|"line"|"metric"|"table"|"none","x_field":"<field>","y_field":"<field>","title":"<title>"}},
  "clarification_needed":false
}}
TODAY: {d['now'].strftime('%Y-%m-%d')}  TY={d['ty']}  LM={d['lm_num']}/{d['lm_year']}
RULES:
1. iCompanyId: ObjectId in Voucher/Item/Business; string OR ObjectId in ItemQuantityTracker
2. $sum/$avg must be {{"$sum":"$fieldName"}} (with $ prefix)
3. Every agg must end with $project removing _id
4. issueDate is Date object — use ISO strings, agent converts
5. Use ItemQuantityTracker for date/product queries (integer year/month)
6. Never project itemList/transactions/tax/party/voucherList
7. x_field/y_field = exact field names from your $project
8. Always filter ItemQuantityTracker by year — never sum all years"""

def parse_plan(text: str) -> Dict:
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try: return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {"query_type":"none","answer_template":"Could not parse response.",
            "chart_suggestion":{"type":"none"},"clarification_needed":False}

VALID_COLS = {
    "Voucher","Item","Business","ItemQuantityTracker","ItemSummary","Contact",
    "Account","IBranch","IUser","ICompany","ItemGroup","company_data",
    "voucher_count","AccountGroup",
}

def resolve_col(col):
    if col in VALID_COLS: return col
    return {c.lower():c for c in VALID_COLS}.get((col or "").lower(), col)

def execute_plan(plan, db, date_type):
    qt = plan.get("query_type")
    if qt not in ("aggregate","find"): return [], None
    col = resolve_col(plan.get("collection",""))
    try:
        if qt == "aggregate" and plan.get("pipeline"):
            pipe = dt_conv(plan["pipeline"]) if date_type=="date_object" else plan["pipeline"]
            return agg(db, col, pipe), None
        if qt == "find" and plan.get("find_query") is not None:
            fq  = dt_conv([plan["find_query"]])[0] if date_type=="date_object" else plan["find_query"]
            srt = list(plan["sort"].items()) if plan.get("sort") else None
            return find(db, col, fq, plan.get("projection"), srt, plan.get("limit",100)), None
    except Exception as e:
        return [], str(e)
    return [], None


# ═══════════════════════════ Intent detection ════════════════════════════════

_ANALYTICS_KW = re.compile(
    r"total|sales|revenue|purchase|trend|monthly|voucher|customer|supplier|"
    r"product|stock|inventory|unpaid|outstanding|overdue|avg|average|count|"
    r"how many|top|best|most|compare|vs|breakdown|profit|amount|analyse|analyze", re.I)

_LOOKUP_KW = re.compile(
    r"\b(find|show|get|what is|what are|name of|details|info|record|lookup|fetch|which)\b", re.I)


# ═══════════════════════════ Main Agent ══════════════════════════════════════

class MongoAIAgent:
    def __init__(self):
        self.client           = get_mongo_client()
        self.llm              = None
        self.history          = []
        self.date_type        = "date_object"
        self.stats            = {}
        self.collection_stats = {}
        if self.client:
            try:
                self.date_type        = detect_date_type(self.client)
                self.stats            = get_stats(self.client)
                self.collection_stats = self.stats
                print(f"[Agent] Connected. issueDate={self.date_type}")
            except Exception as e:
                print(f"[Agent] Init error: {e}")

    def is_connected(self): return self.client is not None

    def refresh_schema(self):
        if self.client:
            self.date_type        = detect_date_type(self.client)
            self.stats            = get_stats(self.client)
            self.collection_stats = self.stats

    def init_llm(self):
        try: self.llm = get_llm(); return True
        except: return False

    def query(self, question: str) -> Dict:
        assert AGENT_VERSION == "4.3", f"Wrong agent version: {AGENT_VERSION}"

        if not self.llm and not self.init_llm():
            return {"error": "GROQ_API_KEY not configured."}

        q_low = question.lower().strip()
        db    = get_db(self.client) if self.client else None

        # ── Step 0: Direct record lookup (only for non-analytics questions) ───
        direct = self._direct_id_or_collection_query(question, db)
        if direct:
            return direct

        # ── Step 1: Global schema shortcuts (no company needed) ───────────────
        sc = schema_shortcut(q_low)
        if sc:
            results, err = execute_plan(sc, db, self.date_type)
            answer = self._answer(question, sc, results, err)
            chart  = self._chart(results, sc["chart_suggestion"])
            self.history.append({"q": question, "a": sc["answer_template"][:80]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":sc,"db_error":err}

        # ── Step 1b: Company ranking ───────────────────────────────────────────
        _rp = (r"(companies|company).*(most|top|highest|ranked?|maximum|max|list).*(voucher|sales)|"
               r"(most|top|highest).*(voucher|sales).*(compan)|"
               r"which compan.*(most|highest|top).*(voucher|sales)|"
               r"rank.*compan.*voucher|list compan.*voucher|compan.*most.*created|"
               r"top\s*\d*\s*compan.*(sales|voucher)|compan.*by.*(sales|voucher)|"
               r"list.*compan.*(most|sales|voucher)|compan.*(created|having).*(sales|voucher)|"
               r"show.*compan.*(sales|voucher)|compan.*with.*most.*(sales|voucher)|"
               r"(sales|voucher).*(compan).*(list|rank|top|most)")
        _has_co  = "compan" in q_low
        _has_met = any(w in q_low for w in ["voucher","sales","invoice"])
        _has_rnk = any(w in q_low for w in ["list","most","top","highest","rank","maximum","best","created"])
        if (bool(re.search(_rp, q_low)) or (_has_co and _has_met and _has_rnk)) and db:
            print(f"[v4.3] Company ranking: '{question[:70]}'")
            _vtype = "purchase" if "purchase" in q_low else "sales"
            _nm    = re.search(r"top\s+(\d+)", q_low)
            _lim   = min(int(_nm.group(1)), 50) if _nm else 20
            results, chart_sug = companies_by_voucher_count(db, _vtype, _lim)
            results = [deep_sanitize(r) for r in results]
            answer  = self._answer_company_ranking(results, question)
            chart   = self._chart(results, chart_sug)
            plan    = {"query_type":"direct","collection":"Voucher",
                       "answer_template":"Companies ranked by sales vouchers.",
                       "chart_suggestion":chart_sug,"clarification_needed":False}
            self.history.append({"q": question, "a": plan["answer_template"]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":plan,"db_error":None}

        # ── Step 2: Resolve company ────────────────────────────────────────────
        company = None

        # Hex ID in question + analytics keywords → resolve as company filter
        hex_match = re.search(r'\b([0-9a-fA-F]{24})\b', question)
        if hex_match and self.client and _ANALYTICS_KW.search(question):
            print(f"[v4.3] Hex ID + analytics → resolve_company_by_hex")
            company = resolve_company_by_hex(self.client, hex_match.group(1))

        # No hex ID — fuzzy name extraction
        if company is None:
            cname = extract_company_name(question)
            if cname and self.client:
                company = resolve_company(self.client, cname)
                if company is None:
                    company = resolve_company(self.client, question)
                if company is None:
                    return {
                        "type":"answer",
                        "answer":(f"❌ No company matching **\"{cname}\"** found.\n\n"
                                  f"**Top companies by sales:**\n"
                                  f"• M/S DIPSHI - ESTIMATE (40,255 vouchers)\n"
                                  f"• HIRAKA JEWELS (34,998)  •  Bhakti Parshwanath (30,707)\n"
                                  f"• NAMO-ESTIMATE (27,478)  •  VAIBHAV FASHION (11,735)\n\n"
                                  f"Ask *\"list all companies\"* to see all 135 companies."),
                        "results":[],"chart":None,"plan":{},"db_error":None
                    }

        # ── Step 3: Intent router ─────────────────────────────────────────────
        if db is not None:
            routed = route(question, company, db)
            if routed:
                results_raw, chart_sug = routed
                results = [deep_sanitize(r) for r in results_raw]
                plan    = {"query_type":"direct","collection":"Voucher",
                           "answer_template":f"Query for {company['name'] if company else 'all'}.",
                           "chart_suggestion":chart_sug,"clarification_needed":False}
                if results and "voucher_count" in results[0] and "company" in results[0]:
                    answer = self._answer_company_ranking(results, question)
                else:
                    answer = self._answer(question, plan, results, None, company)
                chart = self._chart(results, chart_sug)
                self.history.append({"q": question, "a": plan["answer_template"][:80]})
                return {"type":"answer","answer":answer,"results":results,
                        "chart":chart,"plan":plan,"db_error":None}

        # ── Step 4: LLM fallback ──────────────────────────────────────────────
        dates      = get_dates()
        sys_prompt = llm_prompt(dates)
        hist       = ("\nPrev:\n" + "\n".join(f"Q:{h['q']}\nA:{h['a']}"
                       for h in self.history[-3:]) if self.history else "")
        user_msg   = f"{SCHEMA_TEXT}\n{hist}\n\nQuestion: {question}"
        try:
            resp = self.llm.invoke([SystemMessage(content=sys_prompt),
                                    HumanMessage(content=user_msg)])
            plan = parse_plan(resp.content)
        except Exception as e:
            return {"error": f"LLM error: {e}"}

        results, err = execute_plan(plan, db, self.date_type)
        if (not results or err) and plan.get("query_type") != "none":
            hint = (f"\n⚠️ Previous attempt failed: {err or 'empty results'}\n"
                    f"Failed pipeline: {json.dumps(plan.get('pipeline'), default=str)[:200]}\n"
                    f"Fix: ensure $sum uses '$fieldName' format.")
            try:
                resp2 = self.llm.invoke([SystemMessage(content=sys_prompt),
                                         HumanMessage(content=user_msg + hint)])
                plan  = parse_plan(resp2.content)
                results, err = execute_plan(plan, db, self.date_type)
            except: pass

        answer = self._answer(question, plan, results, err, company)
        chart  = self._chart(results, plan.get("chart_suggestion",{}))
        self.history.append({"q": question, "a": plan.get("answer_template","")[:80]})
        return {"type":"answer","answer":answer,"results":results,
                "chart":chart,"plan":plan,"db_error":err}


    def _direct_id_or_collection_query(self, question: str, db) -> Optional[Dict]:
        """Only intercept LOOKUP questions. Analytics questions pass through."""
        if db is None: return None
        q = question.strip()

        SKIP_FIELDS = {
            "_id","__v","id","logoUrl","pancard","gstNo","primaryBranchId",
            "eShopPathName","eShopViewCount","iShopSettings","printSettings",
            "planDetails","companyVoucherSettings","ewayBillEInvoiceSetup",
            "allowDuplicateItems","currencyDecimals","itemQtyDecimals",
            "dateFormat","financialYearStart","bookStartDate","updatedAt",
            "printName","taxDetails","bankDetails","signature","address",
            "defaultTax","itemGroups","__typename","isDeleted","isActive",
            "permissions","settings","config","metadata","createdAt",
        }

        hex_id_match = re.search(r'\b([0-9a-fA-F]{24})\b', q)
        if hex_id_match:
            hex_id = hex_id_match.group(1)
            # Skip if analytics question — let Step 2 handle it
            if _ANALYTICS_KW.search(question) and not _LOOKUP_KW.search(question):
                return None

            obj_id = ObjectId(hex_id)
            col_map = {"icompany":"ICompany","company":"ICompany",
                       "voucher":"Voucher","invoice":"Voucher",
                       "item":"Item","product":"Item",
                       "branch":"IBranch","ibranch":"IBranch",
                       "user":"IUser","iuser":"IUser",
                       "business":"Business","contact":"Contact","account":"Account"}
            q_low = q.lower()
            col   = "ICompany"
            for kw, c in col_map.items():
                if kw in q_low: col = c; break

            try:
                doc = db[col].find_one({"_id": obj_id})
                if not doc:
                    for try_col in ["ICompany","Voucher","IBranch","Item","IUser","Business","Account"]:
                        if try_col == col: continue
                        doc = db[try_col].find_one({"_id": obj_id})
                        if doc: col = try_col; break
                if doc:
                    doc         = deep_sanitize(doc)
                    field_lines = "\n".join(
                        f"• **{k}**: {v}" for k, v in doc.items()
                        if k not in SKIP_FIELDS
                        and v not in (None,"",[],"null")
                        and not isinstance(v, (list,dict))
                    )
                    answer = f"✅ **{col}** record found for id `{hex_id}`:\n\n{field_lines}"
                    plan   = {"query_type":"find","collection":col,
                              "answer_template":f"Found {col} record.",
                              "chart_suggestion":{"type":"none"},"clarification_needed":False}
                    return {"type":"answer","answer":answer,"results":[doc],
                            "chart":None,"plan":plan,"db_error":None}
                else:
                    return {"type":"answer",
                            "answer":f"❌ No document found with `_id = {hex_id}` in any collection.",
                            "results":[],"chart":None,"plan":{},"db_error":None}
            except Exception:
                pass

        # Collection-explicit queries
        col_explicit = None
        for kw, cn in [("icompany","ICompany"),("ibranch","IBranch"),("iuser","IUser"),
                        ("voucher","Voucher"),("item quantitytracker","ItemQuantityTracker"),
                        ("itemquantitytracker","ItemQuantityTracker"),
                        ("item","Item"),("business","Business"),
                        ("account","Account"),("contact","Contact")]:
            if kw in q.lower(): col_explicit = cn; break

        if col_explicit and re.search(r"(find|search|get|show|list|fetch|what|which|name|all)", q.lower()):
            proj = None
            fq   = {}
            if re.search(r"all|list|show all", q.lower()):
                if col_explicit == "ICompany":   proj = {"_id":0,"name":1,"industry":1,"financialYear":1}
                elif col_explicit == "IBranch":  proj = {"_id":0,"name":1,"city":1,"state":1,"code":1}
                elif col_explicit == "IUser":    proj = {"_id":0,"name":1,"phone":1,"lastSignIn":1}
                elif col_explicit == "Item":
                    proj = {"_id":0,"name":1,"availableQty":1,"unit":1}
                    fq   = {"isHidden":False}
                try:
                    rows  = find(db, col_explicit, fq, proj, limit=100)
                    plan  = {"query_type":"find","collection":col_explicit,
                             "answer_template":f"Records from {col_explicit}.",
                             "chart_suggestion":{"type":"table","x_field":"name",
                                                 "y_field":None,"title":f"{col_explicit} Records"},
                             "clarification_needed":False}
                    answer = self._answer(question, plan, rows, None)
                    chart  = self._chart(rows, plan["chart_suggestion"])
                    return {"type":"answer","answer":answer,"results":rows,
                            "chart":chart,"plan":plan,"db_error":None}
                except Exception:
                    pass

        return None

    def _answer_company_ranking(self, results: List, question: str) -> str:
        if not results: return "No company data found."
        q  = question.lower()
        nm = re.search(r"top\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)", q)
        word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                    "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        if nm:
            w      = nm.group(1)
            n_show = int(w) if w.isdigit() else word_map.get(w, 3)
        else:
            n_show = 3
        top            = results[:n_show]
        total_vouchers = sum(r.get("voucher_count",0) for r in results)

        def fmt_amt(amt):
            try: amt = float(amt or 0)
            except: amt = 0.0
            if amt <= 0:            return "₹0"
            if amt >= 1_00_00_000:  return f"₹{amt/1_00_00_000:.2f} crore"
            if amt >= 1_00_000:     return f"₹{amt/1_00_000:.2f} lakh"
            return f"₹{amt:,.0f}"

        lines = [f"**#{i} {r.get('company','Unknown')}** — "
                 f"{int(r.get('voucher_count',0)):,} vouchers "
                 f"({fmt_amt(r.get('total_amount',0))} revenue)"
                 for i, r in enumerate(top, 1)]
        top1 = top[0]
        return (f"**Top {n_show} companies by sales vouchers:**\n\n" + "\n".join(lines)
                + f"\n\n**{top1['company']}** leads with **{int(top1['voucher_count']):,} vouchers** "
                + f"and {fmt_amt(top1.get('total_amount',0))} in sales. "
                + f"Total across all {len(results)} companies: **{total_vouchers:,} vouchers**.")

    def _answer(self, question, plan, results, err, company=None):
        if err:
            return f"⚠️ **Database error:** `{err}`"
        if plan.get("query_type") == "none":
            return plan.get("answer_template","No data found.")

        if not results:
            if company:
                total = company.get("total_vouchers",0)
                nm    = company["name"]
                sc    = company.get("score",1.0)
                tag   = "" if sc > 0.85 else " *(closest match)*"
                if total == 0:
                    return (f"**{nm}**{tag} is in the database but has **zero vouchers** — test account.\n\n"
                            f"**Companies with real data:**\n"
                            f"• M/S DIPSHI - ESTIMATE → 40,255 | HIRAKA JEWELS → 34,998\n"
                            f"• Bhakti Parshwanath → 30,707 | NAMO-ESTIMATE → 27,478")
                return (f"**{nm}**{tag} has {total:,} total vouchers, but none matched this filter.\n\n"
                        f"Try: sales, purchases, customers, trend, revenue, stock")
            return "**No records found.** Query ran but matched 0 documents."

        co      = f" for **{company['name']}**" if company else ""
        sc_note = (f"\n*(matched company: {company['name']})*"
                   if company and company.get("score",1.0) < 0.85 else "")

        MONEY_KEYS = {"amount","revenue","total","price","value","sales","due","paid",
                      "balance","bill","final","cost","tax","discount","subtotal",
                      "net","gross","fee","charge","credit","debit"}
        COUNT_KEYS = {"count","qty","quantity","voucher","order","invoice",
                      "unit","number","num","no","record","item","stock"}

        def fmt_inr(v):
            try: v = float(v or 0)
            except: return str(v)
            if v <= 0:           return "₹0"
            if v >= 1_00_00_000: return f"₹{v/1_00_00_000:,.2f} crore"
            if v >= 1_00_000:    return f"₹{v/1_00_000:,.2f} lakh"
            if v >= 1_000:       return f"₹{v:,.0f}"
            return f"₹{v:.2f}"

        def is_money(fn): return any(k in fn.lower() for k in MONEY_KEYS)
        def is_count(fn): return any(k in fn.lower() for k in COUNT_KEYS)

        def fmt_row(row: dict) -> dict:
            out = {}
            for k, v in row.items():
                if v in (None,"",[],"null",{}): continue
                if isinstance(v, (int,float)):
                    if is_count(k) and not is_money(k.replace("total_","").replace("_total","")):
                        out[k] = f"{int(v):,}"
                    elif is_money(k): out[k] = fmt_inr(v)
                    elif is_count(k): out[k] = f"{int(v):,}"
                    else:             out[k] = v
                else:
                    out[k] = v
            return out

        if len(results) == 1:
            row   = fmt_row(results[0])
            parts = [f"**{k}**: {v}" for k, v in row.items()]
            if parts:
                co_label = f" for **{company['name']}**" if company else ""
                return f"Result{co_label}:\n\n" + "\n".join(f"• {p}" for p in parts)

        formatted_preview = [fmt_row(r) for r in results[:10]]
        prompt = (
            f"Invock ERP analyst. Question: {question}{sc_note}\n"
            f"Company: {company['name'] if company else 'all companies'}\n\n"
            f"Data ({len(results)} records){co} — ALL amounts already formatted in ₹:\n"
            f"{json.dumps(formatted_preview, default=str, indent=2)}\n\n"
            f"STRICT RULES:\n"
            f"1. Copy amounts EXACTLY as shown\n"
            f"2. NEVER re-convert or re-scale any number\n"
            f"3. Name specific customers, products, voucher numbers from data\n"
            f"4. 2-3 sentences max. One business insight. No invented numbers."
        )
        try: return self.llm.invoke([HumanMessage(content=prompt)]).content
        except: return f"Found {len(results)} record(s){co}."

    def _chart(self, results, suggestion):
        if not results or not suggestion or suggestion.get("type") in ("none",None):
            return None
        try:
            clean_rows = []
            for doc in results:
                row = {}
                for k, v in doc.items():
                    if isinstance(v, bool):                     row[k] = str(v)
                    elif isinstance(v, (int,float,type(None))): row[k] = v
                    elif isinstance(v, str):                    row[k] = v
                    elif isinstance(v, dict):                   row[k] = str(v)
                    elif isinstance(v, list):                   row[k] = len(v)
                    else:                                       row[k] = str(v)
                clean_rows.append(row)
            df = pd.DataFrame(clean_rows)
            if df.empty: return None
            for c in df.columns:
                try: df[c] = pd.to_numeric(df[c])
                except: pass
            num = df.select_dtypes(include="number").columns.tolist()
            cat = df.select_dtypes(exclude="number").columns.tolist()
            x   = suggestion.get("x_field")
            y   = suggestion.get("y_field")
            if not x or x not in df.columns: x = cat[0] if cat else (df.columns[0] if len(df.columns) else None)
            if not y or y not in df.columns: y = num[0] if num else (df.columns[1] if len(df.columns)>1 else None)
            return {"type":suggestion.get("type","bar"),"df":df,"x":x,"y":y,
                    "title":suggestion.get("title","Results")}
        except: return None
