"""
MongoDB AI Agent — Invock ERP  v4.4  (production-grade, fully direct-query)

Root causes fixed in v4.4:
  1. ALL collections use $in:[ObjectId,string] for iCompanyId — no format guessing
  2. total_revenue for a specific company queries ALL years (not just current)
     Current year filter only applies to GLOBAL (no company) queries
  3. Business collection also uses $in filter
  4. Item collection also uses $in filter
  5. resolve_company() now returns both obj_id and str_id always
"""
AGENT_VERSION = "4.22"
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
            return uri
        user, pwd, host, db_part, _ = m.groups()
        db_part = db_part or DB_NAME
        print(f"[DoH] Resolving SRV for {host}...")
        req = urllib.request.Request(
            f"https://cloudflare-dns.com/dns-query?name=_mongodb._tcp.{host}&type=SRV",
            headers={"accept":"application/dns-json","User-Agent":"Python/3"})
        resp = _json.loads(urllib.request.urlopen(req, timeout=15).read())
        if not resp.get("Answer"):
            req2 = urllib.request.Request(
                f"https://dns.google/resolve?name=_mongodb._tcp.{host}&type=SRV",
                headers={"User-Agent":"Python/3"})
            resp = _json.loads(urllib.request.urlopen(req2, timeout=15).read())
        if not resp.get("Answer"): return uri
        hosts = []
        for ans in resp["Answer"]:
            if ans.get("type") == 33:
                parts = str(ans["data"]).split()
                if len(parts) == 4:
                    hosts.append(f"{parts[3].rstrip('.')}:{parts[2]}")
        if not hosts: return uri
        print(f"[DoH] Found {len(hosts)} shard(s)")
        rs_name = None
        try:
            req_txt = urllib.request.Request(
                f"https://cloudflare-dns.com/dns-query?name={host}&type=TXT",
                headers={"accept":"application/dns-json","User-Agent":"Python/3"})
            txt_resp = _json.loads(urllib.request.urlopen(req_txt, timeout=10).read())
            for ans in (txt_resp.get("Answer") or []):
                if ans.get("type") == 16:
                    txt = str(ans["data"]).strip('"').strip("'")
                    if "replicaSet=" in txt:
                        rs_name = txt.split("replicaSet=")[1].split("&")[0].strip()
                        break
        except: pass
        params = "tls=true&authSource=admin&tlsAllowInvalidCertificates=true"
        if rs_name: params += f"&replicaSet={rs_name}"
        return f"mongodb://{user}:{pwd}@{','.join(hosts)}/{db_part}?{params}"
    except Exception as e:
        print(f"[DoH] Failed: {e}")
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
    if isinstance(obj, dict):     return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):     return [deep_sanitize(i) for i in obj]
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
    return {"now":now,
            "last_month_start":lms,"last_month_end":lme,"this_month_start":fm,
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

# ═══════════════════════════ Universal iCompanyId Filter ═════════════════════

def cid_filter(obj_id, str_id) -> dict:
    """
    THE ONE TRUE FILTER for iCompanyId across ALL collections.

    CRITICAL: Always include BOTH ObjectId and string forms.
    str(ObjectId('651...')) == '651...' is always True — so do NOT use
    an inequality check to decide whether to add the string.
    Both must be explicitly added so MongoDB matches regardless of
    how iCompanyId was stored (ObjectId or plain string).

    Works for Voucher, ItemQuantityTracker, Business, Item, Account.
    """
    if obj_id is None and str_id is None:
        return {}
    vals = []
    if obj_id is not None:
        vals.append(obj_id)   # ObjectId form — matches ObjectId-stored docs
    if str_id is not None:
        vals.append(str_id)   # string form  — ALWAYS add, matches string-stored docs
    return {"iCompanyId": {"$in": vals}}

# ═══════════════════════════ Company Resolver ═════════════════════════════════

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())).strip()

def clean(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def tri(s):
    return set(s[i:i+3] for i in range(len(s)-2))

def fuzzy(query, candidate):
    q, c = norm(query), norm(candidate)
    if not q or not c: return 0.0
    if q == c: return 1.0
    if q in c or c in q: return 0.93
    qc, cc = clean(query), clean(candidate)
    if qc == cc: return 0.98
    if qc in cc or cc in qc: return 0.91
    tq, tc   = tri(qc), tri(cc)
    tri_sc   = len(tq & tc) / max(len(tq | tc), 1) if tq and tc else 0.0
    qt = {t for t in q.split() if len(t) > 1}
    ct = {t for t in c.split() if len(t) > 1}
    tok_sc   = len(qt & ct) / max(len(qt), len(ct)) if qt and ct else 0.0
    pre_sc   = sum(1 for qw in qt if any(cw.startswith(qw) or qw.startswith(cw)
                                          for cw in ct)) / max(len(qt), 1) if qt else 0.0
    penalty  = min(len(ct - qt) * 0.06, 0.25) if qt and ct else 0.0
    return max(0.0, min(tri_sc*0.50 + tok_sc*0.30 + pre_sc*0.20 - penalty, 1.0))

def _make_company_dict(obj_id, name, total, score):
    """Always return both ObjectId and string forms — cid_filter needs both."""
    str_id = str(obj_id)
    return {"_id_obj": obj_id, "_id_str": str_id,
            "name": name, "total_vouchers": total, "score": score}

def resolve_company(client, name: str) -> Optional[Dict]:
    db = get_db(client)
    all_cos = list(db["ICompany"].find({}, {"_id":1,"name":1}))
    if not all_cos: return None
    scored  = sorted([(fuzzy(name, d.get("name","")), d) for d in all_cos], key=lambda x: -x[0])
    best_sc, best = scored[0]
    print(f"[Fuzzy] '{name}' → top3: {[(round(s,3),d['name']) for s,d in scored[:3]]}")
    if best_sc < 0.15: return None
    obj_id = best["_id"]
    str_id = str(obj_id)
    # Count vouchers using $in — same as query filter — to get accurate total
    n = db["Voucher"].count_documents(
        {"iCompanyId": {"$in": [obj_id, str_id]}}, maxTimeMS=5000)
    print(f"[Company] '{best['name']}' total_vouchers={n}")
    return _make_company_dict(obj_id, best["name"], n, best_sc)

def resolve_company_by_hex(client, hex_id: str) -> Optional[Dict]:
    try:
        db     = get_db(client)
        obj_id = ObjectId(hex_id)
        str_id = hex_id
        doc    = db["ICompany"].find_one({"_id": obj_id}, {"_id":0,"name":1})
        name   = doc.get("name", f"Company {hex_id[:8]}...") if doc else f"Company {hex_id[:8]}..."
        n = db["Voucher"].count_documents(
            {"iCompanyId": {"$in": [obj_id, str_id]}}, maxTimeMS=5000)
        print(f"[HexID] Resolved '{name}' total_vouchers={n}")
        return _make_company_dict(obj_id, name, n, 1.0)
    except Exception as e:
        print(f"[HexID] Failed: {e}")
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
        r"does\s+([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)\s+have",
        r"(?:customers?|suppliers?|vouchers?|sales?|stock|revenue|trend|products?)\s+(?:does|of|for)\s+([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)(?:\s*\?|$)",
        r"([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)\s+(?:has|have|had)\s+(?:how many|\d)",
        r"(?:show|get|give|tell)\s+(?:me\s+)?(?:the\s+)?(?:sales?|revenue|customers?|vouchers?|trend|stock|purchases?)\s+(?:of|for)\s+([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)\s*(?:\?|$)",
    ]
    stopwords = {"company","the","a","an","in","for","of","with","has","have","me","my",
                 "all","this","that","these","those","its","their","collection","icompany",
                 "ibranch","iuser","voucher","item","id","search","find","get","show",
                 "list","fetch","what","which"}
    generic   = {"sales","purchase","voucher","revenue","data","record","item","companies",
                 "trend","customer","invoice","monthly","total","how","many","what","most",
                 "collection","icompany","ibranch","search","find","created","highest","top",
                 "list","show","all","ranked","best","number","count","maximum"}
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

# ═══════════════════════════ Query Builder ════════════════════════════════════

class Q:
    """
    Query builder. Uses cid_filter() for ALL collections.
    cid_filter generates $in:[ObjectId,string] — format-agnostic.
    """

    def __init__(self, db, company: Optional[Dict] = None):
        self.db      = db
        self.company = company
        self.obj_id  = company["_id_obj"] if company else None
        self.str_id  = company["_id_str"] if company else None

    def _cf(self) -> dict:
        """Company filter using $in — works regardless of storage format."""
        return cid_filter(self.obj_id, self.str_id)

    def _mf(self, base: dict) -> dict:
        """Merge company filter into base match dict."""
        base.update(self._cf())
        return base

    # ── Voucher queries ───────────────────────────────────────────────────────

    def voucher_count(self, vtype=None, name="Company"):
        mf = self._mf({"type": vtype, "billFinalAmount":{"$gt":0}} if vtype else {"billFinalAmount":{"$gt":0}})
        rows = agg(self.db, "Voucher", [
            {"$match": mf},
            {"$group": {"_id":None,"total_vouchers":{"$sum":1},
                        "total_amount":{"$sum":"$billFinalAmount"}}},
            {"$project":{"_id":0,"total_vouchers":1,"total_amount":1}}
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
            {"$match": self._mf({"type":"sales","party.name":{"$ne":None},"billFinalAmount":{"$gt":0}})},
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
            {"$match": self._mf({"type":{"$in":["sales","purchase"]},"billFinalAmount":{"$gt":0}})},
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

    # ── ItemQuantityTracker queries ───────────────────────────────────────────

    def monthly_trend(self, years=None, name="Company"):
        d   = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        if self.company:
            # Company-specific: use Voucher.issueDate for exact accuracy
            # This matches MongoDB Compass results precisely
            import datetime as _dt
            _start = _dt.datetime(min(yrs), 1, 1)
            _end   = _dt.datetime(max(yrs)+1, 1, 1)
            rows = agg(self.db,"Voucher",[
                {"$match": self._mf({"type":"sales","billFinalAmount":{"$gt":0},
                                     "issueDate":{"$gte":_start,"$lt":_end}})},
                {"$group": {"_id":{"year":{"$year":"$issueDate"},
                                   "month":{"$month":"$issueDate"}},
                            "amount":{"$sum":"$billFinalAmount"},
                            "count":{"$sum":1}}},
                {"$sort":{"_id.year":1,"_id.month":1}},
                {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month",
                             "amount":1,"count":1}}
            ])
        else:
            # Global: use IQT (faster for all-company aggregation)
            rows = agg(self.db,"ItemQuantityTracker",[
                {"$match": self._mf({"voucherType":"sales","year":{"$in":yrs}})},
                {"$group": {"_id":{"year":"$year","month":"$month"},
                            "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                {"$sort":{"_id.year":1,"_id.month":1}},
                {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1,"qty":1}}
            ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Sales Trend"}

    def total_revenue(self, year=None, name="Company"):
        d = get_dates()
        if self.company:
            # Company-specific: use Voucher.billFinalAmount — this is the
            # FINAL billed amount (includes taxes, charges) — most accurate
            # and consistent with what Invock ERP shows on reports
            mf = self._mf({"type":"sales","billFinalAmount":{"$gt":0}})
            rows = agg(self.db,"Voucher",[
                {"$match": mf},
                {"$group": {"_id":None,
                            "total_revenue":{"$sum":"$billFinalAmount"},
                            "total_vouchers":{"$sum":1}}},
                {"$project":{"_id":0,"total_revenue":1,"total_vouchers":1}}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_revenue",
                          "title":f"{name} — Total Sales Revenue"}
        else:
            # Global query: use ItemQuantityTracker with year filter
            match = {"voucherType":"sales","year": year or d["ty"]}
            rows = agg(self.db,"ItemQuantityTracker",[
                {"$match": self._mf(match)},
                {"$group": {"_id":None,"total_revenue":{"$sum":"$amount"},"total_qty":{"$sum":"$qty"}}},
                {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
            ])
            label = str(year or d["ty"])
            return rows, {"type":"metric","x_field":None,"y_field":"total_revenue",
                          "title":f"{name} — Revenue ({label})"}

    def top_products(self, by="amount", limit=15, name="Company"):
        d  = get_dates()
        yf = "amount" if by == "amount" else "qty"
        # For company queries: last 2 years. For global: current year only.
        match = {"voucherType":"sales"}
        if self.company:
            match["year"] = {"$in": [d["ty"]-1, d["ty"]]}
        else:
            match["year"] = d["ty"]
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._mf(match)},
            {"$group": {"_id":"$itemId","amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
            {"$sort":{yf:-1}},{"$limit":limit},
            {"$project":{"_id":0,"item":"$_id","amount":1,"qty":1}}
        ])
        return rows, {"type":"bar","x_field":"item","y_field":yf,
                      "title":f"{name} — Top {limit} Products"}

    def purchase_trend(self, years=None, name="Company"):
        d   = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        rows = agg(self.db,"ItemQuantityTracker",[
            {"$match": self._mf({"voucherType":"purchase","year":{"$in":yrs}})},
            {"$group": {"_id":{"year":"$year","month":"$month"},"amount":{"$sum":"$amount"}}},
            {"$sort":{"_id.year":1,"_id.month":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1}}
        ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Purchase Trend"}

    # ── Item / Stock ──────────────────────────────────────────────────────────

    def stock(self, name="Company"):
        rows = find(self.db,"Item",self._mf({"isHidden":False,"availableQty":{"$gt":0}}),
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",-1)],limit=100)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Stock / Inventory"}

    def low_stock(self, threshold=10, name="Company"):
        rows = find(self.db,"Item",
            self._mf({"isHidden":False,"availableQty":{"$gt":0,"$lte":threshold}}),
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",1)],limit=50)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Low Stock (≤{threshold})"}

    # ── Business ──────────────────────────────────────────────────────────────

    def customer_list(self, name="Company"):
        rows = find(self.db,"Business",
            self._mf({"relationType":{"$in":["customer","both"]}}),
            {"_id":0,"name":1,"city":1,"state":1,"relationType":1},
            sort=[("name",1)],limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Customer List"}

    def supplier_list(self, name="Company"):
        rows = find(self.db,"Business",
            self._mf({"relationType":{"$in":["supplier","both"]}}),
            {"_id":0,"name":1,"city":1,"state":1},
            sort=[("name",1)],limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Supplier List"}


# ═══════════════════════════ Intent Router ════════════════════════════════════

def route(question: str, company: Optional[Dict], db) -> Optional[Tuple]:
    q    = question.lower().strip()
    n    = company["name"] if company else "All Companies"
    qb   = Q(db, company)
    d    = get_dates()
    has  = lambda *ws: any(w in q for w in ws)
    miss = lambda *ws: not any(w in q for w in ws)

    # ── Company ranking ───────────────────────────────────────────────────────
    if re.search(r"(companies|company).*(most|top|highest|ranked?|maximum|max|list).*(voucher|sales|invoice)|"
                 r"(most|top|highest).*(voucher|sales).*(compan)|"
                 r"which compan.*(most|highest|top).*(voucher|sales)|"
                 r"rank.*compan.*voucher|list compan.*voucher|compan.*most.*created|"
                 r"top\s+\d*\s*compan.*(sales|voucher)|compan.*by.*(sales|voucher)|"
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

    # ── Sales for specific month+year ────────────────────────────────────────
    month_names = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
                   "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
                   "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
                   "sep":9,"oct":10,"nov":11,"dec":12,
                   # common typos / alternate spellings
                   "januray":1,"januaray":1,"janaury":1,
                   "februray":2,"februvary":2,"februrary":2,"febuary":2,"feburary":2,
                   "marchh":3,"aprrl":4,"appril":4,
                   "septembar":9,"setember":9,"septmber":9,
                   "octomber":10,"octobar":10,"novembar":11,"decembar":12}
    # Build regex from all known month keys
    _all_months = "|".join(sorted(month_names.keys(), key=len, reverse=True))
    _month_pat = re.search(rf"\b({_all_months})\b", q)
    _year_pat  = re.search(r"\b(20\d{2})\b", q)
    # Fallback: if no month matched, try fuzzy match for common typos
    if not _month_pat and _year_pat:
        import difflib as _dl
        _words = q.split()
        _all_month_names = list(month_names.keys())
        for _w in _words:
            if len(_w) >= 3:
                _close = _dl.get_close_matches(_w, _all_month_names, n=1, cutoff=0.75)
                if _close:
                    class _FakeMatch:
                        def group(self, n): return _close[0]
                    _month_pat = _FakeMatch()
                    break
    if _month_pat and _year_pat:
        _m = month_names.get(_month_pat.group(1).lower())
        _y = int(_year_pat.group(1))
        print(f"[Route] Month={_m} Year={_y} q='{q[:60]}'")
        if _m and _y:
            import calendar as _cal
            _start = datetime(_y, _m, 1)
            _end   = datetime(_y, _m, _cal.monthrange(_y, _m)[1], 23, 59, 59)
            print(f"[Route] Date range: {_start} to {_end}")
            _has_list    = has("list","show","give","all")
            _has_voucher = has("voucher","invoice","bill")
            print(f"[Route] has_list={_has_list} has_voucher={_has_voucher}")
            if _has_list and _has_voucher:
                # List individual vouchers for this month
                _vt = "purchase" if has("purchase") else "sales"
                mf = qb._mf({"type":_vt,
                             "billFinalAmount":{"$gt":0},
                             "issueDate":{"$gte":_start,"$lte":_end}})
                try:
                    rows = find(db,"Voucher",mf,
                        proj={"_id":0,"voucherNo":1,"issueDate":1,"billFinalAmount":1,
                              "dueAmount":1,"status":1,"party.name":1},
                        sort=[("issueDate",1)],limit=100)
                    print(f"[Route] list-voucher returned {len(rows)} rows")
                    if rows:
                        print(f"[Route] First row: {rows[0]}")
                except Exception as _e:
                    print(f"[Route] list-voucher ERROR: {_e}")
                    rows = []
                mon_label = _cal.month_name[_m]
                return rows, {"type":"table","x_field":"voucherNo","y_field":"billFinalAmount",
                              "title":f"{n} — Vouchers {mon_label} {_y}"}
            if has("sales","revenue","voucher","amount","total","how many","count"):
                # Use Voucher with date range for accuracy
                mf = qb._mf({"type":"sales",
                             "issueDate":{"$gte":_start,"$lte":_end}})
                rows = agg(db,"Voucher",[
                    {"$match": mf},
                    {"$group":{"_id":None,
                               "total_revenue":{"$sum":"$billFinalAmount"},
                               "total_vouchers":{"$sum":1}}},
                    {"$project":{"_id":0,"total_revenue":1,"total_vouchers":1}}
                ])
                mon_label = _cal.month_name[_m]
                return rows, {"type":"metric","x_field":None,"y_field":"total_revenue",
                              "title":f"{n} — Sales {mon_label} {_y}"}
            if has("purchase"):
                mf = qb._mf({"type":"purchase",
                             "issueDate":{"$gte":_start,"$lte":_end}})
                rows = agg(db,"Voucher",[
                    {"$match": mf},
                    {"$group":{"_id":None,
                               "total_purchases":{"$sum":"$billFinalAmount"},
                               "total_vouchers":{"$sum":1}}},
                    {"$project":{"_id":0,"total_purchases":1,"total_vouchers":1}}
                ])
                mon_label = _cal.month_name[_m]
                return rows, {"type":"metric","x_field":None,"y_field":"total_purchases",
                              "title":f"{n} — Purchases {mon_label} {_y}"}

    # ── Revenue / Sales total ─────────────────────────────────────────────────
    if re.search(r"total.*(revenue|sales|amount)|revenue.*total|(sales|revenue).*this year|ytd|year.*to.*date", q):
        yr = d["ty"] if has("this year","ytd","year to date",str(d["ty"])) else None
        return qb.total_revenue(yr, n)

    # ── Purchase total ────────────────────────────────────────────────────────
    if re.search(r"total.*purchase|purchase.*total|purchase.*this year", q) and miss("voucher","count"):
        yr    = d["ty"] if has("this year",str(d["ty"])) else None
        match = {"voucherType":"purchase"}
        if yr: match["year"] = yr
        rows = agg(db,"ItemQuantityTracker",[
            {"$match": qb._mf(match)},
            {"$group": {"_id":None,"total_purchases":{"$sum":"$amount"}}},
            {"$project":{"_id":0,"total_purchases":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"total_purchases",
                      "title":f"{n} — Total Purchases"}

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
        if re.search(r"how many|count|number", q):
            # Use Voucher party.name distinct — Voucher $in filter proven correct
            rows = agg(db,"Voucher",[
                {"$match": qb._mf({"type":"sales","party.name":{"$ne":None}})},
                {"$group": {"_id":"$party.name"}},
                {"$count": "total_customers"}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_customers",
                          "title":f"{n} — Total Unique Customers"}
        return qb.customer_list(n)

    # ── Supplier list / count ─────────────────────────────────────────────────
    if re.search(r"list.*supplier|show.*supplier|all.*supplier|supplier.*list|"
                 r"how many.*supplier|count.*supplier", q):
        if re.search(r"how many|count|number", q):
            cf = qb._cf()
            match = {"relationType":{"$in":["supplier","both"]}}
            match.update(cf)
            rows = agg(db,"Business",[{"$match":match},{"$count":"total_suppliers"}])
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
        return plan("find","IBranch",fq={},
            proj={"_id":0,"name":1,"city":1,"state":1,"code":1},
            limit=300,tmpl="All branches.",ct="table",title="All Branches")

    if re.search(r"\buser|users|staff|employee\b", q) and miss("company with","in company","sales","voucher"):
        return plan("find","IUser",fq={},
            proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},
            limit=500,tmpl="All users.",ct="table",title="All Users")

    if re.search(r"^(list |show |get )?(all )?compan", q) and miss("with","in company","for","sales","voucher","most","top","rank"):
        return plan("find","ICompany",fq={},
            proj={"_id":0,"name":1,"industry":1,"financialYear":1},
            limit=200,tmpl="All companies.",ct="table",title="All Companies")

    if re.search(r"how many compan|no of compan|number of compan|count.*compan|compan.*count|"
                  r"show.*compan|total.*compan|list.*how many|compan.*total", q) and        miss("with","in","for","sales","voucher","most","top","rank","customer","supplier"):
        return plan("aggregate","ICompany",
            pipe=[{"$count":"total_companies"}],
            tmpl="Total companies.",ct="metric",y="total_companies",
            title="Total Companies")

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
            tmpl=f"Top 15 products.",ct="bar",x="item",y="revenue",
            title=f"Top Products by Revenue ({d['ty']})")

    if re.search(r"unpaid|outstanding|overdue", q) and miss("company","with","in"):
        return plan("find","Voucher",
            fq={"status":"unpaid","iCompanyId":{"$ne":None}},
            proj={"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,"issueDate":1},
            sort={"dueAmount":-1},limit=50,
            tmpl="Unpaid invoices.",ct="table",title="Unpaid Invoices")

    if re.search(r"stock|inventory|available.*qty", q) and miss("company","with","in"):
        return plan("find","Item",
            fq={"isHidden":False,"availableQty":{"$gt":0}},
            proj={"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort={"availableQty":-1},limit=100,
            tmpl="Current stock.",ct="table",title="Stock / Inventory")

    if re.search(r"avg.*order|average.*order|aov", q) and miss("company","with","in"):
        return plan("aggregate","Voucher",
            pipe=[{"$match":{"type":"sales","iCompanyId":{"$ne":None}}},
                  {"$group":{"_id":None,"avg_order_value":{"$avg":"$billFinalAmount"},"orders":{"$sum":1}}},
                  {"$project":{"_id":0,"avg_order_value":1,"orders":1}}],
            tmpl="Average order value.",ct="metric",y="avg_order_value",
            title="Average Order Value (All Companies)")

    # ── List vouchers with date filter ────────────────────────────────────────
    month_map = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
                 "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
                 "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
                 "sep":9,"oct":10,"nov":11,"dec":12}
    _mp = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", q)
    _yp = re.search(r"\b(20\d{2})\b", q)
    if re.search(r"list.*voucher|voucher.*list|show.*voucher|give.*voucher|all.*voucher|voucher.*in", q) and _mp and _yp:
        import datetime as _dt, calendar as _cal
        _m = month_map.get(_mp.group(1).lower())
        _y = int(_yp.group(1))
        if _m and _y:
            _vtype = "purchase" if "purchase" in q else "sales"
            _start = _dt.datetime(_y, _m, 1)
            _end   = _dt.datetime(_y, _m, _cal.monthrange(_y, _m)[1], 23, 59, 59)
            return plan("find","Voucher",
                fq={"type":_vtype,
                    "billFinalAmount":{"$gt":0},
                    "issueDate":{"$gte":_start,"$lte":_end}},
                proj={"_id":0,"voucherNo":1,"issueDate":1,"billFinalAmount":1,
                      "dueAmount":1,"status":1,"party.name":1},
                sort={"issueDate":1},limit=100,
                tmpl=f"{_vtype.title()} vouchers for {_cal.month_name[_m]} {_y}.",
                ct="table",x="voucherNo",y="billFinalAmount",
                title=f"{_vtype.title()} Vouchers — {_cal.month_name[_m]} {_y}")

    if re.search(r"how many (customer|supplier|client)", q) and miss("company","with","in"):
        rel   = "customer" if "customer" in q or "client" in q else "supplier"
        field = f"total_{rel}s"
        return plan("aggregate","Business",
            pipe=[{"$match":{"relationType":rel}},{"$count":field}],
            tmpl=f"Total {rel}s.",ct="metric",y=field,title=f"Total {rel.title()}s")

    return None


# ═══════════════════════════ LLM Fallback ════════════════════════════════════

SCHEMA_TEXT = """
DATABASE: dev-cluster — Invock ERP — Jewellery business, India, amounts in ₹ INR (rupees, NOT paise)

=== VOUCHER collection (1.3M documents) ===
FILTER FIELDS:
  type: string — EXACT values: "sales" | "purchase" | "receipt" | "payment"
  status: string — EXACT values: "unpaid" | "paid" | "partial"
  iCompanyId: ObjectId (e.g. ObjectId("651ea989a7dc3e26bda36036")) — CONFIRMED ObjectId
    Confirmed from real document: iCompanyId: ObjectId('651ea989a7dc3e26bda36036')
  billFinalAmount: USE THIS for all revenue. Zero-amount docs are drafts/test vouchers.
    Always add billFinalAmount:{$gt:0} to exclude drafts.
  lineAmountSum: pre-discount item total (slightly less than billFinalAmount)
    This matches ItemQuantityTracker.amount — do NOT use for final revenue
  issueDate: ISODate object (e.g. ISODate("2024-02-15T04:53:16.000Z"))
  isHidden: boolean — add {isHidden:false} or ignore hidden vouchers
  iBranchId: ObjectId

AMOUNT FIELDS (all in ₹ rupees):
  billFinalAmount: final invoice amount INCLUDING tax — USE THIS for revenue totals
  billItemsPrice: pre-discount item total
  billAmountBeforeTax: amount before tax
  billTaxAmount: tax amount only
  dueAmount: amount still unpaid
  paidAmount: amount already paid
  lineAmountSum: sum of line items (pre-tax)
  lineItemQtySum: total quantity across all items

OTHER FIELDS:
  voucherNo: string (e.g. "SA/2324/PL/1")
  party.name: string — customer/supplier name (may have spaces e.g. " AASIF ")
  party.state: string — customer state
  party.city: string — customer city
  narration: string — notes

DO NOT PROJECT: itemList, transactions, tax, voucherList, otherCharges, expenseList
  (these are large arrays that slow queries and inflate responses)

SAMPLE DOCUMENT:
  {type:"sales", billFinalAmount:11124, status:"unpaid", iCompanyId:ObjectId("651ea..."),
   issueDate:ISODate("2024-02-15"), party:{name:" AASIF ", state:"Kerala"},
   voucherNo:"SA/2324/PL/1", lineItemQtySum:20}

=== ITEMQUANTITYTRACKER collection (2.1M documents) ===
FILTER FIELDS:
  voucherType: string — "sales" | "purchase"
  year: integer (e.g. 2024, 2025) — ALWAYS filter by year, never sum all years
  month: integer 1-12
  iCompanyId: ObjectId (e.g. ObjectId("63a3eac5b03f790f14f2b201")) — ObjectId NOT string
  itemId: ObjectId
  iBranchId: ObjectId

AMOUNT FIELDS:
  amount: number in ₹ rupees — item-level sales amount
  qty: number — quantity sold/purchased

NOTE: amount here is item-level only, slightly lower than Voucher.billFinalAmount
  because it excludes service charges and freight added at invoice level.

SAMPLE DOCUMENTS (both confirmed from real data):
  Sales:   {month:10, year:2020, voucherType:"sales",    iCompanyId:ObjectId("5d92..."), qty:1512, amount:12105.78}
  Purchase:{month:8,  year:2024, voucherType:"purchase", iCompanyId:ObjectId("63a3..."), qty:4,    amount:2400}

NOTE: IQT has data from 2020 onwards. trippingValue format is YYYYFYYY (financial year) — do not use for filtering.
startDate/endDate fields exist but year+month integers are faster and more reliable for filtering.

=== ITEM collection (450K documents) ===
FILTER FIELDS:
  iCompanyId: ObjectId — ObjectId NOT string
  isHidden: boolean — ALWAYS add {isHidden:false} to exclude deleted items
  availableQty: number (can be negative if oversold)

OTHER FIELDS:
  name: string — item name
  skuBarcode: string — SKU code
  unit: string (e.g. "pcs", "pair", "gm")
  unitSellRetailPrice: number in ₹
  unitSellWholeSalePrice: number in ₹
  unitPurchasePrice: number in ₹

=== BUSINESS collection (45K documents) ===
FILTER FIELDS:
  relationType: string — "customer" | "supplier" | "both"
  iCompanyId: ObjectId OR null (mixed — some records have null)
  isHidden: boolean

OTHER FIELDS:
  name: string — business/party name
  city, state: strings
  gstin: string — GST number
  code: string — business code

=== ICOMPANY collection (141 documents) ===
  _id: ObjectId — this IS the company ID used in other collections
  name: string — company name
  industry: string (e.g. "imitation-jewellery", "computers")
  currencyUnit: "INR"
  No iCompanyId field — _id IS the identifier

=== KEY RULES FOR QUERY GENERATION ===
1. Voucher.iCompanyId is ObjectId — CONFIRMED from real data (string returns 0 docs)
2. ItemQuantityTracker.iCompanyId is ObjectId — filter with ObjectId
3. Item.iCompanyId is ObjectId — filter with ObjectId
4. ALWAYS use billFinalAmount for revenue from Voucher (not lineAmountSum)
5. ALWAYS filter by year in ItemQuantityTracker — never aggregate all years
6. NEVER project itemList, transactions, tax, voucherList arrays
7. party.name may have leading/trailing spaces — use $regex or $ne:null not exact match
8. Amounts are in RUPEES — do not divide by 100
9. issueDate is ISODate object — use $gte/$lte with ISODate for date ranges
10. For monthly data: use ItemQuantityTracker with year+month integer filters
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
1. iCompanyId format DIFFERS by collection:
   Voucher: STRING (e.g. "651ea989a7dc3e26bda36036") — agent injects this
   IQT/Item: ObjectId — agent injects this
   Never add iCompanyId yourself — agent always handles it
2. $sum/$avg must be {{"$sum":"$fieldName"}} (with $ prefix)
3. Every agg must end with $project removing _id
4. issueDate is Date object — use ISO strings, agent converts
5. Use ItemQuantityTracker for date/product queries (integer year/month)
6. Never project itemList/transactions/tax/party/voucherList
7. x_field/y_field = exact field names from your $project"""

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
        assert AGENT_VERSION == "4.22", f"Wrong agent version: {AGENT_VERSION}"

        if not self.llm and not self.init_llm():
            return {"error": "GROQ_API_KEY not configured."}

        q_low = question.lower().strip()
        db    = get_db(self.client) if self.client else None

        # ── Step 0: Direct lookup (non-analytics only) ────────────────────────
        direct = self._direct_id_or_collection_query(question, db)
        if direct:
            return direct

        # ── Step 1: Company ranking — MUST run before company resolution ────────
        # "top 5 companies by sales vouchers" contains "compan" which would
        # accidentally fuzzy-match a company name if resolution ran first.
        # Company ranking is a GLOBAL query — no company filter needed.
        _rp = (r"(companies|company).*(most|top|highest|ranked?|maximum|max|list).*(voucher|sales)|"
               r"(most|top|highest).*(voucher|sales).*(compan)|"
               r"top\s*\d*\s*compan.*(sales|voucher)|compan.*by.*(sales|voucher)|"
               r"show.*compan.*(sales|voucher)|compan.*with.*most.*(sales|voucher)")
        _has_co  = "compan" in q_low
        _has_met = any(w in q_low for w in ["voucher","sales","invoice"])
        _has_rnk = any(w in q_low for w in ["list","most","top","highest","rank","maximum","best"])
        if (bool(re.search(_rp, q_low)) or (_has_co and _has_met and _has_rnk)) and db is not None:
            _vtype = "purchase" if "purchase" in q_low else "sales"
            _nm    = re.search(r"top\s+(\d+)", q_low)
            _lim   = min(int(_nm.group(1)), 50) if _nm else 20
            results, chart_sug = companies_by_voucher_count(db, _vtype, _lim)
            results = [deep_sanitize(r) for r in results]
            answer  = self._answer_company_ranking(results, question)
            chart   = self._chart(results, chart_sug)
            plan    = {"query_type":"direct","collection":"Voucher",
                       "answer_template":"Companies ranked by vouchers.",
                       "chart_suggestion":chart_sug,"clarification_needed":False}
            self.history.append({"q": question, "a": plan["answer_template"]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":plan,"db_error":None}

        # ── Step 2: Resolve company (after ranking check) ────────────────────
        # Only runs if question is NOT a global ranking query.
        # FIX: removed aggressive fallback resolve_company(client, question)
        # that was fuzzy-matching company names from generic questions.
        company = None

        # 2a: Hex ID in question + analytics keywords → resolve directly
        hex_match = re.search(r'\b([0-9a-fA-F]{24})\b', question)
        if hex_match and self.client and _ANALYTICS_KW.search(question):
            company = resolve_company_by_hex(self.client, hex_match.group(1))

        # 2b: Fuzzy company name extraction from question
        if company is None:
            cname = extract_company_name(question)
            if cname and self.client:
                company = resolve_company(self.client, cname)
                # NOTE: removed fallback resolve_company(client, full_question)
                # that was accidentally matching generic words as company names
                if company is None:
                    return {
                        "type":"answer",
                        "answer":(f"❌ No company matching **\"{cname}\"** found.\n\n"
                                  f"Ask *\"list all companies\"* to see all available companies."),
                        "results":[],"chart":None,"plan":{},"db_error":None
                    }

        # ── Step 3: Global schema shortcuts (ONLY when no company detected) ───
        # If a company was found in Step 1, skip this entirely so we never
        # return unfiltered global data for a company-specific question.
        if company is None:
            sc = schema_shortcut(q_low)
            if sc:
                results, err = execute_plan(sc, db, self.date_type)
                answer = self._answer(question, sc, results, err)
                chart  = self._chart(results, sc["chart_suggestion"])
                self.history.append({"q": question, "a": sc["answer_template"][:80]})
                return {"type":"answer","answer":answer,"results":results,
                        "chart":chart,"plan":sc,"db_error":err}

        # ── Step 4: Intent router (company-aware) ─────────────────────────────
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

        # ── Step 5: LLM fallback ──────────────────────────────────────────────
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
                        and v not in (None,"",[],"null",{})
                        and not isinstance(v, (list,dict))
                    )
                    answer = f"✅ **{col}** record for id `{hex_id}`:\n\n{field_lines}"
                    plan   = {"query_type":"find","collection":col,
                              "answer_template":f"Found {col} record.",
                              "chart_suggestion":{"type":"none"},"clarification_needed":False}
                    return {"type":"answer","answer":answer,"results":[doc],
                            "chart":None,"plan":plan,"db_error":None}
                return {"type":"answer",
                        "answer":f"❌ No document found with `_id = {hex_id}`.",
                        "results":[],"chart":None,"plan":{},"db_error":None}
            except Exception:
                pass

        col_explicit = None
        for kw, cn in [("icompany","ICompany"),("ibranch","IBranch"),("iuser","IUser"),
                        ("voucher","Voucher"),("item quantitytracker","ItemQuantityTracker"),
                        ("itemquantitytracker","ItemQuantityTracker"),
                        ("item","Item"),("business","Business"),
                        ("account","Account"),("contact","Contact")]:
            if kw in q.lower(): col_explicit = cn; break

        if col_explicit and re.search(r"(find|search|get|show|list|fetch|what|which|name|all)", q.lower()):
            proj = None; fq = {}
            if re.search(r"all|list|show all", q.lower()):
                if col_explicit == "ICompany":  proj = {"_id":0,"name":1,"industry":1,"financialYear":1}
                elif col_explicit == "IBranch": proj = {"_id":0,"name":1,"city":1,"state":1,"code":1}
                elif col_explicit == "IUser":   proj = {"_id":0,"name":1,"phone":1,"lastSignIn":1}
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
                except Exception: pass

        return None

    def _answer_company_ranking(self, results: List, question: str) -> str:
        if not results: return "No company data found."
        q  = question.lower()
        nm = re.search(r"top\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)", q)
        word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                    "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        n_show = 3
        if nm:
            w = nm.group(1)
            n_show = int(w) if w.isdigit() else word_map.get(w, 3)
        top            = results[:n_show]
        total_vouchers = sum(r.get("voucher_count",0) for r in results)

        def fmt_amt(v):
            try: v = float(v or 0)
            except: v = 0.0
            if v <= 0:            return "₹0"
            if v >= 1_00_00_000:  return f"₹{v/1_00_00_000:.2f} crore"
            if v >= 1_00_000:     return f"₹{v/1_00_000:.2f} lakh"
            return f"₹{v:,.0f}"

        lines = [f"**#{i} {r.get('company','?')}** — "
                 f"{int(r.get('voucher_count',0)):,} vouchers "
                 f"({fmt_amt(r.get('total_amount',0))})"
                 for i, r in enumerate(top, 1)]
        top1 = top[0]
        return (f"**Top {n_show} companies by sales vouchers:**\n\n"
                + "\n".join(lines)
                + f"\n\n**{top1['company']}** leads with **{int(top1['voucher_count']):,} vouchers**. "
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
                    return (f"**{nm}**{tag} is in the database but has zero vouchers — test account.")
                return (f"**{nm}**{tag} has {total:,} total vouchers but none matched this filter.\n\n"
                        f"Try: sales, purchases, customers, trend, revenue, stock")
            return "**No records found.** Query ran but matched 0 documents."

        co      = f" for **{company['name']}**" if company else ""
        sc_note = (f"\n*(matched: {company['name']})*"
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

        def fmt_row(row):
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
            # Special case: single metric with clean natural language
            if len(row) == 1:
                k, v = list(row.items())[0]
                co_label = f" for **{company['name']}**" if company else ""
                # Map field names to natural sentences
                if "total_companies" in k:
                    return f"There are **{v}** companies in the system."
                if "total_customers" in k:
                    return f"**{company['name'] if company else 'All companies'}** has **{v}** unique customers."
                if "total_suppliers" in k:
                    return f"**{company['name'] if company else 'All companies'}** has **{v}** suppliers."
                if "total_vouchers" in k:
                    return f"Total vouchers{co_label}: **{v}**"
                return f"**{k.replace('_',' ').title()}**{co_label}: **{v}**"
            parts = [f"**{k}**: {v}" for k, v in row.items()]
            if parts:
                co_label = f" for **{company['name']}**" if company else ""
                return f"Result{co_label}:\n\n" + "\n".join(f"• {p}" for p in parts)

        formatted_preview = [fmt_row(r) for r in results[:10]]
        prompt = (
            f"Invock ERP analyst. Question: {question}{sc_note}\n"
            f"Company: {company['name'] if company else 'all companies'}\n\n"
            f"Data ({len(results)} records){co} — amounts already in ₹:\n"
            f"{json.dumps(formatted_preview, default=str, indent=2)}\n\n"
            f"RULES: Copy amounts exactly. No re-scaling. 2-3 sentences. No invented numbers."
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
