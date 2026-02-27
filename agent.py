"""
MongoDB AI Agent — Invock ERP  (production-grade, fully direct-query)
Every common question answered without LLM pipeline generation.
LLM used ONLY for analysis text, never for building MongoDB filters.
"""
import os, json, re, calendar
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI",
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster")
DB_NAME = "dev-cluster"

# ═══════════════════════════ MongoDB Helpers ══════════════════════════════════

def get_mongo_client():
    try:
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        c.admin.command("ping"); return c
    except Exception as e:
        print(f"[MongoDB] {e}"); return None

def get_db(client): return client[DB_NAME]

def deep_sanitize(obj):
    if isinstance(obj, dict):   return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [deep_sanitize(i) for i in obj]
    if isinstance(obj, ObjectId):  return str(obj)
    if isinstance(obj, datetime):  return obj.strftime("%Y-%m-%d %H:%M:%S")
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
    cols = ["Voucher","Item","Business","ItemQuantityTracker","IUser","IBranch","ICompany"]
    return {c: get_db(client)[c].count_documents({}) for c in cols}

# ═══════════════════════════ Date Helpers ═════════════════════════════════════

def get_dates():
    now = datetime.utcnow()
    fm  = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    lme = fm - timedelta(seconds=1)
    lms = lme.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ys  = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return {
        "now": now, "year_start": ys,
        "last_month_start": lms, "last_month_end": lme,
        "this_month_start": fm,
        "today_start": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "lm_num": lme.month, "lm_year": lme.year,
        "tm_num": now.month, "ty": now.year,
    }

def dt_conv(obj):
    if isinstance(obj, dict): return {k: dt_conv(v) for k, v in obj.items()}
    if isinstance(obj, list): return [dt_conv(i) for i in obj]
    if isinstance(obj, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try: return datetime.strptime(obj, fmt)
            except: pass
    return obj

# ═══════════════════════════ Company Fuzzy Resolver ═══════════════════════════

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())).strip()

def clean(s: str) -> str:
    """Strip all non-alpha for merged-word matching."""
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
    # trigram on cleaned (catches typos + merged words)
    tq, tc = tri(qc), tri(cc)
    tri_sc = len(tq & tc) / max(len(tq | tc), 1) if tq and tc else 0.0
    # token overlap
    qt = {t for t in q.split() if len(t) > 1}
    ct = {t for t in c.split() if len(t) > 1}
    tok_sc = len(qt & ct) / max(len(qt), len(ct)) if qt and ct else 0.0
    pre_sc = sum(1 for qw in qt if any(cw.startswith(qw) or qw.startswith(cw)
                                        for cw in ct)) / max(len(qt), 1) if qt else 0.0
    penalty = min(len(ct - qt) * 0.06, 0.25) if qt and ct else 0.0
    return max(0.0, min(tri_sc * 0.50 + tok_sc * 0.30 + pre_sc * 0.20 - penalty, 1.0))

def resolve_company(client, name: str) -> Optional[Dict]:
    """
    Find best ICompany match for `name`.
    Returns dict with real_id (ObjectId), name, total_vouchers, score.
    """
    db = get_db(client)
    all_cos = list(db["ICompany"].find({}, {"_id": 1, "name": 1}))
    if not all_cos: return None

    scored = sorted([(fuzzy(name, d.get("name", "")), d) for d in all_cos],
                    key=lambda x: -x[0])
    best_sc, best = scored[0]
    print(f"[Fuzzy] '{name}' → top3: {[(round(s,3),d['name']) for s,d in scored[:3]]}")

    if best_sc < 0.15: return None

    obj_id = best["_id"]
    str_id = str(obj_id)
    # Probe: ObjectId vs string — which format does Voucher use?
    n_obj  = db["Voucher"].count_documents({"iCompanyId": obj_id},    maxTimeMS=4000)
    n_str  = db["Voucher"].count_documents({"iCompanyId": str_id},    maxTimeMS=4000)
    real_id  = obj_id if n_obj >= n_str else str_id
    total    = max(n_obj, n_str)
    print(f"[Company] '{best['name']}' obj={n_obj} str={n_str} → using {'ObjectId' if real_id==obj_id else 'string'}")
    return {"real_id": real_id, "_id_obj": obj_id, "_id_str": str_id,
            "name": best["name"], "total_vouchers": total, "score": best_sc}

def extract_company_name(question: str) -> Optional[str]:
    """
    Extract company name from any natural language pattern.
    Handles: 'company with X', 'in company X', 'for X company',
             'company named X', 'company id X', 'X company', plain 'X' after keywords.
    """
    q = question.strip()
    patterns = [
        # "company with/named/called/of/id X"
        r"company\s+(?:with|named?|called?|of|id|having|like)?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s*(?:\?|$|\.|,)",
        # "in/for/of company X"
        r"(?:in|for|of|from)\s+(?:the\s+)?company\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s*(?:\?|$|\.|,)",
        # "in/with/for X company"
        r"(?:in|with|for|from)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s+company\b",
        # "vouchers/sales/records in/of/for X"
        r"(?:vouchers?|sales?|purchases?|records?|items?|data|revenue|customers?|trend|invoices?)\s+(?:in|of|for|from)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s*(?:\?|$|\.|,)",
        # "X's vouchers/data/sales"
        r"([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,40}?)(?:'s)\s+(?:vouchers?|sales?|data|revenue|customers?)",
        # "show/get/fetch X data/sales/vouchers"
        r"(?:show|get|fetch|display|find|list|give)\s+(?:me\s+)?(?:the\s+)?['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.']{2,50}?)['\"]?\s+(?:data|sales|vouchers?|revenue|customers?|trend)",
    ]
    stopwords = {"company","the","a","an","in","for","of","with","has","have",
                 "me","my","all","this","that","these","those","its","their"}
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Remove trailing stop words
            name = re.sub(r'\b(' + '|'.join(stopwords) + r')\b\s*$', '', name, flags=re.I).strip()
            # Reject if too short or is a generic word
            generic = {"sales","purchase","voucher","revenue","data","record","item",
                       "trend","customer","invoice","monthly","total","how","many","what"}
            if len(name) >= 3 and name.lower() not in generic:
                return name
    return None

# ═══════════════════════════ LLM ═════════════════════════════════════════════

def get_llm():
    k = os.getenv("GROQ_API_KEY")
    if not k: raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=k, temperature=0)

# ═══════════════════════════ Direct Query Library ════════════════════════════
# All common questions answered with hardcoded correct pipelines.
# iCompanyId is injected as the real ObjectId — no LLM involved.

class Q:
    """Query builder — returns (results, chart_meta) tuples."""

    def __init__(self, db, cid=None):
        self.db  = db
        self.cid = cid  # real ObjectId or string for iCompanyId filter

    def _mf(self, base: Dict) -> Dict:
        """Merge iCompanyId filter into base match."""
        if self.cid is not None:
            base["iCompanyId"] = self.cid
        return base

    # ── Voucher queries ───────────────────────────────────────────────────────

    def voucher_count(self, vtype=None, name="Company"):
        mf = self._mf({"type": vtype} if vtype else {})
        rows = agg(self.db, "Voucher", [
            {"$match": mf},
            {"$group": {"_id": None,
                        "total_vouchers": {"$sum": 1},
                        "total_amount":   {"$sum": "$billFinalAmount"}}},
            {"$project": {"_id":0,"total_vouchers":1,"total_amount":1}}
        ])
        label = vtype or "all"
        return rows, {"type":"metric","x_field":None,"y_field":"total_vouchers",
                      "title":f"{name} — {label.title()} Vouchers"}

    def vouchers_by_type(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({})},
            {"$group": {"_id":"$type","count":{"$sum":1},
                        "amount":{"$sum":"$billFinalAmount"}}},
            {"$sort":{"count":-1}},
            {"$project":{"_id":0,"type":"$_id","count":1,"amount":1}}
        ])
        return rows, {"type":"bar","x_field":"type","y_field":"count",
                      "title":f"{name} — Vouchers by Type"}

    def voucher_by_status(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({})},
            {"$group": {"_id":"$status","count":{"$sum":1},
                        "amount":{"$sum":"$billFinalAmount"}}},
            {"$sort":{"count":-1}},
            {"$project":{"_id":0,"status":"$_id","count":1,"amount":1}}
        ])
        return rows, {"type":"bar","x_field":"status","y_field":"amount",
                      "title":f"{name} — Vouchers by Status"}

    def top_customers(self, limit=15, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":"sales","party.name":{"$ne":None}})},
            {"$group": {"_id":"$party.name",
                        "revenue":{"$sum":"$billFinalAmount"},
                        "invoices":{"$sum":1}}},
            {"$sort": {"revenue":-1}}, {"$limit": limit},
            {"$project":{"_id":0,"customer":"$_id","revenue":1,"invoices":1}}
        ])
        return rows, {"type":"bar","x_field":"customer","y_field":"revenue",
                      "title":f"{name} — Top {limit} Customers"}

    def top_suppliers(self, limit=15, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":"purchase","party.name":{"$ne":None}})},
            {"$group": {"_id":"$party.name",
                        "amount":{"$sum":"$billFinalAmount"},
                        "invoices":{"$sum":1}}},
            {"$sort": {"amount":-1}}, {"$limit": limit},
            {"$project":{"_id":0,"supplier":"$_id","amount":1,"invoices":1}}
        ])
        return rows, {"type":"bar","x_field":"supplier","y_field":"amount",
                      "title":f"{name} — Top {limit} Suppliers"}

    def unpaid_invoices(self, name="Company"):
        rows = find(self.db, "Voucher",
            self._mf({"status":"unpaid"}),
            {"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,"issueDate":1},
            sort=[("dueAmount",-1)], limit=50)
        return rows, {"type":"table","x_field":"voucherNo","y_field":"dueAmount",
                      "title":f"{name} — Unpaid Invoices"}

    def sales_vs_purchases(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":{"$in":["sales","purchase"]}})},
            {"$group": {"_id":"$type",
                        "total":{"$sum":"$billFinalAmount"},
                        "count":{"$sum":1}}},
            {"$project":{"_id":0,"type":"$_id","total":1,"count":1}}
        ])
        return rows, {"type":"bar","x_field":"type","y_field":"total",
                      "title":f"{name} — Sales vs Purchases"}

    def avg_order_value(self, name="Company"):
        rows = agg(self.db, "Voucher", [
            {"$match": self._mf({"type":"sales"})},
            {"$group": {"_id":None,
                        "avg_order_value":{"$avg":"$billFinalAmount"},
                        "total_orders":{"$sum":1}}},
            {"$project":{"_id":0,"avg_order_value":1,"total_orders":1}}
        ])
        return rows, {"type":"metric","x_field":None,"y_field":"avg_order_value",
                      "title":f"{name} — Average Order Value"}

    # ── ItemQuantityTracker queries ───────────────────────────────────────────

    def monthly_trend(self, years=None, name="Company"):
        d = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        rows = agg(self.db, "ItemQuantityTracker", [
            {"$match": self._mf({"voucherType":"sales","year":{"$in":yrs}})},
            {"$group": {"_id":{"year":"$year","month":"$month"},
                        "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
            {"$sort": {"_id.year":1,"_id.month":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month",
                         "amount":1,"qty":1}}
        ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Sales Trend"}

    def total_revenue(self, year=None, name="Company"):
        d = get_dates()
        mf = {"voucherType":"sales"}
        if year: mf["year"] = year
        rows = agg(self.db, "ItemQuantityTracker", [
            {"$match": self._mf(mf)},
            {"$group": {"_id":None,
                        "total_revenue":{"$sum":"$amount"},
                        "total_qty":{"$sum":"$qty"}}},
            {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
        ])
        label = f"{year}" if year else "All Time"
        return rows, {"type":"metric","x_field":None,"y_field":"total_revenue",
                      "title":f"{name} — Revenue ({label})"}

    def top_products(self, by="amount", limit=15, name="Company"):
        yf = "amount" if by == "amount" else "qty"
        rows = agg(self.db, "ItemQuantityTracker", [
            {"$match": self._mf({"voucherType":"sales"})},
            {"$group": {"_id":"$itemId",
                        "amount":{"$sum":"$amount"},
                        "qty":{"$sum":"$qty"}}},
            {"$sort": {yf:-1}}, {"$limit": limit},
            {"$project":{"_id":0,"item":"$_id","amount":1,"qty":1}}
        ])
        return rows, {"type":"bar","x_field":"item","y_field":yf,
                      "title":f"{name} — Top {limit} Products"}

    def purchase_trend(self, years=None, name="Company"):
        d = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        rows = agg(self.db, "ItemQuantityTracker", [
            {"$match": self._mf({"voucherType":"purchase","year":{"$in":yrs}})},
            {"$group": {"_id":{"year":"$year","month":"$month"},
                        "amount":{"$sum":"$amount"}}},
            {"$sort": {"_id.year":1,"_id.month":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1}}
        ])
        return rows, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{name} — Monthly Purchase Trend"}

    # ── Item / Stock queries ──────────────────────────────────────────────────

    def stock(self, name="Company"):
        q = {"isHidden": False, "availableQty": {"$gt": 0}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db, "Item", q,
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",-1)], limit=100)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Stock / Inventory"}

    def low_stock(self, threshold=10, name="Company"):
        q = {"isHidden": False, "availableQty": {"$gt": 0, "$lte": threshold}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db, "Item", q,
            {"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort=[("availableQty",1)], limit=50)
        return rows, {"type":"table","x_field":"name","y_field":"availableQty",
                      "title":f"{name} — Low Stock Items (≤{threshold})"}

    # ── Business queries ──────────────────────────────────────────────────────

    def customer_list(self, name="Company"):
        q = {"relationType": {"$in": ["customer","both"]}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db, "Business", q,
            {"_id":0,"name":1,"city":1,"state":1,"relationType":1},
            sort=[("name",1)], limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Customer List"}

    def supplier_list(self, name="Company"):
        q = {"relationType": {"$in": ["supplier","both"]}}
        if self.cid: q["iCompanyId"] = self.cid
        rows = find(self.db, "Business", q,
            {"_id":0,"name":1,"city":1,"state":1},
            sort=[("name",1)], limit=200)
        return rows, {"type":"table","x_field":"name","y_field":None,
                      "title":f"{name} — Supplier List"}


# ═══════════════════════════ Intent Router ════════════════════════════════════

def route(question: str, company: Optional[Dict], db) -> Optional[Tuple]:
    """
    Route any question to the correct Q method.
    Returns (results, chart_meta) or None (falls through to LLM).
    """
    q   = question.lower().strip()
    n   = company["name"] if company else "All Companies"
    cid = company["real_id"] if company else None
    qb  = Q(db, cid)
    d   = get_dates()
    has  = lambda *ws: any(w in q for w in ws)
    miss = lambda *ws: not any(w in q for w in ws)

    # ── Voucher counts ────────────────────────────────────────────────────────
    if re.search(r"how many.*(sales|purchase|receipt|payment).*(voucher|invoice|bill|record)|"
                 r"(voucher|invoice|bill).*(count|how many|total number|number of)", q):
        vtype = ("sales" if "sales" in q else
                 "purchase" if "purchase" in q else
                 "receipt" if "receipt" in q else
                 "payment" if "payment" in q else None)
        return qb.voucher_count(vtype, n)

    if re.search(r"how many voucher|voucher count|number of voucher|count.*voucher|total.*voucher", q):
        vtype = "sales" if "sales" in q else "purchase" if "purchase" in q else None
        return qb.voucher_count(vtype, n)

    # ── Revenue / Sales total ─────────────────────────────────────────────────
    if re.search(r"total.*(revenue|sales|amount)|revenue.*total|(sales|revenue).*this year|ytd|year.*to.*date", q):
        yr = d["ty"] if has("this year","ytd","year to date",str(d["ty"])) else None
        return qb.total_revenue(yr, n)

    if re.search(r"total.*purchase|purchase.*total|purchase.*this year", q) and miss("voucher","count"):
        yr = d["ty"] if has("this year",str(d["ty"])) else None
        rows = agg(db, "ItemQuantityTracker", [
            {"$match": ({"iCompanyId":cid,"voucherType":"purchase","year":yr}
                        if cid and yr else
                        {"iCompanyId":cid,"voucherType":"purchase"} if cid else
                        {"voucherType":"purchase","year":yr} if yr else
                        {"voucherType":"purchase"})},
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

    # ── Top products ─────────────────────────────────────────────────────────
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

    # ── Vouchers by type (breakdown) ─────────────────────────────────────────
    if re.search(r"voucher.*type|type.*voucher|voucher.*breakdown|breakdown.*voucher|"
                 r"all.*type.*voucher|what type", q):
        return qb.vouchers_by_type(n)

    # ── Stock / inventory ─────────────────────────────────────────────────────
    if re.search(r"stock|inventory|available.*qty|items.*in.*stock|current.*stock|"
                 r"how many.*item|item.*available|product.*stock", q):
        if re.search(r"low|less|below|shortage|running out", q):
            return qb.low_stock(name=n)
        return qb.stock(n)

    # ── Customers list ────────────────────────────────────────────────────────
    if re.search(r"list.*customer|show.*customer|all.*customer|customer.*list|"
                 r"how many.*customer|count.*customer|number.*customer", q):
        if re.search(r"how many|count|number", q):
            rows = agg(db, "Business", [
                {"$match": ({"iCompanyId":cid,"relationType":{"$in":["customer","both"]}}
                            if cid else {"relationType":{"$in":["customer","both"]}})},
                {"$count": "total_customers"}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_customers",
                          "title":f"{n} — Total Customers"}
        return qb.customer_list(n)

    # ── Supplier list ─────────────────────────────────────────────────────────
    if re.search(r"list.*supplier|show.*supplier|all.*supplier|supplier.*list|"
                 r"how many.*supplier|count.*supplier", q):
        if re.search(r"how many|count|number", q):
            rows = agg(db, "Business", [
                {"$match": ({"iCompanyId":cid,"relationType":{"$in":["supplier","both"]}}
                            if cid else {"relationType":{"$in":["supplier","both"]}})},
                {"$count": "total_suppliers"}
            ])
            return rows, {"type":"metric","x_field":None,"y_field":"total_suppliers",
                          "title":f"{n} — Total Suppliers"}
        return qb.supplier_list(n)

    return None  # No direct match — fall through to LLM


# ═══════════════════════════ Schema-level shortcuts ════════════════════════════

def schema_shortcut(q: str) -> Optional[Dict]:
    """Answer schema-level questions (no company context needed)."""
    d = get_dates()

    def plan(qt, col, pipe=None, fq=None, proj=None, sort=None, limit=100,
             tmpl="", ct="table", x=None, y=None, title=""):
        return {"query_type":qt,"collection":col,"pipeline":pipe,
                "find_query":fq,"projection":proj,"sort":sort,"limit":limit,
                "answer_template":tmpl,
                "chart_suggestion":{"type":ct,"x_field":x,"y_field":y,"title":title},
                "clarification_needed":False}

    has  = lambda *ws: any(w in q for w in ws)
    miss = lambda *ws: not any(w in q for w in ws)

    # Branches
    if re.search(r"branch|branches|location", q) and miss("sales","revenue","voucher","customer","company with","in company"):
        return plan("find","IBranch",fq={},
            proj={"_id":0,"name":1,"city":1,"state":1,"code":1},
            limit=300,tmpl="All branches.",ct="table",title="All Branches")

    # Users
    if re.search(r"\buser|users|staff|employee\b", q) and miss("company with","in company","sales","voucher"):
        return plan("find","IUser",fq={},
            proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},
            limit=500,tmpl="All users.",ct="table",title="All Users")

    # Company list (generic)
    if re.search(r"^(list |show |get |how many )?(all )?compan", q) and miss("with","in company","for","sales","voucher"):
        return plan("find","ICompany",fq={},
            proj={"_id":0,"name":1,"industry":1,"financialYear":1},
            limit=200,tmpl="All companies.",ct="table",title="All Companies")

    # Monthly trend (global)
    if re.search(r"monthly.*trend|trend.*month|last 12 month|12 month|month.*wise", q) and miss("company","with","in","for","purchase"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[
                {"$match":{"voucherType":"sales","year":{"$in":[d["ty"]-1,d["ty"]]}}},
                {"$group":{"_id":{"year":"$year","month":"$month"},
                           "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                {"$sort":{"_id.year":1,"_id.month":1}},
                {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1,"qty":1}}
            ],
            tmpl="Monthly sales trend last 12 months.",
            ct="line",x="month",y="amount",title="Monthly Sales Trend (Last 12 Months)")

    # Total revenue this year
    if re.search(r"total.*(revenue|sales).*year|revenue.*this year|sales.*this year|ytd", q) and miss("company","with","in"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[
                {"$match":{"voucherType":"sales","year":d["ty"]}},
                {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"},"total_qty":{"$sum":"$qty"}}},
                {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
            ],
            tmpl=f"Total sales revenue {d['ty']}.",
            ct="metric",y="total_revenue",title=f"Total Revenue {d['ty']}")

    # Sales last month
    if re.search(r"sales.*last month|last month.*sales|revenue.*last month", q) and miss("company","with"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[
                {"$match":{"voucherType":"sales","year":d["lm_year"],"month":d["lm_num"]}},
                {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"}}},
                {"$project":{"_id":0,"total_revenue":1}}
            ],
            tmpl="Last month sales.",ct="metric",y="total_revenue",
            title=f"Sales — {calendar.month_abbr[d['lm_num']]} {d['lm_year']}")

    # Sales vs purchases (global)
    if re.search(r"sales.*vs.*purchase|purchase.*vs.*sales", q) and miss("company","with","in"):
        return plan("aggregate","Voucher",
            pipe=[
                {"$match":{"type":{"$in":["sales","purchase"]},"iCompanyId":{"$ne":None}}},
                {"$group":{"_id":"$type","total":{"$sum":"$billFinalAmount"},"count":{"$sum":1}}},
                {"$project":{"_id":0,"type":"$_id","total":1,"count":1}}
            ],
            tmpl="Sales vs purchases.",ct="bar",x="type",y="total",
            title="Sales vs Purchases (All Companies)")

    # Top customers (global)
    if re.search(r"top.*customer|best.*customer|customer.*revenue", q) and miss("company","with","in","list","show"):
        return plan("aggregate","Voucher",
            pipe=[
                {"$match":{"type":"sales","iCompanyId":{"$ne":None},"party.name":{"$ne":None}}},
                {"$group":{"_id":"$party.name","revenue":{"$sum":"$billFinalAmount"},"invoices":{"$sum":1}}},
                {"$sort":{"revenue":-1}},{"$limit":15},
                {"$project":{"_id":0,"customer":"$_id","revenue":1,"invoices":1}}
            ],
            tmpl="Top 15 customers by revenue.",ct="bar",x="customer",y="revenue",
            title="Top 15 Customers by Revenue")

    # Top products (global)
    if re.search(r"top.*product|best.*product|most.*sold|which.*product.*sold", q) and miss("company","with","in"):
        return plan("aggregate","ItemQuantityTracker",
            pipe=[
                {"$match":{"voucherType":"sales"}},
                {"$group":{"_id":"$itemId","revenue":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                {"$sort":{"revenue":-1}},{"$limit":15},
                {"$project":{"_id":0,"item":"$_id","revenue":1,"qty":1}}
            ],
            tmpl="Top 15 products by revenue.",ct="bar",x="item",y="revenue",
            title="Top Products by Revenue")

    # Unpaid invoices (global)
    if re.search(r"unpaid|outstanding|overdue", q) and miss("company","with","in"):
        return plan("find","Voucher",
            fq={"status":"unpaid","iCompanyId":{"$ne":None}},
            proj={"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,"issueDate":1},
            sort={"dueAmount":-1},limit=50,
            tmpl="Unpaid invoices.",ct="table",title="Unpaid Invoices")

    # Stock (global)
    if re.search(r"stock|inventory|available.*qty", q) and miss("company","with","in"):
        return plan("find","Item",
            fq={"isHidden":False,"availableQty":{"$gt":0}},
            proj={"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
            sort={"availableQty":-1},limit=100,
            tmpl="Current stock.",ct="table",title="Stock / Inventory")

    # Avg order value (global)
    if re.search(r"avg.*order|average.*order|aov", q) and miss("company","with","in"):
        return plan("aggregate","Voucher",
            pipe=[
                {"$match":{"type":"sales","iCompanyId":{"$ne":None}}},
                {"$group":{"_id":None,"avg_order_value":{"$avg":"$billFinalAmount"},"orders":{"$sum":1}}},
                {"$project":{"_id":0,"avg_order_value":1,"orders":1}}
            ],
            tmpl="Average order value.",ct="metric",y="avg_order_value",
            title="Average Order Value (All Companies)")

    # How many customers/suppliers
    if re.search(r"how many (customer|supplier|client)", q) and miss("company","with","in"):
        rel = "customer" if "customer" in q or "client" in q else "supplier"
        field = f"total_{rel}s"
        return plan("aggregate","Business",
            pipe=[{"$match":{"relationType":rel}},{"$count":field}],
            tmpl=f"Total {rel}s.",ct="metric",y=field,title=f"Total {rel.title()}s")

    return None


# ═══════════════════════════ LLM Fallback ════════════════════════════════════

SCHEMA_TEXT = """
DATABASE: dev-cluster — Invock ERP (Jewellery, India, ₹ INR)
Voucher(1.3M): type,billFinalAmount,dueAmount,paidAmount,status,iCompanyId(ObjectId),issueDate(Date),party.name,voucherNo
ItemQuantityTracker(2.1M): voucherType,month(int),year(int),itemId,qty,amount,iCompanyId
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
1. iCompanyId is ObjectId — never filter it (agent handles this)
2. $sum/$avg must be {{"$sum":"$fieldName"}} (with $ prefix on field)
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
            fq   = dt_conv([plan["find_query"]])[0] if date_type=="date_object" else plan["find_query"]
            proj = plan.get("projection")
            srt  = list(plan["sort"].items()) if plan.get("sort") else None
            return find(db, col, fq, proj, srt, plan.get("limit",100)), None
    except Exception as e:
        return [], str(e)
    return [], None


# ═══════════════════════════ Main Agent ══════════════════════════════════════

class MongoAIAgent:
    def __init__(self):
        self.client     = get_mongo_client()
        self.llm        = None
        self.history    = []
        self.date_type  = "date_object"
        self.stats      = {}
        if self.client:
            try:
                self.date_type = detect_date_type(self.client)
                self.stats     = get_stats(self.client)
                print(f"[Agent] Connected. issueDate={self.date_type}")
            except Exception as e:
                print(f"[Agent] Init error: {e}")

    def is_connected(self): return self.client is not None

    def refresh_schema(self):
        if self.client:
            self.date_type = detect_date_type(self.client)
            self.stats     = get_stats(self.client)

    def init_llm(self):
        try: self.llm = get_llm(); return True
        except: return False

    def query(self, question: str) -> Dict:
        if not self.llm and not self.init_llm():
            return {"error": "GROQ_API_KEY not configured."}

        q_low = question.lower().strip()
        db    = get_db(self.client) if self.client else None

        # ── Step 1: Schema-level shortcut (no company) ────────────────────────
        sc = schema_shortcut(q_low)
        if sc:
            results, err = execute_plan(sc, db, self.date_type)
            answer = self._answer(question, sc, results, err)
            chart  = self._chart(results, sc["chart_suggestion"])
            self.history.append({"q":question,"a":sc["answer_template"][:80]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":sc,"db_error":err}

        # ── Step 2: Detect company name ────────────────────────────────────────
        cname   = extract_company_name(question)
        company = None
        if cname and self.client:
            company = resolve_company(self.client, cname)
            if company is None:
                # Try the whole question as company name too
                company = resolve_company(self.client, question)
            if company is None:
                return {
                    "type":"answer",
                    "answer":(f"❌ No company matching **\"{cname}\"** found.\n\n"
                              f"**Top companies by sales:**\n"
                              f"• M/S DIPSHI - ESTIMATE (40,255 vouchers)\n"
                              f"• HIRAKA JEWELS (34,998) • Bhakti Parshwanath (30,707)\n"
                              f"• NAMO-ESTIMATE (27,478) • VAIBHAV FASHION (11,735)\n\n"
                              f"Ask *\"list all companies\"* to see all 135 companies."),
                    "results":[],"chart":None,"plan":{},"db_error":None
                }

        # ── Step 3: Route to direct query (with or without company) ───────────
        if db:
            routed = route(question, company, db)
            if routed:
                results_raw, chart_sug = routed
                results = [deep_sanitize(r) for r in results_raw]
                plan    = {"query_type":"direct","collection":"Voucher",
                           "answer_template":f"Query for {company['name'] if company else 'all'}.",
                           "chart_suggestion":chart_sug,"clarification_needed":False}
                answer  = self._answer(question, plan, results, None, company)
                chart   = self._chart(results, chart_sug)
                self.history.append({"q":question,"a":plan["answer_template"][:80]})
                return {"type":"answer","answer":answer,"results":results,
                        "chart":chart,"plan":plan,"db_error":None}

        # ── Step 4: LLM fallback for unknown questions ─────────────────────────
        dates      = get_dates()
        sys_prompt = llm_prompt(dates)
        hist       = ("\nPrev:\n" + "\n".join(f"Q:{h['q']}\nA:{h['a']}" for h in self.history[-3:])
                      if self.history else "")
        user_msg   = f"{SCHEMA_TEXT}\n{hist}\n\nQuestion: {question}"
        try:
            resp = self.llm.invoke([SystemMessage(content=sys_prompt),
                                    HumanMessage(content=user_msg)])
            plan = parse_plan(resp.content)
        except Exception as e:
            return {"error": f"LLM error: {e}"}

        results, err = execute_plan(plan, db, self.date_type)

        # Retry once if empty/error
        if (not results or err) and plan.get("query_type") != "none":
            hint = (f"\n⚠️ Previous attempt failed: {err or 'empty results'}\n"
                    f"Failed pipeline: {json.dumps(plan.get('pipeline'),default=str)[:200]}\n"
                    f"Fix: ensure $sum uses '$fieldName' format. Try simpler query.")
            try:
                resp2 = self.llm.invoke([SystemMessage(content=sys_prompt),
                                          HumanMessage(content=user_msg+hint)])
                plan  = parse_plan(resp2.content)
                results, err = execute_plan(plan, db, self.date_type)
            except: pass

        answer = self._answer(question, plan, results, err, company)
        chart  = self._chart(results, plan.get("chart_suggestion",{}))
        self.history.append({"q":question,"a":plan.get("answer_template","")[:80]})
        return {"type":"answer","answer":answer,"results":results,
                "chart":chart,"plan":plan,"db_error":err}

    def _answer(self, question, plan, results, err, company=None):
        if err:
            return f"⚠️ **Database error:** `{err}`"
        if plan.get("query_type") == "none":
            return plan.get("answer_template","No data found.")

        if not results:
            if company:
                total = company.get("total_vouchers", 0)
                nm    = company["name"]
                sc    = company.get("score", 1.0)
                tag   = "" if sc > 0.85 else f" *(closest match to your query)*"
                if total == 0:
                    return (f"**{nm}**{tag} ✓ is in the database but has **zero vouchers** "
                            f"— this is a test/demo account.\n\n"
                            f"**Companies with real data:**\n"
                            f"• M/S DIPSHI - ESTIMATE → 40,255 | HIRAKA JEWELS → 34,998\n"
                            f"• Bhakti Parshwanath → 30,707 | NAMO-ESTIMATE → 27,478\n"
                            f"• VAIBHAV FASHION JEWELLERY → 11,735")
                return (f"**{nm}**{tag} has {total:,} total vouchers, but none match "
                        f"this specific filter.\n\nTry: sales, purchases, customers, trend, revenue, stock")
            return "**No records found.** Query ran but matched 0 documents."

        co = f" for **{company['name']}**" if company else ""
        sc_note = (f"\n*(matched company: {company['name']})*" 
                   if company and company.get("score",1.0) < 0.85 else "")
        prompt = (
            f"Invock ERP data analyst. Question: {question}{sc_note}\n"
            f"Company: {company['name'] if company else 'all companies'}\n\n"
            f"Data ({len(results)} records){co}:\n"
            f"{json.dumps(results[:10], default=str, indent=2)}\n\n"
            f"Write 2-3 precise sentences:\n"
            f"1. Lead with the exact number/amount from data\n"
            f"2. ₹ crore/lakh format (1Cr=10L=1,000,000)\n"
            f"3. Name real top items from data\n"
            f"4. ONE insight. ZERO invented numbers."
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
                    if isinstance(v, bool):  row[k] = str(v)
                    elif isinstance(v, (int,float,type(None))): row[k] = v
                    elif isinstance(v, str): row[k] = v
                    elif isinstance(v, dict): row[k] = str(v)
                    elif isinstance(v, list): row[k] = len(v)
                    else: row[k] = str(v)
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
