"""
MongoDB AI Agent — Invock ERP
Robust direct-query approach for company-specific questions.
"""
import os, json, re, calendar
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI","mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster")
DB_NAME = "dev-cluster"

# ─────────────────────────────── MongoDB helpers ──────────────────────────────

def get_mongo_client():
    try:
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        c.admin.command("ping"); return c
    except Exception as e:
        print(f"[MongoDB] {e}"); return None

def get_db(client): return client[DB_NAME]

def deep_sanitize(obj):
    if isinstance(obj, dict):    return {k: deep_sanitize(v) for k,v in obj.items()}
    if isinstance(obj, list):    return [deep_sanitize(i) for i in obj]
    if isinstance(obj, ObjectId):  return str(obj)
    if isinstance(obj, datetime):  return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, float) and obj != obj: return None
    if isinstance(obj, (str,int,float,bool,type(None))): return obj
    return str(obj)

def run_agg(db, col, pipeline):
    raw = list(db[col].aggregate(pipeline, allowDiskUse=True))
    return [deep_sanitize(d) for d in raw]

def run_find(db, col, query, proj=None, limit=100, sort=None):
    cur = db[col].find(query, proj or {"_id":0})
    if sort: cur = cur.sort(sort)
    return [deep_sanitize(d) for d in cur.limit(limit)]

def detect_date_type(client) -> str:
    try:
        s = get_db(client)["Voucher"].find_one({"type":"sales"},{"_id":0,"issueDate":1})
        if s: return "date_object" if isinstance(s.get("issueDate"), datetime) else "string"
    except: pass
    return "date_object"

def get_collection_stats(client):
    stats = {}
    for col in ["Voucher","Item","Business","ItemQuantityTracker",
                "Contact","Account","IUser","IBranch","ICompany"]:
        try: stats[col] = get_db(client)[col].count_documents({})
        except: stats[col] = 0
    return stats

# ─────────────────────────── Date helpers ─────────────────────────────────────

def get_dates():
    now = datetime.utcnow()
    fm  = now.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    lme = fm - timedelta(seconds=1)
    lms = lme.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    ys  = now.replace(month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
    return {
        "now":now,"year_start":ys,"last_month_start":lms,"last_month_end":lme,
        "this_month_start":fm,
        "today_start":now.replace(hour=0,minute=0,second=0,microsecond=0),
        "last_12m_start":now-timedelta(days=365),
        "lm_num":lme.month,"lm_year":lme.year,"tm_num":now.month,"ty":now.year,
    }

def dt_str_to_obj(obj):
    """Recursively convert ISO date strings to datetime objects in a pipeline."""
    if isinstance(obj, dict):  return {k: dt_str_to_obj(v) for k,v in obj.items()}
    if isinstance(obj, list):  return [dt_str_to_obj(i) for i in obj]
    if isinstance(obj, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S","%Y-%m-%d"]:
            try: return datetime.strptime(obj, fmt)
            except: pass
    return obj

# ─────────────────────────── Company fuzzy match ───────────────────────────────

def normalize(s:str)->str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def fuzzy_score(query:str, candidate:str)->float:
    q,c = normalize(query), normalize(candidate)
    if not q or not c: return 0.0
    if q == c: return 1.0
    if q in c or c in q: return 0.92

    # Handle merged words: "namoestimate" should match "namo estimate"
    # Expand candidate by removing separators so "namo estimate" → "namoestimate"
    c_merged = re.sub(r"\s","",c)
    q_merged = re.sub(r"\s","",q)
    if q_merged == c_merged: return 0.98
    if q_merged in c_merged or c_merged in q_merged: return 0.90

    qt = {t for t in q.split() if len(t)>1}
    ct = {t for t in c.split() if len(t)>1}
    if not qt or not ct: return 0.0
    token_sc  = len(qt&ct) / max(len(qt),len(ct))
    prefix_sc = sum(1 for qw in qt if any(cw.startswith(qw) or qw.startswith(cw) for cw in ct)) / max(len(qt),1)
    penalty   = min(len(ct-qt)*0.08, 0.30)
    def tri(s): return set(s[i:i+3] for i in range(len(s)-2))
    tq,tc = tri(q_merged),tri(c_merged)   # use merged for trigrams
    tri_sc = len(tq&tc)/max(len(tq|tc),1) if tq and tc else 0.0
    return max(0.0, min(token_sc*0.45 + prefix_sc*0.20 + tri_sc*0.35 - penalty, 1.0))

def find_company(client, name:str) -> Optional[Dict]:
    """Return {_id_obj, _id_str, name, voucher_count} for best matching company."""
    db = get_db(client)
    all_cos = list(db["ICompany"].find({}, {"_id":1,"name":1}))
    if not all_cos: return None

    # ── Pass 1: score all candidates with fuzzy ────────────────────────────
    scored = [(fuzzy_score(name, d.get("name","")), d) for d in all_cos]
    scored.sort(key=lambda x:-x[0])
    top = scored[:4]
    print(f"[Fuzzy] '{name}' → {[(round(sc,3), d['name']) for sc,d in top[:3]]}")
    best_sc, best_doc = top[0]

    # ── Pass 2: merged-substring fallback for joined words ─────────────────
    # e.g. "namoeatimative" → strip all non-alpha → "namoeatimative"
    #      vs "namo estimate" → "namoestimate"  trigram overlap catches it
    if best_sc < 0.18:
        q_clean = re.sub(r"[^a-z0-9]", "", normalize(name))
        if len(q_clean) >= 5:
            for d in all_cos:
                c_clean = re.sub(r"[^a-z0-9]", "", normalize(d.get("name","")))
                # substring match
                if q_clean in c_clean or c_clean in q_clean:
                    best_sc, best_doc = 0.85, d
                    print(f"[Fuzzy-substring] '{name}' → '{d['name']}' via merged substring")
                    break
                # trigram on cleaned strings
                def tri3(s): return set(s[i:i+3] for i in range(len(s)-2))
                tq, tc = tri3(q_clean), tri3(c_clean)
                sc = len(tq&tc)/max(len(tq|tc),1) if tq and tc else 0.0
                if sc > best_sc:
                    best_sc, best_doc = sc, d

    if best_sc < 0.18:
        return None
    obj_id = best_doc["_id"]
    str_id = str(obj_id)
    # Probe which format Voucher uses — try ALL voucher types
    n_obj = db["Voucher"].count_documents({"iCompanyId": obj_id}, maxTimeMS=4000)
    n_str = db["Voucher"].count_documents({"iCompanyId": str_id}, maxTimeMS=4000)
    real_id   = obj_id if n_obj >= n_str else str_id
    total_all = max(n_obj, n_str)
    print(f"[Company] '{best_doc['name']}' ObjectId:{n_obj} string:{n_str} total_all:{total_all}")
    return {"_id_obj":obj_id, "_id_str":str_id, "real_id":real_id,
            "name":best_doc["name"], "total_vouchers":total_all, "score":best_sc}

def extract_company_name(q:str) -> Optional[str]:
    patterns = [
        r"company\s+(?:with|named?|called?|of)?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,40}?)['\"]?(?:\s*\?|$|\.|,)",
        r"(?:in|for|of)\s+company\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,40}?)['\"]?(?:\s*\?|$|\.|,)",
        r"(?:in|with|for)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,40}?)['\"]?\s+company(?:\s*\?|$|\.|,)?",
        r"(?:vouchers?|sales?|records?|items?|purchases?|data)\s+(?:in|of|for)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,40}?)['\"]?(?:\s*\?|$|\.|,)",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = re.sub(r'\s*(company|the|a|an|in|for|of|with|has|have)\s*$','',m.group(1).strip(),flags=re.I).strip()
            if len(name)>=3: return name
    return None

# ─────────────────────────── LLM ──────────────────────────────────────────────

def get_llm():
    k = os.getenv("GROQ_API_KEY")
    if not k: raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=k, temperature=0)

# ─────────────────────────── Valid collections ────────────────────────────────

VALID_COLS = {
    "Voucher","Item","Business","ItemQuantityTracker","ItemSummary","Contact",
    "Account","IBranch","IUser","ICompany","ItemGroup","company_data",
    "voucher_count","AccountGroup",
}

# ─────────────────────────── Schema for LLM ───────────────────────────────────

SCHEMA = """
DATABASE: dev-cluster — Invock ERP (Jewellery/textiles, India, ₹ INR)

═══ Voucher (1.3M docs) — every financial transaction
  type              "sales"|"purchase"|"receipt"|"payment"|"credit note"|"debit note"|"journal"
  billFinalAmount   float  — total invoice ₹
  billItemsPrice    float  — subtotal
  billTaxAmount     float
  dueAmount         float  — outstanding
  paidAmount        float
  status            "unpaid"|"paid"|"partial"
  iCompanyId        ObjectId  — ALWAYS filter as ObjectId (not string)
  iBranchId         ObjectId
  issueDate         Date object
  lineItemQtySum    float
  party.name        string  — customer/supplier name
  SAFE projection: {_id:0,type:1,voucherNo:1,issueDate:1,billFinalAmount:1,
    billTaxAmount:1,status:1,dueAmount:1,paidAmount:1,iCompanyId:1,lineItemQtySum:1}
  NEVER project: itemList,transactions,tax,voucherList,otherCharges,party

═══ ItemQuantityTracker (2.1M docs) — BEST for date/product analytics
  voucherType  "sales"|"purchase"|"stock_adjustment"
  month  float 1-12,  year  int,  itemId  string
  qty  float,  amount  float,  iCompanyId  ObjectId

═══ Item (450K)  name,skuBarcode,availableQty,unit,iCompanyId,isHidden
═══ Business (45K)  name,relationType("customer"|"supplier"|"both"),city,state,iCompanyId
═══ IBranch (264)  name,city,state,code,iCompanyId
═══ IUser (399)  name,phone,lastSignIn
═══ ICompany (135)  name,industry,financialYear
═══ ItemGroup (1.8K)  name,taxPercentage,hsn
═══ voucher_count (720K)  vouchercount int, iCompanyId ObjectId
═══ Account (51K)  name,accountGroupName,balance,iCompanyId

═══ QUESTION → PIPELINE MAP (follow exactly):
  total sales           → Voucher agg: $match{type:"sales"}, $group{_id:null,$sum:billFinalAmount}, $project{_id:0,total_sales:1}
  total purchases       → Voucher agg: $match{type:"purchase"}, $group, $project{total_purchases:1}
  sales vs purchases    → Voucher agg: $match{type:{$in:["sales","purchase"]}}, $group{_id:"$type",$sum:billFinalAmount}, $project{_id:0,type:"$_id",amount:1}
  top products qty      → ItemQuantityTracker: $match{voucherType:"sales"}, $group{_id:"$itemId",$sum:"$qty"}, sort desc, $project{_id:0,item:"$_id",total_qty:1}
  top products revenue  → ItemQuantityTracker: $match{voucherType:"sales"}, $group{_id:"$itemId",$sum:"$amount"}, sort desc, $project{_id:0,item:"$_id",revenue:1}
  monthly trend         → ItemQuantityTracker: $match{voucherType:"sales"}, $group{_id:{year:"$year",month:"$month"},$sum:"$amount"}, sort {_id:1}, $project{_id:0,year:"$_id.year",month:"$_id.month",amount:1}
  top customers         → Voucher: $match{type:"sales"}, $group{_id:"$party.name",$sum:"$billFinalAmount"}, sort desc, $project{_id:0,customer:"$_id",revenue:1}
  count customers       → Business: $match{relationType:"customer"}, $count → "customer_count"
  avg order value       → Voucher: $match{type:"sales"}, $group{_id:null,$avg:"$billFinalAmount"}, $project{_id:0,avg_order_value:1}
  unpaid invoices       → Voucher: find{status:"unpaid"}, SAFE PROJECTION, limit 50
  stock/inventory       → Item: find{isHidden:false}, sort{availableQty:-1}, projection{_id:0,name:1,skuBarcode:1,availableQty:1,unit:1}
  branches              → IBranch: find{}, projection{_id:0,name:1,city:1,state:1,code:1}
  users                 → IUser: find{}, projection{_id:0,name:1,phone:1,lastSignIn:1}
  companies             → ICompany: find{}, projection{_id:0,name:1,industry:1,financialYear:1}
  sales this year       → ItemQuantityTracker: $match{voucherType:"sales",year:TY}, $group, $sum amount
  sales last month      → ItemQuantityTracker: $match{voucherType:"sales",year:LM_YEAR,month:LM_NUM}, $group, $sum amount
  trend 12 months       → ItemQuantityTracker: $match{voucherType:"sales",year:{$in:[TY-1,TY]}}, group by year+month
"""

def build_prompt(dates, date_type):
    d = dates
    return f"""You are a senior MongoDB analyst for Invock ERP.
Return ONLY a single valid JSON object. No markdown, no backticks, no explanation outside JSON.

JSON format:
{{
  "query_type": "aggregate"|"find"|"none",
  "collection": "<exact collection name>",
  "pipeline": [...] | null,
  "find_query": {{...}} | null,
  "projection": {{"_id":0,...}} | null,
  "sort": {{...}} | null,
  "limit": 50,
  "answer_template": "<one sentence>",
  "chart_suggestion": {{
    "type": "bar"|"line"|"metric"|"table"|"none",
    "x_field": "<exact output field name>",
    "y_field": "<exact output field name>",
    "title": "<chart title>"
  }},
  "clarification_needed": false
}}

DATE CONTEXT (today = {d['now'].strftime('%Y-%m-%d')}):
  TY={d['ty']}  LM_NUM={d['lm_num']}  LM_YEAR={d['lm_year']}
  last_month: {d['last_month_start'].strftime('%Y-%m-%dT%H:%M:%S')} to {d['last_month_end'].strftime('%Y-%m-%dT%H:%M:%S')}
  year_start: {d['year_start'].strftime('%Y-%m-%dT%H:%M:%S')}

CRITICAL RULES:
1. Use ItemQuantityTracker for all date/product queries (has integer year/month — no date conversion needed)
2. For Voucher date queries, issueDate is a DATE OBJECT — always pass ISO strings, agent converts to datetime
3. Collection names are CASE-SENSITIVE
4. Every aggregation MUST end with $project that excludes _id and names all output fields
5. Never include itemList/transactions/tax/party/voucherList in Voucher projections
6. chart x_field and y_field must be EXACT output field names from your $project stage
7. For count queries: use $count stage, project the count field
8. clarification_needed is ALWAYS false
9. Do NOT filter iCompanyId — the agent handles company filtering separately"""

def parse_json(text:str)->Dict:
    text = re.sub(r"```(?:json)?\n?","",text.strip()).strip("`").strip()
    try: return json.loads(text)
    except:
        m = re.search(r"\{.*\}",text,re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {"query_type":"none","answer_template":"Parse error.",
            "chart_suggestion":{"type":"none"},"clarification_needed":False}

# ──────────────────────────── DIRECT QUERY ENGINE ─────────────────────────────
# Handles company-specific questions without relying on LLM for filter values

class DirectQueryEngine:
    """Execute precise queries directly — no LLM needed for well-known patterns."""

    def __init__(self, client, date_type):
        self.db        = get_db(client)
        self.date_type = date_type

    def _cid_filter(self, company):
        """Return the correct iCompanyId filter using the real_id."""
        return {"iCompanyId": company["real_id"]}

    def voucher_count(self, company, vtype=None):
        f = self._cid_filter(company)
        if vtype: f["type"] = vtype
        count = self.db["Voucher"].count_documents(f)
        agg = run_agg(self.db,"Voucher",[
            {"$match": f},
            {"$group": {"_id": None,
                        "total_vouchers": {"$sum": 1},
                        "total_amount":   {"$sum": "$billFinalAmount"}}},
            {"$project": {"_id":0,"total_vouchers":1,"total_amount":1}}
        ])
        data = agg if agg else [{"total_vouchers": count, "total_amount": 0}]
        label = vtype or "all"
        return data, {
            "type":"metric","x_field":None,"y_field":"total_vouchers",
            "title":f"{company['name']} — {label.title()} Vouchers"
        }

    def voucher_by_status(self, company, status=None):
        f = self._cid_filter(company)
        if status: f["status"] = status
        data = run_agg(self.db,"Voucher",[
            {"$match": f},
            {"$group": {"_id":"$status","count":{"$sum":1},"amount":{"$sum":"$billFinalAmount"}}},
            {"$project":{"_id":0,"status":"$_id","count":1,"amount":1}},
            {"$sort":{"count":-1}}
        ])
        return data, {"type":"bar","x_field":"status","y_field":"amount",
                      "title":f"{company['name']} — Vouchers by Status"}

    def top_customers(self, company, limit=10):
        f = {**self._cid_filter(company), "type":"sales"}
        data = run_agg(self.db,"Voucher",[
            {"$match": f},
            {"$group": {"_id":"$party.name","revenue":{"$sum":"$billFinalAmount"},
                        "count":{"$sum":1}}},
            {"$sort":{"revenue":-1}},{"$limit":limit},
            {"$project":{"_id":0,"customer":"$_id","revenue":1,"count":1}}
        ])
        return data, {"type":"bar","x_field":"customer","y_field":"revenue",
                      "title":f"{company['name']} — Top Customers"}

    def top_products(self, company, by="amount", limit=10):
        f = {**self._cid_filter(company), "voucherType":"sales"}
        yf = "amount" if by=="amount" else "qty"
        data = run_agg(self.db,"ItemQuantityTracker",[
            {"$match": f},
            {"$group": {"_id":"$itemId", yf:{"$sum":f"${yf}"}}},
            {"$sort":{yf:-1}},{"$limit":limit},
            {"$project":{"_id":0,"item":"$_id",yf:1}}
        ])
        return data, {"type":"bar","x_field":"item","y_field":yf,
                      "title":f"{company['name']} — Top Products"}

    def monthly_trend(self, company, years=None):
        d = get_dates()
        yrs = years or [d["ty"]-1, d["ty"]]
        f = {**self._cid_filter(company), "voucherType":"sales",
             "year":{"$in": yrs}}
        data = run_agg(self.db,"ItemQuantityTracker",[
            {"$match": f},
            {"$group": {"_id":{"year":"$year","month":"$month"},
                        "amount":{"$sum":"$amount"}}},
            {"$sort":{"_id":1}},
            {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1}}
        ])
        return data, {"type":"line","x_field":"month","y_field":"amount",
                      "title":f"{company['name']} — Monthly Sales Trend"}

    def total_revenue(self, company, year=None):
        d = get_dates()
        f = {**self._cid_filter(company), "voucherType":"sales"}
        if year: f["year"] = year or d["ty"]
        data = run_agg(self.db,"ItemQuantityTracker",[
            {"$match": f},
            {"$group": {"_id":None,"total_revenue":{"$sum":"$amount"}}},
            {"$project":{"_id":0,"total_revenue":1}}
        ])
        return data, {"type":"metric","x_field":None,"y_field":"total_revenue",
                      "title":f"{company['name']} — Total Revenue"}

    def unpaid_invoices(self, company):
        f = {**self._cid_filter(company), "status":"unpaid"}
        data = run_find(self.db,"Voucher",f,
                        {"_id":0,"voucherNo":1,"billFinalAmount":1,"dueAmount":1,
                         "issueDate":1,"status":1},limit=50,sort=[("dueAmount",-1)])
        return data, {"type":"table","x_field":"voucherNo","y_field":"dueAmount",
                      "title":f"{company['name']} — Unpaid Invoices"}

    def handle(self, question:str, company:Dict):
        """Route company-specific question to the right direct query."""
        q = question.lower()
        d = get_dates()

        # voucher count by type
        if re.search(r"how many.*(sales|purchase).*(voucher|invoice|bill)", q) or \
           re.search(r"(sales|purchase).*(voucher|invoice|bill).*(count|how many|total number)", q):
            vtype = "sales" if "sales" in q else "purchase" if "purchase" in q else None
            return self.voucher_count(company, vtype)

        if re.search(r"how many.*voucher|voucher.*count|number of.*voucher|count.*voucher", q):
            vtype = "sales" if "sales" in q else "purchase" if "purchase" in q else None
            return self.voucher_count(company, vtype)

        if re.search(r"top.*customer|best.*customer|customer.*revenue|customer.*sales", q):
            return self.top_customers(company)

        if re.search(r"top.*product|best.*product|product.*revenue|product.*qty|item.*sold", q):
            by = "qty" if "qty" in q or "quantity" in q else "amount"
            return self.top_products(company, by)

        if re.search(r"monthly.*trend|trend.*month|sales.*month|month.*sales", q):
            return self.monthly_trend(company)

        if re.search(r"total.*revenue|revenue.*total|total.*sales|sales.*total", q):
            yr = d["ty"] if "this year" in q or "ytd" in q else None
            return self.total_revenue(company, yr)

        if re.search(r"unpaid|outstanding|due|overdue", q):
            return self.unpaid_invoices(company)

        if re.search(r"status|paid|partial", q):
            return self.voucher_by_status(company)

        # Default: sales voucher count
        vtype = "sales" if "sales" in q else "purchase" if "purchase" in q else None
        return self.voucher_count(company, vtype)


# ──────────────────────────── Agent ───────────────────────────────────────────

class MongoAIAgent:
    def __init__(self):
        self.client   = get_mongo_client()
        self.llm      = None
        self.history  = []
        self.collection_stats = {}
        self.date_type = "date_object"
        if self.client:
            try:
                self.collection_stats = get_collection_stats(self.client)
                self.date_type = detect_date_type(self.client)
                print(f"[Agent] issueDate type: {self.date_type}")
            except Exception as e:
                print(f"[Agent] init error: {e}")

    def is_connected(self): return self.client is not None

    def refresh_schema(self):
        if self.client:
            self.collection_stats = get_collection_stats(self.client)
            self.date_type = detect_date_type(self.client)

    def init_llm(self):
        try: self.llm = get_llm(); return True
        except: return False

    # ── Keyword shortcut for simple schema-level queries ──────────────────────
    def _shortcut(self, q:str) -> Optional[Dict]:
        has = lambda *ws: any(w in q for w in ws)
        hasnot = lambda *ws: not any(w in q for w in ws)

        def plan(qt,col,pipe=None,fq=None,proj=None,limit=100,
                 tmpl="",ct="table",x=None,y=None,title=""):
            return {"query_type":qt,"collection":col,"pipeline":pipe,
                    "find_query":fq,"projection":proj,"limit":limit,
                    "answer_template":tmpl,
                    "chart_suggestion":{"type":ct,"x_field":x,"y_field":y,"title":title},
                    "clarification_needed":False}

        # Only fire shortcuts for clearly generic (no specific company) questions
        if has("branch","branches") and hasnot("sales","revenue","voucher","customer","company","with","in","for"):
            return plan("find","IBranch",fq={},proj={"_id":0,"name":1,"city":1,"state":1,"code":1},
                        limit=300,tmpl="All branches.",ct="table",title="All Branches")

        if has("user","users") and hasnot("sales","revenue","voucher","company","with","in","for"):
            return plan("find","IUser",fq={},proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},
                        limit=500,tmpl="All users.",ct="table",title="All Users")

        if re.match(r"^(list |show |get )?(all )?compan", q) and hasnot("with","in","for","sales","voucher"):
            return plan("find","ICompany",fq={},proj={"_id":0,"name":1,"industry":1,"financialYear":1},
                        limit=200,tmpl="All companies.",ct="table",title="All Companies")

        d = get_dates()

        # Monthly sales trend (last 12 months)
        if re.search(r"monthly.*trend|trend.*month|month.*sales|sales.*trend|last 12 month", q):
            yrs = [d["ty"]-1, d["ty"]]
            return plan("aggregate","ItemQuantityTracker",
                pipe=[
                    {"$match":{"voucherType":"sales","year":{"$in":yrs}}},
                    {"$group":{"_id":{"year":"$year","month":"$month"},
                               "amount":{"$sum":"$amount"},"qty":{"$sum":"$qty"}}},
                    {"$sort":{"_id.year":1,"_id.month":1}},
                    {"$project":{"_id":0,"year":"$_id.year","month":"$_id.month",
                                 "amount":1,"qty":1}}
                ],
                tmpl="Monthly sales trend for last 12 months.",
                ct="line",x="month",y="amount",title="Monthly Sales Trend")

        # Total sales this year
        if re.search(r"total.*revenue.*year|revenue.*this year|total.*sales.*year|sales.*this year|ytd", q):
            return plan("aggregate","ItemQuantityTracker",
                pipe=[
                    {"$match":{"voucherType":"sales","year":d["ty"]}},
                    {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"},
                               "total_qty":{"$sum":"$qty"}}},
                    {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
                ],
                tmpl=f"Total sales revenue for {d['ty']}.",
                ct="metric",y="total_revenue",title=f"Total Revenue {d['ty']}")

        # Total sales last month
        if re.search(r"sales.*last month|last month.*sales|revenue.*last month", q):
            return plan("aggregate","ItemQuantityTracker",
                pipe=[
                    {"$match":{"voucherType":"sales","year":d["lm_year"],"month":d["lm_num"]}},
                    {"$group":{"_id":None,"total_revenue":{"$sum":"$amount"},
                               "total_qty":{"$sum":"$qty"}}},
                    {"$project":{"_id":0,"total_revenue":1,"total_qty":1}}
                ],
                tmpl=f"Total sales for last month ({d['lm_num']}/{d['lm_year']}).",
                ct="metric",y="total_revenue",
                title=f"Sales - {calendar.month_abbr[d['lm_num']]} {d['lm_year']}")

        # Total sales vs purchases
        if re.search(r"sales.*vs.*purchase|purchase.*vs.*sales|compare.*sales|sales.*comparison", q):
            return plan("aggregate","Voucher",
                pipe=[
                    {"$match":{"type":{"$in":["sales","purchase"]},
                               "iCompanyId":{"$ne":None}}},
                    {"$group":{"_id":"$type",
                               "total":{"$sum":"$billFinalAmount"},
                               "count":{"$sum":1}}},
                    {"$project":{"_id":0,"type":"$_id","total":1,"count":1}}
                ],
                tmpl="Sales vs purchases comparison.",
                ct="bar",x="type",y="total",title="Sales vs Purchases")

        # Top customers
        if re.search(r"top.*customer|best.*customer|customer.*revenue|customer.*sales", q) and hasnot("company","with","in"):
            return plan("aggregate","Voucher",
                pipe=[
                    {"$match":{"type":"sales","iCompanyId":{"$ne":None},
                               "party.name":{"$ne":None}}},
                    {"$group":{"_id":"$party.name",
                               "revenue":{"$sum":"$billFinalAmount"},
                               "invoices":{"$sum":1}}},
                    {"$sort":{"revenue":-1}},{"$limit":15},
                    {"$project":{"_id":0,"customer":"$_id","revenue":1,"invoices":1}}
                ],
                tmpl="Top customers by revenue.",
                ct="bar",x="customer",y="revenue",title="Top 15 Customers by Revenue")

        # Top products
        if re.search(r"top.*product|best.*product|most.*sold|product.*revenue|which product", q) and hasnot("company","with","in"):
            return plan("aggregate","ItemQuantityTracker",
                pipe=[
                    {"$match":{"voucherType":"sales"}},
                    {"$group":{"_id":"$itemId",
                               "revenue":{"$sum":"$amount"},
                               "qty":{"$sum":"$qty"}}},
                    {"$sort":{"revenue":-1}},{"$limit":15},
                    {"$project":{"_id":0,"item":"$_id","revenue":1,"qty":1}}
                ],
                tmpl="Top 15 products by revenue.",
                ct="bar",x="item",y="revenue",title="Top Products by Revenue")

        # Avg order value
        if re.search(r"avg.*order|average.*order|order.*value|aov", q):
            return plan("aggregate","Voucher",
                pipe=[
                    {"$match":{"type":"sales","iCompanyId":{"$ne":None}}},
                    {"$group":{"_id":None,
                               "avg_order_value":{"$avg":"$billFinalAmount"},
                               "total_orders":{"$sum":1}}},
                    {"$project":{"_id":0,"avg_order_value":1,"total_orders":1}}
                ],
                tmpl="Average order value.",
                ct="metric",y="avg_order_value",title="Average Order Value")

        # Unpaid invoices
        if re.search(r"unpaid|outstanding|overdue|due amount", q):
            return plan("find","Voucher",
                fq={"status":"unpaid","iCompanyId":{"$ne":None}},
                proj={"_id":0,"voucherNo":1,"billFinalAmount":1,
                      "dueAmount":1,"issueDate":1,"status":1},
                limit=50,tmpl="Unpaid invoices.",
                ct="table",title="Unpaid Invoices")

        # Stock / inventory
        if re.search(r"stock|inventory|available.*qty|items.*stock", q):
            return plan("find","Item",
                fq={"isHidden":False,"availableQty":{"$gt":0}},
                proj={"_id":0,"name":1,"skuBarcode":1,"availableQty":1,"unit":1},
                limit=50,tmpl="Items with stock.",
                ct="table",title="Inventory / Stock")

        # Total purchases this year
        if re.search(r"total.*purchase.*year|purchase.*this year", q):
            return plan("aggregate","ItemQuantityTracker",
                pipe=[
                    {"$match":{"voucherType":"purchase","year":d["ty"]}},
                    {"$group":{"_id":None,"total_purchases":{"$sum":"$amount"}}},
                    {"$project":{"_id":0,"total_purchases":1}}
                ],
                tmpl=f"Total purchases for {d['ty']}.",
                ct="metric",y="total_purchases",title=f"Total Purchases {d['ty']}")

        if re.match(r"^how many (customer|supplier)", q):
            rel = "customer" if "customer" in q else "supplier"
            field = f"total_{rel}s"
            return plan("aggregate","Business",
                        pipe=[{"$match":{"relationType":rel}},{"$count":field}],
                        tmpl=f"Total {rel}s.",ct="metric",y=field,title=f"Total {rel.title()}s")
        return None

    # ── Main query ─────────────────────────────────────────────────────────────
    def query(self, question:str) -> Dict:
        if not self.llm:
            if not self.init_llm():
                return {"error":"GROQ_API_KEY not configured."}

        q_lower = question.lower().strip()

        # 1. Simple shortcut (no company context needed)
        shortcut = self._shortcut(q_lower)
        if shortcut:
            results, err = self._execute(shortcut)
            answer = self._answer(question, shortcut, results, err)
            chart  = self._make_chart(results, shortcut["chart_suggestion"])
            self.history.append({"q":question,"a":shortcut["answer_template"][:80]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":shortcut,"db_error":err}

        # 2. Detect if question is company-specific
        cname = extract_company_name(question)
        company = None
        if cname and self.client:
            company = find_company(self.client, cname)
            if company is None:
                return {
                    "type":"answer",
                    "answer": f"❌ No company matching **\"{cname}\"** found in the database.\n\nTry: *\"list all companies\"* to see available names.",
                    "results":[],"chart":None,"plan":{},"db_error":None
                }

        # 3. Company-specific: use DirectQueryEngine (no LLM for filters)
        if company and self.client:
            print(f"[Agent] Company resolved: '{company['name']}' real_id={company['real_id']!r} total_vouchers={company['total_vouchers']}")
            dqe = DirectQueryEngine(self.client, self.date_type)
            try:
                results_raw, chart_sug = dqe.handle(question, company)
                plan = {"query_type":"direct","collection":"Voucher",
                        "answer_template": f"Direct query for {company['name']}.",
                        "chart_suggestion": chart_sug, "clarification_needed": False}
                results = [deep_sanitize(r) for r in results_raw]
                answer  = self._answer(question, plan, results, None, company)
                chart   = self._make_chart(results, chart_sug)
                self.history.append({"q":question,"a":plan["answer_template"][:80]})
                return {"type":"answer","answer":answer,"results":results,
                        "chart":chart,"plan":plan,"db_error":None}
            except Exception as e:
                print(f"[Agent] DirectQueryEngine error: {e}")
                # Fall through to LLM

        # 4. General question — use LLM
        dates = get_dates()
        system_prompt = build_prompt(dates, self.date_type)
        hist = ""
        if self.history:
            hist = "\nPrevious:\n" + "\n".join(f"Q:{h['q']}\nA:{h['a']}" for h in self.history[-3:])
        user_msg = f"{SCHEMA}\n{hist}\n\nQuestion: {question}"
        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt),
                                    HumanMessage(content=user_msg)])
            plan = parse_json(resp.content)
        except Exception as e:
            return {"error":f"LLM error: {e}"}

        results, err = self._execute(plan)

        # 5. Auto-retry if empty
        if not results and not err and plan.get("query_type") != "none":
            results, err, plan = self._retry(question, plan, system_prompt, user_msg, dates)

        answer = self._answer(question, plan, results, err)
        chart  = self._make_chart(results, plan.get("chart_suggestion",{}))
        self.history.append({"q":question,"a":plan.get("answer_template","")[:80]})
        return {"type":"answer","answer":answer,"results":results,
                "chart":chart,"plan":plan,"db_error":err}

    def _resolve_col(self, col):
        if col in VALID_COLS: return col
        return {c.lower():c for c in VALID_COLS}.get((col or "").lower(), col)

    def _execute(self, plan) -> tuple:
        qt = plan.get("query_type")
        if qt not in ("aggregate","find") or not self.client: return [], None
        col = self._resolve_col(plan.get("collection",""))
        plan["collection"] = col
        db = get_db(self.client)
        try:
            if qt == "aggregate" and plan.get("pipeline"):
                pipe = dt_str_to_obj(plan["pipeline"]) if self.date_type=="date_object" else plan["pipeline"]
                return run_agg(db, col, pipe), None
            if qt == "find" and plan.get("find_query") is not None:
                fq   = dt_str_to_obj([plan["find_query"]])[0] if self.date_type=="date_object" else plan["find_query"]
                proj = plan.get("projection")
                srt  = list(plan["sort"].items()) if plan.get("sort") else None
                lim  = plan.get("limit", 100)
                return run_find(db, col, fq, proj, lim, srt), None
        except Exception as e:
            return [], str(e)
        return [], None

    def _retry(self, question, orig, sys_p, usr_msg, dates):
        d = dates
        hint = f"""
⚠️ RETRY — returned 0 results or had pipeline error.
Failed: {json.dumps(orig.get("pipeline"), default=str)[:200]}

CORRECT pipeline patterns (copy exactly):

Monthly trend:
[
  {{"$match":{{"voucherType":"sales","year":{{"$in":[{d['ty']-1},{d['ty']}]}}}}}},
  {{"$group":{{"_id":{{"year":"$year","month":"$month"}},"amount":{{"$sum":"$amount"}},"qty":{{"$sum":"$qty"}}}}}},
  {{"$sort":{{"_id.year":1,"_id.month":1}}}},
  {{"$project":{{"_id":0,"year":"$_id.year","month":"$_id.month","amount":1,"qty":1}}}}
]

Total this year:
[
  {{"$match":{{"voucherType":"sales","year":{d['ty']}}}}},
  {{"$group":{{"_id":null,"total_revenue":{{"$sum":"$amount"}}}}}},
  {{"$project":{{"_id":0,"total_revenue":1}}}}
]

Top customers (Voucher):
[
  {{"$match":{{"type":"sales","iCompanyId":{{"$ne":null}}}}}},
  {{"$group":{{"_id":"$party.name","revenue":{{"$sum":"$billFinalAmount"}},"count":{{"$sum":1}}}}}},
  {{"$sort":{{"revenue":-1}}}},{{"$limit":15}},
  {{"$project":{{"_id":0,"customer":"$_id","revenue":1,"count":1}}}}
]

RULES: 
- $group _id can be null or a field path like "$type"
- $sum MUST be {{"$sum": "$fieldName"}} — NOT {{"$sum": "fieldName"}}
- Every pipeline MUST end with $project that hides _id

Question: {question}"""
        try:
            resp = self.llm.invoke([SystemMessage(content=sys_p),
                                    HumanMessage(content=usr_msg+hint)])
            plan2 = parse_json(resp.content)
            r, e = self._execute(plan2)
            return r, e, plan2
        except Exception as e:
            return [], str(e), orig

    def _answer(self, question, plan, results, db_error, company=None):
        if db_error:
            return f"⚠️ **Database error:** `{db_error}`"
        if plan.get("query_type") == "none":
            return plan.get("answer_template","No data found.")

        if not results:
            if company:
                total = company.get("total_vouchers", 0)
                name  = company["name"]
                score = company.get("score", 1.0)
                matched = f"**{name}**" if score > 0.9 else f"**{name}** *(closest match to your query)*"
                if total == 0:
                    return (
                        f"{matched} is registered in the database ✓ but has "
                        f"**zero vouchers of any type** — this is a test/demo account "
                        f"with no real transactions.\n\n"
                        f"**Companies with real sales data you can query:**\n"
                        f"• M/S DIPSHI - ESTIMATE → 40,255 sales vouchers\n"
                        f"• HIRAKA JEWELS (NP) - ESTIMATE → 34,998 sales vouchers\n"
                        f"• Bhakti Parshwanath → 30,707 sales vouchers\n"
                        f"• NAMO-ESTIMATE → 27,478 sales vouchers\n"
                        f"• VAIBHAV FASHION JEWELLERY → 11,735 sales vouchers"
                    )
                return (
                    f"{matched} has **{total:,} total vouchers**, but none match "
                    f"this specific filter.\n\nTry asking about: "
                    f"*purchases, customers, monthly trend, revenue, unpaid invoices*"
                )
            return "**No records found.** The query ran but matched zero documents."

        co = f" for **{company['name']}**" if company else ""
        prompt = (
            f"You are a precise data analyst for Invock ERP (India, ₹ INR).\n"
            f"Question: {question}\n"
            f"Company: {company['name'] if company else 'all companies'}\n\n"
            f"Data ({len(results)} records){co}:\n"
            f"{json.dumps(results[:10], default=str, indent=2)}\n\n"
            f"STRICT RULES — violations will be flagged by the reviewer:\n"
            f"1. ONLY use numbers that appear in the data above — NEVER invent figures\n"
            f"2. State counts/totals precisely with ₹ crore/lakh formatting\n"
            f"3. If data shows a count (like total_vouchers), lead with that exact number\n"
            f"4. Name real top performers from the data with exact figures\n"
            f"5. ONE actionable insight based on real numbers only\n"
            f"6. Keep to 3 sentences max. No hallucination. No generic advice."
        )
        try: return self.llm.invoke([HumanMessage(content=prompt)]).content
        except: return f"Found {len(results)} record(s){co}."

    def _make_chart(self, results, suggestion):
        if not results or not suggestion or suggestion.get("type") in ("none",None):
            return None
        try:
            clean = []
            for doc in results:
                row = {}
                for k,v in doc.items():
                    if isinstance(v,bool): row[k]=str(v)
                    elif isinstance(v,(int,float,type(None))): row[k]=v
                    elif isinstance(v,str): row[k]=v
                    elif isinstance(v,dict): row[k]=str(v)
                    elif isinstance(v,list): row[k]=len(v)
                    else: row[k]=str(v)
                clean.append(row)
            df = pd.DataFrame(clean)
            if df.empty: return None
            for c in df.columns:
                try: df[c]=pd.to_numeric(df[c])
                except: pass
            num = df.select_dtypes(include="number").columns.tolist()
            cat = df.select_dtypes(exclude="number").columns.tolist()
            x = suggestion.get("x_field"); y = suggestion.get("y_field")
            if not x or x not in df.columns: x = cat[0] if cat else (df.columns[0] if len(df.columns) else None)
            if not y or y not in df.columns: y = num[0] if num else (df.columns[1] if len(df.columns)>1 else None)
            return {"type":suggestion.get("type","bar"),"df":df,"x":x,"y":y,
                    "title":suggestion.get("title","Results")}
        except: return None

# ── Company knowledge base (built from debug_all_companies.py output) ─────────
# Maps company name → sales voucher count for instant context
COMPANY_VOUCHER_COUNTS = {
    "M/S DIPSHI - ESTIMATE": 40255,
    "HIRAKA JEWELS (NP) - ESTIMATE": 34998,
    "Bhakti Parshwanath": 30707,
    "NAMO-ESTIMATE": 27478,
    "DIPSHI CREATION PRIVATE LIMITED": 21350,
    "VAIBHAV FASHION JEWELLERY": 11735,
    "SHUBHAM JEWELLERY-Estimate": 9385,
    "Old G.S. Est": 7879,
    "KLITZ EST": 6759,
    "KIRAN ENTERPRISE": 5022,
    "SHUBH NX": 4086,
    "SFJ Est": 2936,
    "SANTOSH JEWELLERS": 823,
    "Gsh": 693,
    "SALONI FASHION JEWELLERY": 1712,
    "Sanskriti": 1495,
    "Santosh": 1306,
    "HASU JEWELLERS": 1924,
    "Web Enhancement UI": 444,
    "Aman Demo Company": 913,
    # Companies with 0 sales vouchers (test/empty accounts)
    "NAMO SHIVAYA": 0,
    "Shraddha Jewellery": 0,
    "API Testing": 0,
    "Aman/Invock": 0,
    "HUSSAIN/CCC": 0,
}
