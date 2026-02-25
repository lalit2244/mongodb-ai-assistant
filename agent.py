"""
MongoDB AI Agent - Production Grade
Database: dev-cluster (Invock ERP)
"""
import os, json, re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster")
DB_NAME = "dev-cluster"

def get_mongo_client():
    try:
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        c.admin.command("ping")
        return c
    except Exception as e:
        print(f"[MongoDB] {e}")
        return None

def get_db(client): return client[DB_NAME]

def deep_sanitize(obj):
    if isinstance(obj, dict):   return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [deep_sanitize(i) for i in obj]
    if isinstance(obj, ObjectId): return str(obj)
    if isinstance(obj, datetime): return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, float) and obj != obj: return None   # NaN
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    return str(obj)

def force_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make every column Arrow/PyArrow serialisable."""
    for col in df.columns:
        # Mixed-type object columns (e.g. bool stored as string) → all str
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x) if not isinstance(x, (str, type(None))) else x
            )
        # Explicit bool columns → str to avoid Arrow bool/str confusion
        elif df[col].dtype == bool:
            df[col] = df[col].astype(str)
    return df

def execute_agg(client, col, pipeline):
    try:
        raw = list(get_db(client)[col].aggregate(pipeline, allowDiskUse=True))
        return [deep_sanitize(d) for d in raw], None
    except Exception as e:
        return [], str(e)

def execute_find(client, col, query, proj=None, limit=50):
    try:
        raw = list(get_db(client)[col].find(query, proj or {"_id": 0}).limit(limit))
        return [deep_sanitize(d) for d in raw], None
    except Exception as e:
        return [], str(e)

def detect_date_type(client) -> str:
    try:
        s = get_db(client)["Voucher"].find_one({"type": "sales"}, {"_id": 0, "issueDate": 1})
        if s:
            return "date_object" if isinstance(s.get("issueDate"), datetime) else "string"
    except: pass
    return "string"

def get_collection_stats(client):
    stats = {}
    for col in ["Voucher","Item","Business","ItemQuantityTracker","ItemSummary","Contact","Account","IUser","IBranch","ICompany"]:
        try: stats[col] = get_db(client)[col].count_documents({})
        except: stats[col] = 0
    return stats

def get_dates():
    now = datetime.utcnow()
    fm  = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    lme = fm - timedelta(seconds=1)
    lms = lme.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ys  = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return {
        "now": now, "year_start": ys,
        "last_month_start": lms, "last_month_end": lme,
        "this_month_start": fm,  "today_start": now.replace(hour=0,minute=0,second=0,microsecond=0),
        "last_12m_start": now - timedelta(days=365),
        "lm_num": lme.month, "lm_year": lme.year,
        "tm_num": now.month, "ty": now.year,
    }

def get_llm():
    k = os.getenv("GROQ_API_KEY")
    if not k: raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=k, temperature=0)

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
DATABASE: dev-cluster — Invock ERP (Jewellery, India, ₹ INR)

══ Voucher (1,317,598 docs) ── ALL financial transactions
  type              "sales"|"purchase"|"receipt"|"payment"|"credit note"|"debit note"|"journal"
  voucherNo         string  e.g. "SA/2324/PL/1"
  issueDate         DATE OBJECT ← use Python datetime in pipeline via DATETIME_VARS below
  billFinalAmount   float  ← total invoice amount (revenue)
  billItemsPrice    float  ← subtotal
  billTaxAmount     float
  billDiscountAmount float
  dueAmount         float  ← outstanding
  paidAmount        float
  status            "unpaid"|"paid"|"partial"
  iCompanyId        string  (filter $ne null)
  iBranchId         string
  lineItemQtySum    float
  lineAmountSum     float
  party.name        string  ← customer/supplier name inside party sub-doc

  SAFE find projection (NEVER include itemList,transactions,tax,voucherList,otherCharges,party):
  {"_id":0,"type":1,"voucherNo":1,"issueDate":1,"billFinalAmount":1,
   "billItemsPrice":1,"billTaxAmount":1,"status":1,"dueAmount":1,
   "paidAmount":1,"iCompanyId":1,"iBranchId":1,"lineItemQtySum":1}

══ ItemQuantityTracker (2,135,362 docs) ← PREFER for date-based product analytics
  voucherType  "sales"|"purchase"|"stock_adjustment"
  month        float  1.0-12.0
  year         integer  e.g. 2024
  itemId       string
  qty          float   ← quantity
  amount       float   ← revenue for this item
  iCompanyId   string

══ Item (450,407 docs) ← product catalog
  name, skuBarcode, itemCode  string
  unit  "pcs"|"gm"
  unitPurchasePrice, unitSellWholeSalePrice, unitSellRetailPrice  float
  availableQty  float  ← current stock
  taxPercentage float
  iCompanyId, iBranchId, itemGroupId  string
  isHidden, isService  boolean

══ Business (45,119 docs) ← customers & suppliers
  name, aliasName  string
  relationType  "customer"|"supplier"|"both"
  city, state  string
  iCompanyId   string

══ ItemSummary (147,384 docs)
  type  "sales"|"purchase"
  issueDateMonth  int  YYYYMM
  issueDateYr     int
  itemId, itemGroupId, iCompanyId  string
  amount, qty  float

══ Other:
  Account(51K)   name, accountGroupName, balance, iCompanyId
  IBranch(264)   name, city, state, code, iCompanyId
  IUser(399)     name, phone, lastSignIn  (phoneVerified is string "true"/"false")
  ICompany(135)  name, industry, financialYear
  ItemGroup(1.8K) name, taxPercentage, hsn
  Contact(39K)   name, phone, email
  voucher_count(720K)  vouchercount int, iCompanyId string
  company_data(48)     name, items, business, user, expiryDate

══ QUESTION → EXACT COLLECTION (MANDATORY — do not deviate):
  "total sales"/"revenue"         → Voucher, type="sales", $sum billFinalAmount → metric
  "total purchases"               → Voucher, type="purchase", $sum billFinalAmount → metric
  "sales vs purchases"            → Voucher, match type in ["sales","purchase"], group "$type", $sum billFinalAmount → bar
  "which product sold most"       → ItemQuantityTracker, voucherType="sales", group itemId, $sum qty, sort desc → bar
  "top products by revenue"       → ItemQuantityTracker, voucherType="sales", group itemId, $sum amount → bar
  "monthly sales trend"           → ItemQuantityTracker, voucherType="sales", group {year,month}, $sum amount, sort → line
  "top customers"                 → Voucher, type="sales", group "$party.name", $sum billFinalAmount → bar
  "how many customers"            → Business, relationType="customer", $count → metric (output: customer_count)
  "list customers"                → Business, find {relationType:"customer"}, projection {_id:0,name:1,city:1,state:1}
  "how many suppliers"            → Business, relationType="supplier", $count → metric
  "avg order value"               → Voucher, type="sales", $avg billFinalAmount → metric
  "unpaid invoices"               → Voucher, find {status:"unpaid"}, SAFE PROJECTION ONLY
  "stock"/"inventory"             → Item, find {isHidden:false}, projection {_id:0,name:1,skuBarcode:1,availableQty:1,unit:1}, sort availableQty desc
  "how many branches"/"branches"/"list branches" → IBranch ONLY, find {}, projection {_id:0,name:1,city:1,state:1,code:1}, NO other collection
  "how many users"/"list users"   → IUser ONLY, find {}, projection {_id:0,name:1,phone:1,lastSignIn:1}
  "companies"/"company list"      → ICompany ONLY, find {}, projection {_id:0,name:1,industry:1,financialYear:1}
  "voucher count by company"      → voucher_count, group iCompanyId, $sum vouchercount → bar
  "sales last month"              → ItemQuantityTracker, year=LM_YEAR, month=LM_NUM, voucherType="sales", $sum amount → metric
  "total revenue this year"       → ItemQuantityTracker, year=TY, voucherType="sales", $sum amount → metric
  "sales trend 12 months"         → ItemQuantityTracker, year in [TY-1,TY], voucherType="sales", group {year,month}, $sum amount → line
"""

def build_prompt(dates, date_type):
    d = dates
    lms = d["last_month_start"].strftime("%Y-%m-%dT%H:%M:%S")
    lme = d["last_month_end"].strftime("%Y-%m-%dT%H:%M:%S")
    ys  = d["year_start"].strftime("%Y-%m-%dT%H:%M:%S")
    l12 = d["last_12m_start"].strftime("%Y-%m-%dT%H:%M:%S")
    ts  = d["today_start"].strftime("%Y-%m-%dT%H:%M:%S")
    now = d["now"].strftime("%Y-%m-%dT%H:%M:%S")

    date_note = f"""issueDate is stored as MongoDB Date object.
In aggregation $match, use ISODate strings like: {{"$gte": ISODate("{lms}"), "$lt": ISODate("{lme}")}}
In Python pymongo pipeline write as strings — the agent converts them to datetime automatically.
PREFER ItemQuantityTracker for date queries — uses integer year/month, no date conversion needed."""

    return f"""You are a senior MongoDB data analyst for Invock ERP jewellery business (India, ₹ INR).
Return ONLY a single valid JSON object. No markdown, no backticks, no explanation.

{{
  "query_type": "aggregate" | "find" | "none",
  "collection": "<exact collection name>",
  "pipeline": [...] | null,
  "find_query": {{...}} | null,
  "projection": {{"_id": 0}} | null,
  "limit": 50,
  "answer_template": "<one sentence>",
  "chart_suggestion": {{
    "type": "bar" | "line" | "pie" | "metric" | "table" | "none",
    "x_field": "<exact output field name after $project>",
    "y_field": "<exact output field name after $project>",
    "title": "<descriptive title>"
  }},
  "clarification_needed": false
}}

DATE INFO:
  LM_NUM={d['lm_num']}  LM_YEAR={d['lm_year']}  TM_NUM={d['tm_num']}  TY={d['ty']}
  last_month_start="{lms}"  last_month_end="{lme}"
  year_start="{ys}"  last_12m_start="{l12}"
  today_start="{ts}"  now="{now}"

{date_note}

STRICT RULES:
1. Collection names CASE-SENSITIVE: Voucher, Item, Business, ItemQuantityTracker, ItemSummary, Contact, Account, IBranch, IUser, ICompany, ItemGroup, company_data, voucher_count
2. Every aggregate pipeline MUST end with {{"$project": {{"_id": 0, "field1":1, "field2":1, ...}}}} — only scalar output fields, no nested objects
3. For Voucher find: use SAFE PROJECTION only (listed in schema)
4. For date queries use ItemQuantityTracker (year/month integers) — much simpler
5. For "top customers": Voucher aggregate, group by "$party.name", $sum billFinalAmount → project as {{party_name, total_revenue}}
6. For "sales vs purchases": Voucher aggregate, match type in ["sales","purchase"], group by "$type", $sum billFinalAmount → project as {{voucher_type, total}}
7. For "monthly trend": ItemQuantityTracker, group by {{year:"$year",month:"$month"}}, $sum amount → project as {{year,month,total_amount}}, sort year/month asc
8. x_field/y_field must match EXACT field names in the $project output
9. For metric chart: single number result, y_field = the numeric output field
10. Always filter iCompanyId: {{$ne: null}} to exclude test records
11. Limit to 20 results unless more asked
12. clarification_needed = false always"""

def parse_json(text):
    text = re.sub(r"```(?:json)?\n?", "", text.strip()).strip("`").strip()
    try: return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {"query_type":"none","collection":None,"pipeline":None,"find_query":None,
            "answer_template":"Could not parse response.","chart_suggestion":{"type":"none"},"clarification_needed":False}

def convert_dt_strings(obj):
    """Recursively convert ISO date strings in a pipeline to datetime objects."""
    if isinstance(obj, dict):  return {k: convert_dt_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [convert_dt_strings(i) for i in obj]
    if isinstance(obj, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try: return datetime.strptime(obj, fmt)
            except: pass
    return obj

# ── Agent ─────────────────────────────────────────────────────────────────────
VALID_COLS = {
    "Voucher","Item","Business","ItemQuantityTracker","ItemSummary","Contact",
    "Account","IBranch","IUser","ICompany","ItemGroup","company_data","voucher_count",
    "AccountGroup","IBank","ItemColor","ItemCollection","ItemTag","ItemTagItem",
    "IRoleConfig","ItemAttribute","ItemAttributeValue","ItemService","demoContact",
    "PartyTag","AccessToken","BusinessContact","VersionTracking"
}

class MongoAIAgent:
    def __init__(self):
        self.client     = get_mongo_client()
        self.llm        = None
        self.history    = []
        self.collection_stats = {}
        self.date_type  = "string"
        if self.client:
            try:
                self.collection_stats = get_collection_stats(self.client)
                self.date_type = detect_date_type(self.client)
                print(f"[Agent] issueDate type: {self.date_type}")
            except Exception as e:
                print(f"[Agent] init: {e}")

    def is_connected(self): return self.client is not None

    def _keyword_shortcut(self, q: str):
        """Hardcoded plans for simple questions — bypasses LLM entirely."""
        def plan(qt, col, pipeline=None, find_query=None, proj=None, limit=50,
                 template="", chart_type="table", x=None, y=None, title=""):
            return {"query_type": qt, "collection": col, "pipeline": pipeline,
                    "find_query": find_query, "projection": proj, "limit": limit,
                    "answer_template": template,
                    "chart_suggestion": {"type": chart_type, "x_field": x, "y_field": y, "title": title},
                    "clarification_needed": False}
        if any(w in q for w in ["branch","branches","location"]) and not any(w in q for w in ["sales","revenue","voucher","customer"]):
            return plan("find","IBranch",find_query={},proj={"_id":0,"name":1,"city":1,"state":1,"code":1,"gstBusinessType":1},limit=100,template="All branches.",chart_type="table",title="All Branches")
        if any(w in q for w in ["user","users","staff"]) and not any(w in q for w in ["sales","revenue","voucher"]):
            return plan("find","IUser",find_query={},proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},limit=100,template="All users.",chart_type="table",title="All Users")
        if any(w in q for w in ["compan","companies"]) and "most voucher" not in q:
            return plan("find","ICompany",find_query={},proj={"_id":0,"name":1,"industry":1,"financialYear":1},limit=100,template="All companies.",chart_type="table",title="All Companies")
        if "how many customer" in q:
            return plan("aggregate","Business",pipeline=[{"$match":{"relationType":"customer"}},{"$count":"customer_count"}],template="Total customers.",chart_type="metric",y="customer_count",title="Total Customers")
        if "how many supplier" in q:
            return plan("aggregate","Business",pipeline=[{"$match":{"relationType":"supplier"}},{"$count":"supplier_count"}],template="Total suppliers.",chart_type="metric",y="supplier_count",title="Total Suppliers")
        return None

    def refresh_schema(self):
        if self.client:
            self.collection_stats = get_collection_stats(self.client)
            self.date_type = detect_date_type(self.client)

    def init_llm(self):
        try: self.llm = get_llm(); return True
        except: return False

    def query(self, question: str) -> Dict:
        if not self.llm:
            if not self.init_llm():
                return {"error": "GROQ_API_KEY not configured."}

        # ── Keyword shortcut: force correct collection before LLM call ─────────
        q_lower = question.lower().strip()
        shortcut = self._keyword_shortcut(q_lower)
        if shortcut:
            results, err = self._run(shortcut)
            answer = self._answer(question, shortcut, results, err)
            chart  = self._chart(results, shortcut.get("chart_suggestion", {}))
            self.history.append({"q": question, "a": shortcut.get("answer_template","")[:100]})
            return {"type":"answer","answer":answer,"results":results,"chart":chart,"plan":shortcut,"db_error":err}

        dates = get_dates()
        system_prompt = build_prompt(dates, self.date_type)

        hist = ""
        if self.history:
            hist = "\nConversation history:\n" + "\n".join(f"Q:{h['q']}\nA:{h['a']}" for h in self.history[-3:])

        user_msg = f"{SCHEMA}\n{hist}\n\nQuestion: {question}"

        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
            plan = parse_json(resp.content)
        except Exception as e:
            return {"error": f"LLM error: {e}"}

        results, err = self._run(plan)

        if not results and not err and plan.get("query_type") != "none":
            results, err, plan = self._retry(question, plan, system_prompt, user_msg, dates)

        answer = self._answer(question, plan, results, err)
        chart  = self._chart(results, plan.get("chart_suggestion", {}))
        self.history.append({"q": question, "a": plan.get("answer_template","")[:100]})

        return {"type":"answer","answer":answer,"results":results,"chart":chart,"plan":plan,"db_error":err}

    def _resolve_col(self, col):
        if col in VALID_COLS: return col
        lm = {c.lower(): c for c in VALID_COLS}
        return lm.get((col or "").lower(), col)

    def _run(self, plan):
        qt = plan.get("query_type")
        if qt not in ("aggregate","find") or not self.client: return [], None
        col = self._resolve_col(plan.get("collection",""))
        plan["collection"] = col
        if qt == "aggregate" and plan.get("pipeline"):
            pipeline = convert_dt_strings(plan["pipeline"]) if self.date_type == "date_object" else plan["pipeline"]
            return execute_agg(self.client, col, pipeline)
        if qt == "find" and plan.get("find_query") is not None:
            fq = convert_dt_strings([plan["find_query"]])[0] if self.date_type == "date_object" else plan["find_query"]
            return execute_find(self.client, col, fq, plan.get("projection"), plan.get("limit",50))
        return [], None

    def _retry(self, question, orig, system_prompt, user_msg, dates):
        d = dates
        hint = f"""
⚠️ RETRY — '{orig.get("collection")}' returned 0 results.
Bad pipeline: {json.dumps(orig.get("pipeline"), default=str)[:300]}

Try ItemQuantityTracker instead (uses integers, no date issues):
  For last month: {{"voucherType":"sales","year":{d['lm_year']},"month":{d['lm_num']}}}
  For this year:  {{"voucherType":"sales","year":{d['ty']}}}
  For 12-month trend: match year in [{d['ty']-1},{d['ty']}], group by year+month

Or for Voucher without date filter:
  Sales vs purchase total: group by "$type" (no date needed)
  Top customers: group by "$party.name" (no date needed)

Question: {question}"""
        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg + hint)])
            plan2 = parse_json(resp.content)
            r, e = self._run(plan2)
            return r, e, plan2
        except Exception as e:
            return [], str(e), orig

    def _answer(self, question, plan, results, db_error):
        if db_error: return f"⚠️ **Database error:** {db_error}"
        if plan.get("query_type") == "none": return plan.get("answer_template","No relevant data found.")
        if not results: return f"No records found for: **{question}**\n\nThe query ran but matched no documents."
        prompt = (
            f"You are a CFO presenting to investors for Invock ERP (jewellery business, India, ₹ INR).\n"
            f"Question: {question}\n\n"
            f"Data ({len(results)} records):\n{json.dumps(results[:15], default=str, indent=2)}\n\n"
            f"Write 3-5 sentences of sharp business analysis:\n"
            f"• Use ₹ with crore/lakh formatting (1Cr=10M, 1L=100K)\n"
            f"• Name top performers with exact figures\n"
            f"• Identify a key trend or opportunity\n"
            f"• End with one actionable insight\n"
            f"Be precise and impressive — this is for a recruiter demo."
        )
        try: return self.llm.invoke([HumanMessage(content=prompt)]).content
        except: return f"Found {len(results)} records. {plan.get('answer_template','')}"

    def _chart(self, results, suggestion):
        if not results or not suggestion or suggestion.get("type") in ("none",None): return None
        try:
            clean = []
            for doc in results:
                row = {}
                for k, v in doc.items():
                    if isinstance(v, (int, float, type(None))): row[k] = v
                    elif isinstance(v, bool): row[k] = str(v)
                    elif isinstance(v, str): row[k] = v
                    elif isinstance(v, dict): row[k] = str(v)
                    elif isinstance(v, list): row[k] = len(v)
                    else: row[k] = str(v)
                clean.append(row)
            df = pd.DataFrame(clean)
            if df.empty: return None
            df = force_arrow_safe(df)
            # numeric coercion
            for c in df.columns:
                try: df[c] = pd.to_numeric(df[c])
                except: pass
            num = df.select_dtypes(include="number").columns.tolist()
            cat = df.select_dtypes(exclude="number").columns.tolist()
            x = suggestion.get("x_field"); y = suggestion.get("y_field")
            if not x or x not in df.columns: x = cat[0] if cat else (df.columns[0] if len(df.columns) else None)
            if not y or y not in df.columns: y = num[0] if num else (df.columns[1] if len(df.columns)>1 else None)
            return {"type": suggestion.get("type","bar"), "df": df, "x": x, "y": y,
                    "title": suggestion.get("title","Results")}
        except: return None