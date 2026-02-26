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

MONGODB_URI = os.getenv("MONGODB_URI","mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster")
DB_NAME = "dev-cluster"

# ── MongoDB helpers ────────────────────────────────────────────────────────────

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
    if isinstance(obj, dict):    return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [deep_sanitize(i) for i in obj]
    if isinstance(obj, ObjectId): return str(obj)
    if isinstance(obj, datetime): return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, float) and obj != obj: return None
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    return str(obj)

def execute_agg(client, col, pipeline):
    try:
        raw = list(get_db(client)[col].aggregate(pipeline, allowDiskUse=True))
        return [deep_sanitize(d) for d in raw], None
    except Exception as e:
        return [], str(e)

def execute_find(client, col, query, proj=None, limit=50):
    try:
        raw = list(get_db(client)[col].find(query, proj or {"_id":0}).limit(limit))
        return [deep_sanitize(d) for d in raw], None
    except Exception as e:
        return [], str(e)

def detect_date_type(client) -> str:
    try:
        s = get_db(client)["Voucher"].find_one({"type":"sales"},{"_id":0,"issueDate":1})
        if s: return "date_object" if isinstance(s.get("issueDate"), datetime) else "string"
    except: pass
    return "string"

def get_collection_stats(client):
    stats = {}
    for col in ["Voucher","Item","Business","ItemQuantityTracker","ItemSummary",
                "Contact","Account","IUser","IBranch","ICompany"]:
        try: stats[col] = get_db(client)[col].count_documents({})
        except: stats[col] = 0
    return stats

# ── Fuzzy name resolver ────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def fuzzy_score(query: str, candidate: str) -> float:
    """
    Detailed similarity score 0–1.
    Priority: exact > full-token-overlap > partial-token > trigram.
    Crucially: "namo shivaya" must score higher against "NAMO SHIVAYA"
    than against "NAMO-ESTIMATE".
    """
    q, c = normalize(query), normalize(candidate)
    if not q or not c: return 0.0
    if q == c: return 1.0                        # exact
    if q in c or c in q: return 0.92            # one contains the other

    qt = [t for t in q.split() if len(t) > 1]  # keep order
    ct = [t for t in c.split() if len(t) > 1]
    qt_set, ct_set = set(qt), set(ct)

    if not qt_set or not ct_set: return 0.0

    # ── Full token overlap (both tokens must match) ────────────────────────
    exact_overlap = len(qt_set & ct_set)
    token_score   = exact_overlap / max(len(qt_set), len(ct_set))

    # ── How many query tokens appear as prefix of candidate tokens ─────────
    prefix_matches = 0
    for qw in qt_set:
        for cw in ct_set:
            if cw.startswith(qw) or qw.startswith(cw):
                prefix_matches += 1; break
    prefix_score = prefix_matches / max(len(qt_set), 1)

    # ── Penalty when candidate has many extra tokens (avoids false matches) ─
    extra_tokens = len(ct_set - qt_set)
    penalty = min(extra_tokens * 0.08, 0.30)

    # ── Trigram similarity (catches typos like "eatimaye" → "estimate") ────
    def tgrams(s): return set(s[i:i+3] for i in range(len(s) - 2))
    tq, tc = tgrams(q), tgrams(c)
    tri = len(tq & tc) / max(len(tq | tc), 1) if (tq and tc) else 0.0

    # Weighted combination — token match matters most
    combined = (token_score * 0.55) + (prefix_score * 0.25) + (tri * 0.20) - penalty
    return max(0.0, min(combined, 1.0))

def find_best_company(client, user_name: str) -> Optional[Dict]:
    """
    Find the ICompany that best matches user_name.
    Always scores ALL candidates found by regex, returns the HIGHEST scorer.
    Never returns a match below the confidence threshold.
    """
    db        = get_db(client)
    norm_q    = normalize(user_name)
    words     = [w for w in norm_q.split() if len(w) > 2]

    # 1. Exact case-insensitive match
    doc = db["ICompany"].find_one(
        {"name": {"$regex": f"^{re.escape(user_name)}$", "$options": "i"}},
        {"_id": 1, "name": 1})
    if doc:
        return {"_id": str(doc["_id"]), "_id_obj": doc["_id"],
                "name": doc["name"], "score": 1.0}

    # 2. Collect ALL candidates that contain ANY query word
    candidate_set: Dict[str, dict] = {}   # deduplicate by _id string
    for word in words:
        docs = list(db["ICompany"].find(
            {"name": {"$regex": word, "$options": "i"}},
            {"_id": 1, "name": 1}))
        for d in docs:
            candidate_set[str(d["_id"])] = d

    # 3. Also grab top-200 for full fuzzy scan (catches names with no word overlap)
    all_docs = list(db["ICompany"].find({}, {"_id": 1, "name": 1}))
    for d in all_docs:
        candidate_set[str(d["_id"])] = d

    # 4. Score every candidate and pick the best
    best_score, best_doc = 0.0, None
    for d in candidate_set.values():
        sc = fuzzy_score(user_name, d.get("name", ""))
        if sc > best_score:
            best_score, best_doc = sc, d

    # Debug: print top matches
    top = sorted([(fuzzy_score(user_name, d.get("name","")), d.get("name",""))
                  for d in candidate_set.values()], reverse=True)[:4]
    print(f"[Fuzzy] '{user_name}' top matches: {top}")

    # Require meaningful confidence — avoids false positives
    threshold = 0.30
    if best_score >= threshold and best_doc:
        return {"_id": str(best_doc["_id"]), "_id_obj": best_doc["_id"],
                "name": best_doc["name"], "score": best_score}
    return None

def extract_company_name(question: str) -> Optional[str]:
    """
    Extract a company/org name from the question.
    Handles patterns like:
      'in company Namo eatimaye'  → 'Namo eatimaye'
      'company with Namo eatimaye' → 'Namo eatimaye'
      'for NAMO SHIVAYA'          → 'NAMO SHIVAYA'
      'vouchers of XYZ Ltd'       → 'XYZ Ltd'
    """
    # Strip question words that confuse the pattern
    q = question.strip()

    patterns = [
        # "company with/named/called X"
        r"company\s+(?:with|named?|called?|of)?\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,}?)['\"]?(?:\s*\?|$|\.|,)",
        # "in/for/of company X"
        r"(?:in|for|of)\s+company\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,}?)['\"]?(?:\s*\?|$|\.|,)",
        # "in/with/for X company"
        r"(?:in|with|for)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,}?)['\"]?\s+company(?:\s*\?|$|\.|,)?",
        # "vouchers/sales/records in/of X"
        r"(?:vouchers?|sales?|records?|items?|purchases?)\s+(?:in|of|for)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,}?)['\"]?(?:\s*\?|$|\.|,)",
        # "X company has/have/contains"
        r"['\"]?([A-Za-z0-9][A-Za-z0-9 /\-&.]{2,}?)['\"]?\s+(?:company|org|organisation|organization)",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # strip trailing noise words
            name = re.sub(r'\s*(company|the|a|an|in|for|of|with|has|have)\s*$',
                          '', name, flags=re.I).strip()
            if len(name) >= 3:
                return name
    return None

# ── Date helpers ───────────────────────────────────────────────────────────────

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
        "last_12m_start": now - timedelta(days=365),
        "lm_num": lme.month, "lm_year": lme.year,
        "tm_num": now.month, "ty": now.year,
    }

def convert_dt_strings(obj):
    if isinstance(obj, dict):  return {k: convert_dt_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [convert_dt_strings(i) for i in obj]
    if isinstance(obj, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S","%Y-%m-%d"]:
            try: return datetime.strptime(obj, fmt)
            except: pass
    return obj

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    k = os.getenv("GROQ_API_KEY")
    if not k: raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=k, temperature=0)

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
DATABASE: dev-cluster — Invock ERP (Jewellery, India, ₹ INR)

══ Voucher (1,317,598 docs)
  type              "sales"|"purchase"|"receipt"|"payment"|"credit note"|"debit note"|"journal"
  voucherNo         string
  issueDate         DATE OBJECT
  billFinalAmount   float  ← revenue total
  billItemsPrice    float
  billTaxAmount     float
  billDiscountAmount float
  dueAmount         float
  paidAmount        float
  status            "unpaid"|"paid"|"partial"
  iCompanyId        string  ← COMPANY FILTER KEY
  iBranchId         string
  lineItemQtySum    float
  party.name        string

  SAFE find projection: {"_id":0,"type":1,"voucherNo":1,"issueDate":1,"billFinalAmount":1,
   "billTaxAmount":1,"status":1,"dueAmount":1,"paidAmount":1,"iCompanyId":1,"lineItemQtySum":1}
  NEVER include: itemList,transactions,tax,voucherList,otherCharges,party

══ ItemQuantityTracker (2,135,362 docs) ← BEST for date/product analytics
  voucherType  "sales"|"purchase"|"stock_adjustment"
  month        float  1-12
  year         int
  itemId       string
  qty          float
  amount       float
  iCompanyId   string

══ Item (450,407 docs)
  name, skuBarcode, itemCode  string
  unit  "pcs"|"gm"
  unitPurchasePrice, unitSellWholeSalePrice, unitSellRetailPrice  float
  availableQty  float
  iCompanyId, iBranchId  string
  isHidden, isService  boolean

══ Business (45,119 docs)
  name, aliasName  string
  relationType  "customer"|"supplier"|"both"
  city, state  string
  iCompanyId   string

══ ICompany (135 docs)
  name  string  ← company name
  industry  string
  financialYear  string
  _id  ObjectId  ← use str(_id) as iCompanyId filter in other collections

══ IBranch(264)  name,city,state,code,iCompanyId
══ IUser(399)    name,phone,lastSignIn
══ ItemGroup(1.8K) name,taxPercentage,hsn
══ voucher_count(720K) vouchercount int, iCompanyId string
══ company_data(48) name,items,business,user,expiryDate
══ ItemSummary(147K) type,issueDateMonth,issueDateYr,itemId,amount,qty,iCompanyId

══ QUESTION → COLLECTION MAP:
  total sales              → Voucher, type="sales", $sum billFinalAmount
  total purchases          → Voucher, type="purchase", $sum billFinalAmount
  sales vs purchases       → Voucher, group "$type", $sum billFinalAmount → bar
  top products qty         → ItemQuantityTracker, voucherType="sales", group itemId, $sum qty
  top products revenue     → ItemQuantityTracker, voucherType="sales", group itemId, $sum amount
  monthly trend            → ItemQuantityTracker, voucherType="sales", group {year,month}, $sum amount
  top customers            → Voucher, type="sales", group "$party.name", $sum billFinalAmount
  count customers          → Business, relationType="customer", $count
  avg order value          → Voucher, type="sales", $avg billFinalAmount
  unpaid invoices          → Voucher, find status="unpaid", SAFE PROJECTION
  stock/inventory          → Item, isHidden=false, sort availableQty desc
  branches                 → IBranch, find all
  users                    → IUser, find all
  companies                → ICompany, find all
  vouchers IN A COMPANY    → Voucher, filter iCompanyId="<resolved_id>", SAFE PROJECTION or count
  sales last month         → ItemQuantityTracker, year=LM_YEAR, month=LM_NUM, $sum amount
  revenue this year        → ItemQuantityTracker, year=TY, $sum amount
  trend 12 months          → ItemQuantityTracker, year in [TY-1,TY], group {year,month}, $sum amount
"""

def build_prompt(dates, date_type, company_context=""):
    d = dates
    lms = d["last_month_start"].strftime("%Y-%m-%dT%H:%M:%S")
    lme = d["last_month_end"].strftime("%Y-%m-%dT%H:%M:%S")
    ys  = d["year_start"].strftime("%Y-%m-%dT%H:%M:%S")
    l12 = d["last_12m_start"].strftime("%Y-%m-%dT%H:%M:%S")
    ts  = d["today_start"].strftime("%Y-%m-%dT%H:%M:%S")

    company_note = ""
    if company_context:
        company_note = f"\n⚠️ COMPANY RESOLVED: {company_context}\nUse the iCompanyId above to filter Voucher/Item/etc.\n"

    return f"""You are a senior MongoDB analyst for Invock ERP (jewellery business, India, ₹ INR).
Return ONLY one valid JSON object. No markdown, no backticks, no explanation.

{{
  "query_type": "aggregate" | "find" | "count" | "none",
  "collection": "<exact collection name>",
  "pipeline": [...] | null,
  "find_query": {{...}} | null,
  "projection": {{"_id":0}} | null,
  "limit": 50,
  "answer_template": "<one sentence>",
  "chart_suggestion": {{
    "type": "bar"|"line"|"pie"|"metric"|"table"|"none",
    "x_field": "<exact output field>",
    "y_field": "<exact output field>",
    "title": "<title>"
  }},
  "clarification_needed": false
}}

{company_note}
DATE CONTEXT:
  LM_NUM={d['lm_num']} LM_YEAR={d['lm_year']} TM_NUM={d['tm_num']} TY={d['ty']}
  last_month: "{lms}" to "{lme}"
  year_start: "{ys}"  last_12m: "{l12}"  today: "{ts}"

DATE RULE: issueDate is MongoDB Date object. ALWAYS prefer ItemQuantityTracker
  (integer year/month fields — no date conversion needed).

STRICT RULES:
1. Collections CASE-SENSITIVE: Voucher,Item,Business,ItemQuantityTracker,ItemSummary,
   Contact,Account,IBranch,IUser,ICompany,ItemGroup,company_data,voucher_count
2. Aggregations MUST end with: {{"$project":{{"_id":0,"field1":1,"field2":1,...}}}}
3. Voucher find: SAFE PROJECTION ONLY (no itemList/transactions/tax/party/voucherList)
4. For company-specific queries use iCompanyId filter (provided above if resolved)
5. For counts: use aggregate with $count stage → project as {{count_field: 1}}
6. chart x_field/y_field = EXACT output field names from $project
7. clarification_needed = false always
8. Filter iCompanyId: {{$ne:null}} for analytics to skip test data
9. NEVER hallucinate data — only describe what the query would return"""

def parse_json(text: str) -> Dict:
    text = re.sub(r"```(?:json)?\n?","",text.strip()).strip("`").strip()
    try: return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {"query_type":"none","collection":None,"pipeline":None,"find_query":None,
            "answer_template":"Could not parse response.",
            "chart_suggestion":{"type":"none"},"clarification_needed":False}

# ── Valid collections ─────────────────────────────────────────────────────────

VALID_COLS = {
    "Voucher","Item","Business","ItemQuantityTracker","ItemSummary","Contact",
    "Account","IBranch","IUser","ICompany","ItemGroup","company_data","voucher_count",
    "AccountGroup","IBank","ItemColor","ItemCollection","ItemTag","ItemTagItem",
    "IRoleConfig","ItemAttribute","ItemAttributeValue","ItemService","demoContact",
    "PartyTag","AccessToken","BusinessContact","VersionTracking"
}

# ── Agent class ───────────────────────────────────────────────────────────────

class MongoAIAgent:
    def __init__(self):
        self.client    = get_mongo_client()
        self.llm       = None
        self.history   = []
        self.collection_stats = {}
        self.date_type = "string"
        if self.client:
            try:
                self.collection_stats = get_collection_stats(self.client)
                self.date_type = detect_date_type(self.client)
                print(f"[Agent] issueDate type: {self.date_type}")
            except Exception as e:
                print(f"[Agent] init: {e}")

    def is_connected(self): return self.client is not None

    def refresh_schema(self):
        if self.client:
            self.collection_stats = get_collection_stats(self.client)
            self.date_type = detect_date_type(self.client)

    def init_llm(self):
        try: self.llm = get_llm(); return True
        except: return False

    # ── Keyword shortcut (no LLM needed for simple questions) ─────────────────
    def _shortcut(self, q: str):
        def p(qt,col,pipeline=None,find_query=None,proj=None,limit=50,
              template="",ct="table",x=None,y=None,title=""):
            return {"query_type":qt,"collection":col,"pipeline":pipeline,
                    "find_query":find_query,"projection":proj,"limit":limit,
                    "answer_template":template,
                    "chart_suggestion":{"type":ct,"x_field":x,"y_field":y,"title":title},
                    "clarification_needed":False}

        has = lambda *ws: any(w in q for w in ws)
        hasnt = lambda *ws: not any(w in q for w in ws)

        if has("branch","branches") and hasnt("sales","revenue","voucher","customer"):
            return p("find","IBranch",find_query={},
                     proj={"_id":0,"name":1,"city":1,"state":1,"code":1,"gstBusinessType":1},
                     limit=300,template="All branches.",ct="table",title="All Branches")

        if has("user","users") and hasnt("sales","revenue","voucher"):
            return p("find","IUser",find_query={},
                     proj={"_id":0,"name":1,"phone":1,"lastSignIn":1},
                     limit=500,template="All users.",ct="table",title="All Users")

        # Only shortcut to list ALL companies if no specific company name mentioned
        if has("compan","companies") and hasnt("most voucher","sales","revenue","with","in","for","of"):
            # make sure it's a generic "list companies" not "company with X"
            generic = any(p in q for p in ["list compan","all compan","how many compan",
                                            "show compan","compan list","compan count"])
            if generic:
                return p("find","ICompany",find_query={},
                         proj={"_id":0,"name":1,"industry":1,"financialYear":1},
                         limit=200,template="All companies.",ct="table",title="All Companies")

        if "how many customer" in q:
            return p("aggregate","Business",
                     pipeline=[{"$match":{"relationType":"customer"}},{"$count":"customer_count"}],
                     template="Total customers.",ct="metric",y="customer_count",title="Total Customers")

        if "how many supplier" in q:
            return p("aggregate","Business",
                     pipeline=[{"$match":{"relationType":"supplier"}},{"$count":"supplier_count"}],
                     template="Total suppliers.",ct="metric",y="supplier_count",title="Total Suppliers")
        return None

    # ── Main query entry point ─────────────────────────────────────────────────
    def query(self, question: str) -> Dict:
        if not self.llm:
            if not self.init_llm():
                return {"error":"GROQ_API_KEY not configured."}

        q_lower = question.lower().strip()

        # 1. Try keyword shortcut first
        shortcut = self._shortcut(q_lower)
        if shortcut:
            results, err = self._run(shortcut)
            answer = self._honest_answer(question, shortcut, results, err)
            chart  = self._chart(results, shortcut.get("chart_suggestion",{}))
            self.history.append({"q":question,"a":shortcut.get("answer_template","")[:100]})
            return {"type":"answer","answer":answer,"results":results,
                    "chart":chart,"plan":shortcut,"db_error":err}

        # 2. Company name resolution — detect if question mentions a specific company
        company_context = ""
        resolved_company = None

        # Trigger company lookup if question contains company-related keywords
        COMPANY_TRIGGERS = [
            "company","compan","namo","shivaya","invock","aman","shraddha",
            "hussain","qamber","creative","waja","vouchers in","vouchers of",
            "vouchers for","sales in","purchases in","items in","records in",
            "estimate","in company","with company","for company","of company",
        ]
        should_resolve = any(kw in q_lower for kw in COMPANY_TRIGGERS)

        if self.client and should_resolve:
            cname = extract_company_name(question)
            if cname:
                match = find_best_company(self.client, cname)
                if match:
                    resolved_company = match
                    company_context = (
                        f"User mentioned company: '{cname}'\n"
                        f"Best match found: '{match['name']}' (iCompanyId = \"{match['_id']}\")\n"
                        f"Use this iCompanyId to filter queries."
                    )
                    print(f"[Agent] Resolved '{cname}' → '{match['name']}' ({match['_id']})")
                else:
                    # Company name given but NOT found → honest answer, no hallucination
                    return {
                        "type":"answer",
                        "answer": (f"❌ Could not find a company matching **\"{cname}\"** in the database.\n\n"
                                   f"Please check the company name and try again. "
                                   f"You can ask *\"list all companies\"* to see available names."),
                        "results": [], "chart": None, "plan": {}, "db_error": None
                    }

        # 3. Build prompt and call LLM
        dates = get_dates()
        system_prompt = build_prompt(dates, self.date_type, company_context)
        hist = ""
        if self.history:
            hist = "\nConversation:\n" + "\n".join(f"Q:{h['q']}\nA:{h['a']}" for h in self.history[-3:])
        user_msg = f"{SCHEMA}\n{hist}\n\nQuestion: {question}"

        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
            plan = parse_json(resp.content)
        except Exception as e:
            return {"error":f"LLM error: {e}"}

        # 4. Inject resolved company id into pipeline/query if available
        if resolved_company:
            real_id = self._get_real_company_id(resolved_company)
            resolved_company["real_id"] = real_id  # store for answer generation
            plan = self._inject_company_id(plan, real_id)

        results, err = self._run(plan)

        # 5. Retry if empty
        if not results and not err and plan.get("query_type") != "none":
            results, err, plan = self._retry(question, plan, system_prompt, user_msg, dates, resolved_company)

        # 6. Last resort: if company resolved but still no results, do a direct count
        if not results and not err and resolved_company:
            results, err, plan = self._direct_company_count(question, resolved_company)

        answer = self._honest_answer(question, plan, results, err, resolved_company)
        chart  = self._chart(results, plan.get("chart_suggestion",{}))
        self.history.append({"q":question,"a":plan.get("answer_template","")[:100]})

        return {"type":"answer","answer":answer,"results":results,
                "chart":chart,"plan":plan,"db_error":err}

    def _get_real_company_id(self, resolved: Dict) -> Any:
        """
        Return the iCompanyId value in the exact format Voucher collection uses.
        Debug confirmed: Voucher.iCompanyId is stored as ObjectId (not string).
        So we always prefer the raw ObjectId.
        """
        if not self.client or not resolved: return resolved.get("_id")
        db    = get_db(self.client)
        obj_id = resolved.get("_id_obj")   # raw ObjectId  ← this is what works
        str_id = resolved.get("_id")       # string version

        # Try ObjectId first (confirmed format from debug_company.py)
        if obj_id:
            try:
                n = db["Voucher"].count_documents({"iCompanyId": obj_id}, maxTimeMS=3000)
                if n > 0:
                    print(f"[Agent] iCompanyId=ObjectId, {n} vouchers for '{resolved['name']}'")
                    return obj_id
            except Exception as e:
                print(f"[Agent] ObjectId probe error: {e}")

        # Fallback: string
        if str_id:
            try:
                n = db["Voucher"].count_documents({"iCompanyId": str_id}, maxTimeMS=3000)
                if n > 0:
                    print(f"[Agent] iCompanyId=string, {n} vouchers for '{resolved['name']}'")
                    return str_id
            except: pass

        # Last resort: scan distinct and find exact match
        try:
            ids = db["Voucher"].distinct("iCompanyId")
            for vid in ids:
                if str(vid) == str_id:
                    print(f"[Agent] iCompanyId found via scan: {vid!r} for '{resolved['name']}'")
                    return vid
        except: pass

        # Return ObjectId anyway (best guess)
        print(f"[Agent] WARNING: falling back to ObjectId for '{resolved['name']}'")
        return obj_id or str_id

    def _inject_company_id(self, plan: Dict, company_id: Any) -> Dict:
        """Inject iCompanyId filter into pipeline $match or find_query."""
        try:
            if plan.get("query_type") == "aggregate" and plan.get("pipeline"):
                pipe = plan["pipeline"]
                for stage in pipe:
                    if "$match" in stage:
                        stage["$match"]["iCompanyId"] = company_id
                        return plan
                plan["pipeline"] = [{"$match": {"iCompanyId": company_id}}] + pipe
            elif plan.get("query_type") == "find":
                fq = plan.get("find_query") or {}
                fq["iCompanyId"] = company_id
                plan["find_query"] = fq
        except: pass
        return plan

    def _direct_company_count(self, question: str, resolved_company: Dict):
        """
        Bypass LLM entirely — directly count/fetch Vouchers for this company.
        Tries every possible iCompanyId format.
        """
        if not self.client: return [], None, {}
        db = get_db(self.client)
        q_lower = question.lower()

        # Determine voucher type from question
        if "sales" in q_lower:
            v_filter = {"type": "sales"}
            v_label = "sales"
        elif "purchase" in q_lower:
            v_filter = {"type": "purchase"}
            v_label = "purchase"
        else:
            v_filter = {}
            v_label = "all"

        # Try ObjectId first (confirmed by debug_company.py output), then string
        obj_id = resolved_company.get("_id_obj")
        str_id = resolved_company.get("real_id") or resolved_company.get("_id", "")
        comp_name = resolved_company["name"]
        id_candidates = [c for c in [obj_id, str_id] if c is not None]

        for cid in id_candidates:
            try:
                q = dict(v_filter)
                q["iCompanyId"] = cid
                count = db["Voucher"].count_documents(q, maxTimeMS=5000)
                if count > 0:
                    pipeline = [
                        {"$match": q},
                        {"$group": {"_id": None,
                                    "total_vouchers": {"$sum": 1},
                                    "total_amount":   {"$sum": "$billFinalAmount"}}},
                        {"$project": {"_id":0,"total_vouchers":1,"total_amount":1}}
                    ]
                    agg = list(db["Voucher"].aggregate(pipeline))
                    result_plan = {
                        "query_type": "aggregate", "collection": "Voucher",
                        "pipeline": pipeline, "find_query": None,
                        "answer_template": f"Count of {v_label} vouchers for {comp_name}.",
                        "chart_suggestion": {"type":"metric","x_field":None,
                                             "y_field":"total_vouchers",
                                             "title":f"{comp_name} — {v_label.title()} Vouchers"},
                        "clarification_needed": False
                    }
                    print(f"[Agent] Direct count: {count} {v_label} vouchers for '{comp_name}' (cid={cid!r})")
                    return ([deep_sanitize(r) for r in agg]
                            if agg else [{"total_vouchers": count, "total_amount": 0}]), None, result_plan
            except Exception as e:
                print(f"[Agent] Direct count error with cid={cid!r}: {e}")

        # Nothing found — company exists but genuinely has no vouchers of this type
        result_plan = {
            "query_type": "none", "collection": "Voucher",
            "answer_template": f"No {v_label} vouchers for '{comp_name}'.",
            "chart_suggestion": {"type":"none"}, "clarification_needed": False
        }
        return [], None, result_plan

    def _resolve_col(self, col):
        if col in VALID_COLS: return col
        lm = {c.lower():c for c in VALID_COLS}
        return lm.get((col or "").lower(), col)

    def _run(self, plan):
        qt = plan.get("query_type")
        if qt not in ("aggregate","find") or not self.client: return [], None
        col = self._resolve_col(plan.get("collection",""))
        plan["collection"] = col
        if qt == "aggregate" and plan.get("pipeline"):
            pipe = convert_dt_strings(plan["pipeline"]) if self.date_type=="date_object" else plan["pipeline"]
            return execute_agg(self.client, col, pipe)
        if qt == "find" and plan.get("find_query") is not None:
            fq = convert_dt_strings([plan["find_query"]])[0] if self.date_type=="date_object" else plan["find_query"]
            return execute_find(self.client, col, fq, plan.get("projection"), plan.get("limit",50))
        return [], None

    def _retry(self, question, orig, system_prompt, user_msg, dates, resolved_company=None):
        d = dates
        hint = f"""
⚠️ RETRY — '{orig.get("collection")}' returned 0 results.
Pipeline: {json.dumps(orig.get("pipeline"), default=str)[:250]}

Fixes to try:
1. Use ItemQuantityTracker with integer year/month (no date conversion needed):
   last month: {{"voucherType":"sales","year":{d['lm_year']},"month":{d['lm_num']}}}
   this year:  {{"voucherType":"sales","year":{d['ty']}}}
2. Remove $ne null iCompanyId filter if too strict
3. For "sales vs purchases": group Voucher by "$type", no date filter
4. For company count: just count Voucher with iCompanyId={resolved_company['_id'] if resolved_company else 'X'}

Question: {question}"""
        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg+hint)])
            plan2 = parse_json(resp.content)
            if resolved_company:
                real_id = resolved_company.get("real_id", resolved_company.get("_id"))
                plan2 = self._inject_company_id(plan2, real_id)
            r, e = self._run(plan2)
            return r, e, plan2
        except Exception as e:
            return [], str(e), orig

    def _honest_answer(self, question, plan, results, db_error, resolved_company=None):
        """Generate answer — honest about what was/wasn't found, no hallucination."""
        if db_error:
            return f"⚠️ **Database error:** `{db_error}`"

        if plan.get("query_type") == "none":
            return plan.get("answer_template","I couldn't find relevant data for that question.")

        if not results:
            if resolved_company:
                return (
                    f"**{resolved_company['name']}** was found in the database ✓, "
                    f"but has **no matching records** for this query.\n\n"
                    f"This company may be a test/demo account, or it may not have "
                    f"any transactions of this type recorded yet."
                )
            return (
                f"**No records found.**\n\n"
                f"The query ran successfully but matched zero documents. "
                f"Try adjusting your filters or check if data exists for this period."
            )

        # Build honest, grounded analysis
        company_note = f" for **{resolved_company['name']}**" if resolved_company else ""
        prompt = (
            f"You are a data analyst for Invock ERP (jewellery business, India, ₹ INR).\n"
            f"Question: {question}\n"
            f"Company context: {resolved_company['name'] if resolved_company else 'all companies'}\n\n"
            f"Actual query results ({len(results)} records){company_note}:\n"
            f"{json.dumps(results[:10], default=str, indent=2)}\n\n"
            f"STRICT RULES for your response:\n"
            f"1. Only describe what is ACTUALLY in the data above — no guessing or inventing figures\n"
            f"2. If results are a count, state the count clearly\n"
            f"3. Use ₹ with crore/lakh formatting\n"
            f"4. Name actual top performers from the data\n"
            f"5. Give one actionable business insight based on real numbers\n"
            f"6. If data is insufficient to draw conclusions, say so honestly\n"
            f"Keep it 3-4 sentences. Be precise."
        )
        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content
        except:
            return f"Found {len(results)} record(s){company_note}."

    def _chart(self, results, suggestion):
        if not results or not suggestion or suggestion.get("type") in ("none",None): return None
        try:
            clean = []
            for doc in results:
                row = {}
                for k,v in doc.items():
                    if isinstance(v, bool):           row[k] = str(v)
                    elif isinstance(v,(int,float,type(None))): row[k] = v
                    elif isinstance(v, str):           row[k] = v
                    elif isinstance(v, dict):          row[k] = str(v)
                    elif isinstance(v, list):          row[k] = len(v)
                    else:                              row[k] = str(v)
                clean.append(row)
            df = pd.DataFrame(clean)
            if df.empty: return None
            for c in df.columns:
                try: df[c] = pd.to_numeric(df[c])
                except: pass
            num = df.select_dtypes(include="number").columns.tolist()
            cat = df.select_dtypes(exclude="number").columns.tolist()
            x = suggestion.get("x_field"); y = suggestion.get("y_field")
            if not x or x not in df.columns: x = cat[0] if cat else (df.columns[0] if len(df.columns) else None)
            if not y or y not in df.columns: y = num[0] if num else (df.columns[1] if len(df.columns)>1 else None)
            return {"type":suggestion.get("type","bar"),"df":df,"x":x,"y":y,
                    "title":suggestion.get("title","Results")}
        except: return None
