# Invock ERP — MongoDB AI Assistant

> A production-grade AI-powered chatbot that answers natural language questions about Invock ERP data — revenue, vouchers, inventory, customers, and more — without writing a single line of MongoDB query manually.

**Live Demo:** [https://app-ai-assistant-mdxdkvxtnv4p83e8kuy7nb.streamlit.app/](https://app-ai-assistant-mdxdkvxtnv4p83e8kuy7nb.streamlit.app/)

---

## Overview

This project is a full-stack AI assistant built on top of a real MongoDB database powering Invock ERP — a jewellery industry ERP system used by 135 companies across India. The chatbot interprets plain English questions and returns accurate, formatted answers with charts and tables — no SQL, no query builder, no manual filters.

The system processes **4.1 million records** across 9 collections, handles multi-tenant data isolation, formats all monetary values in Indian notation (crore/lakh), and falls back to a 70B parameter LLM only when no hardcoded query matches.

---

## Database at a Glance

| Collection | Records | Purpose |
|---|---|---|
| Voucher | 1,317,608 | Sales invoices, purchase bills, receipts, payments |
| ItemQuantityTracker | 2,135,365 | Pre-aggregated product movement by month/year |
| Item | 450,407 | Product catalog with real-time stock levels |
| Business | 45,119 | Customers and suppliers |
| Contact | 39,148 | Individual people at businesses |
| Account | 51,601 | Chart of accounts (cash, bank, debtors, creditors) |
| IBranch | 265 | Physical company branches |
| IUser | 399 | Staff with system access |
| ICompany | 135 | Tenant companies using Invock ERP |

---

## Features

- **Natural language querying** — Ask in plain English, get instant answers
- **Zero LLM query generation** for common questions — all critical queries are hardcoded and correct
- **20+ direct query methods** — revenue, unpaid invoices, top customers, stock, monthly trends, and more
- **Company ranking** — Rank all 135 companies by sales vouchers, revenue, or purchase volume
- **Fuzzy company name matching** — Handles typos, abbreviations, and partial names
- **Indian number formatting** — ₹1,633.16 crore, ₹71.23 lakh, ₹8,500 — never raw millions
- **Interactive charts** — Bar, line, and metric cards rendered per query type
- **Multi-tenant isolation** — Every query scoped to the correct company via iCompanyId
- **ObjectId lookup** — Search any collection directly by MongoDB ObjectId
- **Conversation history** — Last 3 turns passed to LLM for contextual follow-ups
- **Version guard** — Crashes loudly if wrong file is deployed, preventing silent bugs

---

## Architecture

```
User Question
      │
      ▼
Step 0 ── ObjectId detected? ──────────────► Direct collection lookup
      │
      ▼
Step 1 ── Global schema shortcut? ─────────► IBranch / IUser / ICompany /
      │    (no company needed)                revenue / trends / unpaid
      ▼
Step 1b ── Company ranking pattern? ───────► companies_by_voucher_count()
      │     (before name extraction)          Python-side join, real names
      ▼
Step 2 ── Company name detected? ──────────► Fuzzy resolve → iCompanyId
      │
      ▼
Step 3 ── Intent router match? ────────────► 20+ hardcoded Q class methods
      │
      ▼
Step 4 ── LLM fallback ─────────────────────► Groq LLaMA-3.3-70B
               (unknown questions only)        Auto-retry on empty result
```

The key design principle: **the LLM never builds MongoDB queries for known question types.** This eliminates hallucinated pipelines, wrong field names, and incorrect number scaling entirely.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI / LLM | Groq API — LLaMA-3.3-70B-Versatile |
| LLM Framework | LangChain |
| Database | MongoDB Atlas (PyMongo) |
| Language | Python 3.10+ |
| Deployment | Streamlit Cloud |
| Environment | python-dotenv |

---

## Query Pipeline — Detailed

### Step 0 — ObjectId Lookup
Detects any 24-character hex string in the question and queries the appropriate collection directly. A SKIP_FIELDS list hides 30+ internal fields from the output so only meaningful data is shown.

### Step 1 — Schema Shortcuts (13 global patterns)
Handles questions that need no company context: list all branches, all users, all companies, global revenue this year, last month sales, sales vs purchases, top products, unpaid invoices, stock levels, and average order value. All use hardcoded correct pipelines — zero LLM involvement.

### Step 1b — Company Ranking
Intercepts questions like *"list companies with most sales vouchers"* before the name extractor can misfire. Groups 1.3M Voucher documents by iCompanyId in MongoDB, then does a Python-side join with ICompany to resolve real names. Returns a ranked table with voucher count and total amount.

### Step 2 — Company Name Extraction + Fuzzy Resolution
Five conservative regex patterns extract company names only when clearly preceded by a preposition or possessive. A 20+ word generic reject list prevents "sales", "most", "created" from being treated as company names. Fuzzy matching uses trigram similarity (50%), token overlap (30%), and prefix matching (20%) with a penalty for extra words.

iCompanyId format is auto-detected: the system probes both ObjectId and string format and uses whichever returns more Voucher documents for that company.

### Step 3 — Intent Router (20+ methods)
The `route()` function maps question patterns to Q class methods. Every method uses a correct, tested MongoDB aggregation or find query.

**Available query methods:**

| Method | What It Does |
|---|---|
| `voucher_count` | Total vouchers filtered by type |
| `vouchers_by_type` | Breakdown across sales / purchase / receipt / payment |
| `voucher_by_status` | Paid / unpaid / partial breakdown |
| `top_customers` | Ranked by revenue from sales vouchers |
| `top_suppliers` | Ranked by amount from purchase vouchers |
| `unpaid_invoices` | Filtered by status=unpaid, sorted by dueAmount |
| `sales_vs_purchases` | Side-by-side comparison |
| `avg_order_value` | Uses MongoDB $avg operator |
| `monthly_trend` | From ItemQuantityTracker integer year/month fields |
| `total_revenue` | Sum of amount from ItemQuantityTracker |
| `top_products` | Grouped by itemId, sortable by revenue or quantity |
| `purchase_trend` | Monthly purchase volume |
| `stock` | Items where availableQty > 0 |
| `low_stock` | Items below configurable threshold |
| `customer_list` | Business collection where relationType = customer |
| `supplier_list` | Business collection where relationType = supplier |

### Step 4 — LLM Fallback
Only fires for questions that match none of the above. Sends schema context, recent conversation history, and strict rules to LLaMA-3.3-70B via Groq. Auto-retries once with the error details if the first attempt returns empty results.

---

## Number Formatting

All monetary values are pre-formatted in Python before the LLM sees them. This permanently prevents the LLM from re-scaling or misinterpreting raw integers.

| Raw Value | Formatted Output |
|---|---|
| 1,633,160,000 | ₹1,633.16 crore |
| 71,230,000 | ₹71.23 lakh |
| 84,109 | ₹84,109 |
| 834,474 (quantity) | 834,474 |

**Money detection keywords (21):** amount, revenue, total, price, value, sales, due, paid, balance, bill, final, cost, tax, discount, subtotal, net, gross, fee, charge, credit, debit

**Count detection keywords (10):** count, qty, quantity, voucher, order, invoice, unit, number, no, record

COUNT always takes priority over MONEY — `total_qty: 834474` is never formatted as ₹8.34 lakh.

---

## Sample Questions

**Global queries**
```
What is the total revenue this year?
Show monthly sales trend for last 12 months
List all branches
Top 15 customers by revenue
Show all unpaid invoices
Sales vs purchases comparison
Average order value across all companies
```

**Company ranking**
```
List companies with most sales vouchers
Top 5 companies by purchase volume
Which company has created the most invoices?
Rank companies by sales
```

**Company-specific queries**
```
Show revenue for Hiraka Jewels
Top customers of Bhakti Parshwanath
Unpaid invoices for NAMO-ESTIMATE
Monthly sales trend for Vaibhav Fashion
Stock levels for Dipshi
How many vouchers does Hiraka have?
Sales vs purchases for Bhakti
```

**ObjectId lookup**
```
Name of company with id 651ea98aa7dc3e26bda3603b
Which branch has id 65cda589189c7507008979f4
```

---

## Project Structure

```
├── agent.py          # Core AI agent — query logic, fuzzy matching, LLM integration
├── app.py            # Streamlit frontend — UI, charts, sidebar stats, chat interface
├── .env              # Environment variables (not committed)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- MongoDB Atlas cluster
- Groq API key — free tier at [console.groq.com](https://console.groq.com)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/invock-ai-assistant.git
cd invock-ai-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your credentials
echo "MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/<db>" >> .env
echo "GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx" >> .env

# 4. Run locally
streamlit run app.py
```

### requirements.txt

```
streamlit
pymongo
python-dotenv
langchain
langchain-groq
pandas
```

---

## Environment Variables

| Variable | Description | Where to Get |
|---|---|---|
| `MONGODB_URI` | Full MongoDB Atlas connection string | MongoDB Atlas dashboard |
| `GROQ_API_KEY` | Groq API key | console.groq.com |

For **Streamlit Cloud** deployment: add both variables under **Settings → Secrets** in your app dashboard.

---

## Key Design Decisions

**Why hardcoded queries instead of full LLM generation?**
Early versions let the LLM generate all MongoDB pipelines. This caused wrong field names, missing `$` prefixes in `$sum`, incorrect date filtering, and revenue values scaled 100x because the LLM misidentified rupees as paise. Hardcoded queries for the 20 most common question types eliminated every one of these bugs permanently.

**Why Groq instead of OpenAI?**
Groq runs LLaMA-3.3-70B at ~500 tokens/second with a generous free tier. For a project where the LLM fires rarely anyway, speed and cost matter more than marginal accuracy gains.

**Why fuzzy matching instead of exact string match?**
135 company names include entries like "M/S DIPSHI - ESTIMATE", "Bhakti Parshwanath Jewellers Pvt Ltd", "HIRAKA JEWELS". Users type partial names, abbreviations, and typos. Trigram similarity handles all of these while the 0.15 threshold prevents false matches.

**Why ItemQuantityTracker instead of Voucher for date queries?**
ItemQuantityTracker stores pre-aggregated data with integer `year` and `month` fields. Filtering `{year: 2025, month: 3}` is dramatically faster than parsing `issueDate` Date objects across 1.3 million Voucher documents.

**Why Python-side join for company ranking?**
MongoDB `$lookup` across 1.3M documents and 135 companies hits memory limits and is slow. The approach used here — aggregate iCompanyId counts in one query, load all 135 ICompany names in a second query, join in Python — is faster, simpler, and more reliable.

---

## Bugs Found and Fixed During Development

| Bug | Root Cause | Fix |
|---|---|---|
| Revenue showing ₹841,096 crore | Raw integer sent to LLM; LLM divided by 100 assuming paise | Pre-format all money fields in Python before LLM sees them |
| ObjectId strings in answers | ObjectId not serialized before passing to LLM | deep_sanitize() converts every ObjectId to string |
| "most created sales" matched as company name | Greedy regex `show me (.*)` | Switched to 5 preposition-anchored patterns with generic reject list |
| Company ranking returning raw Voucher documents | LLM fallback fired before ranking intercept | Added Step 1b that fires before extract_company_name |
| Contact/Account showing 0 in sidebar | Collections not in get_stats() list | Added Contact + Account, switched to estimated_document_count() |

Each bug pointed to the same lesson: for predictable, high-stakes data queries, deterministic code beats probabilistic LLM generation.

---

## Audit Results

Automated audit of agent.py v4.1 across 55 critical checks:

- **Infrastructure** — 8/8 passing
- **Company Resolution** — 8/8 passing
- **Query Pipeline** — 8/8 passing
- **Direct Query Methods** — 16/16 passing
- **Number Formatting** — 8/8 passing
- **Security & Reliability** — 7/7 passing

**Score: 55/55 — 100% of critical checks passing**

---

## Live Demo

[https://app-ai-assistant-mdxdkvxtnv4p83e8kuy7nb.streamlit.app/](https://app-ai-assistant-mdxdkvxtnv4p83e8kuy7nb.streamlit.app/)

The live instance connects to a real MongoDB Atlas cluster with 4.1 million records from Invock ERP. All 135 companies, 1.3M vouchers, and 2.1M item tracker records are live and queryable.

---

*Built with Python · MongoDB · Streamlit · LangChain · Groq LLaMA-3.3-70B*
