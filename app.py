"""
Invock AI Analytics — Streamlit App (Production)
"""
import os, json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Invock AI Analytics", page_icon="💎",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
:root{--bg:#07090f;--sf:#0f1520;--s2:#151e2e;--s3:#1a2538;--bd:#1e3050;
  --bl:#3b82f6;--pu:#8b5cf6;--gr:#10b981;--gd:#f59e0b;
  --tx:#e2e8f0;--mu:#64748b;--re:#ef4444;}
html,body,.stApp{background:var(--bg)!important;color:var(--tx)!important;font-family:'Plus Jakarta Sans',sans-serif;}
#MainMenu,footer,header,.stDeployButton{visibility:hidden;display:none;}
[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--bd)!important;}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
.stTextInput>div>div>input{background:var(--s2)!important;color:var(--tx)!important;border:1px solid var(--bd)!important;border-radius:10px!important;font-size:15px!important;padding:12px 16px!important;}
.stTextInput>div>div>input:focus{border-color:var(--bl)!important;box-shadow:0 0 0 3px rgba(59,130,246,.2)!important;}
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#3b82f6)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(59,130,246,.35)!important;}
.mu{background:var(--s2);border:1px solid var(--bd);border-left:4px solid var(--bl);border-radius:14px;padding:16px 20px;margin:14px 0;}
.ma{background:linear-gradient(135deg,#0d1f35,#0f1e30);border:1px solid #1e3a5f;border-left:4px solid var(--gr);border-radius:14px;padding:16px 20px;margin:14px 0;line-height:1.75;}
.me{background:#120a0a;border:1px solid var(--re);border-left:4px solid var(--re);border-radius:14px;padding:14px 20px;margin:14px 0;}
.kpi-wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:16px 0 20px;}
.kpi{background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:22px 16px;text-align:center;position:relative;overflow:hidden;transition:transform .2s;}
.kpi:hover{transform:translateY(-3px);}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#3b82f6,#8b5cf6);}
.kpi-v{font-family:'JetBrains Mono',monospace;font-size:1.7rem;font-weight:700;color:#60a5fa;}
.kpi-l{font-size:11px;color:var(--mu);text-transform:uppercase;letter-spacing:1.2px;margin-top:8px;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:var(--s2);border-radius:12px;padding:4px;gap:4px;border:1px solid var(--bd);}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--mu)!important;border-radius:8px!important;font-weight:500!important;padding:8px 18px!important;}
.stTabs [aria-selected="true"]{background:var(--s3)!important;color:#60a5fa!important;font-weight:700!important;}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid var(--bd)!important;}
.th{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.tbadge{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid var(--bl);border-radius:20px;padding:3px 14px;font-size:12px;font-family:'JetBrains Mono',monospace;font-weight:600;}
.sr{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1a2538;}
.sl{color:#64748b;font-size:12px;}.sv{color:#60a5fa;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:#1e3050;border-radius:3px}
.lab{font-size:11px;font-weight:700;letter-spacing:1.5px;color:#475569;text-transform:uppercase;margin:14px 0 6px;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
COLORS = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4",
          "#84cc16","#f97316","#ec4899","#14b8a6","#a78bfa","#fb7185"]

BASE_LAYOUT = dict(
    paper_bgcolor="#0f1520", plot_bgcolor="#0d1b2a",
    font=dict(family="Plus Jakarta Sans", color="#cbd5e1", size=12),
    title_font=dict(size=16, color="#60a5fa", family="Plus Jakarta Sans"),
    legend=dict(bgcolor="#0f1520", bordercolor="#1e3050", borderwidth=1,
                font=dict(color="#94a3b8", size=11)),
    margin=dict(l=70, r=40, t=65, b=90),
    hoverlabel=dict(bgcolor="#1a2538", bordercolor="#3b82f6",
                    font=dict(color="white", size=12)),
)

MONEY_KW = {"amount","price","revenue","sales","bill","balance","total",
            "final","due","paid","cost","profit","avg","average","sum"}
ID_COLS   = {"pincode","code","year","month","_id","index","id","flag","count_flag"}

@st.cache_resource
def load_agent():
    from agent import MongoAIAgent
    return MongoAIAgent()

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_money(col: str) -> bool:
    return any(w in col.lower() for w in MONEY_KW)

def fmt_inr(val, col="") -> str:
    if not isinstance(val, (int, float)): return str(val)
    if is_money(col):
        if abs(val) >= 1e7: return f"₹{val/1e7:.2f} Cr"
        if abs(val) >= 1e5: return f"₹{val/1e5:.1f} L"
        return f"₹{val:,.0f}"
    if isinstance(val, float): return f"{val:,.2f}"
    return f"{val:,}"

def safe_df(results: list) -> pd.DataFrame:
    """Arrow-safe DataFrame — no ObjectIds, no mixed bools."""
    rows = []
    for doc in results:
        row = {}
        for k, v in doc.items():
            if v is None:                       row[k] = None
            elif isinstance(v, bool):           row[k] = str(v)
            elif isinstance(v, (int, float)):   row[k] = v
            elif isinstance(v, str):            row[k] = v
            elif isinstance(v, dict):           row[k] = str(v)
            elif isinstance(v, list):           row[k] = len(v)
            else:                               row[k] = str(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            try:    df[c] = pd.to_numeric(df[c])
            except: pass
    return df

def pick_xy(df: pd.DataFrame, hint_x=None, hint_y=None):
    """
    Deterministically pick the right x (label) and y (metric) columns.
    Uses hints first, then keyword matching, never picks ID/pincode cols as y.
    """
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()

    # -- pick Y (the numeric metric we want to plot) --
    y = None
    # 1. use hint if valid and not an ID col
    if hint_y and hint_y in df.columns and hint_y.lower() not in ID_COLS:
        try:
            pd.to_numeric(df[hint_y])
            y = hint_y
        except: pass
    # 2. find numeric col whose name contains a money/metric keyword
    if not y:
        for kw in ["amount","revenue","total","sales","qty","count","sum",
                   "balance","price","bill","final","avg","average"]:
            for col in num:
                if kw in col.lower() and col.lower() not in ID_COLS:
                    y = col; break
            if y: break
    # 3. biggest-value numeric col that's not an ID
    if not y:
        best_val, best_col = -1, None
        for col in num:
            if col.lower() in ID_COLS: continue
            try:
                mx = df[col].abs().max()
                if mx > best_val: best_val, best_col = mx, col
            except: pass
        y = best_col
    # 4. last resort
    if not y and num: y = num[0]

    # -- pick X (the label / category axis) --
    x = None
    # 1. use hint if valid
    if hint_x and hint_x in df.columns: x = hint_x
    # 2. keyword match
    if not x:
        for kw in ["name","party","customer","supplier","type","item","product",
                   "company","branch","city","state","month","year","group"]:
            for col in cat:
                if kw in col.lower():
                    x = col; break
            if x: break
    # 3. first categorical
    if not x and cat: x = cat[0]
    # 4. first non-y numeric (for scatter etc.)
    if not x and num:
        x = next((c for c in num if c != y), None)

    return x, y

def styled_table(df: pd.DataFrame, title=""):
    n = len(df)
    st.markdown(f'<div class="th"><span style="color:#94a3b8;font-size:14px;font-weight:600;">'
                f'{title}</span><span class="tbadge">{n:,} records</span></div>',
                unsafe_allow_html=True)
    disp = df.copy()
    for col in disp.select_dtypes(include="number").columns:
        if is_money(col):
            disp[col] = disp[col].apply(lambda v: fmt_inr(v, col) if pd.notna(v) else "—")
    st.dataframe(disp, use_container_width=True, height=min(520, 60 + n * 38))

def make_layout(title, x_label="", y_label="", money_y=False):
    """Build a fresh plotly layout dict for each chart."""
    lo = dict(**BASE_LAYOUT)
    lo["title_text"] = title
    lo["xaxis"] = dict(
        gridcolor="#1e3050", linecolor="#1e3050",
        tickfont=dict(color="#64748b", size=10),
        title=dict(text=x_label, font=dict(color="#94a3b8", size=12)),
    )
    lo["yaxis"] = dict(
        gridcolor="#1e3050", linecolor="#1e3050",
        tickfont=dict(color="#64748b", size=10),
        title=dict(text=y_label, font=dict(color="#94a3b8", size=12)),
    )
    if money_y:
        lo["yaxis"]["tickprefix"] = "₹"
        lo["yaxis"]["tickformat"] = ",.0f"
    return lo

# ── Chart renderer ─────────────────────────────────────────────────────────────

def render_charts(results, chart_meta, plan, chart_key=""):
    if not results: return
    df = safe_df(results)
    if df.empty: return

    # Unique key prefix for all plotly charts in this render call
    import hashlib, time
    key_base = chart_key or hashlib.md5(
        (str(results[:2]) + str(time.time())).encode()).hexdigest()[:8]

    cs      = (plan or {}).get("chart_suggestion") or {}
    stype   = cs.get("type", "bar")
    title   = cs.get("title") or "Query Results"
    hint_x  = cs.get("x_field")
    hint_y  = cs.get("y_field")

    # chart_meta carries the df from agent._chart (may have better x/y)
    if chart_meta:
        hint_x = hint_x or chart_meta.get("x")
        hint_y = hint_y or chart_meta.get("y")

    x, y = pick_xy(df, hint_x, hint_y)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    money_y  = is_money(y) if y else False

    # ── 1. Pure metric (single number) ────────────────────────────────────────
    if stype == "metric" or (len(results) == 1 and num_cols and not cat_cols):
        st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)
        display = [c for c in num_cols if c.lower() not in ID_COLS][:4] or num_cols[:1]
        n_c = max(1, len(display))
        cols = st.columns(n_c)
        for i, cn in enumerate(display):
            try:    val = float(df[cn].iloc[0])
            except: val = 0
            with cols[i]:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-v">{fmt_inr(val, cn)}</div>'
                    f'<div class="kpi-l">{cn.replace("_"," ")}</div></div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── 2. Full chart suite when we have both x and y ─────────────────────────
    if x and y and x in df.columns and y in df.columns:

        # sort for bar/donut: descending by y
        try:    df_desc = df.sort_values(y, ascending=False).reset_index(drop=True)
        except: df_desc = df.copy()

        # sort for line: ascending by x (time order)
        try:
            # For monthly trends, create a sortable key
            if "month" in df.columns and "year" in df.columns:
                df_asc = df.copy()
                df_asc["_sort"] = df_asc["year"] * 100 + df_asc["month"]
                df_asc = df_asc.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
            else:
                df_asc = df.sort_values(x).reset_index(drop=True)
        except: df_asc = df.copy()

        x_label = x.replace("_"," ").title()
        y_label = ("₹ " if money_y else "") + y.replace("_"," ").title()

        # choose tab order based on chart type hint
        if stype == "line":
            tab_order = ["📈 Line","📊 Bar","🥧 Donut","📋 Table"]
        elif stype in ("table","none"):
            tab_order = ["📋 Table","📊 Bar","🥧 Donut","📈 Line"]
        else:
            tab_order = ["📊 Bar","🥧 Donut","📈 Line","📋 Table"]

        tabs = st.tabs(tab_order)
        tab_map = {name: tab for name, tab in zip(tab_order, tabs)}

        # ── TABLE ──────────────────────────────────────────────────────────────
        with tab_map["📋 Table"]:
            styled_table(df_desc, title)

        # ── BAR ────────────────────────────────────────────────────────────────
        with tab_map["📊 Bar"]:
            top = df_desc.head(20)
            # Truncate long labels for readability
            top = top.copy()
            top[x] = top[x].astype(str).str[:28]

            fig = go.Figure()
            for i, row in top.iterrows():
                fig.add_trace(go.Bar(
                    x=[row[x]], y=[row[y]],
                    name=str(row[x]),
                    marker_color=COLORS[i % len(COLORS)],
                    text=fmt_inr(row[y], y),
                    textposition="outside",
                    textfont=dict(size=9, color="#cbd5e1"),
                    hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,.0f}}<extra></extra>",
                    showlegend=False,
                ))
            lo = make_layout(title, x_label, y_label, money_y)
            lo["xaxis"]["tickangle"] = -38
            lo["barmode"] = "group"
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{key_base}")

        # ── DONUT ──────────────────────────────────────────────────────────────
        with tab_map["🥧 Donut"]:
            top10 = df_desc.head(10).copy()
            top10[x] = top10[x].astype(str).str[:22]
            total = top10[y].sum()

            # custom hover
            hover = [f"<b>{r[x]}</b><br>{fmt_inr(r[y], y)}<br>{r[y]/total*100:.1f}%"
                     for _, r in top10.iterrows()]

            fig = go.Figure(go.Pie(
                labels=top10[x],
                values=top10[y],
                hole=0.50,
                marker=dict(colors=COLORS[:len(top10)],
                            line=dict(color="#07090f", width=2)),
                textinfo="percent",
                textfont=dict(size=11, color="white"),
                hovertext=hover,
                hoverinfo="text",
                pull=[0.04 if i == 0 else 0.01 for i in range(len(top10))],
                sort=False,   # already sorted desc
                direction="clockwise",
            ))
            # annotation in center
            fig.add_annotation(
                text=f"<b>{fmt_inr(total, y)}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#60a5fa"),
            )
            lo = make_layout(f"{title} — Top {len(top10)}", money_y=money_y)
            lo["showlegend"] = True
            lo["legend"]["orientation"] = "v"
            lo["legend"]["x"] = 1.02
            lo["margin"] = dict(l=20, r=160, t=65, b=20)
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True, key=f"donut_{key_base}")

        # ── LINE ───────────────────────────────────────────────────────────────
        with tab_map["📈 Line"]:
            # For monthly data build a proper x label like "Jan 2024"
            df_line = df_asc.copy()
            if "month" in df_line.columns and "year" in df_line.columns:
                import calendar
                df_line["period"] = df_line.apply(
                    lambda r: f"{calendar.month_abbr[int(r['month'])]} {int(r['year'])}", axis=1)
                lx = "period"
            else:
                lx = x

            lx_label = lx.replace("_"," ").title()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_line[lx], y=df_line[y],
                mode="lines+markers",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8, color="#60a5fa",
                            line=dict(color="#07090f", width=2)),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
                hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,.0f}}<extra></extra>",
                name=y_label,
            ))
            # Add data labels on each point
            for _, row in df_line.iterrows():
                fig.add_annotation(
                    x=row[lx], y=row[y],
                    text=fmt_inr(row[y], y),
                    showarrow=False,
                    yshift=14,
                    font=dict(size=8, color="#94a3b8"),
                )
            lo = make_layout(title, lx_label, y_label, money_y)
            lo["xaxis"]["tickangle"] = -35
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True, key=f"line_{key_base}")

    # ── 3. Only numeric columns (multi-KPI) ───────────────────────────────────
    elif num_cols:
        display = [c for c in num_cols if c.lower() not in ID_COLS][:4]
        if not display: display = num_cols[:1]
        n_c = max(1, len(display))
        cols = st.columns(n_c)
        for i, cn in enumerate(display):
            try:
                val = float(df[cn].iloc[0]) if len(df) == 1 else float(df[cn].sum())
            except: val = 0
            with cols[i]:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-v">{fmt_inr(val, cn)}</div>'
                    f'<div class="kpi-l">{cn.replace("_"," ")}</div></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        styled_table(df, title)

    # ── 4. Only text (table) ──────────────────────────────────────────────────
    else:
        styled_table(df, title)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px;">
      <div style="font-size:1.9rem;font-weight:800;font-family:'JetBrains Mono',monospace;
           background:linear-gradient(135deg,#60a5fa,#a78bfa);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">💎 InvockAI</div>
      <div style="color:#475569;font-size:11px;margin-top:6px;letter-spacing:1.5px;font-weight:600;">
           ERP ANALYTICS DASHBOARD</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    groq_key = st.text_input("🔑 Groq API Key", type="password",
                              value=os.getenv("GROQ_API_KEY",""), placeholder="gsk_...")
    if groq_key: os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    agent = load_agent()

    if agent.is_connected():
        st.markdown('<div style="color:#10b981;font-weight:700;font-size:13px;">● MongoDB Connected</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="color:#475569;font-size:11px;margin-bottom:4px;">'
                    f'dev-cluster · issueDate: <b style="color:#60a5fa">{agent.date_type}</b></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="lab">📊 Live Stats</div>', unsafe_allow_html=True)
        for col, label in [
            ("Voucher","📄 Vouchers"),("Item","💎 Products"),
            ("Business","🏢 Businesses"),("ItemQuantityTracker","📦 Item Tracker"),
            ("Contact","👤 Contacts"),("Account","💰 Accounts"),
            ("IBranch","🏪 Branches"),("IUser","👥 Users"),
        ]:
            cnt = agent.collection_stats.get(col, 0)
            st.markdown(f'<div class="sr"><span class="sl">{label}</span>'
                        f'<span class="sv">{cnt:,}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            agent.refresh_schema(); st.cache_resource.clear(); st.rerun()
    else:
        st.error("❌ MongoDB Disconnected")

    st.divider()
    st.markdown('<div class="lab">💡 Try These</div>', unsafe_allow_html=True)
    EXAMPLES = [
        "Show total sales last month",
        "Which product sold the most?",
        "Top 10 customers by revenue",
        "How many customers?",
        "Average order value this year",
        "Monthly sales trend last 12 months",
        "Show unpaid invoices",
        "Total purchases this year",
        "Sales vs purchases comparison",
        "Top 5 products by quantity",
        "Companies with most vouchers",
        "Total revenue this year",
        "Items with highest stock",
        "How many branches?",
        "List all users",
    ]
    for ex in EXAMPLES:
        if st.button(ex, key=f"q_{ex[:22]}", use_container_width=True):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []; st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:10px 0 22px;">
  <div style="font-size:2.1rem;font-weight:800;font-family:'JetBrains Mono',monospace;
       background:linear-gradient(135deg,#60a5fa,#818cf8,#a78bfa);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
       💎 Invock AI Analytics</div>
  <div style="color:#475569;font-size:13px;margin-top:6px;">
       Natural language → MongoDB → Charts · Tables · Insights</div>
</div>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    r = msg["role"]
    if r == "user":
        st.markdown(f'<div class="mu"><span style="color:#60a5fa;font-weight:700;font-size:12px;">'
                    f'YOU</span><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
    elif r == "assistant":
        st.markdown(f'<div class="ma"><span style="color:#10b981;font-weight:700;font-size:12px;">'
                    f'AI ANALYSIS</span><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("results"):
            render_charts(msg["results"], msg.get("chart_data"), msg.get("plan"),
                          chart_key=f"msg{idx}")
        if msg.get("plan"):
            with st.expander("🔍 MongoDB Query Plan"):
                safe_plan = {k:v for k,v in (msg["plan"] or {}).items() if k != "df"}
                st.json(safe_plan)
    elif r == "error":
        st.markdown(f'<div class="me">❌ {msg["content"]}</div>', unsafe_allow_html=True)

# Input row
pending = st.session_state.pop("pending_question", None)
c1, c2  = st.columns([6, 1])
with c1:
    user_input = st.text_input(
        "q", value=pending or "",
        placeholder="Ask anything — sales, revenue, products, customers, trends...",
        label_visibility="collapsed", key="main_input")
with c2:
    send = st.button("Ask ➤", use_container_width=True)

def process(q: str):
    if not q.strip(): return
    st.session_state.messages.append({"role":"user","content":q})
    if not os.getenv("GROQ_API_KEY"):
        st.session_state.messages.append(
            {"role":"error","content":"Please enter your Groq API key in the sidebar."})
        st.rerun(); return
    with st.spinner("🧠 Analysing your data..."):
        a = load_agent(); a.llm = None
        resp = a.query(q)
    if "error" in resp:
        st.session_state.messages.append({"role":"error","content":resp["error"]})
    else:
        st.session_state.messages.append({
            "role":"assistant",
            "content":resp.get("answer","No answer."),
            "results":resp.get("results",[]),
            "chart_data":resp.get("chart"),
            "plan":resp.get("plan"),
        })
    st.rerun()

if send and user_input: process(user_input)
elif pending:           process(pending)

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
      <div style="font-size:3.5rem;margin-bottom:20px;">💎</div>
      <div style="font-size:1.3rem;color:#94a3b8;font-weight:600;margin-bottom:10px;">
           Welcome to Invock AI Analytics</div>
      <div style="color:#475569;font-size:14px;margin-bottom:30px;">
           Ask in plain English — get charts, tables & business insights instantly</div>
      <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
        <span style="background:#0f1520;border:1px solid #1e3050;border-radius:20px;padding:7px 18px;font-size:12px;color:#64748b;">📊 Sales</span>
        <span style="background:#0f1520;border:1px solid #1e3050;border-radius:20px;padding:7px 18px;font-size:12px;color:#64748b;">💰 Revenue</span>
        <span style="background:#0f1520;border:1px solid #1e3050;border-radius:20px;padding:7px 18px;font-size:12px;color:#64748b;">📦 Inventory</span>
        <span style="background:#0f1520;border:1px solid #1e3050;border-radius:20px;padding:7px 18px;font-size:12px;color:#64748b;">🏢 Customers</span>
        <span style="background:#0f1520;border:1px solid #1e3050;border-radius:20px;padding:7px 18px;font-size:12px;color:#64748b;">📈 Trends</span>
      </div>
    </div>""", unsafe_allow_html=True)
