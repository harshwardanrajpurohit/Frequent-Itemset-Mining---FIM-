"""
FIM Comparative Analytics - Midnight Pro Edition (Debugged)
Theme: Ultra-Modern SaaS Dark Mode
Algorithms: Brute Force, Apriori, FP-Growth
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import itertools
import plotly.express as px
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ==========================================
# 1. PAGE CONFIGURATION & MIDNIGHT CSS
# ==========================================
st.set_page_config(page_title="FIM Analytics Pro", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

def inject_midnight_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #09090B !important; color: #FAFAFA !important; }
    section[data-testid="stSidebar"] { background-color: #121214 !important; border-right: 1px solid #27272A !important; }
    
    .saas-card {
        background-color: #18181B; border: 1px solid #27272A; border-radius: 12px;
        padding: 20px; transition: all 0.2s ease; height: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .saas-card:hover { border-color: #52525B; transform: translateY(-2px); }
    
    .kpi-title { color: #A1A1AA; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { color: #FAFAFA; font-size: 2.2rem; font-weight: 700; line-height: 1.2; letter-spacing: -1px; }
    .kpi-accent { background: linear-gradient(to right, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .dash-title { font-size: 2.5rem; font-weight: 800; color: #FAFAFA; letter-spacing: -1px; margin-bottom: 0px; }
    .dash-sub { color: #A1A1AA; font-size: 1.1rem; margin-top: 5px; margin-bottom: 25px; font-weight: 400; }
    
    .stButton > button[kind="primary"] {
        background: #FAFAFA !important; color: #09090B !important; border-radius: 8px !important;
        font-weight: 600 !important; border: none !important; padding: 0.75rem 1.5rem !important; transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover { background: #E4E4E7 !important; transform: scale(1.02); }
    
    [data-testid="stFileUploadDropzone"] { background-color: #18181B !important; border: 1px dashed #3F3F46 !important; border-radius: 8px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; background-color: transparent; border-bottom: 1px solid #27272A; }
    .stTabs [data-baseweb="tab"] { color: #A1A1AA; font-weight: 500; padding: 12px 4px; }
    .stTabs [aria-selected="true"] { color: #FAFAFA !important; border-bottom: 2px solid #FAFAFA !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================
def generate_synthetic_data(num_txns=3000):
    items = ['MacBook Air', 'Magic Mouse', 'Keychron K2', 'LG Monitor', 'Anker Hub', 'AirPods', 'Desk Mat', 'Logitech MX', 'Standing Desk', 'Webcam']
    return [random.sample(items, random.randint(1, 6)) for _ in range(num_txns)]

@st.cache_data
def load_data(file_source=None):
    transactions = generate_synthetic_data() if file_source is None else []
    if file_source:
        content = file_source.getvalue().decode("utf-8")
        for line in content.splitlines():
            line = line.strip().strip('"').strip("'")
            if line: transactions.append([item.strip() for item in line.split(",") if item.strip()])
                
    if not transactions: return None, None, None
    
    all_items = [item for t in transactions for item in t]
    counts = pd.Series(all_items).value_counts()
    
    metrics = {
        "total": len(transactions),
        "unique": len(counts),
        "counts": counts,
        "avg_basket": round(len(all_items) / len(transactions), 2)
    }
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    return transactions, sparse_df, metrics

# ==========================================
# 3. ALGORITHM EXECUTION
# ==========================================
def run_brute_force(transactions, min_support):
    limit = min(500, len(transactions))
    txns = transactions[:limit]
    n = len(txns)
    start = time.perf_counter()
    
    item_counts = pd.Series([item for t in txns for item in t]).value_counts()
    top_items = item_counts.head(10).index.tolist()
    
    freq_itemsets = []
    txn_sets = [set(t) for t in txns]
    
    for size in range(1, 4):
        for combo in itertools.combinations(top_items, size):
            combo_set = frozenset(combo)
            support = sum(1 for t in txn_sets if combo_set.issubset(t)) / n
            if support >= min_support:
                freq_itemsets.append({"itemsets": combo_set, "support": support})
                
    return pd.DataFrame(freq_itemsets), time.perf_counter() - start, limit

def compare_algorithms(transactions, sparse_df, min_support):
    results = {}
    
    start = time.perf_counter()
    results['FP-Growth'] = {"df": fpgrowth(sparse_df, min_support=min_support, use_colnames=True), "time": time.perf_counter() - start, "color": "#3B82F6"}
    
    start = time.perf_counter()
    results['Apriori'] = {"df": apriori(sparse_df, min_support=min_support, use_colnames=True), "time": time.perf_counter() - start, "color": "#8B5CF6"}
    
    df_bf, t_bf, limit = run_brute_force(transactions, min_support)
    results['Brute Force'] = {"df": df_bf, "time": t_bf, "limit": limit, "color": "#EF4444"}
    
    return results

def get_rules(freq_df, min_conf):
    if freq_df.empty: return pd.DataFrame()
    rules = association_rules(freq_df, metric="confidence", min_threshold=min_conf)
    if rules.empty: return pd.DataFrame()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).round(3)

# ==========================================
# 4. DARK MODE VISUALIZATIONS (DEBUGGED)
# ==========================================
def plot_efficiency(results):
    df = pd.DataFrame([{"Algorithm": k, "Time (Seconds)": v['time'], "Color": v['color']} for k,v in results.items()])
    fig = px.bar(df, x="Time (Seconds)", y="Algorithm", orientation='h', color="Algorithm", color_discrete_map={k:v for k,v in zip(df['Algorithm'], df['Color'])}, text_auto='.4f')
    fig.update_layout(title="Execution Time (Lower is Better)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA", showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    fig.update_xaxes(showgrid=True, gridcolor="#27272A")
    return fig

def plot_heatmap(transactions, top_n=10):
    all_items = [item for t in transactions for item in t]
    top_items = pd.Series(all_items).value_counts().head(top_n).index.tolist()
    
    matrix = pd.DataFrame(0, index=top_items, columns=top_items)
    for t in transactions:
        t_set = set(t)
        for item1 in top_items:
            if item1 in t_set:
                for item2 in top_items:
                    if item2 in t_set:
                        matrix.loc[item1, item2] += 1
                        
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale='Magma', title=f"Item Co-occurrence Density")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA")
    return fig

def plot_rule_scatter(rules_df):
    """
    BUG FIX: Removed marginal_x and marginal_y. Plotly Express throws ValueError 
    when combining marginal histograms with continuous color scales.
    """
    fig = px.scatter(
        rules_df, x="support", y="confidence", size="lift", color="lift", 
        hover_name="antecedents", hover_data=["consequents"], 
        color_continuous_scale="Plasma", title="Rule Distribution Matrix"
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA")
    fig.update_xaxes(showgrid=True, gridcolor="#27272A")
    fig.update_yaxes(showgrid=True, gridcolor="#27272A")
    return fig

# ==========================================
# 5. MAIN UI 
# ==========================================
def main():
    inject_midnight_css()
    
    st.markdown('<div class="dash-title">FIM Analytics Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">High-performance evaluation of Association Rule algorithms.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color:#FAFAFA; font-weight:700;'>⚙️ Parameters</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])
        
        st.markdown("---")
        min_support = st.slider("Minimum Support", 0.01, 0.40, 0.05, 0.01)
        min_conf = st.slider("Minimum Confidence", 0.10, 1.00, 0.40, 0.05)
        
        st.markdown("""
        <div style="background:#18181B; border:1px solid #27272A; padding:12px; border-radius:8px; margin-bottom:20px;">
            <span style="color:#FBBF24; font-size:12px; font-weight:700;">💡 PRO TIP</span><br>
            <span style="color:#A1A1AA; font-size:12px;">Lower support values exponentially increase Apriori execution time.</span>
        </div>
        """, unsafe_allow_html=True)
        
        run_btn = st.button("🚀 Run Analytics", type="primary", use_container_width=True)

    if 'txns' not in st.session_state or uploaded_file:
        txns, sparse_df, metrics = load_data(uploaded_file)
        st.session_state.update({'txns': txns, 'sparse': sparse_df, 'metrics': metrics})

    metrics = st.session_state['metrics']
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="saas-card"><div class="kpi-title">Dataset Size</div><div class="kpi-value">{metrics["total"]:,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="saas-card"><div class="kpi-title">Unique Items</div><div class="kpi-value">{metrics["unique"]:,}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="saas-card"><div class="kpi-title">Avg Basket</div><div class="kpi-value">{metrics["avg_basket"]}</div></div>', unsafe_allow_html=True)
    
    k4_placeholder = k4.empty()
    if not run_btn and 'results' not in st.session_state:
        k4_placeholder.markdown('<div class="saas-card"><div class="kpi-title">Active Itemsets</div><div class="kpi-value" style="color:#3F3F46;">-</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Executing Algorithmic Benchmarks..."):
            res = compare_algorithms(st.session_state['txns'], st.session_state['sparse'], min_support)
            st.session_state['results'] = res
            st.session_state['rules'] = get_rules(res['FP-Growth']['df'], min_conf)

    if 'results' in st.session_state:
        res = st.session_state['results']
        fp_df = res['FP-Growth']['df']
        k4_placeholder.markdown(f'<div class="saas-card"><div class="kpi-title">Active Itemsets</div><div class="kpi-value kpi-accent">{len(fp_df):,}</div></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["⚡ Efficiency Matrix", "📊 Visualization", "🗃️ Results Ledger"])
        
        with tab1:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.plotly_chart(plot_efficiency(res), use_container_width=True)
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="saas-card">
                    <h4 style="color:#FAFAFA; margin-bottom:15px; margin-top:0;">Execution Insights</h4>
                    <p style="color:#A1A1AA; font-size:14px;">🥇 <b>FP-Growth</b> dominated with <span style="color:#3B82F6;">{res['FP-Growth']['time']:.4f}s</span> by mapping the dataset into memory via a Trie structure.</p>
                    <p style="color:#A1A1AA; font-size:14px;">🥈 <b>Apriori</b> completed in <span style="color:#8B5CF6;">{res['Apriori']['time']:.4f}s</span>, limited by candidate generation overhead.</p>
                    <p style="color:#A1A1AA; font-size:14px;">⚠️ <b>Brute Force</b> was capped at {res['Brute Force']['limit']} transactions to prevent OOM errors.</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(plot_heatmap(st.session_state['txns']), use_container_width=True)
            with v2:
                rules_df = st.session_state.get('rules', pd.DataFrame())
                if not rules_df.empty:
                    st.plotly_chart(plot_rule_scatter(rules_df), use_container_width=True)
                else:
                    st.info("No strong rules found to visualize at current thresholds.")

        with tab3:
            st.markdown("<p style='color:#A1A1AA;'>Despite structural differences, Apriori and FP-Growth yield mathematically identical itemsets.</p>", unsafe_allow_html=True)
            if not fp_df.empty:
                display_df = fp_df.copy()
                display_df['itemsets'] = display_df['itemsets'].apply(lambda x: ", ".join(list(x)))
                display_df = display_df.sort_values('support', ascending=False)
                
                st.dataframe(
                    display_df,
                    column_config={"support": st.column_config.ProgressColumn("Support", format="%.3f", min_value=0, max_value=float(display_df['support'].max()))},
                    use_container_width=True, hide_index=True
                )

if __name__ == "__main__":
    main()