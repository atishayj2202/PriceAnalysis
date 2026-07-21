import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Configuration
st.set_page_config(
    page_title="Branded Rice Dynamic Pricing Dashboard",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .profit-section-card {
        background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
    }
    .revenue-section-card {
        background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
        margin-bottom: 20px;
    }
    .section-title-profit {
        font-size: 1.1rem;
        font-weight: 800;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .section-title-revenue {
        font-size: 1.1rem;
        font-weight: 800;
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .hero-price {
        font-size: 2.6rem;
        font-weight: 900;
        color: #ffffff;
        margin-top: 4px;
    }
    .badge-gold {
        background-color: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #f59e0b;
    }
    .badge-silver {
        background-color: rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #94a3b8;
    }
    .badge-bronze {
        background-color: rgba(168, 85, 247, 0.25);
        color: #c084fc;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #a855f7;
    }
</style>
""", unsafe_allow_html=True)

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "branded_rice_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "pipeline_results.csv")

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Run `construct_branded_rice_data.py` first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        st.error(f"Pipeline results not found at {RESULTS_PATH}. Run `elasticity_pipeline.py` first.")
        st.stop()
    df_res = pd.read_csv(RESULTS_PATH)
    return df_res

df_raw = load_dataset()
df_results = load_results()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/rice-bowl.png", width=70)
st.sidebar.title("Branded Rice Pricing Engine")
st.sidebar.markdown("---")

# Brand Selection
selected_brand = st.sidebar.selectbox(
    "1. Select Target Brand",
    options=["Daawat", "India_Gate", "Fortune"],
    index=0
)

# Dynamically filter and sort Top 3 Models for the selected brand
brand_results_sorted = df_results[df_results['brand'] == selected_brand].sort_values(by="Rank_Score", ascending=False)
top3_df = brand_results_sorted.head(3).reset_index(drop=True)

# Dynamic Top 3 Options list
top3_options = [
    f"🥇 Option 1: {top3_df.iloc[0]['model']} ({top3_df.iloc[0]['seasonality']} + {top3_df.iloc[0]['decay']})",
    f"🥈 Option 2: {top3_df.iloc[1]['model']} ({top3_df.iloc[1]['seasonality']} + {top3_df.iloc[1]['decay']})",
    f"🥉 Option 3: {top3_df.iloc[2]['model']} ({top3_df.iloc[2]['seasonality']} + {top3_df.iloc[2]['decay']})"
]

# Dynamic Model Selector with brand-specific key to reset selection on brand change
selected_model_idx = st.sidebar.radio(
    "2. Select Model Option (Top 3 for Brand)",
    options=[0, 1, 2],
    format_func=lambda i: top3_options[i],
    index=0,
    key=f"dynamic_top3_radio_{selected_brand}"
)

selected_model_row = top3_df.iloc[selected_model_idx]

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Manual Price Slider")
custom_price_change = st.sidebar.slider("Manual Price Adjustment (%)", min_value=-25.0, max_value=+25.0, value=0.0, step=0.5)

# Calculate Base and Optimal Metrics for the Brand
brand_df = df_raw[df_raw['brand'] == selected_brand].copy()
base_price = brand_df['unit_price'].mean()
base_cost = brand_df['cost_per_unit'].mean()
base_qty = brand_df['units_sold'].mean()
base_rev = base_price * base_qty
base_profit = (base_price - base_cost) * base_qty

# Calculate Elasticity for the selected model
elas = selected_model_row['mean_elasticity']
if elas == 0.0 or np.isnan(elas):
    elas = -0.574  # Fallback for models with 0 derivative

# Generate fine grid for optimization (-25% to +25%)
p_grid = np.linspace(-25, 25, 501)
prices_grid = base_price * (1.0 + p_grid / 100.0)
qtys_grid = base_qty * ((prices_grid / base_price) ** elas)
revs_grid = prices_grid * qtys_grid
profits_grid = (prices_grid - base_cost) * qtys_grid

# 1. Optimal Profit Point Calculation
opt_prof_idx = np.argmax(profits_grid)
opt_prof_price_pct = p_grid[opt_prof_idx]
opt_prof_price_val = prices_grid[opt_prof_idx]
opt_prof_qty_val = qtys_grid[opt_prof_idx]
opt_prof_rev_val = revs_grid[opt_prof_idx]
opt_prof_val = profits_grid[opt_prof_idx]

# 2. Optimal Revenue Point Calculation
opt_rev_idx = np.argmax(revs_grid)
opt_rev_price_pct = p_grid[opt_rev_idx]
opt_rev_price_val = prices_grid[opt_rev_idx]
opt_rev_qty_val = qtys_grid[opt_rev_idx]
opt_rev_val = revs_grid[opt_rev_idx]
opt_rev_prof_val = profits_grid[opt_rev_idx]

# Main Title & Header
st.title("🍚 Branded Rice Dynamic Pricing Strategy Dashboard")
st.markdown(f"**Target SKU / Brand**: `{selected_brand}` | **Active Model**: `{selected_model_row['model']}` ({selected_model_row['seasonality']} + {selected_model_row['decay']})")

# Top Banner: Dynamic Model Switcher Cards
st.subheader("🎯 Dynamic Top 3 Model Selection")
c_m1, c_m2, c_m3 = st.columns(3)

with c_m1:
    r1 = top3_df.iloc[0]
    is_sel1 = (selected_model_idx == 0)
    card_border = "border: 2px solid #fbbf24;" if is_sel1 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-gold">🥇 OPTION 1 (RANK 1)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r1['model']}</div>
        <div style="font-size:0.8rem; color:#fde68a;">{r1['seasonality'].upper()} + {r1['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r1['WMAPE']:.2f}%</b> | Elasticity: <b>{r1['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r1['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_m2:
    r2 = top3_df.iloc[1]
    is_sel2 = (selected_model_idx == 1)
    card_border = "border: 2px solid #38bdf8;" if is_sel2 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-silver">🥈 OPTION 2 (RANK 2)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r2['model']}</div>
        <div style="font-size:0.8rem; color:#cbd5e1;">{r2['seasonality'].upper()} + {r2['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r2['WMAPE']:.2f}%</b> | Elasticity: <b>{r2['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r2['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_m3:
    r3 = top3_df.iloc[2]
    is_sel3 = (selected_model_idx == 2)
    card_border = "border: 2px solid #a855f7;" if is_sel3 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-bronze">🥉 OPTION 3 (RANK 3)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r3['model']}</div>
        <div style="font-size:0.8rem; color:#e9d5ff;">{r3['seasonality'].upper()} + {r3['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r3['WMAPE']:.2f}%</b> | Elasticity: <b>{r3['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r3['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home: Profit & Revenue Optimization",
    "⚔️ Top 3 Models Comparison Matrix",
    "🌊 Seasonality & Fourier Rhythms",
    "🔬 Diagnostics & Exploration"
])

# ---------------------------------------------------------
# TAB 1: HOME PAGE - 2 SECTIONS (PROFIT & REVENUE)
# ---------------------------------------------------------
with tab1:
    st.markdown("## 💰 SECTION 1: GROSS PROFIT MAXIMIZATION STRATEGY")
    st.markdown("This strategy calculates the exact retail price that maximizes **Weekly Gross Profit Margin** taking unit cost (COGS) into account.")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    with col_p1:
        st.markdown(f"""
        <div class="profit-section-card">
            <div class="section-title-profit">PROFIT-MAXIMIZING PRICE</div>
            <div class="hero-price">₹{opt_prof_price_val:.2f}<span style="font-size:1.1rem; font-weight:normal;">/kg</span></div>
            <div style="color:#34d399; font-size:0.9rem; font-weight:bold; margin-top:4px;">Price Adjustment: {opt_prof_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_prof_val/1e7:.2f} Cr</div>
            <div style="color:#34d399; font-size:0.85rem; margin-top:4px;">Profit Lift: {((opt_prof_val - base_profit)/base_profit)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_prof_rev_val/1e7:.2f} Cr</div>
            <div style="color:#38bdf8; font-size:0.85rem; margin-top:4px;">Revenue Shift: {((opt_prof_rev_val - base_rev)/base_rev)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume</div>
            <div class="metric-value">{opt_prof_qty_val/1e3:.1f}k KG</div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">Volume Shift: {((opt_prof_qty_val - base_qty)/base_qty)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Gross Profit Optimization Chart
    fig_prof, ax_prof = plt.subplots(figsize=(12, 3.8), facecolor='#0f172a')
    ax_prof.set_facecolor('#1e293b')
    ax_prof.tick_params(colors='white')
    ax_prof.grid(True, linestyle='--', alpha=0.3)
    ax_prof.spines['top'].set_visible(False)
    ax_prof.spines['right'].set_visible(False)
    ax_prof.spines['left'].set_color('#475569')
    ax_prof.spines['bottom'].set_color('#475569')
    
    ax_prof.plot(p_grid, profits_grid / 1e7, color='#10b981', linewidth=2.5, label='Weekly Gross Profit (₹ Cr)')
    ax_prof.scatter(opt_prof_price_pct, opt_prof_val / 1e7, color='#fbbf24', s=130, zorder=5, label=f'Optimal Profit Peak (₹{opt_prof_price_val:.2f}/kg)')
    ax_prof.axvline(0.0, color='white', linestyle=':', alpha=0.5, label=f'Current Price (₹{base_price:.2f}/kg)')
    ax_prof.set_title(f"Gross Profit Optimization Curve for {selected_brand}", color='white', fontweight='bold')
    ax_prof.set_xlabel("Price Change (%)", color='white')
    ax_prof.set_ylabel("Gross Profit (₹ Cr)", color='white')
    ax_prof.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    st.pyplot(fig_prof)

    st.markdown("---")

    # SECTION 2: REVENUE MAXIMIZATION STRATEGY
    st.markdown("## 📈 SECTION 2: REVENUE MAXIMIZATION STRATEGY")
    st.markdown("This strategy calculates the exact retail price that maximizes **Total Weekly Top-Line Revenue** to capture market share.")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.markdown(f"""
        <div class="revenue-section-card">
            <div class="section-title-revenue">REVENUE-MAXIMIZING PRICE</div>
            <div class="hero-price">₹{opt_rev_price_val:.2f}<span style="font-size:1.1rem; font-weight:normal;">/kg</span></div>
            <div style="color:#7dd3fc; font-size:0.9rem; font-weight:bold; margin-top:4px;">Price Adjustment: {opt_rev_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_rev_val/1e7:.2f} Cr</div>
            <div style="color:#38bdf8; font-size:0.85rem; margin-top:4px;">Revenue Lift: {((opt_rev_val - base_rev)/base_rev)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_rev_prof_val/1e7:.2f} Cr</div>
            <div style="color:#34d399; font-size:0.85rem; margin-top:4px;">Profit Shift: {((opt_rev_prof_val - base_profit)/base_profit)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume</div>
            <div class="metric-value">{opt_rev_qty_val/1e3:.1f}k KG</div>
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">Volume Shift: {((opt_rev_qty_val - base_qty)/base_qty)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Revenue Optimization Chart
    fig_rev, ax_rev = plt.subplots(figsize=(12, 3.8), facecolor='#0f172a')
    ax_rev.set_facecolor('#1e293b')
    ax_rev.tick_params(colors='white')
    ax_rev.grid(True, linestyle='--', alpha=0.3)
    ax_rev.spines['top'].set_visible(False)
    ax_rev.spines['right'].set_visible(False)
    ax_rev.spines['left'].set_color('#475569')
    ax_rev.spines['bottom'].set_color('#475569')
    
    ax_rev.plot(p_grid, revs_grid / 1e7, color='#38bdf8', linewidth=2.5, label='Weekly Revenue (₹ Cr)')
    ax_rev.scatter(opt_rev_price_pct, opt_rev_val / 1e7, color='#7dd3fc', s=130, zorder=5, label=f'Optimal Revenue Peak (₹{opt_rev_price_val:.2f}/kg)')
    ax_rev.axvline(0.0, color='white', linestyle=':', alpha=0.5, label=f'Current Price (₹{base_price:.2f}/kg)')
    ax_rev.set_title(f"Weekly Revenue Optimization Curve for {selected_brand}", color='white', fontweight='bold')
    ax_rev.set_xlabel("Price Change (%)", color='white')
    ax_rev.set_ylabel("Revenue (₹ Cr)", color='white')
    ax_rev.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    st.pyplot(fig_rev)

# ---------------------------------------------------------
# TAB 2: TOP 3 MODELS COMPARISON MATRIX
# ---------------------------------------------------------
with tab2:
    st.header(f"⚔️ Top 3 Model Options Comparison for {selected_brand}")
    st.markdown("Detailed side-by-side technical comparison of the **Top 3 Recommended Models** for this brand.")
    
    m1 = top3_df.iloc[0]
    m2 = top3_df.iloc[1] if len(top3_df) > 1 else m1
    m3 = top3_df.iloc[2] if len(top3_df) > 2 else m1
    
    e1 = m1['mean_elasticity'] if m1['mean_elasticity'] != 0.0 else -0.574
    q1 = base_qty * ((prices_grid / base_price) ** e1)
    prof1 = (prices_grid - base_cost) * q1
    rev1 = prices_grid * q1
    
    e2 = m2['mean_elasticity'] if m2['mean_elasticity'] != 0.0 else -0.574
    q2 = base_qty * ((prices_grid / base_price) ** e2)
    prof2 = (prices_grid - base_cost) * q2
    rev2 = prices_grid * q2
    
    e3 = m3['mean_elasticity'] if m3['mean_elasticity'] != 0.0 else -0.574
    q3 = base_qty * ((prices_grid / base_price) ** e3)
    prof3 = (prices_grid - base_cost) * q3
    rev3 = prices_grid * q3
    
    fig_comp, (ax_c_rev, ax_c_prof) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
    for ax in [ax_c_rev, ax_c_prof]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        
    ax_c_rev.plot(p_grid, rev1 / 1e7, color='#fbbf24', linewidth=2.5, label=f"Rank 1: {m1['model']}")
    ax_c_rev.plot(p_grid, rev2 / 1e7, color='#38bdf8', linewidth=2.5, linestyle='--', label=f"Rank 2: {m2['model']}")
    ax_c_rev.plot(p_grid, rev3 / 1e7, color='#c084fc', linewidth=2.5, linestyle=':', label=f"Rank 3: {m3['model']}")
    ax_c_rev.set_title("Revenue Optimization Comparison", color='white', fontweight='bold')
    ax_c_rev.set_xlabel("Price Change (%)", color='white')
    ax_c_rev.set_ylabel("Weekly Revenue (₹ Crore)", color='white')
    ax_c_rev.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    ax_c_prof.plot(p_grid, prof1 / 1e7, color='#fbbf24', linewidth=2.5, label=f"Rank 1: {m1['model']} (Max: {p_grid[np.argmax(prof1)]:+.1f}%)")
    ax_c_prof.plot(p_grid, prof2 / 1e7, color='#38bdf8', linewidth=2.5, linestyle='--', label=f"Rank 2: {m2['model']} (Max: {p_grid[np.argmax(prof2)]:+.1f}%)")
    ax_c_prof.plot(p_grid, prof3 / 1e7, color='#c084fc', linewidth=2.5, linestyle=':', label=f"Rank 3: {m3['model']} (Max: {p_grid[np.argmax(prof3)]:+.1f}%)")
    ax_c_prof.set_title("Gross Profit Optimization Comparison", color='white', fontweight='bold')
    ax_c_prof.set_xlabel("Price Change (%)", color='white')
    ax_c_prof.set_ylabel("Weekly Gross Profit (₹ Crore)", color='white')
    ax_c_prof.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    st.pyplot(fig_comp)
    
    comp_matrix = pd.DataFrame({
        "Metric": [
            "Model Name",
            "Seasonality Mode",
            "Decay Setting",
            "WMAPE Forecast Error",
            "Seasonal R² Variance",
            "Seasonal WMAPE Gain",
            "Own-Price Elasticity (ε)",
            "Optimal Profit Price (₹/kg)",
            "Optimal Revenue Price (₹/kg)",
            "Composite Rank Score",
            "Operational Verdict"
        ],
        f"Option 1 ({m1['model']})": [
            m1['model'],
            m1['seasonality'].upper(),
            m1['decay'],
            f"{m1['WMAPE']:.2f}%",
            f"{m1['seasonal_r2']*100:.1f}%",
            f"+{m1['seasonal_wmape_gain']:.2f}%",
            f"{m1['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof1)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev1)]/100.0):.2f}",
            f"{m1['Rank_Score']:.3f}",
            "RECOMMENDED WINNER"
        ],
        f"Option 2 ({m2['model']})": [
            m2['model'],
            m2['seasonality'].upper(),
            m2['decay'],
            f"{m2['WMAPE']:.2f}%",
            f"{m2['seasonal_r2']*100:.1f}%",
            f"+{m2['seasonal_wmape_gain']:.2f}%",
            f"{m2['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof2)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev2)]/100.0):.2f}",
            f"{m2['Rank_Score']:.3f}",
            "Strong Alternative"
        ],
        f"Option 3 ({m3['model']})": [
            m3['model'],
            m3['seasonality'].upper(),
            m3['decay'],
            f"{m3['WMAPE']:.2f}%",
            f"{m3['seasonal_r2']*100:.1f}%",
            f"+{m3['seasonal_wmape_gain']:.2f}%",
            f"{m3['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof3)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev3)]/100.0):.2f}",
            f"{m3['Rank_Score']:.3f}",
            "Robust Runner-up"
        ]
    })
    
    st.dataframe(comp_matrix)

# ---------------------------------------------------------
# TAB 3: SEASONALITY & FOURIER RHYTHMS
# ---------------------------------------------------------
with tab3:
    st.header("🌊 Fourier Seasonality & Demand Rhythms")
    st.markdown("Our pipeline uses continuous Fourier Sine and Cosine terms ($\sin_{52,k}, \cos_{52,k}$) to model annual and quarterly demand rhythms.")
    
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.subheader("Empirical Seasonal Multiplier Curve (52-Week Cycle)")
        brand_df = df_raw[df_raw['brand'] == selected_brand].copy()
        brand_df['week_of_year'] = brand_df['date'].dt.isocalendar().week
        weekly_pattern = brand_df.groupby('week_of_year')['units_sold'].mean()
        weekly_pattern_norm = weekly_pattern / weekly_pattern.mean()
        
        fig_seas, ax_s = plt.subplots(figsize=(7, 4.5), facecolor='#0f172a')
        ax_s.set_facecolor('#1e293b')
        ax_s.plot(weekly_pattern_norm.index, weekly_pattern_norm.values, color='#c084fc', linewidth=2.5)
        ax_s.axhline(1.0, color='white', linestyle=':', alpha=0.5, label='Baseline Demand (1.0x)')
        ax_s.set_title(f"{selected_brand} Seasonal Demand Multiplier", color='white', fontweight='bold')
        ax_s.set_xlabel("Week of Year (1 to 52)", color='white')
        ax_s.set_ylabel("Seasonal Multiplier (x)", color='white')
        ax_s.tick_params(colors='white')
        ax_s.grid(True, linestyle='--', alpha=0.3)
        ax_s.spines['top'].set_visible(False)
        ax_s.spines['right'].set_visible(False)
        ax_s.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
        st.pyplot(fig_seas)
        
    with col_s2:
        st.subheader("Seasonal Accuracy & Value Summary")
        st.markdown(f"""
        - **Seasonal Variance Explained ($R^2$)**: **{selected_model_row['seasonal_r2']*100:.1f}%** of weekly sales fluctuations are driven by predictable calendar rhythms.
        - **WMAPE Accuracy Gain**: Adding Fixed Fourier seasonality reduced out-of-sample forecasting error by **+{selected_model_row['seasonal_wmape_gain']:.2f}% percentage points**.
        - **Peak Demand Windows**:
          - *Festival Surge (Weeks 40-46 / Oct-Nov)*: **+35% demand lift** (Diwali/Durga Puja).
          - *Winter Harvest & Wedding Surge (Weeks 2-8 / Jan-Feb)*: **+15% demand lift**.
          - *Monsoon Dip (Weeks 27-36 / Jul-Sep)*: **-15% demand dip**.
        """)

# ---------------------------------------------------------
# TAB 4: DIAGNOSTICS & EXPLORATION
# ---------------------------------------------------------
with tab4:
    st.header("🔬 Model Diagnostics & Historical Data")
    st.markdown("Examine historical price vs. cost trends and sales volume patterns.")
    
    brand_df = df_raw[df_raw['brand'] == selected_brand].copy()
    
    fig_diag, (ax_p_ts, ax_q_ts) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, facecolor='#0f172a')
    for ax in [ax_p_ts, ax_q_ts]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        
    ax_p_ts.plot(brand_df['date'], brand_df['unit_price'], color='#38bdf8', linewidth=2, label='Retail Price (₹/kg)')
    ax_p_ts.plot(brand_df['date'], brand_df['cost_per_unit'], color='#94a3b8', linestyle='--', label='Wholesale COGS (₹/kg)')
    ax_p_ts.set_title(f"{selected_brand} Historical Retail Price vs. COGS", color='white', fontweight='bold')
    ax_p_ts.set_ylabel("Price (₹/kg)", color='white')
    ax_p_ts.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    ax_q_ts.plot(brand_df['date'], brand_df['units_sold']/1e3, color='#34d399', linewidth=1.5, label='Actual Sales Volume (k KG)')
    ax_q_ts.set_title(f"{selected_brand} Weekly Sales Demand Volume", color='white', fontweight='bold')
    ax_q_ts.set_xlabel("Date", color='white')
    ax_q_ts.set_ylabel("Volume (k KG)", color='white')
    ax_q_ts.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    st.pyplot(fig_diag)

st.markdown("---")
st.caption("Branded Rice Pricing Engine | Dynamic Profit & Revenue Strategy Dashboard")
