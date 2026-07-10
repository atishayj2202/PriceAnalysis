import sys
import os
# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agents.coordinator_agent import CoordinatorAgent

st.set_page_config(
    page_title="Demand Approximation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
    <style>
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #4e73df;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e222b 0%, #171923 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    .highlight-title {
        font-size: 1rem;
        font-weight: bold;
        color: #4e73df;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .highlight-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1cc88a;
        margin-top: 8px;
    }
    .highlight-desc {
        font-size: 0.9rem;
        color: #a0aec0;
        margin-top: 6px;
    }
    .hard-stop-card {
        background-color: #2d1b1e;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #e74a3b;
        color: #fed7d7;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(231, 74, 59, 0.1);
    }
    .caution-card {
        background-color: #2d261b;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #f6c23e;
        color: #feebc8;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .badge-high {
        background-color: rgba(28, 200, 138, 0.2);
        color: #1cc88a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #1cc88a;
    }
    .badge-medium {
        background-color: rgba(246, 194, 62, 0.2);
        color: #f6c23e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #f6c23e;
    }
    .badge-low {
        background-color: rgba(231, 74, 59, 0.2);
        color: #e74a3b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #e74a3b;
    }
    .badge-prov {
        background-color: rgba(78, 115, 223, 0.2);
        color: #4e73df;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #4e73df;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load either CSV or Excel
def load_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        else:
            st.sidebar.error(f"Unsupported file format: {uploaded_file.name}")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None

# App Title
st.title("📈 Demand Approximation & Pricing Engine")
st.markdown("### *Pricing Analytics & Multi-Agent Human Alignment Dashboard*")
st.markdown("---")

coordinator = CoordinatorAgent()

# Sidebar Configuration
st.sidebar.header("📁 Data Source Selection")
data_mode = st.sidebar.radio("Choose Input Mode", ["Load Mock Data Presets", "Upload Custom Files (CSV/Excel)"])

df_dict = {
    'sales': None,
    'competitor': None,
    'promotions': None,
    'inventory': None,
    'lifecycle': None,
    'sentiment': None
}

if data_mode == "Load Mock Data Presets":
    st.sidebar.subheader("Presets Configuration")
    category = st.sidebar.selectbox("Product Category", ["electronics", "fmcg"])
    
    if category == "electronics":
        product = st.sidebar.selectbox("Product SKU", ["mobile_phone", "laptop", "iphone_14_pro"])
    else:
        product = st.sidebar.selectbox("Product SKU", ["rice", "shampoo", "face_wash", "fiama_body_wash", "ghevar"])
        
    scenario = st.sidebar.selectbox("Market Condition", ["stable", "inflation", "promo_heavy", "competitor_war"])
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_base_path = os.path.join(base_dir, "MockData", category, product, scenario)
    
    if os.path.exists(mock_base_path):
        df_dict['sales'] = pd.read_csv(os.path.join(mock_base_path, 'sales_demand.csv'))
        df_dict['competitor'] = pd.read_csv(os.path.join(mock_base_path, 'competitor_pricing.csv'))
        df_dict['promotions'] = pd.read_csv(os.path.join(mock_base_path, 'marketing_promotions.csv'))
        df_dict['inventory'] = pd.read_csv(os.path.join(mock_base_path, 'inventory_status.csv'))
        df_dict['lifecycle'] = pd.read_csv(os.path.join(mock_base_path, 'product_lifecycle.csv'))
        df_dict['sentiment'] = pd.read_csv(os.path.join(mock_base_path, 'consumer_sentiment.csv'))
        st.sidebar.success(f"Loaded preset: {product} ({scenario})")
    else:
        st.sidebar.error("Preset files not found.")
        
else:
    st.sidebar.subheader("Upload Datasets")
    sales_file = st.sidebar.file_uploader("1. Historical Sales & Demand * (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['sales'] = load_uploaded_file(sales_file)
        
    comp_file = st.sidebar.file_uploader("2. Competitor Pricing (Optional) (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['competitor'] = load_uploaded_file(comp_file)
        
    promo_file = st.sidebar.file_uploader("3. Promotions & Marketing (Optional) (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['promotions'] = load_uploaded_file(promo_file)
        
    inv_file = st.sidebar.file_uploader("4. Inventory Status (Optional) (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['inventory'] = load_uploaded_file(inv_file)
        
    life_file = st.sidebar.file_uploader("5. Product Lifecycle Metadata (Optional) (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['lifecycle'] = load_uploaded_file(life_file)
        
    sent_file = st.sidebar.file_uploader("6. Consumer Sentiment / CCI (Optional) (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    df_dict['sentiment'] = load_uploaded_file(sent_file)

# Stop execution if sales demand file is not loaded
if df_dict['sales'] is None:
    st.info("👋 Please upload a mandatory Historical Sales & Demand CSV/Excel (*) in the sidebar, or select a Mock Data preset to get started.")
    st.stop()

# Run agent analysis
try:
    analysis_summary = coordinator.run_analysis(df_dict)
except Exception as e:
    st.error(f"Analysis failed during coordinator execution: {str(e)}")
    st.stop()

factor_results = analysis_summary['factor_results']
weights = analysis_summary['weights']
clean_sales = analysis_summary['clean_sales']
e_base = analysis_summary['e_base']
e_ci = analysis_summary['e_ci']
mape = analysis_summary['mape']
p_base = analysis_summary['p_base']
q_base = analysis_summary['q_base']
cost_base = analysis_summary['cost_base']

# Reset sandbox session state values when product SKU changes to prevent out-of-bounds StreamlitValueAboveMaxError
cat_val = category if 'category' in locals() else ""
prod_val = product if 'product' in locals() else ""
scen_val = scenario if 'scenario' in locals() else ""
sku_key = f"{cat_val}_{prod_val}_{scen_val}" if data_mode == "Load Mock Data Presets" else "custom_upload"

if 'prev_sku_key' not in st.session_state:
    st.session_state.prev_sku_key = sku_key

if st.session_state.prev_sku_key != sku_key:
    st.session_state.prev_sku_key = sku_key
    st.session_state.sim_change_pct = 0.0
    st.session_state.sim_price_val = float(round(p_base, 2))

# Pre-calculate Optimal Price points
x_range = np.linspace(-30.0, 30.0, 100)
opt_profit_x = 0.0
max_profit_val = -float('inf')
opt_rev_x = 0.0
max_rev_val = -float('inf')

for x in x_range:
    proj_vals = coordinator.project_demand(analysis_summary, x, is_promo_active=False)
    # Profit optimization
    if proj_vals['profit_new'] > max_profit_val:
        max_profit_val = proj_vals['profit_new']
        opt_profit_x = x
    # Revenue optimization
    if proj_vals['rev_new'] > max_rev_val:
        max_rev_val = proj_vals['rev_new']
        opt_rev_x = x

# TABS SEPARATION
tab_result, tab_curves, tab_trends, tab_debug, tab_audit, tab_doc = st.tabs([
    "🎯 Results & Simulation", 
    "📈 Demand & Profit Curves", 
    "📊 Historical Trends",
    "🐞 Debug Calculations",
    "🔍 Spikes & Run Logs",
    "📖 Documentation"
])

# ==============================================================================
# TAB 1: RESULT
# ==============================================================================
with tab_result:
    # Default all human override variables to NO since the checklist inputs are removed
    h1_val = "NO"
    h2_val = "NO"
    h3_val = "NO"
    h4_val = "NO"
    u1_val = "NO"
    u2_val = "NO"

    st.subheader("🕹️ Simulation Sandbox")
    st.markdown("Adjust price percentage or enter a target price below to analyze projections.")
    
    # Two-way sync session state init
    if 'sim_change_pct' not in st.session_state:
        st.session_state.sim_change_pct = 0.0

    # Synchronization functions
    def on_change_slider():
        st.session_state.sim_price_val = float(round(p_base * (1.0 + st.session_state.sim_change_pct / 100.0), 2))

    def on_change_price():
        st.session_state.sim_change_pct = float(round(((st.session_state.sim_price_val - p_base) / p_base) * 100.0, 2))

    # Initialize price input state if needed
    if 'sim_price_val' not in st.session_state:
        st.session_state.sim_price_val = float(round(p_base, 2))

    st.slider(
        "Price Change Percentage (%)",
        min_value=-50.0,
        max_value=50.0,
        step=0.5,
        key="sim_change_pct",
        on_change=on_change_slider
    )
    
    st.number_input(
        f"Target Price ($) [Base: ${p_base:.2f}]",
        min_value=float(round(p_base * 0.5, 2)),
        max_value=float(round(p_base * 1.5, 2)),
        step=0.01,
        key="sim_price_val",
        on_change=on_change_price
    )
    
    # Promotion Simulation Toggle
    promo_toggle = False
    if df_dict['promotions'] is not None and len(df_dict['promotions']) > 0:
        last_promo = int(df_dict['promotions']['is_promo'].iloc[-1])
        promo_toggle = st.checkbox("Simulate active promotion week?", value=(last_promo == 1))
    else:
        promo_toggle = st.checkbox("Simulate active promotion week?", value=False)

    # 2. Pre-calculate & Run Projections
    sim_x = st.session_state.sim_change_pct
    sim_proj = coordinator.project_demand(analysis_summary, sim_x, is_promo_active=promo_toggle)
    
    # 3. Optimal Price Cards Recommendation
    st.markdown("---")
    st.subheader("🎯 Optimal Price Recommendations")
    
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        p_opt_profit = p_base * (1.0 + opt_profit_x / 100.0)
        proj_opt_profit = coordinator.project_demand(analysis_summary, opt_profit_x, is_promo_active=promo_toggle)
        profit_inc_pct = proj_opt_profit['profit_increase_pct']
        
        st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-title">💰 Optimal Price for Max Profit</div>
                <div class="highlight-value">${p_opt_profit:.2f} <span style="font-size: 1.2rem; color: #a0aec0;">({opt_profit_x:+.1f}%)</span></div>
                <div class="highlight-desc">Expected profit increase of <b>{profit_inc_pct:+.2f}%</b> (weekly profit: ${proj_opt_profit['profit_new']:,.2f} vs. ${proj_opt_profit['profit_base']:,.2f} base).</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c_opt2:
        p_opt_rev = p_base * (1.0 + opt_rev_x / 100.0)
        proj_opt_rev = coordinator.project_demand(analysis_summary, opt_rev_x, is_promo_active=promo_toggle)
        rev_inc_pct = proj_opt_rev['rev_increase_pct']
        
        st.markdown(f"""
            <div class="highlight-card" style="border-left: 6px solid #4e73df;">
                <div class="highlight-title" style="color: #4e73df;">📢 Optimal Price for Max Revenue</div>
                <div class="highlight-value" style="color: #4e73df;">${p_opt_rev:.2f} <span style="font-size: 1.2rem; color: #a0aec0;">({opt_rev_x:+.1f}%)</span></div>
                <div class="highlight-desc">Expected revenue increase of <b>{rev_inc_pct:+.2f}%</b> (weekly revenue: ${proj_opt_rev['rev_new']:,.2f} vs. ${proj_opt_rev['rev_base']:,.2f} base).</div>
            </div>
        """, unsafe_allow_html=True)

    # 4. KPI Outputs of Selection
    st.markdown("---")
    st.subheader("📊 Simulation Projections for Selected Price")
    
    c_res1, c_res2, c_res3 = st.columns(3)
    with c_res1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Simulated Price</div>
                <div class="metric-value">${sim_proj['p_new']:.2f}</div>
                <div style="color: #4e73df; font-weight: bold;">{sim_x:+.2f}% Change</div>
            </div>
        """, unsafe_allow_html=True)
    with c_res2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #1cc88a;">
                <div class="metric-label">Revenue Change</div>
                <div class="metric-value">{sim_proj['rev_increase']:+,.2f}</div>
                <div style="color: {'#1cc88a' if sim_proj['rev_increase_pct'] >= 0 else '#e74a3b'}; font-weight: bold;">
                    {sim_proj['rev_increase_pct']:+.2f}% vs Base
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c_res3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f6c23e;">
                <div class="metric-label">Profit Change</div>
                <div class="metric-value">{sim_proj['profit_increase']:+,.2f}</div>
                <div style="color: {'#1cc88a' if sim_proj['profit_increase_pct'] >= 0 else '#e74a3b'}; font-weight: bold;">
                    {sim_proj['profit_increase_pct']:+.2f}% vs Base
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 3. Model & Factor Confidence Level Table
    st.markdown("---")
    st.subheader("📋 Model & Factor Confidence Metrics")
    
    # Calculate Overall Model Confidence Score based on active factor weights and reliability
    rel_map = {"HIGH": 0.88, "MEDIUM": 0.60, "LOW": 0.38, "PROVISIONAL": 0.50}
    weighted_conf = 0.0
    for key, res in factor_results.items():
        w = weights.get(key, 0.0)
        rel_str = res['reliability'].split()[0]  # get HIGH, MEDIUM, LOW
        rel_score = rel_map.get(rel_str, 0.5)
        weighted_conf += w * rel_score
    
    overall_conf_pct = weighted_conf * 100.0
    
    # Display overall confidence level
    overall_badge = ""
    if overall_conf_pct >= 75.0:
        overall_badge = '<span class="badge-high">HIGH CONFIDENCE</span>'
    elif overall_conf_pct >= 50.0:
        overall_badge = '<span class="badge-medium">MEDIUM CONFIDENCE</span>'
    else:
        overall_badge = '<span class="badge-low">LOW/PROVISIONAL</span>'
        
    st.markdown(f"**Overall Model Confidence Score: {overall_conf_pct:.1f}%** &nbsp;&nbsp;&nbsp; {overall_badge}", unsafe_allow_html=True)
    
    # Small coloured table for factor confidences using Streamlit Native DataFrame
    metrics_data = []
    for key, res in factor_results.items():
        status = res['status']
        weight_pct = weights.get(key, 0.0) * 100.0
        rel_str = res['reliability']
        
        status_label = "Active" if "Used" in status else "Inactive"
        if "Left Out" in status:
            rel_label = "EXCLUDED"
        else:
            rel_label = rel_str
            
        factor_val = res.get('factor_value', 1.0)
        val_str = f"{factor_val:.3f}" if isinstance(factor_val, (float, np.floating, np.float64)) else str(factor_val)
            
        metrics_data.append({
            "Factor": key.capitalize(),
            "Factor Value": val_str,
            "Reliability": rel_label,
            "Normalized Weight": f"{weight_pct:.1f}%",
            "Model Status": status_label,
            "Details": res['details']
        })
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # 4. Fixed Human Checklist Factors
    st.markdown("---")
    st.subheader("⚠️ Human Judgment Overrides (Cannot be automated)")
    st.markdown("The following qualitative factors require human oversight. If any are checked as `YES` or `UNKNOWN`, the model blocks automated deployment:")

    # Map state checks
    override_states = []
    c_check1, c_check2 = st.columns(2)
    with c_check1:
        if h1_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H1 - Brand perception:</b> Current crisis flagged as "{h1_val}" (NPS drop / PR event). Hold pricing changes!</div>', unsafe_allow_html=True)
            override_states.append("H1")
        else:
            st.markdown('<b>H1 - Brand perception:</b> Cleared (stable NPS, no brand threats).', unsafe_allow_html=True)
            
        if h2_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H2 - Substitute availability:</b> Competitor launch flagged as "{h2_val}" (new entry). Adjust e_eff manually!</div>', unsafe_allow_html=True)
            override_states.append("H2")
        else:
            st.markdown('<b>H2 - Substitute availability:</b> Cleared (stable assortment, no new entrants).', unsafe_allow_html=True)
            
        if h3_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H3 - Channel/Placement:</b> Organic position drop flagged as "{h3_val}" (>5 rank drop). Postpone price experiment!</div>', unsafe_allow_html=True)
            override_states.append("H3")
        else:
            st.markdown('<b>H3 - Channel/Placement:</b> Cleared (rank stable, shelf placement secure).', unsafe_allow_html=True)

    with c_check2:
        if h4_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H4 - Regulatory & policy:</b> Tariff or price ceiling risk flagged as "{h4_val}". Escalate to legal!</div>', unsafe_allow_html=True)
            override_states.append("H4")
        else:
            st.markdown('<b>H4 - Regulatory & policy:</b> Cleared (compliance cleared, no pending controls).', unsafe_allow_html=True)
            
        if u1_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>U1 - Perceived fairness:</b> Price gouging backlash risk flagged as "{u1_val}". Block increases!</div>', unsafe_allow_html=True)
            override_states.append("U1")
        else:
            st.markdown('<b>U1 - Perceived fairness:</b> Cleared (sentiment stable, pricing within norms).', unsafe_allow_html=True)
            
        if u2_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>U2 - Word of mouth:</b> Viral social media spike flagged as "{u2_val}". Run spike cleaning!</div>', unsafe_allow_html=True)
            override_states.append("U2")
        else:
            st.markdown('<b>U2 - Word of mouth:</b> Cleared (social trends stable, normal demand).', unsafe_allow_html=True)

    # Display execution clearance status
    st.markdown("---")
    # Build list of active hard stops
    hard_stops = []
    if abs(sim_proj['e_eff']) > 5.0:
        hard_stops.append("CRITICAL: Effective elasticity exceeds 5.0 (|e_eff| > 5.0). Estimates likely corrupted.")
    if abs(e_ci[1] - e_ci[0]) > 1.5:
        hard_stops.append("WARNING: 95% Confidence Interval width for e exceeds 1.5. Model has high uncertainty.")
    if abs(sim_x) > 25.0:
        hard_stops.append("WARNING: Price change is outside the safe extrapolation range of 25%. Requires manual review.")
    if df_dict['sentiment'] is not None and len(df_dict['sentiment']) > 0:
        cci_series = df_dict['sentiment']['cci_current']
        if len(cci_series) >= 5:
            if (cci_series.iloc[-5] - cci_series.iloc[-1]) > 10.0:
                hard_stops.append("CRITICAL: CCI drop exceeds 10 points in a single month. Macro demand climate highly unstable.")
    if (clean_sales['is_spike'] & (clean_sales['spike_type'].isna())).any():
        hard_stops.append("CRITICAL: Uncategorized spike detected in the sales series. Model run blocked.")
    if len(override_states) > 0:
        hard_stops.append(f"HUMAN SIGN-OFF BLOCKED: The following checklist items are triggered: {', '.join(override_states)}.")

    if len(hard_stops) > 0:
        st.markdown("<div class='hard-stop-card'><h3>⚠️ Hard Stop Active - Automated execution is BLOCKED</h3><ul>" + 
                    "".join([f"<li>{stop}</li>" for stop in hard_stops]) + 
                    "</ul></div>", unsafe_allow_html=True)
    else:
        st.success("✅ **CLEARED FOR EXECUTION:** No Hard Stops active and human checklist items are clear.")

# ==============================================================================
# TAB 2: ANALYSIS CURVES
# ==============================================================================
with tab_curves:
    st.markdown("### 📊 Demand & Profit Projection Curves")
    st.markdown("Review demand elasticity bands and profit optimization curves across the price change range (±30%).")
    
    # Plot weekly demand curve (full width)
    q_proj_range = []
    q_p10_range = []
    q_p90_range = []
    
    for x in x_range:
        temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle)
        q_proj_range.append(temp_p['q_new'])
        q_p10_range.append(temp_p['q_p10'])
        q_p90_range.append(temp_p['q_p90'])
        
    fig_d, ax_d = plt.subplots(figsize=(12, 4))
    fig_d.patch.set_facecolor('#121212')
    ax_d.set_facecolor('#1e1e1e')
    
    # Color ticks and labels
    ax_d.tick_params(colors='#a0aec0')
    ax_d.xaxis.label.set_color('#ffffff')
    ax_d.yaxis.label.set_color('#ffffff')
    
    ax_d.plot(x_range, q_p90_range, linestyle='--', color='#1cc88a', label="P90 (Upper Bound)")
    ax_d.plot(x_range, q_proj_range, linewidth=3, color='#4e73df', label="P50 (Median Projected)")
    ax_d.plot(x_range, q_p10_range, linestyle='--', color='#e74a3b', label="P10 (Lower Bound)")
    ax_d.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_d.set_title("Weekly Demand (Quantity) vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_d.set_xlabel("Price Change Percentage (%)")
    ax_d.set_ylabel("Units Sold / Demand (Qty)")
    ax_d.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_d.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='upper right', labelcolor='#ffffff')
    st.pyplot(fig_d)

    st.markdown("---")

    # Plot weekly profit curve (full width)
    profit_range = []
    for x in x_range:
        temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle)
        profit_range.append(temp_p['profit_new'])
        
    fig_p, ax_p = plt.subplots(figsize=(12, 4))
    fig_p.patch.set_facecolor('#121212')
    ax_p.set_facecolor('#1e1e1e')
    
    # Color ticks and labels
    ax_p.tick_params(colors='#a0aec0')
    ax_p.xaxis.label.set_color('#ffffff')
    ax_p.yaxis.label.set_color('#ffffff')
    
    ax_p.plot(x_range, profit_range, linewidth=3, color='#f6c23e', label="Projected Profit")
    
    max_idx = np.argmax(profit_range)
    opt_x = x_range[max_idx]
    opt_profit = profit_range[max_idx]
    ax_p.scatter([opt_x], [opt_profit], color='#1cc88a', s=100, zorder=5, label=f"Optimal ({opt_x:+.1f}%)")
    ax_p.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_p.set_title("Projected Profit vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_p.set_xlabel("Price Change Percentage (%)")
    ax_p.set_ylabel("Weekly Profit Amount")
    ax_p.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_p.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='lower center', labelcolor='#ffffff')
    st.pyplot(fig_p)

    st.markdown("---")

    # Plot weekly revenue curve (full width)
    revenue_range = []
    for x in x_range:
        temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle)
        revenue_range.append(temp_p['rev_new'])
        
    fig_r, ax_r = plt.subplots(figsize=(12, 4))
    fig_r.patch.set_facecolor('#121212')
    ax_r.set_facecolor('#1e1e1e')
    
    # Color ticks and labels
    ax_r.tick_params(colors='#a0aec0')
    ax_r.xaxis.label.set_color('#ffffff')
    ax_r.yaxis.label.set_color('#ffffff')
    
    ax_r.plot(x_range, revenue_range, linewidth=3, color='#36b9cc', label="Projected Revenue")
    
    max_rev_idx = np.argmax(revenue_range)
    opt_rev_x = x_range[max_rev_idx]
    opt_revenue = revenue_range[max_rev_idx]
    ax_r.scatter([opt_rev_x], [opt_revenue], color='#1cc88a', s=100, zorder=5, label=f"Max Revenue ({opt_rev_x:+.1f}%)")
    ax_r.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_r.set_title("Projected Revenue vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_r.set_xlabel("Price Change Percentage (%)")
    ax_r.set_ylabel("Weekly Revenue Amount")
    ax_r.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_r.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='lower center', labelcolor='#ffffff')
    st.pyplot(fig_r)

# ==============================================================================
# TAB 3: HISTORICAL TRENDS
# ==============================================================================
with tab_trends:
    st.markdown("### 📊 Historical Parameter Trends")
    st.markdown("Track how unit pricing, competitor pricing, product cost, and rolling elasticity evolve over the 3-year historical dataset.")
        # 1. Prepare Price & Cost time-series data
    df_price_cost = pd.DataFrame({
        'date': pd.to_datetime(clean_sales['date']),
        'Own Price': pd.to_numeric(clean_sales['unit_price'], errors='coerce'),
        'Unit Cost': pd.to_numeric(clean_sales['cost_per_unit'], errors='coerce')
    })
    
    if df_dict.get('competitor') is not None:
        df_comp = df_dict['competitor'].copy()
        df_comp['date'] = pd.to_datetime(df_comp['date'])
        df_comp['comp_price_avg'] = pd.to_numeric(df_comp['comp_price_avg'], errors='coerce')
        df_comp['comp_price_min'] = pd.to_numeric(df_comp.get('comp_price_min', df_comp['comp_price_avg']), errors='coerce')
        df_comp['comp_price_max'] = pd.to_numeric(df_comp.get('comp_price_max', df_comp['comp_price_avg']), errors='coerce')
        
        # Group by date and take mean to avoid duplicates if custom data contains multiple entries per date
        df_comp_agg = df_comp.groupby('date').agg({
            'comp_price_avg': 'mean',
            'comp_price_min': 'mean',
            'comp_price_max': 'mean'
        }).reset_index()
        
        df_price_cost = pd.merge(df_price_cost, df_comp_agg, on='date', how='left')
        df_price_cost.rename(columns={
            'comp_price_avg': 'Competitor Price',
            'comp_price_min': 'Competitor Min',
            'comp_price_max': 'Competitor Max'
        }, inplace=True)
        
    # 2. Compute rolling elasticity (using covariance formula for speed)
    def compute_rolling_e(df_sales, window=24):
        dates = []
        e_vals = []
        for i in range(window, len(df_sales) + 1):
            window_df = df_sales.iloc[i-window:i]
            # only compute on clean rows with positive values
            clean_w = window_df[
                (window_df['exclude_from_regression'] == False) & 
                (window_df['units_sold'] > 0) & 
                (window_df['unit_price'] > 0)
            ]
            if len(clean_w) < 10:
                continue
            try:
                log_q = np.log(clean_w['units_sold'])
                log_p = np.log(clean_w['unit_price'])
                cov = np.cov(log_p, log_q)
                # Enforce a price variance threshold to avoid division by near-zero float limits
                e = cov[0, 1] / cov[0, 0] if cov[0, 0] > 1e-4 else -1.5
                if not np.isnan(e) and not np.isinf(e):
                    # Clamp values between -10.0 and +2.0 to prevent scaling corruption on the plot
                    e = max(-10.0, min(2.0, e))
                    e_vals.append(e)
                    dates.append(pd.to_datetime(df_sales['date'].iloc[i-1]))
            except:
                pass
        return pd.DataFrame({'date': dates, 'rolling_elasticity': e_vals})
        
    # 3. Compute EMA elasticity (weights recent data more heavily)
    def compute_ema_e(df_sales, window=24):
        alpha = 2.0 / (window + 1)
        dates = []
        ema_vals = []
        
        # Calculate exponentially weighted elasticity step by step
        for i in range(12, len(df_sales) + 1):
            sub_df = df_sales.iloc[:i]
            # only compute on clean rows with positive values
            clean_w = sub_df[
                (sub_df['exclude_from_regression'] == False) & 
                (sub_df['units_sold'] > 0) & 
                (sub_df['unit_price'] > 0)
            ]
            if len(clean_w) < 10:
                continue
            try:
                log_q = np.log(clean_w['units_sold'].values)
                log_p = np.log(clean_w['unit_price'].values)
                n = len(clean_w)
                
                # Exponential weights: larger weight for more recent dates (which are at the end)
                w = np.array([(1.0 - alpha)**(n - 1 - k) for k in range(n)])
                w /= np.sum(w)
                
                # Weighted means
                mu_p = np.sum(w * log_p)
                mu_q = np.sum(w * log_q)
                
                # Weighted covariance & variance
                cov_pq = np.sum(w * (log_p - mu_p) * (log_q - mu_q))
                var_p = np.sum(w * (log_p - mu_p)**2)
                
                e_ema = cov_pq / var_p if var_p > 1e-4 else -1.5
                if not np.isnan(e_ema) and not np.isinf(e_ema):
                    e_ema = max(-10.0, min(2.0, e_ema))
                    ema_vals.append(e_ema)
                    dates.append(pd.to_datetime(df_sales['date'].iloc[i-1]))
            except:
                pass
        return pd.DataFrame({'date': dates, 'ema_elasticity': ema_vals})
        
    df_rolling_e = compute_rolling_e(clean_sales)
    df_ema_e = compute_ema_e(clean_sales)
    
    # Plot Prices & Cost over time (full width)
    fig_pc, ax_pc = plt.subplots(figsize=(12, 4))
    fig_pc.patch.set_facecolor('#121212')
    ax_pc.set_facecolor('#1e1e1e')
    
    # Color ticks and labels
    ax_pc.tick_params(colors='#a0aec0')
    ax_pc.xaxis.label.set_color('#ffffff')
    ax_pc.yaxis.label.set_color('#ffffff')
    
    ax_pc.plot(df_price_cost['date'].values, df_price_cost['Own Price'].values, linewidth=3, color='#4e73df', label="Own Price")
    ax_pc.plot(df_price_cost['date'].values, df_price_cost['Unit Cost'].values, linestyle=':', linewidth=2, color='#e74a3b', label="Unit Cost")
    
    if 'Competitor Price' in df_price_cost.columns:
        ax_pc.plot(df_price_cost['date'].values, df_price_cost['Competitor Price'].values, linestyle='--', linewidth=2, color='#f6c23e', label="Competitor Price (Avg)")
        if 'Competitor Min' in df_price_cost.columns and 'Competitor Max' in df_price_cost.columns:
            # Plot competitor pricing range envelope (gold gradient)
            ax_pc.fill_between(
                df_price_cost['date'].values,
                df_price_cost['Competitor Min'].values,
                df_price_cost['Competitor Max'].values,
                color='#f6c23e',
                alpha=0.15,
                label="Competitor Price Range"
            )
        
    ax_pc.set_title("Pricing & Cost Evolution Over Time", color='#ffffff', fontsize=12, pad=10)
    ax_pc.set_xlabel("Timeline")
    ax_pc.set_ylabel("Price / Cost per Unit")
    ax_pc.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_pc.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='upper right', labelcolor='#ffffff')
    plt.xticks(rotation=15)
    st.pyplot(fig_pc)
    
    st.markdown("---")
    
    # Plot Rolling Elasticity over time (full width)
    fig_el, ax_el = plt.subplots(figsize=(12, 4))
    fig_el.patch.set_facecolor('#121212')
    ax_el.set_facecolor('#1e1e1e')
    
    # Color ticks and labels
    ax_el.tick_params(colors='#a0aec0')
    ax_el.xaxis.label.set_color('#ffffff')
    ax_el.yaxis.label.set_color('#ffffff')
    
    # Plot rolling elasticity (green)
    if len(df_rolling_e) > 0:
        ax_el.plot(df_rolling_e['date'].values, df_rolling_e['rolling_elasticity'].values, linewidth=3, color='#1cc88a', label="Rolling 24-W Elasticity")
        
    # Plot EMA elasticity (cyan)
    if len(df_ema_e) > 0:
        ax_el.plot(df_ema_e['date'].values, df_ema_e['ema_elasticity'].values, linewidth=2.5, color='#17a2b8', label="Exponentially Weighted (EMA)")
        
    # Plot normal/baseline elasticity (horizontal line, now exponentially weighted in the main agent)
    ax_el.axhline(y=e_base, color='#f6c23e', linestyle='-', linewidth=2, label=f"EMA Elasticity (Main Agent: {e_base:.2f})")
        
    # Reference bounds (dashed lines)
    ax_el.axhline(y=-0.2, linestyle='--', color='#718096', alpha=0.7, label="Lower Bound (-0.2)")
    ax_el.axhline(y=-4.0, linestyle='--', color='#e74a3b', alpha=0.7, label="Upper Bound (-4.0)")
        
    ax_el.set_title("Elasticity Metric Comparisons Over Time", color='#ffffff', fontsize=12, pad=10)
    ax_el.set_xlabel("Timeline")
    ax_el.set_ylabel("Elasticity Value (e)")
    ax_el.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_el.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='lower left', labelcolor='#ffffff')
    plt.xticks(rotation=15)
    st.pyplot(fig_el)
        
    # 4. Plot Quantity Sold over time (with spike highlight markers)
    st.markdown("---")
    st.markdown("### 📦 Quantity Sold & Spike Evolution")
    st.markdown("Track weekly historical demand volume alongside automated spike detections and stockout anomalies on the timeline.")
    
    # Pre-compute anomaly arrays
    is_spike_arr = clean_sales['is_spike'].values if 'is_spike' in clean_sales.columns else np.zeros(len(clean_sales), dtype=bool)
    spike_type_arr = clean_sales['spike_type'].values if 'spike_type' in clean_sales.columns else np.array([None]*len(clean_sales))
    
    stockout_idx = np.zeros(len(clean_sales), dtype=bool)
    if df_dict.get('inventory') is not None:
        df_inv = df_dict['inventory'].copy()
        df_inv['date'] = pd.to_datetime(df_inv['date'])
        df_sales_dates = pd.DataFrame({'date': pd.to_datetime(clean_sales['date'])})
        df_inv_merged = pd.merge(df_sales_dates, df_inv, on='date', how='left')
        
        stock_level = df_inv_merged['units_in_stock'].values
        daily_sales = df_inv_merged['avg_daily_sales_14d'].values
        coverage = stock_level / np.maximum(0.1, daily_sales)
        stockout_idx = coverage < 7.0
        
    fig_q, ax_q = plt.subplots(figsize=(12, 4))
    fig_q.patch.set_facecolor('#121212')
    ax_q.set_facecolor('#1e1e1e')
    
    # Ticks and labels styling
    ax_q.tick_params(colors='#a0aec0')
    ax_q.xaxis.label.set_color('#ffffff')
    ax_q.yaxis.label.set_color('#ffffff')
    
    dates_arr = pd.to_datetime(clean_sales['date']).values
    q_arr = clean_sales['units_sold'].values
    
    # Plot baseline line
    ax_q.plot(dates_arr, q_arr, linewidth=2.5, color='#36b9cc', label="Units Sold", alpha=0.7)
    
    # Plot normal points
    normal_idx = ~is_spike_arr
    ax_q.scatter(dates_arr[normal_idx], q_arr[normal_idx], color='#36b9cc', s=25, zorder=3, label="Normal Demand")
    
    # Highlight Type A spikes
    type_a_idx = is_spike_arr & (spike_type_arr == 'Type A')
    if type_a_idx.any():
        ax_q.scatter(dates_arr[type_a_idx], q_arr[type_a_idx], color='#e74a3b', s=60, edgecolor='#ffffff', zorder=4, label="Type A Spike (Transient Hype)")
        
    # Highlight Type B spikes
    type_b_idx = is_spike_arr & (spike_type_arr == 'Type B')
    if type_b_idx.any():
        ax_q.scatter(dates_arr[type_b_idx], q_arr[type_b_idx], color='#f6c23e', marker='s', s=70, edgecolor='#ffffff', zorder=4, label="Type B Spike (Structural Shift)")
        
    # Highlight Stockouts
    if stockout_idx.any():
        ax_q.scatter(dates_arr[stockout_idx], q_arr[stockout_idx], color='#e74a3b', marker='x', s=80, linewidth=2.5, zorder=5, label="Stockout Risk (< 7d coverage)")
            
    ax_q.set_title("Units Sold Evolution & Anomaly Timeline", color='#ffffff', fontsize=12, pad=10)
    ax_q.set_xlabel("Timeline")
    ax_q.set_ylabel("Quantity Sold (units)")
    ax_q.grid(True, color='#2d3748', linestyle='-', linewidth=0.5)
    ax_q.legend(facecolor='#1e1e1e', edgecolor='#2d3748', loc='upper right', labelcolor='#ffffff')
    plt.xticks(rotation=15)
    st.pyplot(fig_q)
    
    # Summary of spikes & anomalies
    st.markdown("#### 🚨 Historical Anomaly Summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    num_type_a = int(np.sum(type_a_idx))
    num_type_b = int(np.sum(type_b_idx))
    num_stockouts = int(np.sum(stockout_idx))
    total_anoms = num_type_a + num_type_b + num_stockouts
    
    with col_a:
        st.metric("Type A Spikes", f"{num_type_a} Weeks", "Transient hype (Filtered)")
    with col_b:
        st.metric("Type B Shifts", f"{num_type_b} Weeks", "Structural break (Rebased)")
    with col_c:
        st.metric("Stockout Incidents", f"{num_stockouts} Weeks", "Supply constraints")
    with col_d:
        st.metric("Total Anomalies", f"{total_anoms} Weeks", "Flagged timeline activity")
        
    # Flagged anomaly log details table
    anomaly_rows = []
    for idx, row in clean_sales.iterrows():
        reasons = []
        if is_spike_arr[idx]:
            reasons.append(f"{spike_type_arr[idx]} Spike")
        if stockout_idx[idx]:
            reasons.append("Low Stock / Stockout Risk")
            
        if reasons:
            anomaly_rows.append({
                'Date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                'Units Sold': int(row['units_sold']),
                'Unit Price': f"${row['unit_price']:.2f}",
                'Anomaly Type': " & ".join(reasons),
                'Action Taken': "Excluded from Regressions" if row['exclude_from_regression'] else "Retained / Rebaselined"
            })
            
    if anomaly_rows:
        df_anoms = pd.DataFrame(anomaly_rows)
        st.markdown("**Detail Log of Flagged Anomalies:**")
        st.dataframe(df_anoms, use_container_width=True)
    else:
        st.success("No anomalies detected in the historical sales timeline.")

# TAB 4: DEBUG CALCULATIONS
# ==============================================================================
with tab_debug:
    st.markdown("### 🔄 Data Transformation Pipeline Audit")
    st.markdown("Track how raw sales logs undergo filtering, rebaselining, and regressions to yield final baseline model parameters.")
    
    # 1. Pipeline step-by-step metric cards
    col_step1, col_step2, col_step3, col_step4 = st.columns(4)
    with col_step1:
        st.metric("1. Raw Input History", f"{len(df_dict['sales'])} Weeks", "Initial upload/preset")
    with col_step2:
        discounted_rebaseline = int((clean_sales['rebaseline_weight_multiplier'] < 1.0).sum()) if 'rebaseline_weight_multiplier' in clean_sales.columns else 0
        st.metric("2. Structural Breaks", f"{discounted_rebaseline} Weeks", "Discounted prior data (0.2x)")
    with col_step3:
        st.metric("3. Cleaned Dataset", f"{len(clean_sales)} Weeks", "Outlier-cleaned history")
    with col_step4:
        active_wls = len(clean_sales[(clean_sales['exclude_from_regression'] == False) & (clean_sales['units_sold'] > 0)])
        st.metric("4. WLS Input Samples", f"{active_wls} Weeks", "Excluding Type A & 0s")
        
    # 2. Detailed pipeline description table
    st.markdown("#### ⚙️ Step-by-Step Data Transformations")
    pipeline_steps = [
        {
            "Pipeline Phase": "1. Raw Sales Load",
            "Input Data Source": "sales_demand.csv",
            "Operations Performed": "Reads date, units_sold, unit_price, cost_per_unit.",
            "Output State": f"{len(df_dict['sales'])} raw rows loaded."
        },
        {
            "Pipeline Phase": "2. Outlier Detection (Z-Score)",
            "Input Data Source": "sales_demand.csv",
            "Operations Performed": "Computes rolling 4-week mean and std. Flags WoW change > WoW velocity threshold.",
            "Output State": f"{clean_sales['is_spike'].sum()} weeks flagged as anomalies."
        },
        {
            "Pipeline Phase": "3. Spike Classification",
            "Input Data Source": "Spike Detector output",
            "Operations Performed": "Classifies spikes. Type A (transient, reverts within 3 weeks). Type B (structural shift, stays high 4+ weeks).",
            "Output State": f"Type A (excluded): {clean_sales['exclude_from_regression'].sum()} | Type B (break): {'Yes' if discounted_rebaseline > 0 else 'No'}."
        },
        {
            "Pipeline Phase": "4. Re-baselining Weight Discount",
            "Input Data Source": "Spike Classification output",
            "Operations Performed": "If Type B break occurs, scales down weights of all prior data to 0.2 to focus on new market reality.",
            "Output State": f"Discounted {discounted_rebaseline} weeks prior to break."
        },
        {
            "Pipeline Phase": "5. WLS Base Model Regression",
            "Input Data Source": "Rebaselined weights output",
            "Operations Performed": "Converts to ln(Q) and ln(P). Fits Weighted Least Squares (WLS) model: ln(Q) ~ ln(P) with 52-w half-life + break weights.",
            "Output State": f"Base Elasticity e = {e_base:.3f} | R2 = {factor_results['elasticity']['r2']:.4f} | MAPE = {mape:.2f}%."
        },
        {
            "Pipeline Phase": "6. Residual Multi-Agent Fit",
            "Input Data Source": "OLS residual errors",
            "Operations Performed": "Extracts residuals: log_q_actual - log_q_fitted. Dispatches residuals to 7 sub-agents in parallel to fit optional factor regressions.",
            "Output State": "Weights (R2_i / sum(R2)) and multipliers generated."
        }
    ]
    st.table(pd.DataFrame(pipeline_steps))
    
    st.markdown("---")
    st.markdown("### 🐞 Simulation Formula Breakdown")
    st.markdown("Below is the step-by-step breakdown of how the simulated demand quantity (**Q_new**) is calculated from the master demand projection formula:")
    
    # Master formula inputs
    e_val = factor_results['elasticity']['factor_value']
    C_val = factor_results['competitor']['factor_value'] if 'Used' in factor_results['competitor']['status'] else 1.0
    L_val = factor_results['lifecycle']['factor_value'] if 'Used' in factor_results['lifecycle']['status'] else 1.0
    X_val = factor_results['sentiment']['factor_value'] if 'Used' in factor_results['sentiment']['status'] else 1.0
    S_val = factor_results['seasonality']['factor_value'] if 'Used' in factor_results['seasonality']['status'] else 1.0
    I_val = factor_results['inventory']['factor_value'] if 'Used' in factor_results['inventory']['status'] else 1.0
    
    lift_m_val = 0.0
    if promo_toggle:
        lift_m_val = factor_results['promotions']['factor_value'] if 'Used' in factor_results['promotions']['status'] else 0.1275 # default fallback
        
    formula_data = [
        {"Parameter": "Base Quantity (Q_base)", "Formula Symbol": "Q_base", "Value": f"{q_base:.2f}", "Description": "Average sales of last 4 historical weeks"},
        {"Parameter": "Base Price (P_base)", "Formula Symbol": "P_base", "Value": f"${p_base:.2f}", "Description": "Average unit price of last 4 historical weeks"},
        {"Parameter": "New Price (P_new)", "Formula Symbol": "P_new", "Value": f"${sim_proj['p_new']:.2f}", "Description": "Simulated selling price (Base * (1 + x/100))"},
        {"Parameter": "Base Elasticity (e)", "Formula Symbol": "e", "Value": f"{e_val:.3f}", "Description": "OLS log-linear price sensitivity slope"},
        {"Parameter": "Competitor Modifier (C)", "Formula Symbol": "C", "Value": f"{C_val:.3f}", "Description": "Elasticity modifier based on price Gap vs competitor"},
        {"Parameter": "Lifecycle Modifier (L)", "Formula Symbol": "L", "Value": f"{L_val:.3f}", "Description": "Elasticity modifier based on product age phase"},
        {"Parameter": "Sentiment Modifier (X)", "Formula Symbol": "X", "Value": f"{X_val:.3f}", "Description": "Elasticity modifier based on Consumer Confidence Index"},
        {"Parameter": "Effective Elasticity (e_eff)", "Formula Symbol": "e_eff = e * C * L * X", "Value": f"{sim_proj['e_eff']:.3f}", "Description": "Final exponent used in projection"},
        {"Parameter": "Price Ratio Exponent", "Formula Symbol": "(P_new / P_base)^e_eff", "Value": f"{((sim_proj['p_new'] / p_base) ** sim_proj['e_eff']):.4f}", "Description": "Price ratio raised to effective elasticity"},
        {"Parameter": "Seasonality Multiplier (S)", "Formula Symbol": "S", "Value": f"{S_val:.3f}", "Description": "Aggregated seasonal factor for current week of year"},
        {"Parameter": "Inventory Multiplier (I)", "Formula Symbol": "I", "Value": f"{I_val:.3f}", "Description": "Scarcity demand multiplier if stock coverage < 7 days"},
        {"Parameter": "Promotions Lift (1 + lift_M)", "Formula Symbol": "1 + lift_M", "Value": f"{1.0 + lift_m_val:.3f}", "Description": "Sales volume lift factor if promotion is active"},
        {"Parameter": "Projected Demand (Q_new)", "Formula Symbol": "Q_proj", "Value": f"{sim_proj['q_new']:.2f}", "Description": "Calculated demand: Q_base * Ratio^e_eff * S * I * (1 + lift_M)"},
    ]
    
    st.table(pd.DataFrame(formula_data))
    
    # 2. Historical Week-by-Week Breakdown Table
    st.markdown("---")
    st.markdown("### 📂 Historical Week-by-Week Data Audit")
    st.markdown("This table contains the week-by-week values computed for all factors over the entire 3-year historical dataset:")
    
    # Construct df_debug
    df_debug = pd.DataFrame({
        'Date': clean_sales['date'],
        'Units Sold (Q)': clean_sales['units_sold'],
        'Own Price (P)': clean_sales['unit_price'],
        'Unit Cost': clean_sales['cost_per_unit'],
        'Excluded Spike': clean_sales['exclude_from_regression']
    })
    
    if 'residuals' in clean_sales.columns:
        df_debug['Residual (log)'] = clean_sales['residuals']
        
    try:
        df_debug['Week of Year'] = pd.to_datetime(df_debug['Date']).dt.isocalendar().week
        if 'week_averages' in factor_results['seasonality']:
            week_avg = factor_results['seasonality']['week_averages']
            df_debug['Seasonality S_t'] = df_debug['Week of Year'].map(lambda w: np.exp(week_avg.get(w, 0.0)))
    except:
        df_debug['Seasonality S_t'] = 1.0
        
    if df_dict.get('competitor') is not None:
        df_comp = df_dict['competitor'].copy()
        df_comp['date'] = pd.to_datetime(df_comp['date'])
        df_comp_avg = df_comp.groupby('date')['comp_price_avg'].mean().reset_index()
        df_debug = pd.merge(df_debug, df_comp_avg, left_on=pd.to_datetime(df_debug['Date']), right_on='date', how='left').drop(columns=['date'])
        df_debug.rename(columns={'comp_price_avg': 'Comp Price'}, inplace=True)
        df_debug['Comp Price'] = df_debug['Comp Price'].fillna(p_base)
        gap = (df_debug['Own Price (P)'] - df_debug['Comp Price']) / df_debug['Comp Price']
        gap_clamped = np.clip(gap, -0.5, 0.5)
        df_debug['Competitor C_t'] = 1.0 + 0.2 * np.sign(gap_clamped) * np.minimum(np.abs(gap_clamped), 0.5)
    else:
        df_debug['Comp Price'] = p_base
        df_debug['Competitor C_t'] = 1.0
        
    if df_dict.get('promotions') is not None:
        df_promo = df_dict['promotions'].copy()
        df_promo['date'] = pd.to_datetime(df_promo['date'])
        df_debug = pd.merge(df_debug, df_promo[['date', 'is_promo', 'marketing_spend']], left_on=pd.to_datetime(df_debug['Date']), right_on='date', how='left').drop(columns=['date'])
        df_debug.rename(columns={'is_promo': 'Promo Active', 'marketing_spend': 'Marketing Spend'}, inplace=True)
    else:
        df_debug['Promo Active'] = 0
        df_debug['Marketing Spend'] = 0.0
        
    if df_dict.get('inventory') is not None:
        df_inv = df_dict['inventory'].copy()
        df_inv['date'] = pd.to_datetime(df_inv['date'])
        df_debug = pd.merge(df_debug, df_inv[['date', 'units_in_stock', 'avg_daily_sales_14d']], left_on=pd.to_datetime(df_debug['Date']), right_on='date', how='left').drop(columns=['date'])
        df_debug.rename(columns={'units_in_stock': 'Stock Level', 'avg_daily_sales_14d': 'Daily Sales Rate'}, inplace=True)
        coverage = df_debug['Stock Level'] / np.maximum(0.1, df_debug['Daily Sales Rate'])
        df_debug['Inventory I_t'] = np.where(coverage < 7, 1.15, 1.0)
    else:
        df_debug['Stock Level'] = q_base * 3
        df_debug['Daily Sales Rate'] = q_base / 7.0
        df_debug['Inventory I_t'] = 1.0
        
    if df_dict.get('lifecycle') is not None:
        try:
            launch_date = pd.to_datetime(df_dict['lifecycle']['launch_date'].iloc[0])
            ages_months = [(pd.to_datetime(d) - launch_date).days / 30.4 for d in df_debug['Date']]
            df_debug['Age (Months)'] = ages_months
            df_debug['Lifecycle L_t'] = df_debug['Age (Months)'].map(lambda a: 0.7 if a < 6 else (0.85 if a < 18 else (1.0 if a < 36 else 1.2)))
        except:
            df_debug['Lifecycle L_t'] = 1.0
    else:
        df_debug['Lifecycle L_t'] = 1.0
        
    if df_dict.get('sentiment') is not None:
        df_sent = df_dict['sentiment'].copy()
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        df_debug = pd.merge(df_debug, df_sent[['date', 'cci_current', 'cci_baseline']], left_on=pd.to_datetime(df_debug['Date']), right_on='date', how='left').drop(columns=['date'])
        df_debug.rename(columns={'cci_current': 'CCI Current', 'cci_baseline': 'CCI Baseline'}, inplace=True)
        sig = (df_debug['CCI Current'] - df_debug['CCI Baseline']) / df_debug['CCI Baseline']
        df_debug['Sentiment X_t'] = 1.0 + 0.1 * sig
    else:
        df_debug['CCI Current'] = 100.0
        df_debug['CCI Baseline'] = 100.0
        df_debug['Sentiment X_t'] = 1.0
        
    # Reorder and format columns for readability
    cols_order = [
        'Date', 'Units Sold (Q)', 'Own Price (P)', 'Unit Cost', 'Residual (log)', 
        'Comp Price', 'Competitor C_t', 'Promo Active', 'Marketing Spend', 
        'Stock Level', 'Inventory I_t', 'Lifecycle L_t', 'CCI Current', 'Sentiment X_t',
        'Seasonality S_t', 'Excluded Spike'
    ]
    df_debug_final = df_debug[[c for c in cols_order if c in df_debug.columns]]
    st.dataframe(df_debug_final, use_container_width=True)
    
    # 3. Output raw analysis_summary dictionary metadata
    st.markdown("---")
    st.markdown("### 📊 Raw Analysis Summary Metadata")
    st.markdown("Below is the raw metadata dictionary (`analysis_summary`) generated by the Coordinate Agent for verification:")
    
    # Include all analysis_summary items including clean_sales
    debug_metadata = analysis_summary
    
    # Convert non-serializable datatypes (like numpy floats, dataframes, series) to native types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                import statsmodels.api as sm
                if isinstance(obj, sm.regression.linear_model.RegressionResultsWrapper):
                    return f"<RegressionResultsWrapper: R2={obj.rsquared:.4f}, nobs={obj.nobs}>"
            except:
                pass
            return obj
            
    st.json(make_serializable(debug_metadata))

# ==============================================================================
# TAB 5: AUDIT & SPIKES
# ==============================================================================
with tab_audit:
    st.markdown("### 🔍 Spikes and Data Cleaning Report")
    st.markdown("Coordinator pre-processing identifies outliers using rolling Z-scores and WoW velocity threshold metrics.")
    
    spikes_detected = clean_sales[clean_sales['is_spike'] == True]
    if len(spikes_detected) > 0:
        st.write(f"⚠️ **{len(spikes_detected)} anomalous demand spikes identified and cleaned:**")
        st.dataframe(spikes_detected[['date', 'units_sold', 'unit_price', 'spike_type', 'exclude_from_regression']])
    else:
        st.success("✅ Data cleaning complete: No demand spikes detected in history.")
        
    st.code(f"""
[RUN LOG - SIMULATION METADATA]
Product Base Price: {p_base:.2f}
Product Base Weekly Quantity: {q_base:.1f}
Product Base Unit Cost: {cost_base:.2f}
Baseline Elasticity e: {e_base:.3f} (95% CI: [{e_ci[0]:.3f}, {e_ci[1]:.3f}])
Effective Elasticity e_eff: {sim_proj['e_eff']:.3f}
Simulated Price: {sim_proj['p_new']:.2f} ({sim_x:+.1f}%)
Projected Demand Q_new: {sim_proj['q_new']:.1f} units
P10 / P50 / P90 Scenario Band: [{sim_proj['q_p10']:.1f} | {sim_proj['q_p50']:.1f} | {sim_proj['q_p90']:.1f}]
Expected Revenue Delta: {sim_proj['rev_increase']:+,.2f} ({sim_proj['rev_increase_pct']:+.1f}%)
Expected Profit Delta: {sim_proj['profit_increase']:+,.2f} ({sim_proj['profit_increase_pct']:+.1f}%)
Status: {"⛔ BLOCKED (Hard Stop Active)" if len(hard_stops) > 0 else "✅ CLEARED FOR EXECUTION"}
""", language="ini")

# ==============================================================================
# TAB 6: DOCUMENTATION
# ==============================================================================
with tab_doc:
    st.markdown("### 📖 Demand & Price Elasticity Documentation")
    st.markdown("Learn how the multi-agent system cleans data, calculates reliability, fits regressions, and applies modifiers.")

    st.markdown("---")
    st.subheader("🏁 Final Projections Formula")
    st.markdown("The final combined pricing and demand forecasting formula is stated below:")
    
    # Combined Symbolic Formula
    st.markdown("**Symbolic Formula:**")
    st.latex(r"Q_{\text{new}} = Q_{\text{base}} \cdot \left( \frac{P_{\text{new}}}{P_{\text{base}}} \right)^{e_{\text{base}} \cdot C \cdot L \cdot X} \cdot S \cdot I \cdot (1 + \text{lift}_M)")
    
    # Dynamic live numbers calculation
    e_base_val = analysis_summary['e_base']
    C_val = sim_proj['C']
    L_val = sim_proj['L']
    X_val = sim_proj['X']
    S_val = sim_proj['S']
    I_val = sim_proj['I']
    lift_m_val = sim_proj['lift_m']
    q_base_val = analysis_summary['q_base']
    p_base_val = analysis_summary['p_base']
    p_new_val = sim_proj['p_new']
    q_new_val = sim_proj['q_new']
    
    st.markdown("**Live Numeric Calculation (Mapped from Sandbox selections):**")
    st.latex(f"Q_{{\\text{{new}}}} = ({q_base_val:.1f}) \\cdot \\left( \\frac{{{p_new_val:.2f}}}{{{p_base_val:.2f}}} \\right)^{{({e_base_val:.3f}) \\cdot ({C_val:.3f}) \\cdot ({L_val:.3f}) \\cdot ({X_val:.3f})}} \\cdot ({S_val:.3f}) \\cdot ({I_val:.3f}) \\cdot (1 + {lift_m_val:.3f}) = {q_new_val:.1f}")

    st.markdown("#### 📝 Terminology Glossary & Explanations:")
    st.markdown(r"""
* **$Q_{\text{new}}$** (""" + f"{q_new_val:.1f}" + r"""): **Projected Demand Quantity**, representing the expected units sold per week.
* **$Q_{\text{base}}$** (""" + f"{q_base_val:.1f}" + r"""): **Base Weekly Quantity**, the historical baseline quantity of units sold.
* **$P_{\text{new}}$** ($""" + f"{p_new_val:.2f}" + r"""$) / **$P_{\text{base}}$** ($""" + f"{p_base_val:.2f}" + r"""$): **Simulated Price / Baseline Price** ratio.
* **$e_{\text{base}}$** (""" + f"{e_base_val:.3f}" + r"""): **Baseline Price Elasticity**, fitted via **Exponentially Weighted Least Squares (WLS)** regression of $\log(Q)$ vs $\log(P)$ with 24-week EMA-equivalent time-decay weights ($0.9200^{\Delta t}$) to prioritize recent market price sensitivity.
* **$C$** (""" + f"{C_val:.3f}" + r"""): **Competitor Pricing Modifier**, scales the elasticity exponent based on our price gap relative to competitors.
* **$L$** (""" + f"{L_val:.3f}" + r"""): **Product Lifecycle Modifier**, adjusts price sensitivity depending on launch hype or decline saturation phases.
* **$X$** (""" + f"{X_val:.3f}" + r"""): **Consumer Sentiment Modifier**, scales elasticity to adjust for macro buyer confidence (CCI index changes).
* **$S$** (""" + f"{S_val:.3f}" + r"""): **Seasonality Modifier**, multiplies volume to account for yearly calendar cycle fluctuations (like holidays/weather).
* **$I$** (""" + f"{I_val:.3f}" + r"""): **Inventory Stockout Modifier**, accounts for stock out supply caps or low-stock consumer scarcity signals.
* **$\text{lift}_M$** (""" + f"{lift_m_val:.3f}" + r"""): **Promotional Volume Lift**, captures the percentage volume increase when marketing promotion campaigns are active.
""")

    st.markdown("---")
    st.subheader("📋 Factor Regression & Modifier Formulas")
    
    # Table of formulas and symbols
    st.markdown("""
| Factor | Formula | Active State Modifier | Symbols Explanation |
| :--- | :--- | :--- | :--- |
| **$F_1$ - Price Elasticity** | $\\log(Q_t) = \\beta_0 + e_{\\text{base}} \\log(P_t) + \\epsilon_t$ | Baseline Elasticity ($e_{\\text{base}}$) | **$Q_t$**: units sold, **$P_t$**: unit price, **$e_{\\text{base}}$**: base elasticity coefficient, **$\\epsilon_t$**: residual. |
| **$F_2$ - Seasonality** | $\\epsilon_t = \\gamma_0 + \\sum \\gamma_k \\cdot D_{k, t} + u_t$ | $S = \\exp(\\text{residual}_{\\text{week}})$ | **$D_{k, t}$**: week dummies, **$\\gamma_k$**: seasonal residual average. |
| **$F_3$ - Competitor** | $\\epsilon_t = \\delta_0 + \\delta_1 \\cdot \\text{gap}_t + v_t$ | $C = 1.0 + 0.2 \\cdot \\text{sign}(\\text{gap}) \\cdot \\min(\\|\\text{gap}\\|, 0.5)$ | **$\\text{gap}$**: relative price gap to competitor, **$\\delta_1$**: cross-elasticity residual. |
| **$F_4$ - Promotions** | $\\epsilon_t = \\theta_0 + \\theta_1 \\cdot \\text{Promo}_t + w_t$ | $\\text{lift}_M = \\theta_1 \\cdot \\log(1 + \\text{spend})$ | **$\\text{Promo}_t$**: active promo flag, **$\\text{spend}$**: marketing adstock. |
| **$F_5$ - Inventory** | $\\epsilon_t = \\alpha_0 + \\alpha_1 \\cdot \\text{LowStock}_t + z_t$ | $I = 1.15$ (if coverage $< 7$ days) | **$\\text{LowStock}$**: low-stock dummy flag, **$I$**: scarcity demand lift. |
| **$F_6$ - Lifecycle** | $\\epsilon_t = \\lambda_0 + \\sum \\lambda_p \\cdot \\text{Phase}_{p, t} + \\eta_t$ | $L = \\text{Phase Modifier}$ (Hype vs Mature) | **$\\text{Phase}$**: Launch/Growth/Mature/Decline, **$L$**: lifecycle weight. |
| **$F_7$ - Sentiment** | $\\epsilon_t = \\omega_0 + \\omega_1 \\cdot X_{\\text{signal}, t} + e_t$ | $X = 1.0 + 0.1 \\cdot X_{\\text{signal}}$ | **$X_{\\text{signal}}$**: CCI relative index signal, **$X$**: buyer confidence modifier. |
""")

    st.markdown("---")
    st.subheader("⚖️ Normalized Weights Calculation ($w_i$)")
    st.markdown("Weights represent the relative explanatory power of each factor model's regression fit, normalized to sum to exactly 1.0:")
    st.latex(r"w_i = \frac{R^2_i \cdot \mathbb{I}(R^2_i \ge T_i)}{\sum_{j} R^2_j \cdot \mathbb{I}(R^2_j \ge T_j)}")
    st.markdown("""
* **$R^2_i$**: Coefficient of determination (r-squared) of factor $i$'s residual regression fit.
* **$T_i$**: Significance threshold ($T_{\\text{inventory}} = 0.03$, $T_{\\text{seasonality}/ \\text{promotions}/ \\text{sentiment}} = 0.05$).
* **$\\mathbb{I}(\\cdot)$**: Indicator function (1 if $R^2_i \\ge T_i$, 0 otherwise). Excludes noisy fits from weighting.
""")

    st.markdown("---")
    st.subheader("🛡️ Reliability Score Criteria")
    st.markdown("""
Reliability measures the statistical robustness of the regression fits using sample size and variance limits:
* **HIGH CONFIDENCE**: Sourced from high base model parameters with $N \\ge 30$ observations and low variance.
* **MEDIUM CONFIDENCE**: Active factors with high statistical residuals fit significance ($p \\le 0.05$).
* **LOW / PROVISIONAL**: Slashed to provisional if:
  * **Thin Data**: Clean observation count is under 30 ($N < 30$).
  * **High Parameter Uncertainty**: The 95% Confidence Interval width for baseline elasticity exceeds 1.5 ($\text{CI}_{\\text{width}} > 1.5$), or the absolute baseline elasticity estimate $|e_{\\text{base}}| > 5.0$.
""")
