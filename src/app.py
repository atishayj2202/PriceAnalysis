import sys
import os
# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        product = st.sidebar.selectbox("Product SKU", ["mobile_phone", "laptop"])
    else:
        product = st.sidebar.selectbox("Product SKU", ["rice", "shampoo", "face_wash", "fiama_body_wash"])
        
    scenario = st.sidebar.selectbox("Market Condition", ["stable", "inflation", "promo_heavy", "competitor_war"])
    
    mock_base_path = f"/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/MockData/{category}/{product}/{scenario}"
    
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
tab_result, tab_curves, tab_audit = st.tabs([
    "🎯 Results & Simulation", 
    "📈 Demand & Profit Curves", 
    "🔍 Spikes & Run Logs"
])

# ==============================================================================
# TAB 1: RESULT
# ==============================================================================
with tab_result:
    # 1. Inputs Columns (Sandbox and Human Checklist)
    c_sandbox, c_human = st.columns(2)
    
    with c_sandbox:
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
            
    with c_human:
        st.subheader("📋 Human Checklist Inputs")
        st.markdown("Assess the qualitative factors that cannot be automatically modeled:")
        
        h1_val = st.selectbox("H1: Brand / PR crisis event?", ["NO", "YES", "UNKNOWN"])
        h2_val = st.selectbox("H2: Strong new competitor/substitute launched?", ["NO", "YES", "UNKNOWN"])
        h3_val = st.selectbox("H3: Organic search rank dropped > 5 positions?", ["NO", "YES", "UNKNOWN"])
        h4_val = st.selectbox("H4: Pending government price ceiling/tariff?", ["NO", "YES", "UNKNOWN"])
        u1_val = st.selectbox("U1: Active public crisis/gouging backlash risk?", ["NO", "YES", "UNKNOWN"])
        u2_val = st.selectbox("U2: Product experiencing active viral hype?", ["NO", "YES", "UNKNOWN"])

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
    
    # Small coloured table for factor confidences
    table_rows = []
    for key, res in factor_results.items():
        status = res['status']
        weight_pct = weights.get(key, 0.0) * 100.0
        rel_str = res['reliability']
        
        # Color row details based on confidence/status
        row_color = "#1a1e24"
        text_color = "#e5e9f0"
        
        if "Left Out" in status:
            rel_badge = '<span style="color:#718096; font-weight:bold;">EXCLUDED</span>'
            row_color = "#16181d"
            text_color = "#718096"
        elif "HIGH" in rel_str:
            rel_badge = f'<span class="badge-high">{rel_str}</span>'
        elif "MEDIUM" in rel_str:
            rel_badge = f'<span class="badge-medium">{rel_str}</span>'
        elif "LOW" in rel_str:
            rel_badge = f'<span class="badge-low">{rel_str}</span>'
        else:
            rel_badge = f'<span class="badge-prov">{rel_str}</span>'
            
        table_rows.append(f"""
            <tr style="background-color: {row_color}; color: {text_color}; border-bottom: 1px solid #2d3748;">
                <td style="padding: 12px; font-weight: bold;">{key.capitalize()}</td>
                <td style="padding: 12px; text-align: center;">{rel_badge}</td>
                <td style="padding: 12px; text-align: center;">{weight_pct:.1f}%</td>
                <td style="padding: 12px; text-align: center;">{"Active" if "Used" in status else "Inactive"}</td>
                <td style="padding: 12px; font-size: 0.85rem;">{res['details']}</td>
            </tr>
        """)
        
    table_html = f"""
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px; border: 1px solid #2d3748; color: #e5e9f0;">
            <thead>
                <tr style="background-color: #2d3748; border-bottom: 2px solid #4e73df; color: #ffffff;">
                    <th style="padding: 12px; text-align: left;">Factor</th>
                    <th style="padding: 12px; text-align: center; width: 150px;">Reliability</th>
                    <th style="padding: 12px; text-align: center; width: 120px;">Normalized Weight</th>
                    <th style="padding: 12px; text-align: center; width: 120px;">Model Status</th>
                    <th style="padding: 12px; text-align: left;">Details</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

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
            st.markdown('✅ <b>H1 - Brand perception:</b> Stable NPS, no viral brand threats.', unsafe_allow_html=True)
            
        if h2_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H2 - Substitute availability:</b> Competitor launch flagged as "{h2_val}" (new entry). Adjust e_eff manually!</div>', unsafe_allow_html=True)
            override_states.append("H2")
        else:
            st.markdown('✅ <b>H2 - Substitute availability:</b> Assortment stable, no new category entrants.', unsafe_allow_html=True)
            
        if h3_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H3 - Channel/Placement:</b> Organic position drop flagged as "{h3_val}" (>5 rank drop). Postpone price experiment!</div>', unsafe_allow_html=True)
            override_states.append("H3")
        else:
            st.markdown('✅ <b>H3 - Channel/Placement:</b> Rank stable, shelf placement secure.', unsafe_allow_html=True)

    with c_check2:
        if h4_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>H4 - Regulatory & policy:</b> Tariff or price ceiling risk flagged as "{h4_val}". Escalate to legal!</div>', unsafe_allow_html=True)
            override_states.append("H4")
        else:
            st.markdown('✅ <b>H4 - Regulatory & policy:</b> Compliance cleared, no pending price controls.', unsafe_allow_html=True)
            
        if u1_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>U1 - Perceived fairness:</b> Price gouging backlash risk flagged as "{u1_val}". Block increases!</div>', unsafe_allow_html=True)
            override_states.append("U1")
        else:
            st.markdown('✅ <b>U1 - Perceived fairness:</b> Customer sentiment stable, pricing fits normal ranges.', unsafe_allow_html=True)
            
        if u2_val in ["YES", "UNKNOWN"]:
            st.markdown(f'<div class="caution-card">⚠️ <b>U2 - Word of mouth:</b> Viral social media spike flagged as "{u2_val}". Run spike cleaning!</div>', unsafe_allow_html=True)
            override_states.append("U2")
        else:
            st.markdown('✅ <b>U2 - Word of mouth:</b> Social trends stable, normal demand organic run rate.', unsafe_allow_html=True)

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
    st.markdown("Review demand elasticity bands and profit optimization curves across the price change range (\(\pm 30\%\)).")
    
    c_left, c_right = st.columns(2)
    
    with c_left:
        q_proj_range = []
        q_p10_range = []
        q_p90_range = []
        
        for x in x_range:
            temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle)
            q_proj_range.append(temp_p['q_new'])
            q_p10_range.append(temp_p['q_p10'])
            q_p90_range.append(temp_p['q_p90'])
            
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(x=x_range, y=q_p90_range, name="P90 (Upper Bound)", line=dict(dash='dash', color='#1cc88a')))
        fig_demand.add_trace(go.Scatter(x=x_range, y=q_proj_range, name="P50 (Median Projected)", line=dict(width=3, color='#4e73df')))
        fig_demand.add_trace(go.Scatter(x=x_range, y=q_p10_range, name="P10 (Lower Bound)", line=dict(dash='dash', color='#e74a3b')))
        
        fig_demand.add_vline(x=sim_x, line_width=2, line_dash="dot", line_color="#ffffff", annotation_text=f"Simulated ({sim_x:+.1f}%)")
        
        fig_demand.update_layout(
            title="Weekly Demand (Quantity) vs. Price Change %",
            xaxis_title="Price Change Percentage (%)",
            yaxis_title="Units Sold / Demand (Qty)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_demand, use_container_width=True)

    with c_right:
        profit_range = []
        for x in x_range:
            temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle)
            profit_range.append(temp_p['profit_new'])
            
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(x=x_range, y=profit_range, name="Projected Profit", line=dict(width=3, color='#f6c23e')))
        
        max_idx = np.argmax(profit_range)
        opt_x = x_range[max_idx]
        opt_profit = profit_range[max_idx]
        fig_profit.add_trace(go.Scatter(x=[opt_x], y=[opt_profit], mode='markers+text', name="Optimal Price Point", text=["Optimal"], textposition="top center", marker=dict(size=12, color='#1cc88a')))
        
        fig_profit.add_vline(x=sim_x, line_width=2, line_dash="dot", line_color="#ffffff", annotation_text=f"Simulated ({sim_x:+.1f}%)")
        
        fig_profit.update_layout(
            title="Projected Profit vs. Price Change %",
            xaxis_title="Price Change Percentage (%)",
            yaxis_title="Weekly Profit Amount",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_profit, use_container_width=True)

# ==============================================================================
# TAB 3: AUDIT & SPIKES
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
        
    st.markdown("#### 📜 Pricing Simulation Run Log (Audit Trail)")
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
