import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_agents.coordinator_agent import MLCoordinatorAgent

st.set_page_config(
    page_title="ML Pricing & Demand approximation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for ML application
st.markdown("""
    <style>
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background-color: #1a1e27;
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #6366f1;
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
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #312e81;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    .highlight-title {
        font-size: 1rem;
        font-weight: bold;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .highlight-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #34d399;
        margin-top: 8px;
    }
    .highlight-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 6px;
    }
    .hard-stop-card {
        background-color: #451a1a;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #f87171;
        color: #fca5a5;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(248, 113, 113, 0.1);
    }
    .caution-card {
        background-color: #453015;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #fbbf24;
        color: #fde68a;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .badge-high {
        background-color: rgba(52, 211, 153, 0.2);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #34d399;
    }
    .badge-medium {
        background-color: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #fbbf24;
    }
    .badge-low {
        background-color: rgba(248, 113, 113, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #f87171;
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
st.title("🧠 ML / Deep Learning Pricing & Demand Engine")
st.markdown("### *Neural Network Approximations & Multi-Agent Attribution Dashboard*")
st.markdown("---")

coordinator = MLCoordinatorAgent()

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
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# ---------------------------------------------------------
# Sidebar Model Configuration & Hyperparameters
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Configurations")
model_mode = st.sidebar.selectbox(
    "Choose Prediction Architecture",
    ["Modular ML Pipeline", "Joint Neural Network Model"]
)

st.sidebar.subheader("⚙️ Joint NN Hyperparameters")
with st.sidebar.expander("Neural Network Architecture Tuning"):
    hidden_layers_choice = st.selectbox(
        "Hidden Layer Dimensions",
        ["(16, 8)", "(32, 16)", "(64, 32)"],
        index=1
    )
    mlp_hidden_layers = eval(hidden_layers_choice)
    
    mlp_solver = st.selectbox(
        "Optimization Solver",
        ["lbfgs", "adam", "sgd"],
        index=0
    )
    
    mlp_max_iter = st.slider(
        "Max Training Epochs",
        min_value=500,
        max_value=2500,
        value=1500,
        step=100
    )

# Run agent analysis with specified NN params
try:
    with st.spinner("Training Neural Networks and fitting models..."):
        analysis_summary = coordinator.run_analysis(
            df_dict,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_solver=mlp_solver,
            mlp_max_iter=mlp_max_iter
        )
except Exception as e:
    st.error(f"Analysis failed during model training: {str(e)}")
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

# Mode selection mapping
projection_mode = 'modular' if model_mode == "Modular ML Pipeline" else 'joint'

# Reset sandbox session state values when product SKU changes to prevent out-of-bounds StreamlitValueAboveMaxError
cat_val = category if 'category' in locals() else ""
prod_val = product if 'product' in locals() else ""
scen_val = scenario if 'scenario' in locals() else ""
sku_key = f"{cat_val}_{prod_val}_{scen_val}_{projection_mode}" if data_mode == "Load Mock Data Presets" else f"custom_upload_{projection_mode}"

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
    proj_vals = coordinator.project_demand(analysis_summary, x, is_promo_active=False, mode=projection_mode)
    # Profit optimization
    if proj_vals['profit_new'] > max_profit_val:
        max_profit_val = proj_vals['profit_new']
        opt_profit_x = x
    # Revenue optimization
    if proj_vals['rev_new'] > max_rev_val:
        max_rev_val = proj_vals['rev_new']
        opt_rev_x = x

# TABS SEPARATION
tab_result, tab_curves, tab_insights, tab_historical, tab_audit, tab_doc = st.tabs([
    "🎯 ML Results & Simulation", 
    "📈 Demand & Profit Curves", 
    "🧠 NN & ML Insights",
    "📊 Historical fit",
    "🔍 Hard Stops & Auditing",
    "📖 Methodology"
])

# ==============================================================================
# TAB 1: RESULTS & SIMULATION
# ==============================================================================
with tab_result:
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

    # Pre-calculate & Run Projections
    sim_x = st.session_state.sim_change_pct
    sim_proj = coordinator.project_demand(analysis_summary, sim_x, is_promo_active=promo_toggle, mode=projection_mode)
    
    # Optimal Price Cards Recommendation
    st.markdown("---")
    st.subheader("🎯 Optimal Price Recommendations")
    
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        p_opt_profit = p_base * (1.0 + opt_profit_x / 100.0)
        proj_opt_profit = coordinator.project_demand(analysis_summary, opt_profit_x, is_promo_active=promo_toggle, mode=projection_mode)
        profit_inc_pct = proj_opt_profit['profit_increase_pct']
        
        st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-title">💰 Optimal Price for Max Profit</div>
                <div class="highlight-value">${p_opt_profit:.2f} <span style="font-size: 1.2rem; color: #94a3b8;">({opt_profit_x:+.1f}%)</span></div>
                <div class="highlight-desc">Expected profit increase of <b>{profit_inc_pct:+.2f}%</b> (weekly profit: ${proj_opt_profit['profit_new']:,.2f} vs. ${proj_opt_profit['profit_base']:,.2f} base).</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c_opt2:
        p_opt_rev = p_base * (1.0 + opt_rev_x / 100.0)
        proj_opt_rev = coordinator.project_demand(analysis_summary, opt_rev_x, is_promo_active=promo_toggle, mode=projection_mode)
        rev_inc_pct = proj_opt_rev['rev_increase_pct']
        
        st.markdown(f"""
            <div class="highlight-card" style="border-left: 6px solid #818cf8;">
                <div class="highlight-title" style="color: #818cf8;">📢 Optimal Price for Max Revenue</div>
                <div class="highlight-value" style="color: #818cf8;">${p_opt_rev:.2f} <span style="font-size: 1.2rem; color: #94a3b8;">({opt_rev_x:+.1f}%)</span></div>
                <div class="highlight-desc">Expected revenue increase of <b>{rev_inc_pct:+.2f}%</b> (weekly revenue: ${proj_opt_rev['rev_new']:,.2f} vs. ${proj_opt_rev['rev_base']:,.2f} base).</div>
            </div>
        """, unsafe_allow_html=True)

    # KPI Outputs of Selection
    st.markdown("---")
    st.subheader(f"📊 ML Simulation Projections for Selected Price ({model_mode})")
    
    c_res1, c_res2, c_res3 = st.columns(3)
    with c_res1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Simulated Price</div>
                <div class="metric-value">${sim_proj['p_new']:.2f}</div>
                <div style="color: #818cf8; font-weight: bold;">{sim_x:+.2f}% Change</div>
            </div>
        """, unsafe_allow_html=True)
    with c_res2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #34d399;">
                <div class="metric-label">Revenue Change</div>
                <div class="metric-value">{sim_proj['rev_increase']:+,.2f}</div>
                <div style="color: {'#34d399' if sim_proj['rev_increase_pct'] >= 0 else '#f87171'}; font-weight: bold;">
                    {sim_proj['rev_increase_pct']:+.2f}% vs Base
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c_res3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #fbbf24;">
                <div class="metric-label">Profit Change</div>
                <div class="metric-value">{sim_proj['profit_increase']:+,.2f}</div>
                <div style="color: {'#34d399' if sim_proj['profit_increase_pct'] >= 0 else '#f87171'}; font-weight: bold;">
                    {sim_proj['profit_increase_pct']:+.2f}% vs Base
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Model & Factor Confidence Level Table
    st.markdown("---")
    st.subheader("📋 Model & Factor Confidence Metrics")
    
    if projection_mode == 'modular':
        rel_map = {"HIGH": 0.90, "MEDIUM": 0.65, "LOW": 0.50, "PROVISIONAL": 0.55}
        weighted_conf = 0.0
        for key, res in factor_results.items():
            w = weights.get(key, 0.0)
            rel_str = res['reliability'].split()[0]
            rel_score = rel_map.get(rel_str, 0.5)
            weighted_conf += w * rel_score
        
        overall_conf_pct = weighted_conf * 100.0
        
        overall_badge = ""
        if overall_conf_pct >= 75.0:
            overall_badge = '<span class="badge-high">HIGH CONFIDENCE</span>'
        elif overall_conf_pct >= 50.0:
            overall_badge = '<span class="badge-medium">MEDIUM CONFIDENCE</span>'
        else:
            overall_badge = '<span class="badge-low">LOW/PROVISIONAL</span>'
            
        st.markdown(f"**Overall Model Confidence Score (Modular ML): {overall_conf_pct:.1f}%** &nbsp;&nbsp;&nbsp; {overall_badge}", unsafe_allow_html=True)
        
        metrics_data = []
        for key, res in factor_results.items():
            status = res['status']
            weight_pct = weights.get(key, 0.0) * 100.0
            rel_str = res['reliability']
            
            status_label = "Active" if "Used" in status else "Inactive"
            rel_label = "EXCLUDED" if "Left Out" in status else rel_str
                
            factor_val = res.get('factor_value', 1.0)
            val_str = f"{factor_val:.3f}" if isinstance(factor_val, (float, np.floating, np.float64)) else str(factor_val)
                
            metrics_data.append({
                "Factor (ML Agent)": key.capitalize(),
                "Factor Value": val_str,
                "Reliability": rel_label,
                "Normalized Weight": f"{weight_pct:.1f}%",
                "Model Status": status_label,
                "Details": res['details']
            })
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)
    else:
        st.markdown(f"**Joint Neural Network Status: ACTIVE** &nbsp;&nbsp;&nbsp; <span class='badge-high'>JOINT ENSEMBLE ({coordinator.joint_ensemble_size} MLPs)</span>", unsafe_allow_html=True)
        st.markdown(f"**Joint Model fit MAPE: {analysis_summary['joint_mape']:.2f}%** &nbsp;&nbsp;&nbsp; **Joint Model R²: {analysis_summary['joint_r2']:.4f}**")
        
        # Display current features status table
        df_feats = analysis_summary['df_feats']
        last_feats = analysis_summary['base_features']
        
        feat_data = []
        for col in df_feats.columns:
            feat_data.append({
                "Feature Name": col,
                "Baseline Value (Current Week)": f"{last_feats[col]:.4f}",
                "Feature Role": "Price Input" if col == 'log_price' else "Contextual Feature"
            })
        df_feat_table = pd.DataFrame(feat_data)
        st.dataframe(df_feat_table, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 2: DEMAND & PROFIT CURVES
# ==============================================================================
with tab_curves:
    st.subheader("📊 Non-linear Demand & Profit curves (ML Estimates)")
    st.markdown("Review demand elasticity bands and profit optimization curves across the price change range (±30%).")
    
    # Pre-calculate curves
    q_proj_range = []
    q_p10_range = []
    q_p90_range = []
    profit_range = []
    revenue_range = []
    
    for x in x_range:
        temp_p = coordinator.project_demand(analysis_summary, x, is_promo_active=promo_toggle, mode=projection_mode)
        q_proj_range.append(temp_p['q_new'])
        q_p10_range.append(temp_p['q_p10'])
        q_p90_range.append(temp_p['q_p90'])
        profit_range.append(temp_p['profit_new'])
        revenue_range.append(temp_p['rev_new'])
        
    # Plot weekly demand curve
    fig_d, ax_d = plt.subplots(figsize=(12, 4))
    fig_d.patch.set_facecolor('#0f1115')
    ax_d.set_facecolor('#1a1e27')
    
    ax_d.tick_params(colors='#94a3b8')
    ax_d.xaxis.label.set_color('#ffffff')
    ax_d.yaxis.label.set_color('#ffffff')
    
    ax_d.plot(x_range, q_p90_range, linestyle='--', color='#34d399', label="P90 (Upper Bound)")
    ax_d.plot(x_range, q_proj_range, linewidth=3, color='#6366f1', label="P50 (Median Projected)")
    ax_d.plot(x_range, q_p10_range, linestyle='--', color='#f87171', label="P10 (Lower Bound)")
    ax_d.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_d.set_title("Weekly Demand (Quantity) vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_d.set_xlabel("Price Change Percentage (%)")
    ax_d.set_ylabel("Units Sold (Qty)")
    ax_d.grid(True, color='#334155', linestyle='-', linewidth=0.5)
    ax_d.legend(facecolor='#1a1e27', edgecolor='#334155', loc='upper right', labelcolor='#ffffff')
    st.pyplot(fig_d)
    
    st.markdown("---")
    
    # Plot weekly profit curve
    fig_p, ax_p = plt.subplots(figsize=(12, 4))
    fig_p.patch.set_facecolor('#0f1115')
    ax_p.set_facecolor('#1a1e27')
    
    ax_p.tick_params(colors='#94a3b8')
    ax_p.xaxis.label.set_color('#ffffff')
    ax_p.yaxis.label.set_color('#ffffff')
    
    ax_p.plot(x_range, profit_range, linewidth=3, color='#fbbf24', label="Projected Profit")
    
    max_idx = np.argmax(profit_range)
    opt_x = x_range[max_idx]
    opt_profit = profit_range[max_idx]
    ax_p.scatter([opt_x], [opt_profit], color='#34d399', s=100, zorder=5, label=f"Optimal ({opt_x:+.1f}%)")
    ax_p.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_p.set_title("Projected Profit vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_p.set_xlabel("Price Change Percentage (%)")
    ax_p.set_ylabel("Weekly Profit Amount")
    ax_p.grid(True, color='#334155', linestyle='-', linewidth=0.5)
    ax_p.legend(facecolor='#1a1e27', edgecolor='#334155', loc='lower center', labelcolor='#ffffff')
    st.pyplot(fig_p)

    st.markdown("---")
    
    # Plot weekly revenue curve
    fig_r, ax_r = plt.subplots(figsize=(12, 4))
    fig_r.patch.set_facecolor('#0f1115')
    ax_r.set_facecolor('#1a1e27')
    
    ax_r.tick_params(colors='#94a3b8')
    ax_r.xaxis.label.set_color('#ffffff')
    ax_r.yaxis.label.set_color('#ffffff')
    
    ax_r.plot(x_range, revenue_range, linewidth=3, color='#60a5fa', label="Projected Revenue")
    
    max_rev_idx = np.argmax(revenue_range)
    opt_rev_x = x_range[max_rev_idx]
    opt_revenue = revenue_range[max_rev_idx]
    ax_r.scatter([opt_rev_x], [opt_revenue], color='#34d399', s=100, zorder=5, label=f"Max Revenue ({opt_rev_x:+.1f}%)")
    ax_r.axvline(x=sim_x, linestyle=':', color='#ffffff', label=f"Simulated ({sim_x:+.1f}%)")
    
    ax_r.set_title("Projected Revenue vs. Price Change %", color='#ffffff', fontsize=12, pad=10)
    ax_r.set_xlabel("Price Change Percentage (%)")
    ax_r.set_ylabel("Weekly Revenue Amount")
    ax_r.grid(True, color='#334155', linestyle='-', linewidth=0.5)
    ax_r.legend(facecolor='#1a1e27', edgecolor='#334155', loc='lower center', labelcolor='#ffffff')
    st.pyplot(fig_r)

# ==============================================================================
# TAB 3: NEURAL NETWORK & ML INSIGHTS
# ==============================================================================
with tab_insights:
    st.subheader("🧠 Neural Network Structure & Feature Importance")
    
    # Display joint NN model specs
    st.markdown("#### Joint Neural Network Architecture")
    st.write(f"- **Ensemble size**: {coordinator.joint_ensemble_size} Multi-Layer Perceptrons (trained with bootstrapped data resampling)")
    st.write(f"- **Hidden Layer layout**: {mlp_hidden_layers} neurons per layer")
    st.write(f"- **Activation function**: Tanh (hyperbolic tangent)")
    st.write(f"- **Optimization Solver**: {mlp_solver.upper()}")
    st.write(f"- **Max Training Epochs**: {mlp_max_iter}")
    
    st.markdown("---")
    st.markdown("#### Feature Importances")
    
    # Calculate feature importances
    importances = {}
    if projection_mode == 'modular':
        # Normalized R2 values of modular models
        for key, res in factor_results.items():
            if 'Used' in res['status']:
                importances[key] = res['r2']
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
    else:
        # Permutation feature importance for joint NN
        joint_models = analysis_summary['joint_models']
        df_clean = analysis_summary['clean_sales']
        df_dict = analysis_summary['df_dict']
        
        # Build features
        df_feats = coordinator._prepare_joint_features(df_clean, df_dict)
        clean_idx = df_clean[df_clean['exclude_from_regression'] == False].index
        X_joint = df_feats.loc[clean_idx]
        y_joint = np.log(df_clean.loc[clean_idx, 'units_sold'].values)
        
        baseline_mape = analysis_summary['joint_mape']
        
        for col in X_joint.columns:
            # Permute values
            X_shuffled = X_joint.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            
            # Predict
            preds = []
            for mlp in joint_models:
                preds.append(mlp.predict(X_shuffled))
            mean_pred = np.mean(preds, axis=0)
            
            shuffled_mape = np.mean(np.abs((np.exp(y_joint) - np.exp(mean_pred)) / np.exp(y_joint))) * 100.0
            importances[col] = max(0.0, shuffled_mape - baseline_mape)
            
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
                
    if len(importances) > 0:
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        feats_names = [f[0].capitalize().replace('_', ' ') for f in sorted_feats]
        feats_vals = [f[1] * 100.0 for f in sorted_feats]
        
        fig_fi, ax_fi = plt.subplots(figsize=(10, 4))
        fig_fi.patch.set_facecolor('#0f1115')
        ax_fi.set_facecolor('#1a1e27')
        
        ax_fi.tick_params(colors='#94a3b8')
        ax_fi.xaxis.label.set_color('#ffffff')
        ax_fi.yaxis.label.set_color('#ffffff')
        
        y_pos = np.arange(len(feats_names))
        ax_fi.barh(y_pos, feats_vals, align='center', color='#6366f1')
        ax_fi.set_yticks(y_pos)
        ax_fi.set_yticklabels(feats_names)
        ax_fi.invert_yaxis()
        
        ax_fi.set_xlabel("Relative Importance (%)")
        ax_fi.set_title("ML Model Feature Attribution / Importance Plot", color='#ffffff', fontsize=12, pad=10)
        ax_fi.grid(True, color='#334155', linestyle='-', linewidth=0.5, axis='x')
        st.pyplot(fig_fi)
    else:
        st.info("Feature importance is not available for the current configuration.")

# ==============================================================================
# TAB 4: HISTORICAL FIT
# ==============================================================================
with tab_historical:
    st.subheader("📊 Historical Model Predictions vs. Actual Sales")
    
    # Calculate fit
    dates = pd.to_datetime(clean_sales['date'])
    actual_q = clean_sales['units_sold'].values
    
    if projection_mode == 'modular':
        fitted_log_q = analysis_summary['factor_results']['elasticity'].get('fitted_log_q')
        clean_sales_est = analysis_summary['factor_results']['elasticity'].get('clean_sales')
        
        fitted_q = pd.Series(np.nan, index=clean_sales.index)
        
        if fitted_log_q is not None and clean_sales_est is not None:
            fitted_q_est = np.exp(fitted_log_q)
            # Apply multipliers
            S_agent = analysis_summary['factor_results']['seasonality']
            if 'week_averages' in S_agent and 'status' in S_agent and 'Used' in S_agent['status']:
                weeks_est = pd.to_datetime(clean_sales_est['date']).dt.isocalendar().week.values
                fitted_q_est = fitted_q_est * S_agent['week_averages'].loc[weeks_est].values
            
            fitted_q.loc[clean_sales_est.index] = fitted_q_est
    else:
        fitted_q = pd.Series(np.exp(analysis_summary['fitted_joint_log_q']), index=clean_sales.index)
        
    fig_f, ax_f = plt.subplots(figsize=(12, 4))
    fig_f.patch.set_facecolor('#0f1115')
    ax_f.set_facecolor('#1a1e27')
    
    ax_f.tick_params(colors='#94a3b8')
    ax_f.xaxis.label.set_color('#ffffff')
    ax_f.yaxis.label.set_color('#ffffff')
    
    ax_f.plot(dates.values, actual_q, label="Actual Demand", color='#94a3b8', linewidth=2.0)
    ax_f.plot(dates.values, fitted_q.values, label=f"Predicted Demand ({model_mode})", color='#6366f1', linewidth=2.5, linestyle='--')
    
    ax_f.set_title("Timeline Fitting Audit: Actual vs. Fitted Quantity", color='#ffffff', fontsize=12, pad=10)
    ax_f.set_xlabel("Timeline")
    ax_f.set_ylabel("Quantity Sold")
    ax_f.grid(True, color='#334155', linestyle='-', linewidth=0.5)
    ax_f.legend(facecolor='#1a1e27', edgecolor='#334155', loc='upper right', labelcolor='#ffffff')
    st.pyplot(fig_f)
    
    # Calculate regression accuracy metrics
    clean_idx = clean_sales[clean_sales['exclude_from_regression'] == False].index
    y_true_ser = pd.Series(actual_q, index=clean_sales.index).loc[clean_idx]
    y_pred_ser = fitted_q.loc[clean_idx]
    
    combined = pd.DataFrame({'true': y_true_ser, 'pred': y_pred_ser}).dropna()
    y_true = combined['true'].values
    y_pred = combined['pred'].values
    
    if len(y_true) > 0:
        m_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        m_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    else:
        m_mape = 15.0
        m_r2 = 0.0
        rmse = 0.0
    
    c_met1, c_met2, c_met3 = st.columns(3)
    with c_met1:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #6366f1;">
                <div class="metric-label">Overall Fit MAPE</div>
                <div class="metric-value">{m_mape:.2f}%</div>
                <div style="color: #94a3b8;">Average prediction error rate</div>
            </div>
        """, unsafe_allow_html=True)
    with c_met2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #34d399;">
                <div class="metric-label">Model R² Coefficient</div>
                <div class="metric-value">{m_r2:.4f}</div>
                <div style="color: #94a3b8;">Explained variance proportion</div>
            </div>
        """, unsafe_allow_html=True)
    with c_met3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #fbbf24;">
                <div class="metric-label">Root Mean Squared Error (RMSE)</div>
                <div class="metric-value">{rmse:.1f}</div>
                <div style="color: #94a3b8;">Standard deviation of residuals</div>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: HARD STOPS & AUDITING
# ==============================================================================
with tab_audit:
    st.subheader("⚠️ Safety Controls & Hard Stop Auditor")
    st.markdown("The system automatically checks constraints. If critical safety parameters are violated, automated execution is blocked.")
    
    # Checklist overrides definitions (normally NO unless overridden in custom settings)
    h1_val = "NO"
    h2_val = "NO"
    h3_val = "NO"
    h4_val = "NO"
    u1_val = "NO"
    u2_val = "NO"
    
    override_states = []
    
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
        
    # Extrapolation warning
    max_hist_p = clean_sales['unit_price'].max()
    min_hist_p = clean_sales['unit_price'].min()
    if sim_proj['p_new'] > max_hist_p * 1.1 or sim_proj['p_new'] < min_hist_p * 0.9:
        hard_stops.append(f"WARNING: Simulated price (${sim_proj['p_new']:.2f}) is outside historical boundary range [${min_hist_p:.2f} - ${max_hist_p:.2f}]. ML models may suffer from extrapolation drift.")
        
    # Positive elasticity check
    if sim_proj['e_eff'] > 0.0:
        hard_stops.append(f"CRITICAL: Positive price elasticity ({sim_proj['e_eff']:.3f}) detected. Demand increases as price rises, violating classical demand law.")

    if len(hard_stops) > 0:
        st.markdown("<div class='hard-stop-card'><h3>⚠️ Hard Stop Active - Automated execution is BLOCKED</h3><ul>" + 
                    "".join([f"<li>{stop}</li>" for stop in hard_stops]) + 
                    "</ul></div>", unsafe_allow_html=True)
    else:
        st.success("✅ **CLEARED FOR AUTOMATED EXECUTION:** All safety constraint evaluations passed successfully.")

# ==============================================================================
# TAB 6: METHODOLOGY
# ==============================================================================
with tab_doc:
    st.subheader("📖 Machine Learning & Deep Learning Methodology")
    st.markdown("""
    ### 1. Elasticity via Numerical Gradients
    In standard econometrics, price elasticity ($e$) is treated as a constant parameter using linear log-log models:
    $$\log(Q) = \alpha + e \cdot \log(P) + \epsilon$$
    
    However, real consumer response is non-linear and includes threshold effects. Our machine learning framework maps demand as a non-linear network function:
    $$\log(Q) = f_{\text{MLP}}(\log(P))$$
    
    We evaluate the local elasticity dynamically at any target price point $P$ using central finite differences:
    $$e = \frac{f_{\text{MLP}}(\log(P) + h) - f_{\text{MLP}}(\log(P) - h)}{2h}$$
    where $h = 10^{-4}$.
    
    ---
    
    ### 2. Joint Neural Network Ensemble vs. Modular Agents
    - **Modular ML Pipeline**: Replaces statsmodels regressions with independent ML estimators (MLP for price-demand baseline; Random Forest/Gradient Boosting for residuals). It retains the additive and multiplicative separation rules of the demand framework.
    - **Joint Neural Network**: Trains an end-to-end Multi-Layer Perceptron (MLP) ensemble that takes all contextual features (price, competitor gap, promotional flags, inventory coverage, consumer sentiment) simultaneously. This is structurally superior because it automatically fits cross-variable interactions (e.g., promotional elasticities being flatter than non-promotional ones).
    
    ---
    
    ### 3. Uncertainty Estimation via Bootstrapping
    To generate confidence intervals for non-linear predictions, we employ **Deep Ensembles**:
    1. We construct $N=5$ (or $N=10$) replica neural networks.
    2. Each replica is trained on a resampled dataset drawn with replacement (Bootstrapping).
    3. The resampling probability is adjusted based on time weights ($0.95^{\Delta t}$) to emphasize recent market behavior.
    4. The standard deviation of the ensemble forecasts determines the demand confidence intervals (P10/P90).
    """)
