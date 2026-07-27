"""
End-to-End Neural Network & RL Dynamic Pricing Dashboard (Multi-Domain)
========================================================================
Supports BOTH Domains:
  1. 💻 Personal Laptops (Dell, HP, Lenovo, Asus)
  2. 🍚 Branded Basmati Rice (India Gate, Daawat, Fortune)

Key Features:
  - Multi-domain selector (Laptops & Rice)
  - Month + Year Target Date Selector (up to 1 year ahead)
  - Range [-50% to +50%] price optimization grid
  - Industry-standard microeconomic price elasticity curves
  - 5-Model comparison matrix with 3 comparative curve graphs
"""

import os
import sys
import pickle
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Set Page Configuration
st.set_page_config(
    page_title="End-to-End Neural & RL Pricing Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        color: #ffffff !important;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #f8fafc !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .metric-base {
        font-size: 0.88rem;
        color: #e2e8f0 !important;
        margin-top: 4px;
        font-weight: 600;
    }
    .metric-base b, .metric-base span {
        color: #ffffff !important;
    }
    .kpi-change-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 20px;
        border-top: 4px solid #38bdf8;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        color: #ffffff !important;
    }
    .kpi-change-title {
        font-size: 0.85rem;
        font-weight: 800;
        color: #f8fafc !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-change-val {
        font-size: 1.9rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.85rem;
        color: #e2e8f0 !important;
        margin-top: 4px;
        font-weight: 500;
    }
    .profit-section-card {
        background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .revenue-section-card {
        background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .section-title-profit {
        font-size: 1.1rem;
        font-weight: 800;
        color: #34d399 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .section-title-revenue {
        font-size: 1.1rem;
        font-weight: 800;
        color: #7dd3fc !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .hero-price {
        font-size: 2.4rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .fix-badge {
        display: inline-block;
        background: #10b981;
        color: #ffffff !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 8


# ---------------------------------------------------------
# SIDEBAR DOMAIN SELECTION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/brain-mind.png", width=70)
st.sidebar.title("End-to-End NN & RL Engine")
st.sidebar.markdown("---")

domain_option = st.sidebar.selectbox(
    "📦 Select Industry Domain",
    options=["💻 Personal Laptops (Dell, HP, Lenovo, Asus)", "🍚 Branded Basmati Rice (India Gate, Daawat, Fortune)"],
    index=0
)

if "Laptops" in domain_option:
    domain_key = "laptops"
    domain_title = "10-Year Personal Laptop Dataset"
    csv_file = "laptop_pricing_data.csv"
    results_file = "pipeline_results_laptops.csv"
    unit_label = "Units"
    price_unit = "₹"
else:
    domain_key = "rice"
    domain_title = "Branded Basmati Rice Dataset"
    csv_file = "branded_rice_data.csv"
    results_file = "pipeline_results_rice.csv"
    unit_label = "KG"
    price_unit = "₹/kg"

DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "mod_test", csv_file)
MODELS_DIR = os.path.join(BASE_DIR, "models", domain_key)
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, "models")

RESULTS_PATH = os.path.join(BASE_DIR, results_file)

@st.cache_data
def load_domain_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_domain_results(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_raw = load_domain_data(DATA_PATH)
df_results = load_domain_results(RESULTS_PATH)

# Load Scalers and Metadata
scalers_path = os.path.join(MODELS_DIR, "scalers.pkl")
with open(scalers_path, "rb") as f:
    scaler_dict = pickle.load(f)
scaler_X = scaler_dict["scaler_X"]
scaler_y = scaler_dict["scaler_y"]
FEATURE_COLS = scaler_dict["features"]
BRANDS_LIST = scaler_dict.get("brands", list(df_raw['brand'].unique()))

# Load Elasticities
elasticities_path = os.path.join(MODELS_DIR, "elasticities.json")
if os.path.exists(elasticities_path):
    with open(elasticities_path, "r") as f:
        ELASTICITIES = json.load(f)
else:
    ELASTICITIES = {}

def feature_engineering(df):
    df_out = df.copy().sort_values(['brand', 'date']).reset_index(drop=True)

    df_out['year'] = df_out['date'].dt.year
    df_out['month'] = df_out['date'].dt.month
    df_out['week'] = df_out['date'].dt.isocalendar().week.astype(int)
    df_out['month_sin'] = np.sin(2 * np.pi * df_out['month'] / 12.0)
    df_out['month_cos'] = np.cos(2 * np.pi * df_out['month'] / 12.0)
    df_out['week_sin'] = np.sin(2 * np.pi * df_out['week'] / 52.0)
    df_out['week_cos'] = np.cos(2 * np.pi * df_out['week'] / 52.0)

    has_comp3 = 'comp_price_3' in df_out.columns
    df_out['price_ratio_comp1'] = df_out['unit_price'] / (df_out['comp_price_1'] + 1e-5)
    df_out['price_ratio_comp2'] = df_out['unit_price'] / (df_out['comp_price_2'] + 1e-5)
    if has_comp3:
        df_out['price_ratio_comp3'] = df_out['unit_price'] / (df_out['comp_price_3'] + 1e-5)
        df_out['cross_price_diff'] = df_out['unit_price'] - (df_out['comp_price_1'] + df_out['comp_price_2'] + df_out['comp_price_3']) / 3.0
    else:
        df_out['cross_price_diff'] = df_out['unit_price'] - (df_out['comp_price_1'] + df_out['comp_price_2']) / 2.0

    df_out['margin_ratio'] = (df_out['unit_price'] - df_out['cost_per_unit']) / (df_out['unit_price'] + 1e-5)

    brands_list = []
    for brand, grp in df_out.groupby('brand'):
        grp = grp.sort_values('date').copy()
        grp['qty_lag_1'] = grp['units_sold'].shift(1)
        grp['qty_lag_4'] = grp['units_sold'].shift(4)
        grp['qty_lag_13'] = grp['units_sold'].shift(13)
        grp['qty_roll_mean_4'] = grp['units_sold'].shift(1).rolling(window=4, min_periods=1).mean()
        grp['qty_roll_std_4'] = grp['units_sold'].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
        grp['qty_roll_mean_13'] = grp['units_sold'].shift(1).rolling(window=13, min_periods=1).mean()
        grp['price_velocity'] = grp['unit_price'].pct_change().fillna(0)
        max_price_expanding = grp['unit_price'].expanding().max()
        grp['discount_depth'] = (max_price_expanding - grp['unit_price']) / (max_price_expanding + 1e-5)
        grp['price_lag_1'] = grp['unit_price'].shift(1)
        brands_list.append(grp)

    df_out = pd.concat(brands_list, ignore_index=True)
    df_out['log_units_sold'] = np.log1p(df_out['units_sold'])

    brand_dummies = pd.get_dummies(df_out['brand'], prefix='brand', drop_first=False).astype(float)
    df_out = pd.concat([df_out, brand_dummies], axis=1)

    df_out = df_out.dropna(subset=['qty_lag_13']).reset_index(drop=True)
    return df_out

df_engineered = feature_engineering(df_raw)

# Model Classes
class DeepMLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.block1 = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
        )
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        h = torch.relu(self.input_proj(x))
        h = h + self.block1(h)
        h = self.block2(h)
        return self.head(h)

class LSTMAttentionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        return self.fc(attn_out[:, -1, :])

class DQNPricingNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        return self.net(state)

@st.cache_resource
def load_domain_models(m_dir, n_feat):
    mlp = DeepMLPRegressor(n_feat)
    mlp.load_state_dict(torch.load(os.path.join(m_dir, "pytorch_mlp.pt"), map_location='cpu'))
    mlp.eval()

    lstm = LSTMAttentionRegressor(n_feat)
    lstm.load_state_dict(torch.load(os.path.join(m_dir, "lstm_attention.pt"), map_location='cpu'))
    lstm.eval()

    lgb_m = lgb.Booster(model_file=os.path.join(m_dir, "lgb_regressor.txt"))

    with open(os.path.join(m_dir, "rl_meta.json"), "r") as f:
        rl_meta = json.load(f)
    dqn = DQNPricingNetwork(rl_meta["state_dim"], rl_meta["num_actions"])
    dqn.load_state_dict(torch.load(os.path.join(m_dir, "dqn_agent.pt"), map_location='cpu'))
    dqn.eval()
    action_grid = np.array(rl_meta["action_grid"])

    with open(os.path.join(m_dir, "meta_learner.pkl"), "rb") as f:
        meta_lr = pickle.load(f)

    return mlp, lstm, lgb_m, dqn, action_grid, meta_lr

mlp_model, lstm_model, lgb_model, dqn_model, ACTION_GRID, meta_learner = load_domain_models(MODELS_DIR, len(FEATURE_COLS))

def apply_plotly_light_theme(fig, title_text, x_title, y_title, height=420):
    fig.update_layout(
        title=dict(text=title_text, font=dict(color='#ffffff', size=16, family='sans-serif')),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color='#f8fafc', size=13)),
            tickfont=dict(color='#f8fafc', size=11),
            gridcolor='#334155'
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color='#f8fafc', size=13)),
            tickfont=dict(color='#f8fafc', size=11),
            gridcolor='#334155'
        ),
        legend=dict(
            font=dict(color='#ffffff', size=11),
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='#475569'
        ),
        font=dict(color='#ffffff', family='sans-serif'),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        margin=dict(l=40, r=40, t=50, b=40),
        height=height
    )
    return fig

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
selected_brand = st.sidebar.selectbox(
    "1. Select Target Brand / SKU",
    options=BRANDS_LIST,
    index=0
)

# Month & Year selector up to 1 year (12 months ahead from dataset max date)
max_dt = df_raw['date'].max()
future_dates = [max_dt + pd.DateOffset(months=i) for i in range(1, 13)]
future_date_labels = [d.strftime("%B %Y") for d in future_dates]

selected_future_label = st.sidebar.selectbox(
    "2. Select Target Future Month & Year (Up to 1 Year Ahead)",
    options=future_date_labels,
    index=1
)

selected_idx = future_date_labels.index(selected_future_label)
target_dt = future_dates[selected_idx]
target_month_num = target_dt.month
target_year_num = target_dt.year

model_options = [
    "🧠 Option 1: PyTorch Deep MLP (Residual)",
    "🌊 Option 2: LSTM-Attention (8-Step Windows)",
    "🌲 Option 3: LightGBM Gradient Boosted Ensemble",
    "🤖 Option 4: Deep Q-Network (DQN) RL Agent",
    "⚡ Option 5: Neuro-Boost Learned Stacking Hybrid"
]

selected_model_str = st.sidebar.radio(
    "3. Select End-to-End Model Architecture",
    options=model_options,
    index=4
)

# Parse selected model clean name
if "Option 1" in selected_model_str:
    clean_model_name = "PyTorch Deep Neural Network (MLP)"
elif "Option 2" in selected_model_str:
    clean_model_name = "Temporal LSTM-Attention Network"
elif "Option 3" in selected_model_str:
    clean_model_name = "LightGBM Gradient Boosted Ensemble"
elif "Option 4" in selected_model_str:
    clean_model_name = "Deep Q-Network (DQN) RL Agent"
else:
    clean_model_name = "Neuro-Boost Learned Stacking Hybrid"

# Retrieve learned elasticity for active brand & model
active_elasticity = ELASTICITIES.get(selected_brand, {}).get(clean_model_name, -1.45)

# ---------------------------------------------------------
# END-TO-END INFERENCE FUNCTION
# ---------------------------------------------------------

def build_feature_vector(brand, target_month, target_year, price_val, base_row):
    month_sin = np.sin(2 * np.pi * target_month / 12.0)
    month_cos = np.cos(2 * np.pi * target_month / 12.0)
    week_num = int(target_month * 4.33)
    week_sin = np.sin(2 * np.pi * week_num / 52.0)
    week_cos = np.cos(2 * np.pi * week_num / 52.0)

    is_fest = 1 if target_month in [8, 9, 10, 11] else 0
    is_pr = base_row.get('is_promo', 0)
    cogs = base_row['cost_per_unit']

    cp1 = base_row.get('comp_price_1', price_val * 0.95)
    cp2 = base_row.get('comp_price_2', price_val * 0.90)
    cp3 = base_row.get('comp_price_3', price_val * 0.85)

    p_ratio1 = price_val / (cp1 + 1e-5)
    p_ratio2 = price_val / (cp2 + 1e-5)
    p_ratio3 = price_val / (cp3 + 1e-5)
    m_ratio = (price_val - cogs) / (price_val + 1e-5)
    cross_diff = price_val - (cp1 + cp2 + cp3) / 3.0

    brand_eng = df_engineered[df_engineered['brand'] == brand].sort_values('date')
    if len(brand_eng) > 0:
        last_eng = brand_eng.iloc[-1]
        qty_lag_1 = last_eng.get('qty_lag_1', base_row['units_sold'])
        qty_lag_4 = last_eng.get('qty_lag_4', base_row['units_sold'])
        qty_lag_13 = last_eng.get('qty_lag_13', base_row['units_sold'])
        qty_roll_mean_4 = last_eng.get('qty_roll_mean_4', base_row['units_sold'])
        qty_roll_std_4 = last_eng.get('qty_roll_std_4', 0)
        qty_roll_mean_13 = last_eng.get('qty_roll_mean_13', base_row['units_sold'])
        price_velocity = last_eng.get('price_velocity', 0)
        discount_depth = last_eng.get('discount_depth', 0)
        price_lag_1 = last_eng.get('price_lag_1', base_row['unit_price'])
    else:
        qty_lag_1 = base_row['units_sold']
        qty_lag_4 = base_row['units_sold']
        qty_lag_13 = base_row['units_sold']
        qty_roll_mean_4 = base_row['units_sold']
        qty_roll_std_4 = 0
        qty_roll_mean_13 = base_row['units_sold']
        price_velocity = 0
        discount_depth = 0
        price_lag_1 = base_row['unit_price']

    feat_dict = {
        'unit_price': price_val,
        'cost_per_unit': cogs,
        'comp_price_1': cp1,
        'comp_price_2': cp2,
        'comp_price_3': cp3,
        'is_promo': is_pr,
        'is_festival': is_fest,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'week_sin': week_sin,
        'week_cos': week_cos,
        'price_ratio_comp1': p_ratio1,
        'price_ratio_comp2': p_ratio2,
        'price_ratio_comp3': p_ratio3,
        'margin_ratio': m_ratio,
        'cross_price_diff': cross_diff,
        'price_velocity': price_velocity,
        'discount_depth': discount_depth,
        'qty_lag_1': qty_lag_1,
        'qty_lag_4': qty_lag_4,
        'qty_lag_13': qty_lag_13,
        'qty_roll_mean_4': qty_roll_mean_4,
        'qty_roll_std_4': qty_roll_std_4,
        'qty_roll_mean_13': qty_roll_mean_13,
        'price_lag_1': price_lag_1,
    }

    for b in BRANDS_LIST:
        feat_dict[f'brand_{b}'] = 1.0 if brand == b else 0.0

    vec = [feat_dict[col] for col in FEATURE_COLS if col in feat_dict]
    return np.array([vec])


def predict_mlp(raw_vec):
    vec_scaled = scaler_X.transform(raw_vec)
    with torch.no_grad():
        pred_s = mlp_model(torch.tensor(vec_scaled, dtype=torch.float32)).item()
    return max(100.0, scaler_y.inverse_transform([[pred_s]])[0][0])


def predict_lstm(brand, raw_vec):
    brand_eng = df_engineered[df_engineered['brand'] == brand].sort_values('date')
    if len(brand_eng) >= SEQ_LEN:
        hist_features = brand_eng[FEATURE_COLS].values[-(SEQ_LEN - 1):]
        hist_scaled = scaler_X.transform(hist_features)
        curr_scaled = scaler_X.transform(raw_vec)
        window = np.vstack([hist_scaled, curr_scaled])
        with torch.no_grad():
            pred_s = lstm_model(torch.tensor(window, dtype=torch.float32).unsqueeze(0)).item()
        return max(100.0, scaler_y.inverse_transform([[pred_s]])[0][0])
    else:
        return predict_mlp(raw_vec)


def predict_lgb(raw_vec):
    return max(100.0, lgb_model.predict(raw_vec)[0])


def predict_dqn(brand, raw_vec, base_price, cost):
    vec_scaled = scaler_X.transform(raw_vec)
    state = vec_scaled[0]
    with torch.no_grad():
        q_vals = dqn_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        best_action = q_vals.argmax(dim=1).item()
    price_delta = ACTION_GRID[best_action]
    rl_price = base_price * (1.0 + price_delta)
    rl_vec = build_feature_vector(brand, target_month_num, target_year_num, rl_price,
                                   df_raw[df_raw['brand'] == brand].sort_values('date').iloc[-1])
    return predict_mlp(rl_vec)


def predict_hybrid(brand, raw_vec):
    p_mlp = predict_mlp(raw_vec)
    p_lgb = predict_lgb(raw_vec)
    p_lstm = predict_lstm(brand, raw_vec)
    meta_feats = np.array([[p_mlp, p_lgb, p_lstm]])
    return max(100.0, meta_learner.predict(meta_feats)[0])


def predict_end_to_end(brand, target_month, target_year, price_val, base_row):
    raw_vec = build_feature_vector(brand, target_month, target_year, price_val, base_row)

    if "Option 1" in selected_model_str:
        return predict_mlp(raw_vec)
    elif "Option 2" in selected_model_str:
        return predict_lstm(brand, raw_vec)
    elif "Option 3" in selected_model_str:
        return predict_lgb(raw_vec)
    elif "Option 4" in selected_model_str:
        return predict_dqn(brand, raw_vec, price_val, base_row['cost_per_unit'])
    else:
        return predict_hybrid(brand, raw_vec)

# Baseline Metrics
brand_df = df_raw[df_raw['brand'] == selected_brand].sort_values('date').reset_index(drop=True)
last_row = brand_df.iloc[-1]
base_price = last_row['unit_price']
base_cost = last_row['cost_per_unit']
latest_date_str = last_row['date'].strftime('%b %d, %Y')

# Model baseline demand prediction at current base price
base_qty = predict_end_to_end(selected_brand, target_month_num, target_year_num, base_price, last_row)
base_rev = base_price * base_qty
base_profit = (base_price - base_cost) * base_qty

# ---------------------------------------------------------
# INDUSTRY-STANDARD MICROECONOMIC ELASTICITY GRID [-50% TO +50%]
# ---------------------------------------------------------
p_grid = np.linspace(-50, 50, 401)
prices_grid = base_price * (1.0 + p_grid / 100.0)

# Microeconomic Log-Log Demand Elasticity Curve: Q(P) = Q_base * (P / P_base) ^ elas
def compute_elasticity_demand_grid(q_base, p_ratio_grid, elasticity_val):
    return q_base * (p_ratio_grid ** elasticity_val)

p_ratio_grid = 1.0 + p_grid / 100.0
qtys_grid = compute_elasticity_demand_grid(base_qty, p_ratio_grid, active_elasticity)
revs_grid = prices_grid * qtys_grid
profits_grid = (prices_grid - base_cost) * qtys_grid

# Profit Peak
opt_prof_idx = np.argmax(profits_grid)
opt_prof_price_pct = p_grid[opt_prof_idx]
opt_prof_price_val = prices_grid[opt_prof_idx]
opt_prof_qty_val = qtys_grid[opt_prof_idx]
opt_prof_rev_val = revs_grid[opt_prof_idx]
opt_prof_val = profits_grid[opt_prof_idx]

# Revenue Peak
opt_rev_idx = np.argmax(revs_grid)
opt_rev_price_pct = p_grid[opt_rev_idx]
opt_rev_price_val = prices_grid[opt_rev_idx]
opt_rev_qty_val = qtys_grid[opt_rev_idx]
opt_rev_val = revs_grid[opt_rev_idx]
opt_rev_prof_val = profits_grid[opt_rev_idx]

# Header Title
st.title(f"🧠 End-to-End Neural Network & RL Dynamic Pricing Engine")
st.markdown(f"**Domain**: `{domain_title}` | **Target SKU**: `{selected_brand}` | **Target Date**: `{selected_future_label}` | **Active Model**: `{selected_model_str}` | **Base Price**: **{price_unit} {base_price:,.2f}** | **Learned Elasticity (ε)**: `{active_elasticity:.3f}`")

# Top Banner: Model Cards
st.subheader("🎯 5 Research-Backed End-to-End Models Comparison")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("<div class='metric-card'><b>🧠 PyTorch MLP</b><span class='fix-badge'>RESIDUAL</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Residual + LayerNorm + GELU</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='metric-card'><b>🌊 LSTM-Attention</b><span class='fix-badge'>8-STEP</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Bi-LSTM + Self-Attention</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='metric-card'><b>🌲 LightGBM</b><span class='fix-badge'>TUNED</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Early Stopping + Regularized</span></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='metric-card'><b>🤖 DQN Agent</b><span class='fix-badge'>REAL RL</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Deep Q-Network + Replay</span></div>", unsafe_allow_html=True)
with c5:
    st.markdown("<div class='metric-card' style='border-left-color:#fbbf24;'><b>⚡ Neuro-Boost</b><span class='fix-badge'>STACKED</span><br><span style='font-size:0.8rem; color:#fbbf24;'>Learned Ridge Stacking</span></div>", unsafe_allow_html=True)

# Tabs Wiring
tab1, tab_delta, tab2, tab3, tab_eda = st.tabs([
    "🏠 Home: Profit & Revenue Optimization",
    "📊 Selection Change & Delta Analysis",
    "⚔️ 5-Model Industry Benchmark Matrix",
    "🔬 Deep Research & Architecture Diagnostics",
    "📈 EDA & Historical Dataset"
])

# ---------------------------------------------------------
# TAB 1: HOME PAGE
# ---------------------------------------------------------
with tab1:
    st.markdown(f"## 💰 SECTION 1: GROSS PROFIT MAXIMIZATION ({selected_future_label.upper()})")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f"""
        <div class="profit-section-card">
            <div class="section-title-profit">PROFIT-MAXIMIZING PRICE</div>
            <div class="hero-price">{price_unit} {opt_prof_price_val:,.2f}</div>
            <div class="metric-base">Base Price: <b>{price_unit} {base_price:,.2f}</b></div>
            <div style="color:#34d399; font-size:0.9rem; font-weight:bold; margin-top:4px;">Shift: {opt_prof_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_prof_val/1e7:.2f} Cr</div>
            <div class="metric-base">Baseline: <b>₹{base_profit/1e7:.2f} Cr</b></div>
            <div style="color:#34d399; font-size:0.85rem; font-weight:bold; margin-top:4px;">Profit Lift: {((opt_prof_val - base_profit)/(base_profit + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_prof_rev_val/1e7:.2f} Cr</div>
            <div class="metric-base">Baseline: <b>₹{base_rev/1e7:.2f} Cr</b></div>
            <div style="color:#38bdf8; font-size:0.85rem; font-weight:bold; margin-top:4px;">Revenue Shift: {((opt_prof_rev_val - base_rev)/(base_rev + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume (Q_new)</div>
            <div class="metric-value">{opt_prof_qty_val/1e3:.1f}k {unit_label}</div>
            <div class="metric-base">Baseline: <b>{base_qty/1e3:.1f}k {unit_label}</b></div>
            <div style="color:#f1f5f9; font-size:0.85rem; font-weight:bold; margin-top:4px;">Volume Shift: {((opt_prof_qty_val - base_qty)/(base_qty + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(x=p_grid, y=profits_grid/1e7, mode='lines', name='End-to-End Profit Curve', line=dict(color='#10b981', width=3)))
    fig_prof.add_trace(go.Scatter(x=[opt_prof_price_pct], y=[opt_prof_val/1e7], mode='markers+text', name=f'Profit Peak ({price_unit} {opt_prof_price_val:,.2f})', marker=dict(color='#fbbf24', size=14, symbol='star'), text=[f"Max Profit: ₹{opt_prof_val/1e7:.2f} Cr"], textposition="top center"))
    apply_plotly_light_theme(fig_prof, f"End-to-End Gross Profit Curve ({selected_brand} - {selected_future_label})", "Price Shift (%) [-50% to +50%]", "Gross Profit (₹ Crore)")
    st.plotly_chart(fig_prof, width='stretch')

    st.markdown("---")
    st.markdown(f"## 📈 SECTION 2: REVENUE MAXIMIZATION ({selected_future_label.upper()})")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(f"""
        <div class="revenue-section-card">
            <div class="section-title-revenue">REVENUE-MAXIMIZING PRICE</div>
            <div class="hero-price">{price_unit} {opt_rev_price_val:,.2f}</div>
            <div class="metric-base">Base Price: <b>{price_unit} {base_price:,.2f}</b></div>
            <div style="color:#7dd3fc; font-size:0.9rem; font-weight:bold; margin-top:4px;">Shift: {opt_rev_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_rev_val/1e7:.2f} Cr</div>
            <div class="metric-base">Baseline: <b>₹{base_rev/1e7:.2f} Cr</b></div>
            <div style="color:#38bdf8; font-size:0.85rem; font-weight:bold; margin-top:4px;">Revenue Lift: {((opt_rev_val - base_rev)/(base_rev + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_rev_prof_val/1e7:.2f} Cr</div>
            <div class="metric-base">Baseline: <b>₹{base_profit/1e7:.2f} Cr</b></div>
            <div style="color:#34d399; font-size:0.85rem; font-weight:bold; margin-top:4px;">Profit Shift: {((opt_rev_prof_val - base_profit)/(base_profit + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume (Q_new)</div>
            <div class="metric-value">{opt_rev_qty_val/1e3:.1f}k {unit_label}</div>
            <div class="metric-base">Baseline: <b>{base_qty/1e3:.1f}k {unit_label}</b></div>
            <div style="color:#f1f5f9; font-size:0.85rem; font-weight:bold; margin-top:4px;">Volume Shift: {((opt_rev_qty_val - base_qty)/(base_qty + 1e-5))*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=p_grid, y=revs_grid/1e7, mode='lines', name='End-to-End Revenue Curve', line=dict(color='#38bdf8', width=3)))
    fig_rev.add_trace(go.Scatter(x=[opt_rev_price_pct], y=[opt_rev_val/1e7], mode='markers+text', name=f'Revenue Peak ({price_unit} {opt_rev_price_val:,.2f})', marker=dict(color='#7dd3fc', size=14, symbol='diamond'), text=[f"Max Revenue: ₹{opt_rev_val/1e7:.2f} Cr"], textposition="top center"))
    apply_plotly_light_theme(fig_rev, f"End-to-End Revenue Curve ({selected_brand} - {selected_future_label})", "Price Shift (%) [-50% to +50%]", "Weekly Revenue (₹ Crore)")
    st.plotly_chart(fig_rev, width='stretch')

# ---------------------------------------------------------
# TAB 2: SELECTION CHANGE & DELTA ANALYSIS (-50% TO +50%)
# ---------------------------------------------------------
with tab_delta:
    st.header(f"📊 Delta Impact Analysis for Future Target: {selected_future_label}")

    tab_delta_date = st.selectbox(
        "Select Target Future Month & Year (Upto 1 Year Ahead)",
        options=future_date_labels,
        index=selected_idx,
        key="delta_tab_date_selector"
    )

    custom_price_change = st.slider(
        f"💡 Adjust Price Shift for {selected_brand} in {tab_delta_date} (%)",
        min_value=-50.0, max_value=+50.0, value=0.0, step=0.5,
        key=f"slider_{selected_brand}_{tab_delta_date}"
    )

    cust_price_val = base_price * (1.0 + custom_price_change / 100.0)
    cust_p_ratio = 1.0 + custom_price_change / 100.0
    cust_qty_val = base_qty * (cust_p_ratio ** active_elasticity)
    cust_rev_val = cust_price_val * cust_qty_val
    cust_profit_val = (cust_price_val - base_cost) * cust_qty_val

    pct_change_price = custom_price_change
    pct_change_qty = ((cust_qty_val - base_qty) / (base_qty + 1e-5)) * 100.0
    pct_change_rev = ((cust_rev_val - base_rev) / (base_rev + 1e-5)) * 100.0
    pct_change_profit = ((cust_profit_val - base_profit) / (base_profit + 1e-5)) * 100.0

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>1. RETAIL PRICE</div><div class='kpi-change-val'>{price_unit} {cust_price_val:,.2f}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{price_unit} {base_price:,.2f}</b></div><div class='kpi-subtext'><b style='color:#38bdf8;'>Shift: {pct_change_price:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>2. DEMAND QUANTITY (Q_new)</div><div class='kpi-change-val'>{cust_qty_val/1e3:.1f}k {unit_label}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{base_qty/1e3:.1f}k</b></div><div class='kpi-subtext'><b style='color:#34d399;'>Change: {pct_change_qty:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with col_d3:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>3. WEEKLY REVENUE</div><div class='kpi-change-val'>₹{cust_rev_val/1e7:.2f} Cr</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>₹{base_rev/1e7:.2f} Cr</b></div><div class='kpi-subtext'><b style='color:#7dd3fc;'>Change: {pct_change_rev:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with col_d4:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>4. GROSS PROFIT</div><div class='kpi-change-val'>₹{cust_profit_val/1e7:.2f} Cr</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>₹{base_profit/1e7:.2f} Cr</b></div><div class='kpi-subtext'><b style='color:#34d399;'>Change: {pct_change_profit:+.1f}%</b></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 3 Impact Graphs (Stacked Vertically)")

    fig_g1 = go.Figure()
    fig_g1.add_trace(go.Scatter(x=p_grid, y=profits_grid/1e7, mode='lines', name='Profit Curve', line=dict(color='#10b981', width=3)))
    fig_g1.add_trace(go.Scatter(x=[custom_price_change], y=[cust_profit_val/1e7], mode='markers+text', name='Selected Shift', marker=dict(color='#fbbf24', size=14, symbol='star'), text=[f"{pct_change_profit:+.1f}%"], textposition="top center"))
    apply_plotly_light_theme(fig_g1, f"1. Gross Profit Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", "Profit (₹ Cr)")
    st.plotly_chart(fig_g1, width='stretch')

    fig_g2 = go.Figure()
    fig_g2.add_trace(go.Scatter(x=p_grid, y=revs_grid/1e7, mode='lines', name='Revenue Curve', line=dict(color='#38bdf8', width=3)))
    fig_g2.add_trace(go.Scatter(x=[custom_price_change], y=[cust_rev_val/1e7], mode='markers+text', name='Selected Shift', marker=dict(color='#7dd3fc', size=14, symbol='diamond'), text=[f"{pct_change_rev:+.1f}%"], textposition="top center"))
    apply_plotly_light_theme(fig_g2, f"2. Revenue Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", "Revenue (₹ Cr)")
    st.plotly_chart(fig_g2, width='stretch')

    fig_g3 = go.Figure()
    fig_g3.add_trace(go.Scatter(x=p_grid, y=qtys_grid/1e3, mode='lines', name='Demand Curve (Q_new)', line=dict(color='#a855f7', width=3)))
    fig_g3.add_trace(go.Scatter(x=[custom_price_change], y=[cust_qty_val/1e3], mode='markers+text', name='Selected Shift', marker=dict(color='#c084fc', size=14, symbol='square'), text=[f"{pct_change_qty:+.1f}%"], textposition="top center"))
    apply_plotly_light_theme(fig_g3, f"3. Demand Volume (Q_new) Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", f"Demand Volume Q_new (k {unit_label})")
    st.plotly_chart(fig_g3, width='stretch')

# ---------------------------------------------------------
# TAB 3: 5-MODEL BENCHMARK MATRIX & COMPARATIVE CURVES
# ---------------------------------------------------------
with tab2:
    st.header(f"⚔️ 5 End-to-End Model Comparison Matrix ({selected_brand})")

    if df_results is not None:
        brand_res = df_results[df_results['brand'] == selected_brand]
        st.dataframe(brand_res, width='stretch')

        st.markdown("---")
        st.subheader("📊 Metric Comparison Charts")

        fig_wmape = go.Figure()
        fig_wmape.add_trace(go.Bar(
            x=brand_res['model'], y=brand_res['WMAPE'],
            marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
            text=brand_res['WMAPE'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto', textfont=dict(color='white')
        ))
        apply_plotly_light_theme(fig_wmape, f"WMAPE Comparison (Lower is Better) — {selected_brand}", "Model", "WMAPE (%)", height=380)
        st.plotly_chart(fig_wmape, width='stretch')

        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=brand_res['model'], y=brand_res['R2'],
            marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
            text=brand_res['R2'].apply(lambda x: f"{x:.4f}"),
            textposition='auto', textfont=dict(color='white')
        ))
        apply_plotly_light_theme(fig_r2, f"R² Comparison (Higher is Better) — {selected_brand}", "Model", "R²", height=380)
        st.plotly_chart(fig_r2, width='stretch')

        if 'mean_elasticity' in brand_res.columns:
            fig_elas = go.Figure()
            fig_elas.add_trace(go.Bar(
                x=brand_res['model'], y=brand_res['mean_elasticity'],
                marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
                text=brand_res['mean_elasticity'].apply(lambda x: f"{x:.3f}"),
                textposition='auto', textfont=dict(color='white')
            ))
            apply_plotly_light_theme(fig_elas, f"Price Elasticity of Demand — {selected_brand}", "Model", "Elasticity (ε)", height=380)
            st.plotly_chart(fig_elas, width='stretch')

        st.markdown("---")
        st.subheader(f"📈 5-Model Comparative Curves ({selected_brand} - {selected_future_label})")

        # Compute curves for all 5 models using their learned elasticities
        model_names = [
            ("PyTorch Deep MLP", "#38bdf8"),
            ("Temporal LSTM-Attention Network", "#a855f7"),
            ("LightGBM Gradient Boosted Ensemble", "#10b981"),
            ("Deep Q-Network (DQN) RL Agent", "#f59e0b"),
            ("Neuro-Boost Learned Stacking Hybrid", "#ef4444")
        ]

        models_comp = []
        for m_title, color in model_names:
            e_val = ELASTICITIES.get(selected_brand, {}).get(m_title, -1.45)
            q_m = base_qty * (p_ratio_grid ** e_val)
            models_comp.append((m_title, q_m, color))

        fig_comp_profit = go.Figure()
        for name, q_grid_m, color in models_comp:
            prof_grid_m = (prices_grid - base_cost) * q_grid_m
            fig_comp_profit.add_trace(go.Scatter(
                x=p_grid, y=prof_grid_m / 1e7,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        apply_plotly_light_theme(fig_comp_profit, f"1. Gross Profit vs Price Shift Comparison ({selected_brand} - {selected_future_label})", "Price Shift (%) [-50% to +50%]", "Gross Profit (₹ Crore)")
        st.plotly_chart(fig_comp_profit, width='stretch')

        fig_comp_rev = go.Figure()
        for name, q_grid_m, color in models_comp:
            rev_grid_m = prices_grid * q_grid_m
            fig_comp_rev.add_trace(go.Scatter(
                x=p_grid, y=rev_grid_m / 1e7,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        apply_plotly_light_theme(fig_comp_rev, f"2. Weekly Revenue vs Price Shift Comparison ({selected_brand} - {selected_future_label})", "Price Shift (%) [-50% to +50%]", "Weekly Revenue (₹ Crore)")
        st.plotly_chart(fig_comp_rev, width='stretch')

        fig_comp_qty = go.Figure()
        for name, q_grid_m, color in models_comp:
            fig_comp_qty.add_trace(go.Scatter(
                x=p_grid, y=q_grid_m / 1e3,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        apply_plotly_light_theme(fig_comp_qty, f"3. Demand Volume (Q_new) vs Price Shift Comparison ({selected_brand} - {selected_future_label})", "Price Shift (%) [-50% to +50%]", f"Demand Volume Q_new (k {unit_label})")
        st.plotly_chart(fig_comp_qty, width='stretch')
    else:
        st.info("Run `python nn_rl_pricing_engine/train_models.py` to populate pipeline results.")

# ---------------------------------------------------------
# TAB EDA: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
with tab_eda:
    st.header(f"📈 Exploratory Data Analysis: {domain_title}")
    st.markdown("Comprehensive statistical and graphical exploration of prices, volume, seasonality, and price elasticity.")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader("1. Historical Price Trends")
        fig_p = px.line(
            df_raw, x="date", y="unit_price", color="brand",
            title=f"Unit Price Trends ({domain_title})",
            labels={"unit_price": f"Price ({price_unit})", "date": "Date"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_p, width='stretch')
        
    with col_eda2:
        st.subheader("2. Weekly Demand Volume Trends")
        fig_v = px.line(
            df_raw, x="date", y="units_sold", color="brand",
            title=f"Weekly Units Sold ({domain_title})",
            labels={"units_sold": "Units Sold", "date": "Date"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_v, width='stretch')

    col_eda3, col_eda4 = st.columns(2)
    
    with col_eda3:
        st.subheader("3. Monthly Demand Seasonality")
        df_raw['month'] = pd.to_datetime(df_raw['date']).dt.month
        monthly_avg = df_raw.groupby(['month', 'brand'])['units_sold'].mean().reset_index()
        fig_m = px.bar(
            monthly_avg, x="month", y="units_sold", color="brand", barmode="group",
            title="Average Monthly Volume Demand",
            labels={"month": "Month of Year (1-12)", "units_sold": "Avg Weekly Units"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_m, width='stretch')
        
    with col_eda4:
        st.subheader("4. Feature Correlation Matrix")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        corr = df_raw[num_cols].corr()
        fig_c = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Dataset Feature Correlation Heatmap",
            color_continuous_scale="Blues", template="plotly_dark"
        )
        st.plotly_chart(fig_c, width='stretch')

    st.markdown("---")
    st.subheader(f"📋 Historical Raw Dataset Rows for {selected_brand}")
    brand_df_diag = df_raw[df_raw['brand'] == selected_brand].sort_values('date', ascending=False)
    st.dataframe(brand_df_diag, width='stretch')

# ---------------------------------------------------------
# TAB 4: DEEP RESEARCH DIAGNOSTICS
# ---------------------------------------------------------
with tab3:
    st.header("🔬 Deep Neural & Reinforcement Learning Architecture Diagnostics")

    st.markdown(f"""
    ### 📦 Industry Domain: {domain_title}
    - **Active Dataset**: `{csv_file}`
    - **Brands Covered**: {", ".join(BRANDS_LIST)}
    - **Features Count**: {len(FEATURE_COLS)} engineered features
    
    ### 🧠 Model 1: PyTorch Deep MLP (Residual + LayerNorm + GELU)
    - **Architecture**: `Input → Linear(128) → [Residual Block: Linear(128) → LayerNorm → GELU → Dropout(0.15)] → Linear(64) → LayerNorm → GELU → Linear(32) → GELU → Linear(1)`

    ### 🌊 Model 2: LSTM-Attention (8-Step Sliding Windows)
    - **Architecture**: `Input(8, features) → BiLSTM(64, 2 layers) → MultiHead Self-Attention(4 heads) → LayerNorm + Residual → FC(64→32→1)`

    ### 🌲 Model 3: LightGBM Gradient-Boosted Ensemble
    - **Architecture**: 500 boosting rounds, 63 leaves, max_depth=8, with L1/L2 regularization

    ### 🤖 Model 4: Deep Q-Network (DQN) RL Agent
    - **Architecture**: State → Linear(128) → LayerNorm → ReLU → Linear(64) → LayerNorm → ReLU → Q-values(21 actions)

    ### ⚡ Model 5: Neuro-Boost Learned Stacking Hybrid
    - **Architecture**: Ridge regression meta-learner combining out-of-sample predictions from MLP + LGB + LSTM
    """)

st.markdown("---")
st.caption("Multi-Domain Neural & RL Dynamic Pricing Engine v4.0 | Research-Backed | Elasticity Optimized")
