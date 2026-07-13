import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure imports resolve from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.insert(0, src_dir)

from agents.coordinator_agent import CoordinatorAgent
from ml_agents.coordinator_agent import MLCoordinatorAgent

def ensure_dirs():
    output_dir = os.path.join(os.path.dirname(current_dir), "compare_output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def scan_datasets():
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    mock_dir = os.path.join(base_dir, "MockData")
    
    datasets = []
    categories = ["electronics", "fmcg"]
    for cat in categories:
        cat_path = os.path.join(mock_dir, cat)
        if not os.path.exists(cat_path):
            continue
        for prod in os.listdir(cat_path):
            if prod.startswith('.') or not os.path.isdir(os.path.join(cat_path, prod)):
                continue
            prod_path = os.path.join(cat_path, prod)
            for scenario in os.listdir(prod_path):
                if scenario.startswith('.') or not os.path.isdir(os.path.join(prod_path, scenario)):
                    continue
                datasets.append({
                    'category': cat,
                    'product': prod,
                    'scenario': scenario,
                    'path': os.path.join(prod_path, scenario)
                })
    return sorted(datasets, key=lambda x: (x['category'], x['product'], x['scenario']))

def load_data(path):
    files = {
        'sales': 'sales_demand.csv',
        'competitor': 'competitor_pricing.csv',
        'promotions': 'marketing_promotions.csv',
        'inventory': 'inventory_status.csv',
        'lifecycle': 'product_lifecycle.csv',
        'sentiment': 'consumer_sentiment.csv'
    }
    df_dict = {}
    for key, filename in files.items():
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            df_dict[key] = pd.read_csv(filepath)
        else:
            df_dict[key] = None
    return df_dict

def simulate_curves(coordinator, summary, is_promo_active, mode=None):
    price_changes = np.linspace(-30, 30, 121)  # 0.5% increments
    revenues = []
    profits = []
    q_news = []
    
    best_profit = -float('inf')
    best_profit_pct = 0.0
    best_proj_profit = None
    
    best_revenue = -float('inf')
    best_revenue_pct = 0.0
    best_proj_revenue = None
    
    base_profit = None
    base_revenue = None
    
    for pct in price_changes:
        if mode is not None:
            proj = coordinator.project_demand(summary, pct, is_promo_active=is_promo_active, mode=mode)
        else:
            proj = coordinator.project_demand(summary, pct, is_promo_active=is_promo_active)
            
        revenues.append(proj['rev_new'])
        profits.append(proj['profit_new'])
        q_news.append(proj['q_new'])
        
        if abs(pct - 0.0) < 1e-5:
            base_profit = proj['profit_base']
            base_revenue = proj['rev_base']
            
        if proj['profit_new'] > best_profit:
            best_profit = proj['profit_new']
            best_profit_pct = pct
            best_proj_profit = proj
            
        if proj['rev_new'] > best_revenue:
            best_revenue = proj['rev_new']
            best_revenue_pct = pct
            best_proj_revenue = proj
            
    if base_profit is None or base_profit == 0:
        base_profit = profits[60]  # center of simulation
    if base_revenue is None or base_revenue == 0:
        base_revenue = revenues[60]
        
    profit_increase_pct = ((best_profit - base_profit) / abs(base_profit)) * 100.0 if base_profit != 0 else 0.0
    
    return {
        'price_changes': price_changes,
        'revenues': np.array(revenues),
        'profits': np.array(profits),
        'q_news': np.array(q_news),
        'opt_pct': best_profit_pct,       # optimal price chg % for profit
        'opt_profit': best_profit,
        'base_profit': base_profit,
        'base_revenue': base_revenue,
        'opt_revenue': best_proj_profit['rev_new'] if best_proj_profit else revenues[60],
        'profit_increase_pct': profit_increase_pct,
        'opt_proj': best_proj_profit,
        
        # Revenue-based optimization fields
        'opt_rev_pct': best_revenue_pct,  # optimal price chg % for revenue
        'opt_rev_revenue': best_revenue,
    }

def run_comparison():
    output_dir = ensure_dirs()
    datasets = scan_datasets()
    
    print(f"Starting comparison analysis for {len(datasets)} datasets...")
    
    math_coord = CoordinatorAgent()
    ml_coord = MLCoordinatorAgent()
    
    records = []
    summary_records = []
    
    for idx, ds in enumerate(datasets):
        cat = ds['category']
        prod = ds['product']
        scenario = ds['scenario']
        path = ds['path']
        
        print(f"[{idx+1}/{len(datasets)}] Processing {cat}/{prod}/{scenario}...")
        
        df_dict = load_data(path)
        is_promo_active = (scenario == 'promo_heavy')
        
        # 1. Run analysis
        try:
            summary_math = math_coord.run_analysis(df_dict)
            summary_ml = ml_coord.run_analysis(df_dict)
        except Exception as e:
            print(f"  ❌ Error running analysis for {prod}/{scenario}: {str(e)}")
            continue
            
        # 2. Simulate curves
        curve_math = simulate_curves(math_coord, summary_math, is_promo_active)
        curve_ml_mod = simulate_curves(ml_coord, summary_ml, is_promo_active, mode='modular')
        curve_ml_joint = simulate_curves(ml_coord, summary_ml, is_promo_active, mode='joint')
        
        # 3. Compile Master CSV (with detailed weights/factors)
        models_data = [
            ('Math', summary_math, curve_math, None),
            ('ML Modular', summary_ml, curve_ml_mod, 'modular'),
            ('ML Joint', summary_ml, curve_ml_joint, 'joint')
        ]
        
        for model_name, summary, curve, ml_mode in models_data:
            opt_proj = curve['opt_proj']
            
            # Extract factors
            factors = {}
            for factor_key in ['elasticity', 'seasonality', 'competitor', 'promotions', 'inventory', 'lifecycle', 'sentiment']:
                res = summary['factor_results'].get(factor_key, {})
                factors[f'{factor_key}_val'] = res.get('factor_value', np.nan)
                factors[f'{factor_key}_r2'] = res.get('r2', np.nan)
                factors[f'{factor_key}_weight'] = summary['weights'].get(factor_key, np.nan)
                factors[f'{factor_key}_status'] = res.get('status', 'N/A')
                
            if model_name == 'ML Joint':
                for factor_key in ['seasonality', 'competitor', 'promotions', 'inventory', 'lifecycle', 'sentiment']:
                    factors[f'{factor_key}_weight'] = 0.0
                    factors[f'{factor_key}_status'] = 'Joint Ensemble'
                effective_e = opt_proj['e_eff'] if opt_proj else np.nan
            else:
                effective_e = opt_proj['e_eff'] if opt_proj else np.nan
                
            record = {
                'Category': cat,
                'Product': prod,
                'Scenario': scenario,
                'Model': model_name,
                'Base Price ($)': summary['p_base'],
                'Base Quantity': summary['q_base'],
                'Base Cost ($)': summary['cost_base'],
                'Base Elasticity': summary['e_base'] if model_name != 'ML Joint' else np.nan,
                'Effective Elasticity': effective_e,
                'Model R2': summary.get('joint_r2') if model_name == 'ML Joint' else summary['factor_results']['elasticity'].get('r2', np.nan),
                'Model MAPE (%)': summary.get('joint_mape') if model_name == 'ML Joint' else summary.get('mape', np.nan),
                
                # Factors
                'Elasticity Factor Val': factors['elasticity_val'],
                'Elasticity Weight': factors['elasticity_weight'],
                'Seasonality Factor Val': factors['seasonality_val'],
                'Seasonality Weight': factors['seasonality_weight'],
                'Competitor Factor Val': factors['competitor_val'],
                'Competitor Weight': factors['competitor_weight'],
                'Promotions Factor Val': factors['promotions_val'],
                'Promotions Weight': factors['promotions_weight'],
                'Inventory Factor Val': factors['inventory_val'],
                'Inventory Weight': factors['inventory_weight'],
                'Lifecycle Factor Val': factors['lifecycle_val'],
                'Lifecycle Weight': factors['lifecycle_weight'],
                'Sentiment Factor Val': factors['sentiment_val'],
                'Sentiment Weight': factors['sentiment_weight'],
                
                # KPIs
                'Optimal Price Change (%)': curve['opt_pct'],
                'Base Profit ($)': curve['base_profit'],
                'Optimal Profit ($)': curve['opt_profit'],
                'Profit Increment ($)': curve['opt_profit'] - curve['base_profit'],
                'Profit Increment (%)': curve['profit_increase_pct'],
                'Base Revenue ($)': curve['base_revenue'],
                'Optimal Revenue ($)': curve['opt_revenue'],
                'Optimal Price ($)': summary['p_base'] * (1.0 + curve['opt_pct'] / 100.0),
                'Optimal Quantity': opt_proj['q_new'] if opt_proj else np.nan
            }
            records.append(record)
            
        # 4. Compile Pricing Summary CSV (Math vs ML with percent differences)
        p_base = summary_math['p_base']
        
        opt_rev_price_math = p_base * (1.0 + curve_math['opt_rev_pct'] / 100.0)
        opt_rev_price_ml_mod = p_base * (1.0 + curve_ml_mod['opt_rev_pct'] / 100.0)
        opt_rev_price_ml_joint = p_base * (1.0 + curve_ml_joint['opt_rev_pct'] / 100.0)
        
        opt_prof_price_math = p_base * (1.0 + curve_math['opt_pct'] / 100.0)
        opt_prof_price_ml_mod = p_base * (1.0 + curve_ml_mod['opt_pct'] / 100.0)
        opt_prof_price_ml_joint = p_base * (1.0 + curve_ml_joint['opt_pct'] / 100.0)
        
        # Percent differences (ML relative to Math)
        diff_rev_mod = ((opt_rev_price_ml_mod - opt_rev_price_math) / opt_rev_price_math) * 100.0 if opt_rev_price_math != 0 else 0.0
        diff_rev_joint = ((opt_rev_price_ml_joint - opt_rev_price_math) / opt_rev_price_math) * 100.0 if opt_rev_price_math != 0 else 0.0
        
        diff_prof_mod = ((opt_prof_price_ml_mod - opt_prof_price_math) / opt_prof_price_math) * 100.0 if opt_prof_price_math != 0 else 0.0
        diff_prof_joint = ((opt_prof_price_ml_joint - opt_prof_price_math) / opt_prof_price_math) * 100.0 if opt_prof_price_math != 0 else 0.0
        
        sum_rec = {
            'Category': cat,
            'Product': prod,
            'Scenario': scenario,
            'Base Price ($)': p_base,
            
            # Revenue Optimal Prices
            'Math Opt Price (Revenue) ($)': opt_rev_price_math,
            'ML Mod Opt Price (Revenue) ($)': opt_rev_price_ml_mod,
            'ML Joint Opt Price (Revenue) ($)': opt_rev_price_ml_joint,
            
            # Profit Optimal Prices
            'Math Opt Price (Profit) ($)': opt_prof_price_math,
            'ML Mod Opt Price (Profit) ($)': opt_prof_price_ml_mod,
            'ML Joint Opt Price (Profit) ($)': opt_prof_price_ml_joint,
            
            # Percent differences (ML vs Math)
            'Pct Diff (ML Mod vs Math) - Revenue Price (%)': diff_rev_mod,
            'Pct Diff (ML Joint vs Math) - Revenue Price (%)': diff_rev_joint,
            'Pct Diff (ML Mod vs Math) - Profit Price (%)': diff_prof_mod,
            'Pct Diff (ML Joint vs Math) - Profit Price (%)': diff_prof_joint
        }
        summary_records.append(sum_rec)
            
        # 5. Generate Plot with 16:9 TV screen aspect ratio, graphs side-by-side above, table below
        plot_dataset(cat, prod, scenario, summary_math, summary_ml, curve_math, curve_ml_mod, curve_ml_joint, output_dir)
        
    # Write CSVs
    df_compare = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    df_compare.to_csv(csv_path, index=False)
    
    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "optimal_pricing_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    
    print(f"\n✅ Detailed metrics CSV saved to: {csv_path}")
    print(f"✅ Summary pricing CSV saved to: {summary_csv_path}")
    print(f"📊 Visualizations generated for all 32 datasets in: {output_dir}")

def plot_dataset(cat, prod, scenario, summary_math, summary_ml, curve_math, curve_ml_mod, curve_ml_joint, output_dir):
    title_str = f"Agent Pricing Optimization: {cat.upper()} - {prod.replace('_', ' ').title()}"
    subtitle_str = f"{scenario.replace('_', ' ').title()} Scenario Run"
    
    # Setup Figure with exact 16:9 TV screen aspect ratio and dark theme
    fig = plt.figure(figsize=(16, 9), facecolor='#0f0f12')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], wspace=0.18, hspace=0.35)
    
    # 2 side-by-side subplots on top row with dark facecolor
    ax_rev = fig.add_subplot(gs[0, 0], facecolor='#14151b')
    ax_prof = fig.add_subplot(gs[0, 1], facecolor='#14151b')
    
    # KPI Grid table on bottom row (spanning both columns)
    ax_tbl = fig.add_subplot(gs[1, :], facecolor='#0f0f12')
    
    # Style curves subplots
    for ax in [ax_rev, ax_prof]:
        ax.spines['top'].set_color('#2d303f')
        ax.spines['right'].set_color('#2d303f')
        ax.spines['left'].set_color('#2d303f')
        ax.spines['bottom'].set_color('#2d303f')
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.grid(True, linestyle='--', alpha=0.3, color='#4f556f')
        ax.tick_params(colors='#ffffff', labelsize=9.5)
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        ax.set_xlabel("Price Change (%)", fontsize=10, fontweight='medium')
        
    price_changes = curve_math['price_changes']
    
    # --- 1. Top Left: Projected Revenue vs. Price Change ---
    ax_rev.plot(price_changes, curve_math['revenues'], label='Math Model', color='#ff9100', linewidth=2, linestyle='--')
    ax_rev.plot(price_changes, curve_ml_mod['revenues'], label='ML Modular', color='#00e5ff', linewidth=2)
    ax_rev.plot(price_changes, curve_ml_joint['revenues'], label='ML Joint (Ensemble)', color='#00e676', linewidth=2, linestyle='-.')
    
    ax_rev.axvline(0.0, color='#ffffff', linestyle=':', alpha=0.5, label='Base Price (0%)')
    ax_rev.axhline(curve_math['base_revenue'], color='#ff9100', linestyle=':', alpha=0.3)
    
    # Highlight optimal revenue points
    ax_rev.scatter(curve_math['opt_rev_pct'], curve_math['opt_rev_revenue'], color='#ff9100', s=70, zorder=5, edgecolor='#ffffff', linewidth=1)
    ax_rev.scatter(curve_ml_mod['opt_rev_pct'], curve_ml_mod['opt_rev_revenue'], color='#00e5ff', s=70, zorder=5, edgecolor='#ffffff', linewidth=1)
    ax_rev.scatter(curve_ml_joint['opt_rev_pct'], curve_ml_joint['opt_rev_revenue'], color='#00e676', s=70, zorder=5, edgecolor='#ffffff', linewidth=1)
    
    ax_rev.set_ylabel("Projected Revenue ($)", fontsize=10, fontweight='medium')
    ax_rev.set_title("Projected Revenue Curve", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
    legend1 = ax_rev.legend(frameon=True, facecolor='#14151b', edgecolor='#2d303f', loc='upper right', fontsize=8.5)
    plt.setp(legend1.get_texts(), color='w')
    
    # --- 2. Top Right: Projected Profit vs. Price Change ---
    ax_prof.plot(price_changes, curve_math['profits'], label='Math Model', color='#ff9100', linewidth=2, linestyle='--')
    ax_prof.plot(price_changes, curve_ml_mod['profits'], label='ML Modular', color='#00e5ff', linewidth=2)
    ax_prof.plot(price_changes, curve_ml_joint['profits'], label='ML Joint (Ensemble)', color='#00e676', linewidth=2, linestyle='-.')
    
    ax_prof.axvline(0.0, color='#ffffff', linestyle=':', alpha=0.5)
    ax_prof.axhline(curve_math['base_profit'], color='#ff9100', linestyle=':', alpha=0.3)
    
    # Vertical guides for max profits
    ax_prof.axvline(curve_math['opt_pct'], color='#ff9100', linestyle='--', alpha=0.3)
    ax_prof.axvline(curve_ml_mod['opt_pct'], color='#00e5ff', linestyle='--', alpha=0.3)
    ax_prof.axvline(curve_ml_joint['opt_pct'], color='#00e676', linestyle='--', alpha=0.3)
    
    # Highlight optimal profit points
    ax_prof.scatter(curve_math['opt_pct'], curve_math['opt_profit'], color='#ff9100', s=85, zorder=5, edgecolor='#ffffff', linewidth=1)
    ax_prof.scatter(curve_ml_mod['opt_pct'], curve_ml_mod['opt_profit'], color='#00e5ff', s=85, zorder=5, edgecolor='#ffffff', linewidth=1)
    ax_prof.scatter(curve_ml_joint['opt_pct'], curve_ml_joint['opt_profit'], color='#00e676', s=85, zorder=5, edgecolor='#ffffff', linewidth=1)
    
    ax_prof.set_ylabel("Projected Profit ($)", fontsize=10, fontweight='medium')
    ax_prof.set_title("Projected Profit Curve", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
    legend2 = ax_prof.legend(frameon=True, facecolor='#14151b', edgecolor='#2d303f', loc='upper right', fontsize=8.5)
    plt.setp(legend2.get_texts(), color='w')
    
    # --- 3. Bottom: Render Spacious Tabular KPI Layout ---
    ax_tbl.axis('off')
    
    cols = [
        ("Model / Configuration", 0.22, 'left'),
        ("Base Price", 0.08, 'center'),
        ("Opt. Chg %", 0.08, 'center'),
        ("Opt. Price", 0.08, 'center'),
        ("Base Profit", 0.09, 'center'),
        ("Max Profit", 0.09, 'center'),
        ("Profit Impact", 0.10, 'center'),
        ("Profit Inc %", 0.08, 'center'),
        ("Max Revenue", 0.09, 'center'),
        ("Verdict", 0.09, 'center')
    ]
    
    x_positions = []
    curr_x = 0.01
    for name, width, align in cols:
        x_positions.append((curr_x, width, align))
        curr_x += width
        
    p_base = summary_math['p_base']
    cost_base = summary_math['cost_base']
    prof_base = (p_base - cost_base) * summary_math['q_base']
    rev_base = p_base * summary_math['q_base']
    
    models_data = [
        ("Math Model (Traditional)", curve_math, '#ff9100'),
        ("ML Modular (Factor Decay)", curve_ml_mod, '#00e5ff'),
        ("ML Joint Ensemble (Deep)", curve_ml_joint, '#00e676')
    ]
    
    # Header rectangle
    header_rect = plt.Rectangle((0.005, 0.76), 0.99, 0.20, facecolor='#1a1b22', edgecolor='#2d303f', linewidth=1, transform=ax_tbl.transAxes)
    ax_tbl.add_patch(header_rect)
    
    # Draw header text and vertical dividers
    curr_x = 0.01
    for i, (name, width, align) in enumerate(cols):
        x_pos = x_positions[i][0] + (width / 2.0 if align == 'center' else 0.0)
        ax_tbl.text(x_pos, 0.86, name, color='#ffffff', fontsize=9, fontweight='bold',
                    horizontalalignment=align, verticalalignment='center', transform=ax_tbl.transAxes)
        if i < len(cols) - 1:
            curr_x += width
            ax_tbl.plot([curr_x, curr_x], [0.76, 0.96], color='#2d303f', linewidth=1, transform=ax_tbl.transAxes)
            
    # Draw rows
    y_pos = 0.51
    for name, curve, color in models_data:
        # Draw row background
        row_rect = plt.Rectangle((0.005, y_pos - 0.11), 0.99, 0.22, facecolor='#111215', edgecolor='#2d303f', linewidth=0.8, transform=ax_tbl.transAxes)
        ax_tbl.add_patch(row_rect)
        
        opt_pct = curve['opt_pct']
        opt_price = p_base * (1.0 + opt_pct / 100.0)
        opt_profit = curve['opt_profit']
        profit_inc = opt_profit - curve['base_profit']
        profit_inc_pct = curve['profit_increase_pct']
        opt_rev = curve['opt_revenue']
        
        if opt_pct > 1.0:
            verdict = "INCREASE"
            verdict_color = '#00e676'
        elif opt_pct < -1.0:
            verdict = "DECREASE"
            verdict_color = '#ff1744'
        else:
            verdict = "MAINTAIN"
            verdict_color = '#2979ff'
            
        sign = "+" if opt_pct >= 0 else ""
        
        values = [
            (name, color, 'left'),
            (f"${p_base:.2f}", '#ffffff', 'center'),
            (f"{sign}{opt_pct:.1f}%", color, 'center'),
            (f"${opt_price:.2f}", color, 'center'),
            (f"${curve['base_profit']:.2f}", '#ffffff', 'center'),
            (f"${opt_profit:.2f}", color, 'center'),
            (f"+${profit_inc:.2f}" if profit_inc >= 0 else f"-${abs(profit_inc):.2f}", color, 'center'),
            (f"{profit_inc_pct:+.1f}%", color, 'center'),
            (f"${opt_rev:.2f}", '#ffffff', 'center'),
            (verdict, verdict_color, 'center')
        ]
        
        # Text and dividers for each row
        curr_x = 0.01
        for idx, (val, val_color, align) in enumerate(values):
            x_pos = x_positions[idx][0] + (x_positions[idx][1] / 2.0 if align == 'center' else 0.0)
            weight = 'bold' if idx == 0 or idx == 9 else 'normal'
            ax_tbl.text(x_pos, y_pos, val, color=val_color, fontsize=8.5, fontweight=weight,
                        horizontalalignment=align, verticalalignment='center', transform=ax_tbl.transAxes)
            if idx < len(values) - 1:
                curr_x += x_positions[idx][1]
                ax_tbl.plot([curr_x, curr_x], [y_pos - 0.11, y_pos + 0.11], color='#2d303f', linewidth=0.8, transform=ax_tbl.transAxes)
                
        y_pos -= 0.26
        
    # Main visual adjustments
    fig.text(0.5, 0.95, title_str, fontsize=15, fontweight='bold', color='#ffffff', ha='center')
    fig.text(0.5, 0.91, subtitle_str, fontsize=11, fontweight='bold', color='#a0a0a5', ha='center')
    
    plt.subplots_adjust(top=0.88, bottom=0.03, left=0.06, right=0.94)
    
    # Save Image (exact 16:9 TV ratio, dark style)
    filename = f"{cat}_{prod}_{scenario}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0f0f12')
    plt.close()

if __name__ == '__main__':
    run_comparison()
