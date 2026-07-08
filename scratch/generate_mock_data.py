import os
import pandas as pd
import numpy as np
import datetime

# Define base configuration for products
PRODUCTS = {
    'electronics': {
        'mobile_phone': {
            'base_price': 800.0,
            'cost': 500.0,
            'base_q': 150.0,
            'e': -2.5,
            'seasonality_amplitude': 0.15,
            'lifecycle_shift': True,
        },
        'laptop': {
            'base_price': 1200.0,
            'cost': 800.0,
            'base_q': 60.0,
            'e': -1.8,
            'seasonality_amplitude': 0.10,
            'lifecycle_shift': True,
        }
    },
    'fmcg': {
        'rice': {
            'base_price': 3.0,
            'cost': 1.5,
            'base_q': 2000.0,
            'e': -0.4,
            'seasonality_amplitude': 0.05,
            'lifecycle_shift': False,
        },
        'shampoo': {
            'base_price': 6.0,
            'cost': 2.8,
            'base_q': 600.0,
            'e': -0.9,
            'seasonality_amplitude': 0.25,
            'lifecycle_shift': False,
        },
        'face_wash': {
            'base_price': 8.0,
            'cost': 3.8,
            'base_q': 450.0,
            'e': -1.2,
            'seasonality_amplitude': 0.35,
            'lifecycle_shift': False,
        },
        'fiama_body_wash': {
            'base_price': 200.0,
            'cost': 120.0,
            'base_q': 800.0,
            'e': -2.2,
            'seasonality_amplitude': 0.18,
            'lifecycle_shift': False,
        }
    }
}

SCENARIOS = ['stable', 'inflation', 'promo_heavy', 'competitor_war']

# Generate date index (156 weeks starting 3 years ago)
start_date = datetime.date(2023, 7, 1)
dates = [start_date + datetime.timedelta(weeks=i) for i in range(156)]

# Root directory for mock data
MOCK_DATA_ROOT = "/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/MockData"

def simulate_product_data(category, product_name, config, scenario):
    np.random.seed(42 + hash(product_name + scenario) % 1000)
    
    # 1. Base Variables
    base_price = config['base_price']
    cost = config['cost']
    base_q = config['base_q']
    base_e = config['e']
    
    # Initialize lists
    prices = []
    costs = []
    units_sold = []
    comp_prices = []
    is_promo = []
    mkt_spend = []
    stock = []
    avg_daily_sales = []
    cci = []
    google_trends = []
    
    # Setup initial inventory state
    current_stock = base_q * 3  # 3 weeks of stock
    
    # 2. Iterate week by week
    for t in range(156):
        date = dates[t]
        week_of_year = date.isocalendar()[1]
        
        # Product age in months
        age_months = t * 12 / 52
        
        # A. Product Lifecycle Modifier (L)
        if config['lifecycle_shift']:
            if age_months < 6:
                L = 0.70  # Launch
            elif age_months < 18:
                L = 0.85  # Growth
            elif age_months < 36:
                L = 1.0   # Mature
            else:
                L = 1.2   # Decline
        else:
            L = 1.0  # FMCG stays mature
            
        # B. Consumer Sentiment (X) and CCI
        if scenario == 'inflation':
            # Consumer sentiment drops over time
            current_cci = max(80.0, 100.0 - t * 0.12)
            current_cost = cost * (1.0 + t * 0.001)  # supply cost inflation
        else:
            current_cci = 100.0 + np.random.normal(0, 1.5)
            current_cost = cost
            
        X = 1.0 + 0.1 * (current_cci - 100.0) / 100.0
        
        # C. Competitor Price & Gap (C)
        if scenario == 'competitor_war' and t > 80:
            # Competitor starts dumping price
            comp_price = base_price * 0.80 + np.random.normal(0, base_price * 0.02)
        else:
            comp_price = base_price + np.random.normal(0, base_price * 0.03)
            
        # D. Price Decision (Own Price)
        if scenario == 'promo_heavy' and t % 4 == 0:
            # Every 4 weeks we run a promo price cut
            own_price = base_price * 0.82
            promo_flag = 1
            spend = base_price * 2.5
        elif scenario == 'competitor_war' and t > 95:
            # Match competitor price war halfway
            own_price = base_price * 0.90 + np.random.normal(0, base_price * 0.01)
            promo_flag = 0
            spend = 0.0
        else:
            # Normal pricing fluctuations
            own_price = base_price + np.random.normal(0, base_price * 0.02)
            promo_flag = 0
            spend = 0.0
            
        gap = (own_price - comp_price) / comp_price
        C = 1.0 + 0.2 * np.sign(gap) * min(abs(gap), 0.5)
        
        # E. Effective Elasticity
        e_eff = base_e * C * L * X
        
        # F. Seasonality (S)
        S = 1.0 + config['seasonality_amplitude'] * np.sin(2.0 * np.pi * week_of_year / 52.0)
        
        # G. Inventory Coverage (I)
        # Daily sales proxy
        est_daily_sales = base_q / 7.0
        coverage_days = current_stock / max(0.1, est_daily_sales)
        
        # Stockouts and Scarcity
        if coverage_days < 7:
            I_val = 1.15
        else:
            I_val = 1.0
            
        # H. Promo Lift
        lift_M = 0.25 if promo_flag == 1 else 0.0
        if category == 'electronics':
            lift_M *= 1.5  # Electronics react more to promos
            
        # I. Demand Projection (Master Formula)
        Q_proj = base_q * ((own_price / base_price) ** e_eff) * S * I_val * (1.0 + lift_M)
        
        # Add random noise
        noise = np.random.normal(1.0, 0.05)
        Q_actual = Q_proj * noise
        
        # Inject Spikes for testing (S1-S5)
        # We inject an unexplained transient hype spike (Type A) at Week 45
        if t == 45:
            Q_actual = Q_proj * 2.5  # 250% demand spike
            
        # We inject a social media hype spike (Type A) at Week 110
        if t == 110:
            Q_actual = Q_proj * 2.2
            
        # J. Inventory Supply Cap & Update
        if current_stock < 3 * est_daily_sales:  # Severe stockout
            # Exclude week for estimation, clamp units sold
            actual_sold = min(Q_actual, current_stock)
        else:
            actual_sold = Q_actual
            
        actual_sold = max(0, int(actual_sold))
        
        # Replenish stock
        refill = base_q + np.random.normal(0, base_q * 0.1)
        current_stock = max(10, current_stock - actual_sold + refill)
        
        # Append data points
        prices.append(round(own_price, 2))
        costs.append(round(current_cost, 2))
        units_sold.append(actual_sold)
        comp_prices.append(round(comp_price, 2))
        is_promo.append(promo_flag)
        mkt_spend.append(round(spend, 2))
        stock.append(int(current_stock))
        avg_daily_sales.append(round(actual_sold / 7.0, 2))
        cci.append(round(current_cci, 1))
        
        # Google trends index
        gt = 50.0 + (S - 1.0) * 100.0 + np.random.normal(0, 5)
        if t == 110:
            gt *= 3.5  # Social media surge
        google_trends.append(round(min(100.0, max(0.0, gt)), 1))
        
    # Write to target CSVs
    target_dir = os.path.join(MOCK_DATA_ROOT, category, product_name, scenario)
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. sales_demand.csv
    df_sales = pd.DataFrame({
        'date': dates,
        'sku_id': f"{category[:3].upper()}_{product_name.upper()}_001",
        'unit_price': prices,
        'units_sold': units_sold,
        'cost_per_unit': costs
    })
    df_sales.to_csv(os.path.join(target_dir, 'sales_demand.csv'), index=False)
    
    # 2. competitor_pricing.csv
    df_comp = pd.DataFrame({
        'date': dates,
        'comp_price_avg': comp_prices,
        'comp_price_min': [p - round(p*0.05, 2) for p in comp_prices],
        'comp_price_max': [p + round(p*0.05, 2) for p in comp_prices],
    })
    df_comp.to_csv(os.path.join(target_dir, 'competitor_pricing.csv'), index=False)
    
    # 3. marketing_promotions.csv
    df_mkt = pd.DataFrame({
        'date': dates,
        'is_promo': is_promo,
        'marketing_spend': mkt_spend
    })
    df_mkt.to_csv(os.path.join(target_dir, 'marketing_promotions.csv'), index=False)
    
    # 4. inventory_status.csv
    df_inv = pd.DataFrame({
        'date': dates,
        'units_in_stock': stock,
        'avg_daily_sales_14d': avg_daily_sales
    })
    df_inv.to_csv(os.path.join(target_dir, 'inventory_status.csv'), index=False)
    
    # 5. product_lifecycle.csv
    # Just need one SKU row
    df_life = pd.DataFrame({
        'sku_id': [f"{category[:3].upper()}_{product_name.upper()}_001"],
        'launch_date': [dates[0].strftime('%Y-%m-%d')],
        'category': [category.capitalize()]
    })
    df_life.to_csv(os.path.join(target_dir, 'product_lifecycle.csv'), index=False)
    
    # 6. consumer_sentiment.csv
    df_sent = pd.DataFrame({
        'date': dates,
        'cci_current': cci,
        'cci_baseline': [100.0] * 156,
        'google_trends_score': google_trends
    })
    df_sent.to_csv(os.path.join(target_dir, 'consumer_sentiment.csv'), index=False)

def main():
    print("Generating mock data...")
    for category, products in PRODUCTS.items():
        for product_name, config in products.items():
            for scenario in SCENARIOS:
                simulate_product_data(category, product_name, config, scenario)
                print(f"Generated {category}/{product_name}/{scenario}")
    print("Mock data generation complete!")

if __name__ == '__main__':
    main()
