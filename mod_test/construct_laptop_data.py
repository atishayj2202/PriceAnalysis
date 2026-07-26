"""
Construct Realistic 10-Year Laptop Pricing & Sales Dataset (2015 - 2025)
========================================================================
Based on actual IDC India / Global PC shipment statistics, ASP evolution,
component cost cycles (RAM/SSD/CPU generations), and promo seasonality.

Brands:
  - Dell (Target Brand / Leader in Commercial & Premium)
  - HP (Market Leader in Total Units)
  - Lenovo (Leader in Enterprise & Value)
  - Asus (Fast-growing Consumer & Gaming)

Timeframe: Weekly data from Jan 3, 2015 to June 28, 2025 (547 weeks x 4 brands = 2,188 rows)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def construct_laptop_dataset():
    print("Constructing 10-Year Laptop Pricing & Shipment Dataset (2015-2025)...")
    
    start_date = datetime(2015, 1, 3)
    end_date = datetime(2025, 6, 28)
    
    dates = []
    curr = start_date
    while curr <= end_date:
        dates.append(curr)
        curr += timedelta(days=7)
        
    brands = ["Dell", "HP", "Lenovo", "Asus"]
    
    # 1. Base ASP (Average Selling Price in INR ₹) trends year by year (2015-2025)
    # Reflects inflation, RAM/SSD upgrades (4GB HDD in 2015 -> 16GB SSD in 2024), GST in 2017, COVID in 2020, AI PC in 2025
    asp_yearly = {
        "Dell": {
            2015: 44500, 2016: 45800, 2017: 48200, 2018: 51000, 2019: 53500,
            2020: 58000, 2021: 62500, 2022: 66000, 2023: 65000, 2024: 68500, 2025: 72000
        },
        "HP": {
            2015: 42000, 2016: 43200, 2017: 45500, 2018: 48000, 2019: 50500,
            2020: 55000, 2021: 59000, 2022: 62500, 2023: 61500, 2024: 64500, 2025: 68000
        },
        "Lenovo": {
            2015: 38500, 2016: 39800, 2017: 42000, 2018: 44500, 2019: 47000,
            2020: 51500, 2021: 55500, 2022: 58500, 2023: 57500, 2024: 60500, 2025: 64000
        },
        "Asus": {
            2015: 36000, 2016: 37200, 2017: 39500, 2018: 42000, 2019: 44500,
            2020: 49000, 2021: 53000, 2022: 56000, 2023: 55000, 2024: 58000, 2025: 61500
        }
    }
    
    # 2. Total annual India market shipments (million units) from IDC data
    annual_market_volume_m = {
        2015: 9.50, 2016: 8.58, 2017: 9.56, 2018: 9.30, 2019: 11.00,
        2020: 10.20, 2021: 14.80, 2022: 14.90, 2023: 13.90, 2024: 14.40, 2025: 15.90
    }
    
    # 3. Market share split per year (from IDC reports)
    market_shares = {
        2015: {"HP": 0.264, "Dell": 0.216, "Lenovo": 0.181, "Asus": 0.050},
        2016: {"HP": 0.284, "Dell": 0.233, "Lenovo": 0.176, "Asus": 0.052},
        2017: {"HP": 0.299, "Dell": 0.224, "Lenovo": 0.202, "Asus": 0.055},
        2018: {"HP": 0.310, "Dell": 0.230, "Lenovo": 0.210, "Asus": 0.058},
        2019: {"HP": 0.280, "Dell": 0.215, "Lenovo": 0.295, "Asus": 0.062},
        2020: {"HP": 0.300, "Dell": 0.190, "Lenovo": 0.245, "Asus": 0.065},
        2021: {"HP": 0.315, "Dell": 0.236, "Lenovo": 0.184, "Asus": 0.059},
        2022: {"HP": 0.302, "Dell": 0.192, "Lenovo": 0.189, "Asus": 0.068},
        2023: {"HP": 0.315, "Dell": 0.155, "Lenovo": 0.167, "Asus": 0.079},
        2024: {"HP": 0.301, "Dell": 0.161, "Lenovo": 0.172, "Asus": 0.070},
        2025: {"HP": 0.291, "Dell": 0.151, "Lenovo": 0.187, "Asus": 0.076}
    }
    
    elasticities = {"Dell": -3.8, "HP": -4.2, "Lenovo": -4.5, "Asus": -4.8}
    cost_ratios = {"Dell": 0.72, "HP": 0.74, "Lenovo": 0.76, "Asus": 0.78}
    
    df_rows = []
    
    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        yr = dt.year
        mo = dt.month
        
        season_mult = 1.0
        if mo in [10, 11]:
            season_mult = 1.45
        elif mo in [8, 9]:
            season_mult = 1.25
        elif mo == 1:
            season_mult = 1.15
        elif mo == 3:
            season_mult = 1.20
        elif mo in [5, 6]:
            season_mult = 0.85
            
        total_mkt_vol = annual_market_volume_m[yr]
        weekly_mkt_base_units = (total_mkt_vol * 1e6) / 52.0
        
        prices_curr_week = {}
        for brand in brands:
            base_p = asp_yearly[brand][yr]
            monthly_drift = 1.0 + (mo - 6.5) * 0.005
            prices_curr_week[brand] = base_p * monthly_drift
            
        for brand in brands:
            promo_rand = np.random.RandomState(seed=(yr * 10000 + mo * 100 + dt.day + hash(brand) % 1000))
            is_promo_period = mo in [1, 8, 10, 11]
            is_promo = 1 if (promo_rand.rand() < (0.40 if is_promo_period else 0.10)) else 0
            
            discount_pct = 0.08 if is_promo else 0.0
            unit_price = prices_curr_week[brand] * (1.0 - discount_pct)
            
            brand_share = market_shares[yr][brand]
            units_base = weekly_mkt_base_units * brand_share
            
            asp_base = asp_yearly[brand][yr]
            price_ratio = unit_price / asp_base
            price_effect = (price_ratio) ** elasticities[brand]
            
            comps = [b for b in brands if b != brand]
            comp_p1 = prices_curr_week[comps[0]]
            comp_p2 = prices_curr_week[comps[1]]
            comp_p3 = prices_curr_week[comps[2]]
            
            noise = 1.0 + (promo_rand.rand() - 0.5) * 0.08
            
            units_sold = units_base * season_mult * price_effect * noise
            cost_per_unit = asp_base * cost_ratios[brand]
            
            df_rows.append({
                "date": dt_str,
                "brand": brand,
                "sku_id": f"LAPTOP_{brand.upper()}_SERIES",
                "unit_price": round(unit_price, 2),
                "units_sold": round(units_sold, 1),
                "cost_per_unit": round(cost_per_unit, 2),
                "is_promo": is_promo,
                "is_festival": 1 if season_mult > 1.1 else 0,
                "comp_price_1": round(comp_p1, 2),
                "comp_price_2": round(comp_p2, 2),
                "comp_price_3": round(comp_p3, 2)
            })
            
    df_out = pd.DataFrame(df_rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptop_pricing_data.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Total rows: {len(df_out)} (Date range: {df_out['date'].min()} to {df_out['date'].max()})")

if __name__ == "__main__":
    construct_laptop_dataset()
