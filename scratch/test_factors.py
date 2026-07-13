import os
import sys
import pandas as pd

# Ensure imports resolve from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agents.coordinator_agent import CoordinatorAgent
from ml_agents.coordinator_agent import MLCoordinatorAgent

def test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mock_base_path = os.path.join(base_dir, "MockData", "electronics", "mobile_phone", "stable")
    
    df_dict = {
        'sales': pd.read_csv(os.path.join(mock_base_path, 'sales_demand.csv')),
        'competitor': pd.read_csv(os.path.join(mock_base_path, 'competitor_pricing.csv')),
        'promotions': pd.read_csv(os.path.join(mock_base_path, 'marketing_promotions.csv')),
        'inventory': pd.read_csv(os.path.join(mock_base_path, 'inventory_status.csv')),
        'lifecycle': pd.read_csv(os.path.join(mock_base_path, 'product_lifecycle.csv')),
        'sentiment': pd.read_csv(os.path.join(mock_base_path, 'consumer_sentiment.csv'))
    }
    
    print("--- Running Math Agent ---")
    coord_math = CoordinatorAgent()
    summary_math = coord_math.run_analysis(df_dict)
    
    print("Math weights:")
    print(summary_math['weights'])
    print("Math base elasticity:", summary_math['e_base'])
    print("Math factors:")
    for k, v in summary_math['factor_results'].items():
        print(f"  {k}: factor_value={v.get('factor_value')}, r2={v.get('r2')}, status={v.get('status')}")
        
    print("\n--- Running ML Agent ---")
    coord_ml = MLCoordinatorAgent()
    summary_ml = coord_ml.run_analysis(df_dict)
    print("ML weights:")
    print(summary_ml['weights'])
    print("ML base elasticity:", summary_ml['e_base'])
    print("ML factors:")
    for k, v in summary_ml['factor_results'].items():
        print(f"  {k}: factor_value={v.get('factor_value')}, r2={v.get('r2')}, status={v.get('status')}")

if __name__ == '__main__':
    test()
