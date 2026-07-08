import sys
import os
# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils.spike_detector import SpikeDetector
from agents.coordinator_agent import CoordinatorAgent

def test_spike_detector():
    print("Running Spike Detector Test...")
    sales = [100] * 10 + [250] + [100] * 5
    dates = pd.date_range(start='2026-01-01', periods=16, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'sku_id': ['TST_01'] * 16,
        'unit_price': [10.0] * 16,
        'units_sold': sales,
        'cost_per_unit': [5.0] * 16
    })
    
    detector = SpikeDetector()
    df_clean = detector.detect_and_classify_spikes(df)
    
    assert df_clean.loc[10, 'is_spike'] == True, "Spike at index 10 not detected"
    assert df_clean.loc[10, 'spike_type'] == 'Type A', "Spike should be classified as Type A (transient)"
    assert df_clean.loc[10, 'exclude_from_regression'] == True, "Spike should be excluded from regression"
    print("✅ Spike Detector Test passed successfully!")

def test_model_worked_example():
    print("Running Model worked example test...")
    p_base = 200.0
    q_base = 100.0
    cost_base = 80.0
    p_new = 220.0
    
    e = -1.5
    S = 1.2
    C = 1.032
    L = 0.85
    X = 1.0
    I = 1.0
    lift_m = 0.0
    
    e_eff = e * C * L * X
    q_new = q_base * ((p_new / p_base) ** e_eff) * S * I * (1.0 + lift_m)
    profit_base = (p_base - cost_base) * q_base
    profit_new = (p_new - cost_base) * q_new
    
    # Mathematically correct evaluation:
    assert round(e_eff, 3) == -1.316, f"Expected e_eff = -1.316, got {e_eff}"
    assert round(q_new, 1) == 105.9, f"Expected q_new = 105.9, got {q_new}"
    assert int(round(profit_new, 0)) == 14820, f"Expected Profit = 14820, got {profit_new}"
    print("✅ Model worked example test passed successfully!")

def test_multi_agent_coordinator():
    print("Running Multi-Agent Coordinator Test...")
    mock_base_path = "/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/MockData/electronics/mobile_phone/stable"
    assert os.path.exists(mock_base_path), "Mock data preset path does not exist. Run generate_mock_data.py first."
    
    df_dict = {
        'sales': pd.read_csv(os.path.join(mock_base_path, 'sales_demand.csv')),
        'competitor': pd.read_csv(os.path.join(mock_base_path, 'competitor_pricing.csv')),
        'promotions': pd.read_csv(os.path.join(mock_base_path, 'marketing_promotions.csv')),
        'inventory': pd.read_csv(os.path.join(mock_base_path, 'inventory_status.csv')),
        'lifecycle': pd.read_csv(os.path.join(mock_base_path, 'product_lifecycle.csv')),
        'sentiment': pd.read_csv(os.path.join(mock_base_path, 'consumer_sentiment.csv'))
    }
    
    coordinator = CoordinatorAgent()
    summary = coordinator.run_analysis(df_dict)
    
    assert 'clean_sales' in summary
    assert 'factor_results' in summary
    assert 'weights' in summary
    
    sum_w = sum(summary['weights'].values())
    assert abs(sum_w - 1.0) < 1e-4, f"Weights should sum to 1.0, got {sum_w}"
    assert summary['e_base'] < 0, "Base elasticity should be negative"
    
    proj = coordinator.project_demand(summary, 10.0, is_promo_active=False)
    assert proj['q_new'] < summary['q_base'], "Quantity sold should decrease on 10% price increase for mobile phone"
    assert proj['p_new'] > summary['p_base'], "New price should be greater than base price"
    print("✅ Multi-Agent Coordinator Test passed successfully!")

def run_all_tests():
    print("--- STARTING DEMAND APPROXIMATION TESTS ---")
    test_spike_detector()
    test_model_worked_example()
    test_multi_agent_coordinator()
    print("--- ALL TESTS PASSED SUCCESSFULLY! ---")
    return True

if __name__ == '__main__':
    run_all_tests()
