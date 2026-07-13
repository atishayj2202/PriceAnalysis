import sys
import os
import pandas as pd
import numpy as np

# Ensure imports resolve correctly from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_agents.elasticity_agent import MLElasticityAgent
from ml_agents.coordinator_agent import MLCoordinatorAgent

def test_ml_elasticity_agent():
    print("Running ML Elasticity Agent Test...")
    # Generate log-linear price-demand data with elasticity = -1.5
    np.random.seed(42)
    p_vals = np.linspace(10.0, 20.0, 30)
    log_p = np.log(p_vals)
    log_q = 5.0 - 1.5 * log_p + np.random.normal(0, 0.05, 30)
    q_vals = np.exp(log_q)
    
    dates = pd.date_range(start='2026-01-01', periods=30, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'sku_id': ['TST_ML'] * 30,
        'unit_price': p_vals,
        'units_sold': q_vals,
        'cost_per_unit': [5.0] * 30,
        'exclude_from_regression': [False] * 30
    })
    
    agent = MLElasticityAgent()
    res = agent.assess({'sales': df})
    
    print(f"ML Estimated Elasticity: {res['factor_value']:.3f} (R2 = {res['r2']:.4f})")
    assert res['status'] == 'Used', "Status should be 'Used'"
    assert res['factor_value'] < 0.0, "Elasticity should be negative"
    assert abs(res['factor_value'] - (-1.5)) < 0.5, "Elasticity estimate should be close to -1.5"
    assert res['r2'] > 0.8, "R2 should be reasonably high (>0.8)"
    assert 'ci' in res, "Confidence interval should be present"
    print("✅ ML Elasticity Agent Test passed successfully!")

def test_ml_coordinator_and_projection():
    print("Running ML Coordinator Test...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mock_base_path = os.path.join(base_dir, "MockData", "electronics", "mobile_phone", "stable")
    assert os.path.exists(mock_base_path), "Mock data preset path does not exist."
    
    df_dict = {
        'sales': pd.read_csv(os.path.join(mock_base_path, 'sales_demand.csv')),
        'competitor': pd.read_csv(os.path.join(mock_base_path, 'competitor_pricing.csv')),
        'promotions': pd.read_csv(os.path.join(mock_base_path, 'marketing_promotions.csv')),
        'inventory': pd.read_csv(os.path.join(mock_base_path, 'inventory_status.csv')),
        'lifecycle': pd.read_csv(os.path.join(mock_base_path, 'product_lifecycle.csv')),
        'sentiment': pd.read_csv(os.path.join(mock_base_path, 'consumer_sentiment.csv'))
    }
    
    coordinator = MLCoordinatorAgent()
    summary = coordinator.run_analysis(df_dict)
    
    assert 'clean_sales' in summary
    assert 'factor_results' in summary
    assert 'weights' in summary
    assert 'joint_models' in summary
    
    # 1. Test Modular Projection
    proj_modular = coordinator.project_demand(summary, 10.0, is_promo_active=False, mode='modular')
    print(f"Modular projection: base_q={summary['q_base']:.1f}, new_q={proj_modular['q_new']:.1f}")
    assert proj_modular['q_new'] < summary['q_base'], "Demand should decrease with price increase in modular mode"
    assert proj_modular['p_new'] > summary['p_base'], "New price should be higher than base"
    
    # 2. Test Joint NN Projection
    proj_joint = coordinator.project_demand(summary, 10.0, is_promo_active=False, mode='joint')
    print(f"Joint NN projection: base_q={summary['q_base']:.1f}, new_q={proj_joint['q_new']:.1f}")
    assert proj_joint['q_new'] < summary['q_base'], "Demand should decrease with price increase in joint mode"
    assert proj_joint['p_new'] > summary['p_base'], "New price should be higher than base"
    
    print("✅ ML Coordinator and Projection Test passed successfully!")

def test_ml_fallback_behavior():
    print("Running ML Fallback Behavior Test...")
    # Generate sales data, leave other datasets as None
    dates = pd.date_range(start='2026-01-01', periods=20, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'sku_id': ['TST_ML'] * 20,
        'unit_price': np.linspace(10, 15, 20),
        'units_sold': np.linspace(100, 80, 20),
        'cost_per_unit': [5.0] * 20
    })
    
    df_dict = {
        'sales': df,
        'competitor': None,
        'promotions': None,
        'inventory': None,
        'lifecycle': None,
        'sentiment': None
    }
    
    coordinator = MLCoordinatorAgent()
    summary = coordinator.run_analysis(df_dict)
    
    # Test that running projection does not crash and defaults are applied
    proj_mod = coordinator.project_demand(summary, 5.0, mode='modular')
    proj_joint = coordinator.project_demand(summary, 5.0, mode='joint')
    
    assert proj_mod['q_new'] > 0
    assert proj_joint['q_new'] > 0
    print("✅ ML Fallback Behavior Test passed successfully!")

def run_all_ml_tests():
    print("--- STARTING MACHINE LEARNING DEMAND TESTS ---")
    test_ml_elasticity_agent()
    test_ml_coordinator_and_projection()
    test_ml_fallback_behavior()
    print("--- ALL ML TESTS PASSED SUCCESSFULLY! ---")
    return True

if __name__ == '__main__':
    run_all_ml_tests()
