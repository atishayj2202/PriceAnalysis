import numpy as np
import pandas as pd
import concurrent.futures
from utils.spike_detector import SpikeDetector
from agents.factor_agents import (
    ElasticityAgent, SeasonalityAgent, CompetitorAgent,
    PromoAgent, InventoryAgent, LifecycleAgent, SentimentAgent
)

class CoordinatorAgent:
    def __init__(self):
        self.spike_detector = SpikeDetector()
        
        self.agents = {
            'elasticity': ElasticityAgent(),
            'seasonality': SeasonalityAgent(),
            'competitor': CompetitorAgent(),
            'promotions': PromoAgent(),
            'inventory': InventoryAgent(),
            'lifecycle': LifecycleAgent(),
            'sentiment': SentimentAgent()
        }

    def run_analysis(self, df_dict):
        df_sales = df_dict.get('sales')
        if df_sales is None or len(df_sales) == 0:
            raise ValueError("Sales demand data is required for analysis.")
            
        df_clean = self.spike_detector.detect_and_classify_spikes(
            df_sales, 
            df_sent=df_dict.get('sentiment'), 
            df_promo=df_dict.get('promotions')
        )
        
        df_clean['rebaseline_weight_multiplier'] = 1.0
        last_rebaseline = df_clean['rebaseline_start'].dropna()
        if len(last_rebaseline) > 0:
            rebaseline_idx = int(last_rebaseline.iloc[-1])
            # Reduce weight of prior data to 0.2 instead of discarding
            df_clean.loc[:rebaseline_idx - 1, 'rebaseline_weight_multiplier'] = 0.2
            
        elasticity_result = self.agents['elasticity'].assess({'sales': df_clean})
        e_base = elasticity_result['factor_value']
        
        residuals = np.zeros(len(df_clean))
        mape = 15.0
        
        if 'model' in elasticity_result and elasticity_result['model'] is not None:
            model = elasticity_result['model']
            clean_idx = df_clean[df_clean['exclude_from_regression'] == False].index
            log_q_clean = np.log(df_clean.loc[clean_idx, 'units_sold'])
            fitted_clean = model.fittedvalues
            
            residuals_clean = log_q_clean - fitted_clean
            residuals[clean_idx] = residuals_clean.values
            
            q_actual = np.exp(log_q_clean)
            q_fitted = np.exp(fitted_clean)
            mape = np.mean(np.abs((q_actual - q_fitted) / q_actual)) * 100.0
            
            ci = model.conf_int().iloc[1]
            e_ci = (ci[0], ci[1])
        else:
            e_ci = (e_base - 0.3, e_base + 0.3)
            
        df_clean['residuals'] = residuals
        
        factor_results = {'elasticity': elasticity_result}
        dispatch_dict = {
            'sales': df_clean,
            'residuals': residuals,
            'competitor': df_dict.get('competitor'),
            'promotions': df_dict.get('promotions'),
            'inventory': df_dict.get('inventory'),
            'lifecycle': df_dict.get('lifecycle'),
            'sentiment': df_dict.get('sentiment')
        }
        
        agent_keys = ['seasonality', 'competitor', 'promotions', 'inventory', 'lifecycle', 'sentiment']
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.agents[key].assess, dispatch_dict): key 
                for key in agent_keys
            }
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    factor_results[key] = future.result()
                except Exception as ex:
                    factor_results[key] = {
                        'factor_value': self.agents[key].default_val,
                        'r2': 0.0,
                        'reliability': self.agents[key].reliability_score,
                        'status': 'Left Out (Error)',
                        'details': f"Parallel agent failed: {str(ex)}"
                    }
                    
        active_r2 = {}
        for key, res in factor_results.items():
            if 'Used' in res['status'] and res['r2'] > 0:
                active_r2[key] = res['r2']
            else:
                active_r2[key] = 0.0
                
        sum_r2 = sum(active_r2.values())
        weights = {}
        for key in factor_results.keys():
            if sum_r2 > 0:
                weights[key] = active_r2[key] / sum_r2
            else:
                weights[key] = 1.0 if key == 'elasticity' else 0.0
                
        last_4_weeks = df_clean.tail(4)
        p_base = last_4_weeks['unit_price'].mean()
        q_base = last_4_weeks['units_sold'].mean()
        cost_base = last_4_weeks['cost_per_unit'].mean()
        
        analysis_summary = {
            'clean_sales': df_clean,
            'factor_results': factor_results,
            'weights': weights,
            'e_base': e_base,
            'e_ci': e_ci,
            'mape': mape,
            'p_base': p_base,
            'q_base': q_base,
            'cost_base': cost_base
        }
        
        return analysis_summary

    def project_demand(self, summary, price_change_pct, is_promo_active=False):
        p_base = summary['p_base']
        q_base = summary['q_base']
        cost_base = summary['cost_base']
        e_base = summary['e_base']
        e_ci = summary['e_ci']
        results = summary['factor_results']
        
        p_new = p_base * (1.0 + price_change_pct / 100.0)
        
        C = results['competitor']['factor_value'] if 'Used' in results['competitor']['status'] else 1.0
        L = results['lifecycle']['factor_value'] if 'Used' in results['lifecycle']['status'] else 1.0
        X = results['sentiment']['factor_value'] if 'Used' in summary['factor_results']['sentiment']['status'] else 1.0
        
        S = results['seasonality']['factor_value'] if 'Used' in results['seasonality']['status'] else 1.0
        I = results['inventory']['factor_value'] if 'Used' in results['inventory']['status'] else 1.0
        
        lift_m = 0.0
        if is_promo_active:
            lift_m = results['promotions']['factor_value'] if 'Used' in results['promotions']['status'] else 0.0
            if lift_m == 0.0 and 'Used' not in results['promotions']['status']:
                lift_m = 0.1275 
                
        e_eff = e_base * C * L * X
        e_eff_low = e_ci[0] * C * L * X
        e_eff_high = e_ci[1] * C * L * X
        
        q_new = q_base * ((p_new / p_base) ** e_eff) * S * I * (1.0 + lift_m)
        
        q_low = q_base * ((p_new / p_base) ** e_eff_low) * S * I * (1.0 + lift_m)
        q_high = q_base * ((p_new / p_base) ** e_eff_high) * S * I * (1.0 + lift_m)
        
        q_p10 = min(q_new, q_low, q_high)
        q_p90 = max(q_new, q_low, q_high)
        q_p50 = q_new
        
        rev_base = p_base * q_base
        rev_new = p_new * q_new
        rev_increase = rev_new - rev_base
        rev_increase_pct = (rev_increase / rev_base) * 100.0 if rev_base > 0 else 0.0
        
        profit_base = (p_base - cost_base) * q_base
        profit_new = (p_new - cost_base) * q_new
        profit_increase = profit_new - profit_base
        profit_increase_pct = (profit_increase / profit_base) * 100.0 if profit_base > 0 else 0.0
        
        return {
            'p_new': p_new,
            'q_new': q_new,
            'q_p10': q_p10,
            'q_p50': q_p50,
            'q_p90': q_p90,
            'e_eff': e_eff,
            'rev_base': rev_base,
            'rev_new': rev_new,
            'rev_increase': rev_increase,
            'rev_increase_pct': rev_increase_pct,
            'profit_base': profit_base,
            'profit_new': profit_new,
            'profit_increase': profit_increase,
            'profit_increase_pct': profit_increase_pct
        }
