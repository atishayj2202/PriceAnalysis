import numpy as np
import pandas as pd
import concurrent.futures
from utils.spike_detector import SpikeDetector
from ml_agents.elasticity_agent import MLElasticityAgent
from ml_agents.factor_agents import (
    MLSeasonalityAgent, MLCompetitorAgent, MLPromoAgent,
    MLInventoryAgent, MLLifecycleAgent, MLSentimentAgent
)
from sklearn.neural_network import MLPRegressor

class MLCoordinatorAgent:
    def __init__(self):
        self.spike_detector = SpikeDetector()
        
        self.agents = {
            'elasticity': MLElasticityAgent(),
            'seasonality': MLSeasonalityAgent(),
            'competitor': MLCompetitorAgent(),
            'promotions': MLPromoAgent(),
            'inventory': MLInventoryAgent(),
            'lifecycle': MLLifecycleAgent(),
            'sentiment': MLSentimentAgent()
        }
        
        self.joint_ensemble_size = 5

    def _prepare_joint_features(self, df_clean, df_dict):
        """
        Merge and impute all datasets into a single feature dataframe for the joint model.
        """
        n_obs = len(df_clean)
        
        # 1. Base price and sales features
        df_feats = pd.DataFrame({
            'log_price': np.log(df_clean['unit_price'].values),
            'week': pd.to_datetime(df_clean['date']).dt.isocalendar().week.astype(float)
        }, index=df_clean.index)
        
        df_feats['sin_week'] = np.sin(2.0 * np.pi * df_feats['week'] / 52.0)
        df_feats['cos_week'] = np.cos(2.0 * np.pi * df_feats['week'] / 52.0)
        
        # 2. Competitor pricing
        df_comp = df_dict.get('competitor')
        if df_comp is not None and len(df_comp) == n_obs:
            gap = (df_clean['unit_price'].values - df_comp['comp_price_avg'].values) / np.maximum(0.1, df_comp['comp_price_avg'].values)
            df_feats['comp_gap'] = np.clip(gap, -0.5, 0.5)
        else:
            df_feats['comp_gap'] = 0.0
            
        # 3. Promotions
        df_promo = df_dict.get('promotions')
        if df_promo is not None and len(df_promo) == n_obs:
            df_feats['is_promo'] = df_promo['is_promo'].fillna(0).values
            df_feats['marketing_spend'] = df_promo['marketing_spend'].fillna(0.0).values
        else:
            df_feats['is_promo'] = 0.0
            df_feats['marketing_spend'] = 0.0
            
        # 4. Inventory
        df_inv = df_dict.get('inventory')
        if df_inv is not None and len(df_inv) == n_obs:
            coverage = df_inv['units_in_stock'].values / np.maximum(0.1, df_inv['avg_daily_sales_14d'].values)
            df_feats['coverage'] = np.clip(coverage, 0.0, 60.0)
        else:
            df_feats['coverage'] = 15.0  # neutral stock coverage
            
        # 5. Lifecycle
        df_life = df_dict.get('lifecycle')
        if df_life is not None and len(df_life) > 0:
            launch_date = pd.to_datetime(df_life['launch_date'].iloc[0])
            sales_dates = pd.to_datetime(df_clean['date'])
            ages_months = np.array([(d - launch_date).days / 30.4 for d in sales_dates])
            df_feats['age_months'] = ages_months
        else:
            df_feats['age_months'] = 24.0  # mature default
            
        # 6. Sentiment
        df_sent = df_dict.get('sentiment')
        if df_sent is not None and len(df_sent) == n_obs:
            curr = df_sent['cci_current'].values
            base = df_sent['cci_baseline'].values
            df_feats['sent_signal'] = (curr - base) / np.maximum(1.0, base)
            df_feats['trends_score'] = df_sent['google_trends_score'].fillna(50.0).values
        else:
            df_feats['sent_signal'] = 0.0
            df_feats['trends_score'] = 50.0
            
        return df_feats

    def run_analysis(self, df_dict, mlp_hidden_layers=(32, 16), mlp_solver='lbfgs', mlp_max_iter=1500):
        df_sales = df_dict.get('sales')
        if df_sales is None or len(df_sales) == 0:
            raise ValueError("Sales demand data is required for analysis.")
            
        df_clean = self.spike_detector.detect_and_classify_spikes(
            df_sales, 
            df_sent=df_dict.get('sentiment'), 
            df_promo=df_dict.get('promotions')
        )
        
        # Apply rebaseline adjustments (reduce weights of older data if break detected)
        df_clean['rebaseline_weight_multiplier'] = 1.0
        last_rebaseline = df_clean['rebaseline_start'].dropna()
        if len(last_rebaseline) > 0:
            rebaseline_idx = int(last_rebaseline.iloc[-1])
            df_clean.loc[:rebaseline_idx - 1, 'rebaseline_weight_multiplier'] = 0.2
            
        # ---------------------------------------------------------
        # ANALYSIS A: MODULAR ML PIPELINE
        # ---------------------------------------------------------
        # 1. Fit Elasticity Agent (MLP Neural Network)
        elasticity_result = self.agents['elasticity'].assess({'sales': df_clean})
        e_base = elasticity_result['factor_value']
        
        # Compute residuals from elasticity MLP prediction
        residuals = np.zeros(len(df_clean))
        mape = 15.0
        
        if 'fitted_log_q' in elasticity_result:
            clean_sales_est = elasticity_result['clean_sales']
            clean_idx_est = clean_sales_est.index
            log_q_clean = np.log(clean_sales_est['units_sold'])
            fitted_clean = elasticity_result['fitted_log_q']
            
            residuals_clean = log_q_clean - fitted_clean
            residuals[clean_idx_est] = residuals_clean.values
            
            q_actual = np.exp(log_q_clean)
            q_fitted = np.exp(fitted_clean)
            mape = float(np.mean(np.abs((q_actual - q_fitted) / q_actual)) * 100.0)
            
        df_clean['residuals'] = residuals
        
        # 2. Run other ML Factor agents in parallel
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
                        'details': f"Parallel ML agent failed: {str(ex)}"
                    }
                    
        # Dynamically normalize weights using R2 scores of used ML agents
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
                
        # ---------------------------------------------------------
        # ANALYSIS B: JOINT DEEP LEARNING (NEURAL NET) ENSEMBLE
        # ---------------------------------------------------------
        df_feats = self._prepare_joint_features(df_clean, df_dict)
        clean_idx = df_clean[df_clean['exclude_from_regression'] == False].index
        
        X_joint = df_feats.loc[clean_idx]
        y_joint = np.log(df_clean.loc[clean_idx, 'units_sold'].values)
        
        # Setup EMA/Time Weights for resampling
        max_date = pd.to_datetime(df_clean['date']).max()
        t_diff_weeks = (max_date - pd.to_datetime(df_clean['date'])).dt.days / 7.0
        time_weights = 0.95 ** t_diff_weeks
        if 'rebaseline_weight_multiplier' in df_clean.columns:
            time_weights *= df_clean['rebaseline_weight_multiplier'].fillna(1.0).values
        time_weights = time_weights.loc[clean_idx].values
        norm_weights = time_weights / np.sum(time_weights)
        
        joint_models = []
        fitted_joint_q_list = []
        
        np.random.seed(42)
        n_joint_obs = len(X_joint)
        
        if n_joint_obs >= 10:
            for i in range(self.joint_ensemble_size):
                indices = np.random.choice(n_joint_obs, size=n_joint_obs, replace=True, p=norm_weights)
                X_train = X_joint.iloc[indices]
                y_train = y_joint[indices]
                
                mlp = MLPRegressor(
                    hidden_layer_sizes=mlp_hidden_layers,
                    activation='tanh',
                    solver=mlp_solver,
                    max_iter=mlp_max_iter,
                    random_state=100 + i
                )
                mlp.fit(X_train, y_train)
                joint_models.append(mlp)
                
                # Predict on full unresampled data
                fitted_joint_q_list.append(mlp.predict(df_feats))
                
            # Compute fit metrics for joint model
            mean_joint_log_q = np.mean(fitted_joint_q_list, axis=0)
            ss_tot = np.sum((np.log(df_clean['units_sold'].values) - np.mean(np.log(df_clean['units_sold'].values))) ** 2)
            ss_res = np.sum((np.log(df_clean['units_sold'].values) - mean_joint_log_q) ** 2)
            joint_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            joint_mape = float(np.mean(np.abs((df_clean['units_sold'].values - np.exp(mean_joint_log_q)) / df_clean['units_sold'].values)) * 100.0)
        else:
            joint_models = []
            joint_r2 = 0.0
            joint_mape = 20.0
            mean_joint_log_q = np.zeros(len(df_clean))

        # Base properties
        last_4_weeks = df_clean.tail(4)
        p_base = float(last_4_weeks['unit_price'].mean())
        q_base = float(last_4_weeks['units_sold'].mean())
        cost_base = float(last_4_weeks['cost_per_unit'].mean())
        
        # Build baseline feature state based on last row
        base_features = df_feats.iloc[-1].copy()
        
        analysis_summary = {
            'clean_sales': df_clean,
            'factor_results': factor_results,
            'weights': weights,
            'e_base': e_base,
            'e_ci': elasticity_result.get('ci', (e_base - 0.3, e_base + 0.3)),
            'mape': mape,
            'p_base': p_base,
            'q_base': q_base,
            'cost_base': cost_base,
            # Joint model artifacts
            'joint_models': joint_models,
            'joint_r2': joint_r2,
            'joint_mape': joint_mape,
            'base_features': base_features,
            'df_feats': df_feats,
            'fitted_joint_log_q': mean_joint_log_q,
            'df_dict': df_dict
        }
        
        return analysis_summary

    def project_demand(self, summary, price_change_pct, is_promo_active=False, mode='modular'):
        """
        Projects demand, revenue, and profit.
        Supports:
          - mode='modular': Evaluates separate ML agents and combines predictions.
          - mode='joint': Evaluates the joint deep learning neural net ensemble.
        """
        p_base = summary['p_base']
        q_base = summary['q_base']
        cost_base = summary['cost_base']
        
        p_new = p_base * (1.0 + price_change_pct / 100.0)
        
        if mode == 'modular':
            e_base = summary['e_base']
            e_ci = summary['e_ci']
            results = summary['factor_results']
            
            C = results['competitor']['factor_value'] if 'Used' in results['competitor']['status'] else 1.0
            L = results['lifecycle']['factor_value'] if 'Used' in results['lifecycle']['status'] else 1.0
            X = results['sentiment']['factor_value'] if 'Used' in results['sentiment']['status'] else 1.0
            S = results['seasonality']['factor_value'] if 'Used' in results['seasonality']['status'] else 1.0
            I = results['inventory']['factor_value'] if 'Used' in results['inventory']['status'] else 1.0
            
            lift_m = 0.0
            if is_promo_active:
                lift_m = results['promotions']['factor_value'] if 'Used' in results['promotions']['status'] else 0.0
                if lift_m == 0.0 and 'Used' not in results['promotions']['status']:
                    lift_m = 0.1275  # default promo lift if data absent
                    
            e_eff = e_base * C * L * X
            e_eff_low = e_ci[0] * C * L * X
            e_eff_high = e_ci[1] * C * L * X
            
            q_new = q_base * ((p_new / p_base) ** e_eff) * S * I * (1.0 + lift_m)
            q_low = q_base * ((p_new / p_base) ** e_eff_low) * S * I * (1.0 + lift_m)
            q_high = q_base * ((p_new / p_base) ** e_eff_high) * S * I * (1.0 + lift_m)
            
            q_p10 = float(min(q_new, q_low, q_high))
            q_p90 = float(max(q_new, q_low, q_high))
            q_p50 = float(q_new)
            
        else: # joint model mode
            joint_models = summary['joint_models']
            base_features = summary['base_features'].copy()
            
            # Update target price in features
            base_features['log_price'] = np.log(p_new)
            if is_promo_active:
                base_features['is_promo'] = 1.0
                # If promotional spend is recorded, set to historical average promo spend, else default
                df_dict = summary['df_dict']
                df_promo = df_dict.get('promotions')
                if df_promo is not None and len(df_promo) > 0:
                    base_features['marketing_spend'] = float(df_promo[df_promo['is_promo'] == 1]['marketing_spend'].mean())
                else:
                    base_features['marketing_spend'] = 200.0  # proxy marketing spend
            else:
                base_features['is_promo'] = 0.0
                base_features['marketing_spend'] = 0.0
                
            # Reshape features for prediction
            X_sim = base_features.values.reshape(1, -1)
            
            # Predict log demand from ensemble
            predictions = []
            for mlp in joint_models:
                predictions.append(mlp.predict(X_sim)[0])
                
            q_new_log = np.mean(predictions)
            q_std_log = np.std(predictions)
            
            q_new = float(np.exp(q_new_log))
            
            # P10 / P90 demand prediction interval from ensemble std dev
            q_p10 = float(np.exp(q_new_log - 1.28 * q_std_log))
            q_p90 = float(np.exp(q_new_log + 1.28 * q_std_log))
            q_p50 = q_new
            
            # Numerical elasticity at p_new in joint model
            h = 1e-4
            log_p_new = np.log(p_new)
            
            e_eff_list = []
            for mlp in joint_models:
                feat_plus = base_features.copy()
                feat_plus['log_price'] = log_p_new + h
                pred_plus = mlp.predict(feat_plus.values.reshape(1, -1))[0]
                
                feat_minus = base_features.copy()
                feat_minus['log_price'] = log_p_new - h
                pred_minus = mlp.predict(feat_minus.values.reshape(1, -1))[0]
                
                e_mlp = (pred_plus - pred_minus) / (2.0 * h)
                e_eff_list.append(e_mlp)
                
            e_eff = float(np.mean(e_eff_list))
            
            # Mock modifiers for display consistency with UI table
            C, L, X, S, I, lift_m = 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
            
        # Finance metrics calculations
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
            'C': C, 'L': L, 'X': X, 'S': S, 'I': I,
            'lift_m': lift_m,
            'rev_base': rev_base,
            'rev_new': rev_new,
            'rev_increase': rev_increase,
            'rev_increase_pct': rev_increase_pct,
            'profit_base': profit_base,
            'profit_new': profit_new,
            'profit_increase': profit_increase,
            'profit_increase_pct': profit_increase_pct
        }
