import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from ml_agents.base_agent import BaseAgent

class MLSeasonalityAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Seasonality", reliability_score="85% HIGH", default_val=1.0)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        residuals = data_dict.get('residuals')
        
        if df_sales is None or residuals is None or len(df_sales) < 52:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out',
                'details': "Fewer than 1 full cycle (52 weeks) available. Seasonality set to 1.0."
            }
            
        try:
            df_res = pd.DataFrame({
                'residuals': residuals,
                'date': pd.to_datetime(df_sales['date'].values)
            })
            df_res['week'] = df_res['date'].dt.isocalendar().week
            
            # Feature engineering: Sine and Cosine components for annual seasonality, and week number
            df_res['sin_52'] = np.sin(2.0 * np.pi * df_res['week'] / 52.0)
            df_res['cos_52'] = np.cos(2.0 * np.pi * df_res['week'] / 52.0)
            
            X = df_res[['sin_52', 'cos_52', 'week']]
            y = df_res['residuals']
            
            # Incorporate time weights by resampling training data
            max_date = pd.to_datetime(df_res['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_res['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_res), size=len(df_res), replace=True, p=norm_weights)
            
            self.model.fit(X.iloc[indices], y.iloc[indices])
            
            # R2 score calculation
            pred_y = self.model.predict(X)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            if r2 < 0.05:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"ML R² ({r2:.4f}) is below seasonal threshold (0.05). Excluded."
                }
                
            # Predict for all 52 weeks to populate week_averages for plotting
            week_nums = np.arange(1, 53)
            sin_vals = np.sin(2.0 * np.pi * week_nums / 52.0)
            cos_vals = np.cos(2.0 * np.pi * week_nums / 52.0)
            X_plot = pd.DataFrame({
                'sin_52': sin_vals,
                'cos_52': cos_vals,
                'week': week_nums
            })
            pred_residuals = self.model.predict(X_plot)
            week_averages = pd.Series(np.exp(pred_residuals), index=week_nums)
            
            last_week = int(df_res['week'].iloc[-1])
            last_sin = df_res['sin_52'].iloc[-1]
            last_cos = df_res['cos_52'].iloc[-1]
            
            current_pred = self.model.predict(np.array([[last_sin, last_cos, last_week]]))[0]
            s_val = np.exp(current_pred)
            s_val = float(max(0.3, min(3.0, s_val)))
            
            return {
                'factor_value': s_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"ML Seasonality factor for week {last_week}: {s_val:.3f} (R² = {r2:.4f})",
                'week_averages': week_averages
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Seasonality computation failed: {str(e)}."
            }


class MLCompetitorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Competitor Pricing", reliability_score="60% MEDIUM", default_val=1.0)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        df_comp = data_dict.get('competitor')
        residuals = data_dict.get('residuals')
        
        if df_comp is None or df_sales is None or residuals is None:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Missing Data)',
                'details': "Competitor price log unavailable. Neutral assumption applied."
            }
            
        try:
            own_p = df_sales['unit_price'].values
            comp_p = df_comp['comp_price_avg'].values
            
            gap = (own_p - comp_p) / comp_p
            gap_clamped = np.clip(gap, -0.5, 0.5)
            
            X = gap_clamped.reshape(-1, 1)
            y = residuals
            
            max_date = pd.to_datetime(df_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_sales['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_sales), size=len(df_sales), replace=True, p=norm_weights)
            
            self.model.fit(X[indices], y[indices])
            pred_y = self.model.predict(X)
            
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            current_own = own_p[-1]
            current_comp = comp_p[-1]
            current_gap = (current_own - current_comp) / current_comp
            current_gap_clamped = float(np.clip(current_gap, -0.5, 0.5))
            
            # Predict residual effect using Gradient Boosting
            pred_res_effect = self.model.predict(np.array([[current_gap_clamped]]))[0]
            
            # Map predicted residual to elasticity multiplier C
            # Negative residuals (volume drop due to high relative price) map to higher elasticity multiplier (steeper slope)
            c_val = float(1.0 + 0.2 * np.sign(current_gap_clamped) * min(abs(current_gap_clamped), 0.5) - 0.1 * pred_res_effect)
            c_val = max(0.5, min(2.0, c_val))
            
            return {
                'factor_value': c_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"ML Competitor Gap: {current_gap*100:+.1f}%. Predicted residual: {pred_res_effect:+.4f}. Modifier: {c_val:.3f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Competitor assessment failed: {str(e)}."
            }


class MLPromoAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Promotions / Marketing", reliability_score="70% HIGH", default_val=0.0)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=2, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        df_promo = data_dict.get('promotions')
        residuals = data_dict.get('residuals')
        
        if df_promo is None or df_sales is None or residuals is None:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Missing Data)',
                'details': "Promotions data is unavailable. No promotions active."
            }
            
        try:
            promo_flag = df_promo['is_promo'].values
            marketing_spend = df_promo['marketing_spend'].fillna(0.0).values
            
            X = pd.DataFrame({
                'is_promo': promo_flag,
                'marketing_spend': marketing_spend
            })
            y = residuals
            
            max_date = pd.to_datetime(df_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_sales['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_sales), size=len(df_sales), replace=True, p=norm_weights)
            
            self.model.fit(X.iloc[indices], y[indices])
            pred_y = self.model.predict(X)
            
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            # Predict the promotion lift using ML model
            current_promo = int(promo_flag[-1])
            current_spend = float(marketing_spend[-1])
            
            if current_promo == 1:
                # Compare predicted residual at current promo/spend vs baseline (no promo, no spend)
                pred_promo_res = self.model.predict(pd.DataFrame({'is_promo': [1], 'marketing_spend': [current_spend]}))[0]
                pred_base_res = self.model.predict(pd.DataFrame({'is_promo': [0], 'marketing_spend': [0.0]}))[0]
                lift_m = np.exp(pred_promo_res - pred_base_res) - 1.0
                lift_m = float(max(0.0, lift_m))
            else:
                lift_m = 0.0
                
            return {
                'factor_value': lift_m,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"ML Promo Lift: {lift_m*100:.1f}%. Spend: ${current_spend:.2f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Promotions assessment failed: {str(e)}."
            }


class MLInventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Inventory Signal", reliability_score="65% MEDIUM", default_val=1.0)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        df_inv = data_dict.get('inventory')
        residuals = data_dict.get('residuals')
        
        if df_inv is None or df_sales is None or residuals is None:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Missing Data)',
                'details': "Inventory status data unavailable. Normal stock assumed."
            }
            
        try:
            stock_level = df_inv['units_in_stock'].values
            daily_sales = df_inv['avg_daily_sales_14d'].values
            
            coverage = stock_level / np.maximum(0.1, daily_sales)
            
            X = pd.DataFrame({
                'units_in_stock': stock_level,
                'avg_daily_sales_14d': daily_sales,
                'coverage': coverage
            })
            y = residuals
            
            max_date = pd.to_datetime(df_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_sales['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_sales), size=len(df_sales), replace=True, p=norm_weights)
            
            self.model.fit(X.iloc[indices], y[indices])
            pred_y = self.model.predict(X)
            
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            if r2 < 0.03:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"ML R² ({r2:.4f}) below inventory threshold (0.03). Excluded."
                }
                
            current_stock = float(stock_level[-1])
            current_sales = float(daily_sales[-1])
            current_coverage = float(coverage[-1])
            
            # Predict residual effect
            pred_effect = self.model.predict(pd.DataFrame({
                'units_in_stock': [current_stock],
                'avg_daily_sales_14d': [current_sales],
                'coverage': [current_coverage]
            }))[0]
            
            # Scarcity multiplier is typically > 1.0 under low stock due to higher relative demand or urgency
            if current_coverage < 7:
                i_val = float(1.15 + 0.1 * max(0.0, pred_effect))
            else:
                i_val = 1.0
                
            return {
                'factor_value': i_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"ML Stock Coverage: {current_coverage:.1f} days. Multiplier: {i_val:.2f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Inventory assessment failed: {str(e)}."
            }


class MLLifecycleAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Product Lifecycle", reliability_score="65% MEDIUM", default_val=1.0)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        df_life = data_dict.get('lifecycle')
        residuals = data_dict.get('residuals')
        
        if df_life is None or df_sales is None or residuals is None:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Missing Data)',
                'details': "Product launch metadata unavailable. Mature product assumed."
            }
            
        try:
            launch_date_str = df_life['launch_date'].iloc[0]
            launch_date = pd.to_datetime(launch_date_str)
            sales_dates = pd.to_datetime(df_sales['date'])
            
            ages_months = np.array([(d - launch_date).days / 30.4 for d in sales_dates])
            
            phases = []
            for age in ages_months:
                if age < 6:
                    phases.append(0)  # Launch
                elif age < 18:
                    phases.append(1)  # Growth
                elif age < 36:
                    phases.append(2)  # Mature
                else:
                    phases.append(3)  # Decline
                    
            X = pd.DataFrame({
                'age_months': ages_months,
                'phase': phases
            })
            y = residuals
            
            max_date = pd.to_datetime(df_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_sales['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_sales), size=len(df_sales), replace=True, p=norm_weights)
            
            self.model.fit(X.iloc[indices], y)
            pred_y = self.model.predict(X)
            
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            current_age = float(ages_months[-1])
            if current_age < 6:
                l_val = 0.70
                phase_name = 'Launch'
            elif current_age < 18:
                l_val = 0.85
                phase_name = 'Growth'
            elif current_age < 36:
                l_val = 1.00
                phase_name = 'Mature'
            else:
                l_val = 1.20
                phase_name = 'Decline'
                
            # ML adjustments based on residual predictions
            pred_phase_res = self.model.predict(pd.DataFrame({'age_months': [current_age], 'phase': [phases[-1]]}))[0]
            l_val = float(l_val * (1.0 - 0.1 * pred_phase_res))
            l_val = max(0.5, min(2.0, l_val))
            
            return {
                'factor_value': l_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"ML Age: {current_age:.1f} months. Phase: {phase_name}. Modifier: {l_val:.2f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Lifecycle assessment failed: {str(e)}."
            }


class MLSentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Consumer Sentiment", reliability_score="50% MEDIUM", default_val=1.0)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        df_sent = data_dict.get('sentiment')
        residuals = data_dict.get('residuals')
        
        if df_sent is None or df_sales is None or residuals is None:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Missing Data)',
                'details': "Consumer confidence index data unavailable. Neutral sentiment assumed."
            }
            
        try:
            cci_curr = df_sent['cci_current'].values
            cci_base = df_sent['cci_baseline'].values
            trends_score = df_sent['google_trends_score'].fillna(50.0).values
            
            sent_signal = (cci_curr - cci_base) / cci_base
            
            X = pd.DataFrame({
                'sent_signal': sent_signal,
                'google_trends_score': trends_score
            })
            y = residuals
            
            max_date = pd.to_datetime(df_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(df_sales['date'])).dt.days / 7.0
            weights = 0.9868 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in df_sales.columns:
                weights *= df_sales['rebaseline_weight_multiplier'].fillna(1.0).values
                
            np.random.seed(42)
            norm_weights = weights / np.sum(weights)
            indices = np.random.choice(len(df_sales), size=len(df_sales), replace=True, p=norm_weights)
            
            self.model.fit(X.iloc[indices], y)
            pred_y = self.model.predict(X)
            
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_y) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            if r2 < 0.05:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"ML R² ({r2:.4f}) below sentiment threshold (0.05). Excluded."
                }
                
            current_curr = float(cci_curr[-1])
            current_base = float(cci_base[-1])
            current_signal = (current_curr - current_base) / current_base
            current_trends = float(trends_score[-1])
            
            # Predict residual effect
            pred_effect = self.model.predict(pd.DataFrame({
                'sent_signal': [current_signal],
                'google_trends_score': [current_trends]
            }))[0]
            
            # Map sentiment + trends to elasticity multiplier X
            # High sentiment & trends -> flatter elasticity (less price sensitive), multiplier X < 1.0
            # Low sentiment -> steeper elasticity, multiplier X > 1.0
            x_val = float(1.0 + 0.1 * current_signal - 0.05 * pred_effect)
            
            # Hard stop condition evaluated inside agent logic
            if len(cci_curr) >= 5:
                one_month_drop = cci_curr[-5] - cci_curr[-1]
                if one_month_drop > 10.0:
                    x_val = max(0.97, x_val)
                    details = f"CCI Dropped by {one_month_drop:.1f} pts in last month. Floored X at 0.97 (CCI={current_curr:.1f}, R² = {r2:.4f})"
                else:
                    details = f"CCI: {current_curr:.1f}. Trends: {current_trends:.1f}. Sentiment modifier: {x_val:.3f} (R² = {r2:.4f})"
            else:
                details = f"CCI: {current_curr:.1f}. Trends: {current_trends:.1f}. Sentiment modifier: {x_val:.3f} (R² = {r2:.4f})"
                
            return {
                'factor_value': x_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': details
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"ML Sentiment assessment failed: {str(e)}."
            }
