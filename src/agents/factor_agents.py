import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.agents.base_agent import BaseAgent

class ElasticityAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Price Elasticity", reliability_score="88% HIGH", default_val=-1.5)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        if df_sales is None or len(df_sales) == 0:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (No Data)',
                'details': "Sales data is completely missing. Using category proxy."
            }

        # Filter out rows marked for exclusion (Type A spikes)
        clean_sales = df_sales[df_sales['exclude_from_regression'] == False]
            
        n_obs = len(clean_sales)
        if n_obs < 10:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': 'PROVISIONAL',
                'status': 'Used (Proxy)',
                'details': f"Insufficient clean observations ({n_obs} < 10). Using category proxy of {self.default_val}."
            }

        try:
            log_q = np.log(clean_sales['units_sold'])
            log_p = np.log(clean_sales['unit_price'])
            X = sm.add_constant(log_p)
            model = sm.OLS(log_q, X).fit()
            
            e_estimate = model.params.iloc[1] if len(model.params) > 1 else self.default_val
            r2 = model.rsquared
            
            ci = model.conf_int().iloc[1]
            ci_width = ci[1] - ci[0]
            
            reliability = self.reliability_score
            details = f"OLS estimated elasticity: {e_estimate:.3f} (R² = {r2:.4f}, Obs = {n_obs})"
            
            status = 'Used'
            if n_obs < 30:
                reliability = 'PROVISIONAL'
                details += " - Warning: thin data (< 30 observations)."
            if abs(e_estimate) > 5.0 or ci_width > 1.5:
                reliability = 'PROVISIONAL (High Uncertainty)'
                
            return {
                'factor_value': e_estimate,
                'r2': r2,
                'reliability': reliability,
                'status': status,
                'details': details,
                'model': model
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': 'PROVISIONAL',
                'status': 'Used (Proxy)',
                'details': f"OLS regression failed: {str(e)}. Using default proxy."
            }


class SeasonalityAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Seasonality", reliability_score="82% HIGH", default_val=1.0)

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        residuals = data_dict.get('residuals')
        
        if df_sales is None or residuals is None or len(df_sales) < 104:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out',
                'details': "Fewer than 2 full cycles (104 weeks) available. Seasonality set to 1.0."
            }
            
        try:
            df_res = pd.DataFrame({
                'residuals': residuals,
                'date': pd.to_datetime(df_sales['date'].values)
            })
            df_res['week'] = df_res['date'].dt.isocalendar().week
            
            week_averages = df_res.groupby('week')['residuals'].mean()
            
            dummies = pd.get_dummies(df_res['week'], drop_first=True, dtype=float)
            X = sm.add_constant(dummies)
            model = sm.OLS(df_res['residuals'], X).fit()
            r2 = model.rsquared
            
            if r2 < 0.05:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"R² ({r2:.4f}) is below seasonal threshold (0.05). Excluded."
                }
                
            last_week = df_res['week'].iloc[-1]
            s_val = np.exp(week_averages.get(last_week, 0.0))
            s_val = max(0.3, min(3.0, s_val))
            
            return {
                'factor_value': s_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"Seasonality factor for week {last_week}: {s_val:.3f} (R² = {r2:.4f})",
                'week_averages': week_averages
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"Seasonality computation failed: {str(e)}."
            }


class CompetitorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Competitor Pricing", reliability_score="52% MEDIUM", default_val=1.0)

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
            
            X = sm.add_constant(gap_clamped)
            model = sm.OLS(residuals, X).fit()
            r2 = model.rsquared
            
            current_own = own_p[-1]
            current_comp = comp_p[-1]
            current_gap = (current_own - current_comp) / current_comp
            current_gap_clamped = np.clip(current_gap, -0.5, 0.5)
            
            c_val = 1.0 + 0.2 * np.sign(current_gap_clamped) * min(abs(current_gap_clamped), 0.5)
            
            return {
                'factor_value': c_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"Competitor Gap: {current_gap*100:+.1f}%. Modifier: {c_val:.3f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"Competitor assessment failed: {str(e)}."
            }


class PromoAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Promotions / Marketing", reliability_score="65% MEDIUM", default_val=0.0)

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
            
            X = sm.add_constant(promo_flag)
            model = sm.OLS(residuals, X).fit()
            r2 = model.rsquared
            
            b_promo = model.params.iloc[1] if len(model.params) > 1 else 0.0
            lift_m = np.exp(b_promo) - 1.0
            lift_m = max(0.0, lift_m)
            
            current_promo = int(promo_flag[-1])
            
            return {
                'factor_value': lift_m if current_promo == 1 else 0.0,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"Promo Lift: {lift_m*100:.1f}%. Currently active: {'Yes' if current_promo == 1 else 'No'} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"Promotions assessment failed: {str(e)}."
            }


class InventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Inventory Signal", reliability_score="60% MEDIUM", default_val=1.0)

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
            low_stock_dummy = (coverage < 7).astype(float)
            
            X = sm.add_constant(low_stock_dummy)
            model = sm.OLS(residuals, X).fit()
            r2 = model.rsquared
            
            if r2 < 0.03:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"R² ({r2:.4f}) below inventory threshold (0.03). Excluded."
                }
                
            current_coverage = coverage[-1]
            if current_coverage < 7:
                i_val = 1.15
            else:
                i_val = 1.0
                
            return {
                'factor_value': i_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"Current stock coverage: {current_coverage:.1f} days. Multiplier: {i_val:.2f} (R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"Inventory assessment failed: {str(e)}."
            }


class LifecycleAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Product Lifecycle", reliability_score="58% MEDIUM", default_val=1.0)

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
            
            ages_months = [(d - launch_date).days / 30.4 for d in sales_dates]
            
            phases = []
            for age in ages_months:
                if age < 6:
                    phases.append('launch')
                elif age < 18:
                    phases.append('growth')
                elif age < 36:
                    phases.append('mature')
                else:
                    phases.append('decline')
                    
            df_phases = pd.get_dummies(phases, drop_first=True, dtype=float)
            if df_phases.shape[1] > 0:
                X = sm.add_constant(df_phases)
                model = sm.OLS(residuals, X).fit()
                r2 = model.rsquared
            else:
                r2 = 0.0
                
            current_age = ages_months[-1]
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
                
            return {
                'factor_value': l_val,
                'r2': r2,
                'reliability': self.reliability_score,
                'status': 'Used',
                'details': f"Age: {current_age:.1f} months. Phase: {phase_name} (L = {l_val:.2f}, R² = {r2:.4f})"
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (Error)',
                'details': f"Lifecycle assessment failed: {str(e)}."
            }


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Consumer Sentiment", reliability_score="38% LOW", default_val=1.0)

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
            
            sent_signal = (cci_curr - cci_base) / cci_base
            
            X = sm.add_constant(sent_signal)
            model = sm.OLS(residuals, X).fit()
            r2 = model.rsquared
            
            if r2 < 0.05:
                return {
                    'factor_value': self.default_val,
                    'r2': r2,
                    'reliability': self.reliability_score,
                    'status': 'Left Out (Low Explanatory Power)',
                    'details': f"R² ({r2:.4f}) below sentiment threshold (0.05). Excluded."
                }
                
            current_curr = cci_curr[-1]
            current_base = cci_base[-1]
            current_signal = (current_curr - current_base) / current_base
            
            x_val = 1.0 + 0.1 * current_signal
            
            if len(cci_curr) >= 5:
                one_month_drop = cci_curr[-5] - cci_curr[-1]
                if one_month_drop > 10.0:
                    x_val = max(0.97, x_val)
                    details = f"CCI Dropped by {one_month_drop:.1f} pts in last month. Floored X at 0.97 (CCI={current_curr:.1f}, R² = {r2:.4f})"
                else:
                    details = f"CCI: {current_curr:.1f}. Sentiment modifier: {x_val:.3f} (R² = {r2:.4f})"
            else:
                details = f"CCI: {current_curr:.1f}. Sentiment modifier: {x_val:.3f} (R² = {r2:.4f})"
                
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
                'details': f"Sentiment assessment failed: {str(e)}."
            }
