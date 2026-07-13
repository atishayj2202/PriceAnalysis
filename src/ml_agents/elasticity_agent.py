import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from ml_agents.base_agent import BaseAgent

class MLElasticityAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ML Price Elasticity", reliability_score="90% HIGH", default_val=-1.5)
        # Use an ensemble of MLPs to estimate both prediction and uncertainty (elasticity distribution)
        self.n_estimators = 10
        self.h = 1e-4  # finite difference step size

    def assess(self, data_dict):
        df_sales = data_dict.get('sales')
        if df_sales is None or len(df_sales) == 0:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': self.reliability_score,
                'status': 'Left Out (No Data)',
                'details': "Sales data is completely missing. Using default ML proxy."
            }

        # Filter out rows marked for exclusion (Type A spikes) and zero/negative sales/prices
        clean_sales = df_sales[
            (df_sales['exclude_from_regression'] == False) & 
            (df_sales['units_sold'] > 0) & 
            (df_sales['unit_price'] > 0)
        ].copy()
            
        n_obs = len(clean_sales)
        if n_obs < 10:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': 'PROVISIONAL',
                'status': 'Used (Proxy)',
                'details': f"Insufficient clean observations ({n_obs} < 10). Using default ML proxy of {self.default_val}."
            }

        try:
            # Calculate time-decay weights (24-week EMA equivalent decay)
            max_date = pd.to_datetime(clean_sales['date']).max()
            t_diff_weeks = (max_date - pd.to_datetime(clean_sales['date'])).dt.days / 7.0
            weights = 0.9200 ** t_diff_weeks
            if 'rebaseline_weight_multiplier' in clean_sales.columns:
                weights *= clean_sales['rebaseline_weight_multiplier'].fillna(1.0).values

            log_p = np.log(clean_sales['unit_price'].values).reshape(-1, 1)
            log_q = np.log(clean_sales['units_sold'].values)

            # Target base price to evaluate local elasticity
            last_4_weeks = clean_sales.tail(4)
            p_base = last_4_weeks['unit_price'].mean()
            log_p_base = np.log(p_base)

            # Fit bootstrap ensemble of MLP Neural Networks
            models = []
            elasticity_vals = []
            fitted_values_list = []
            
            np.random.seed(42)
            normalized_weights = weights / np.sum(weights)
            
            for i in range(self.n_estimators):
                # Resample with replacement according to time weights
                indices = np.random.choice(n_obs, size=n_obs, replace=True, p=normalized_weights)
                X_train = log_p[indices]
                y_train = log_q[indices]

                # Neural Network definition
                mlp = MLPRegressor(
                    hidden_layer_sizes=(16, 8),
                    activation='tanh',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42 + i
                )
                mlp.fit(X_train, y_train)
                models.append(mlp)

                # Compute local elasticity at log_p_base using finite difference (gradient)
                # e = d(log_q) / d(log_p)
                pred_plus = mlp.predict(np.array([[log_p_base + self.h]]))[0]
                pred_minus = mlp.predict(np.array([[log_p_base - self.h]]))[0]
                e_i = (pred_plus - pred_minus) / (2 * self.h)
                
                # Check for extreme outlier elasticity and clamp to sensible range for stability
                e_i = max(-10.0, min(2.0, e_i))
                elasticity_vals.append(e_i)

                # Predict on full unresampled data
                fitted_values_list.append(mlp.predict(log_p))

            # Ensemble aggregation
            e_estimate = float(np.mean(elasticity_vals))
            e_std = float(np.std(elasticity_vals))
            
            # Predict mean log quantity on unresampled data
            mean_fitted_log_q = np.mean(fitted_values_list, axis=0)
            
            # Compute R^2 of ensemble mean predictions
            ss_tot = np.sum((log_q - np.mean(log_q)) ** 2)
            ss_res = np.sum((log_q - mean_fitted_log_q) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # CI boundaries (1 standard deviation for P10-P90 range representation)
            ci_low = e_estimate - 1.28 * e_std
            ci_high = e_estimate + 1.28 * e_std
            
            reliability = self.reliability_score
            details = f"Neural Net estimated local elasticity: {e_estimate:.3f} (R² = {r2:.4f}, Obs = {n_obs})"
            
            status = 'Used'
            if n_obs < 30:
                reliability = 'PROVISIONAL'
                details += " - Warning: thin data (< 30 observations)."
            if abs(e_estimate) > 5.0 or (ci_high - ci_low) > 1.5:
                reliability = 'PROVISIONAL (High Uncertainty)'

            return {
                'factor_value': e_estimate,
                'r2': r2,
                'reliability': reliability,
                'status': status,
                'details': details,
                'ci': (ci_low, ci_high),
                'models': models,
                'fitted_log_q': mean_fitted_log_q,
                'clean_sales': clean_sales
            }
        except Exception as e:
            return {
                'factor_value': self.default_val,
                'r2': 0.0,
                'reliability': 'PROVISIONAL',
                'status': 'Used (Proxy)',
                'details': f"MLP fitting failed: {str(e)}. Using default ML proxy."
            }
