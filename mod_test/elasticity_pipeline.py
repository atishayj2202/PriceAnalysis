import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

class SeasonalityEngine:
    """Implements 3 ways of Fourier seasonality (sin/cos) modeling"""
    
    @staticmethod
    def create_fourier_features(t, period, K):
        features = {}
        for k in range(1, K + 1):
            features[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
            features[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
        return pd.DataFrame(features)

    @classmethod
    def get_features(cls, n_samples, approach="fixed"):
        t = np.arange(n_samples)
        
        if approach == "fixed":
            # Way 1: Fixed annual seasonality (K=2, 52-week cycle)
            return cls.create_fourier_features(t, period=52, K=2)
            
        elif approach == "multi_period":
            # Way 3: Multi-period seasonality (Annual K=2 + Quarterly K=1)
            f_annual = cls.create_fourier_features(t, period=52, K=2)
            f_quarterly = cls.create_fourier_features(t, period=13, K=1)
            return pd.concat([f_annual, f_quarterly], axis=1)
            
        else:  # "adaptive"
            # Way 2: AICc-optimized annual seasonality (test K=1..5 and find best fit on training data)
            # Will be computed dynamically during training window
            return None

    @classmethod
    def get_adaptive_features(cls, log_q, log_p, competitor_feats, promo_feat, weights=None):
        """Finds the optimal Fourier order K for yearly seasonality using AICc"""
        n = len(log_q)
        best_k = 1
        best_aicc = float('inf')
        
        t = np.arange(n)
        for K in range(1, 6):
            fourier_df = cls.create_fourier_features(t, period=52, K=K)
            X = pd.concat([log_p, competitor_feats, promo_feat, fourier_df], axis=1)
            X = sm.add_constant(X)
            
            if weights is not None:
                model = sm.WLS(log_q, X, weights=weights).fit()
            else:
                model = sm.OLS(log_q, X).fit()
                
            # Compute AICc
            k_params = len(model.params)
            aic = model.aic
            if n - k_params - 1 > 0:
                aicc = aic + (2 * k_params * (k_params + 1)) / (n - k_params - 1)
            else:
                aicc = aic
                
            if aicc < best_aicc:
                best_aicc = aicc
                best_k = K
                
        # Return fourier features with best K
        return cls.create_fourier_features(t, period=52, K=best_k), best_k

class ElasticityPipeline:
    def __init__(self, data_path, brand):
        self.data_path = data_path
        self.brand = brand
        self.df = None
        self.load_data()
        
    def load_data(self):
        df_full = pd.read_csv(self.data_path)
        self.df = df_full[df_full['brand'] == self.brand].reset_index(drop=True)
        # Create log variables
        self.df['log_q'] = np.log(self.df['units_sold'])
        self.df['log_p'] = np.log(self.df['unit_price'])
        self.df['log_comp_1'] = np.log(self.df['comp_price_1'])
        self.df['log_comp_2'] = np.log(self.df['comp_price_2'])
        self.df['log_cost'] = np.log(self.df['cost_per_unit'])
        
    def fit_model(self, model_name, X_train, y_train, weights=None, X_train_first_stage=None):
        """Fits model and returns fitted object, coefficient/elasticity, and predicted log_q"""
        # Exclude constant from ML models
        has_const = 'const' in X_train.columns
        
        # 1. 2SLS Instrumental Variables Model (Industry Standard for Endogeneity)
        if model_name == "IV_2SLS":
            # Stage 1: Regress log_p on log_cost + other features
            # X_train_first_stage has log_cost instead of log_p
            if weights is not None:
                stage1 = sm.WLS(X_train['log_p'], X_train_first_stage, weights=weights).fit()
            else:
                stage1 = sm.OLS(X_train['log_p'], X_train_first_stage).fit()
            log_p_pred = stage1.predict(X_train_first_stage)
            
            # Stage 2: Regress log_q on log_p_pred + other features
            X_stage2 = X_train.copy()
            X_stage2['log_p'] = log_p_pred
            if weights is not None:
                stage2 = sm.WLS(y_train, X_stage2, weights=weights).fit()
            else:
                stage2 = sm.OLS(y_train, X_stage2).fit()
            
            elasticity = stage2.params['log_p']
            return (stage1, stage2), elasticity
            
        # 2. Linear OLS/WLS Model
        elif model_name == "Linear":
            if weights is not None:
                model = sm.WLS(y_train, X_train, weights=weights).fit()
            else:
                model = sm.OLS(y_train, X_train).fit()
            elasticity = model.params['log_p']
            return model, elasticity
            
        # 3. Ridge Regression
        elif model_name == "Ridge":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = Ridge(alpha=1.0)
            if weights is not None:
                model.fit(X_fit, y_train, sample_weight=weights)
            else:
                model.fit(X_fit, y_train)
            # Log-log coefficient is direct elasticity
            price_col_idx = X_fit.columns.get_loc('log_p')
            elasticity = model.coef_[price_col_idx]
            return model, elasticity
            
        # 4. Huber Regressor
        elif model_name == "Huber":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = HuberRegressor(max_iter=1000)
            if weights is not None:
                model.fit(X_fit, y_train, sample_weight=weights)
            else:
                model.fit(X_fit, y_train)
            price_col_idx = X_fit.columns.get_loc('log_p')
            elasticity = model.coef_[price_col_idx]
            return model, elasticity
            
        # 5. Support Vector Regression (SVR)
        elif model_name == "SVR":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            if weights is not None:
                model.fit(X_fit, y_train, sample_weight=weights)
            else:
                model.fit(X_fit, y_train)
            elasticity = self.finite_difference_elasticity(model, X_fit)
            return model, elasticity
            
        # 6. Random Forest (RF)
        elif model_name == "RF":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
            if weights is not None:
                model.fit(X_fit, y_train, sample_weight=weights)
            else:
                model.fit(X_fit, y_train)
            elasticity = self.finite_difference_elasticity(model, X_fit)
            return model, elasticity
            
        # 7. Gradient Boosting (GB)
        elif model_name == "GB":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            if weights is not None:
                model.fit(X_fit, y_train, sample_weight=weights)
            else:
                model.fit(X_fit, y_train)
            elasticity = self.finite_difference_elasticity(model, X_fit)
            return model, elasticity
            
        # 8. MLP Neural Network (MLP)
        elif model_name == "MLP":
            X_fit = X_train.drop(columns=['const']) if has_const else X_train
            model = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=200, random_state=42, activation='tanh', solver='adam')
            if weights is not None:
                p = weights / np.sum(weights)
                indices = np.random.RandomState(42).choice(len(X_fit), size=len(X_fit), p=p)
                X_resampled = X_fit.iloc[indices]
                y_resampled = y_train.iloc[indices]
                model.fit(X_resampled, y_resampled)
            else:
                model.fit(X_fit, y_train)
            elasticity = self.finite_difference_elasticity(model, X_fit)
            return model, elasticity
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
    def predict_model(self, model_name, fitted_model, X_test, X_test_first_stage=None):
        """Predicts log_q for X_test"""
        has_const = 'const' in X_test.columns
        
        if model_name == "IV_2SLS":
            stage1, stage2 = fitted_model
            # Predict log_p first
            log_p_pred = stage1.predict(X_test_first_stage)
            X_stage2 = X_test.copy()
            X_stage2['log_p'] = log_p_pred
            return stage2.predict(X_stage2)[0]
            
        elif model_name == "Linear":
            return fitted_model.predict(X_test)[0]
            
        else:
            X_fit = X_test.drop(columns=['const']) if has_const else X_test
            return fitted_model.predict(X_fit)[0]
            
    def finite_difference_elasticity(self, model, X_train, h=1e-4):
        """Estimates elasticity via finite difference of log_p at the average feature values"""
        # Average row
        mean_row = X_train.mean().to_frame().T
        
        # Perturb log_p
        mean_row_plus = mean_row.copy()
        mean_row_minus = mean_row.copy()
        
        mean_row_plus['log_p'] += h
        mean_row_minus['log_p'] -= h
        
        pred_plus = model.predict(mean_row_plus)[0]
        pred_minus = model.predict(mean_row_minus)[0]
        
        return (pred_plus - pred_minus) / (2 * h)

    def run_backtest(self, model_name, seasonality_approach, use_decay):
        """Runs rolling backtest using a 104-week window"""
        n_weeks = len(self.df)
        train_window = 104
        test_weeks = range(train_window, n_weeks)
        
        actual_quantities = []
        predicted_quantities = []
        elasticities = []
        adaptive_k_list = []
        
        # Basic controls: competitor prices and promotions
        competitor_cols = ['log_comp_1', 'log_comp_2']
        promo_col = ['is_promo']
        
        for idx in test_weeks:
            # 1. Split training and test slices
            train_slice = self.df.iloc[idx - train_window : idx].reset_index(drop=True)
            test_row = self.df.iloc[idx : idx + 1].reset_index(drop=True)
            
            y_train = train_slice['log_q']
            
            # Apply time decay weights if requested (0.95 half-life weight)
            weights = None
            if use_decay:
                t_diff = np.arange(train_window)[::-1]
                weights = 0.95 ** t_diff
                # Normalize weights to sum to N
                weights = weights / np.mean(weights)
                
            # 2. Extract Fourier Seasonality Features
            fourier_k = None
            if seasonality_approach == "baseline":
                fourier_train = pd.DataFrame()
                fourier_test = pd.DataFrame()
            elif seasonality_approach == "adaptive":
                # Find best K on training slice
                fourier_train, fourier_k = SeasonalityEngine.get_adaptive_features(
                    y_train, train_slice['log_p'], train_slice[competitor_cols], train_slice[promo_col], weights
                )
                adaptive_k_list.append(fourier_k)
                # Compute fourier for the test row index (which is index train_window in the t sequence)
                fourier_test = SeasonalityEngine.create_fourier_features(np.array([train_window]), period=52, K=fourier_k)
            else:
                # fixed or multi_period
                fourier_full = SeasonalityEngine.get_features(train_window + 1, approach=seasonality_approach)
                fourier_train = fourier_full.iloc[:train_window].reset_index(drop=True)
                fourier_test = fourier_full.iloc[train_window : train_window + 1].reset_index(drop=True)
                
            # Assemble feature matrices
            X_train = pd.concat([train_slice['log_p'], train_slice[competitor_cols], train_slice[promo_col], fourier_train], axis=1)
            X_train = sm.add_constant(X_train)
            
            X_test = pd.concat([test_row['log_p'], test_row[competitor_cols], test_row[promo_col], fourier_test], axis=1)
            X_test = sm.add_constant(X_test, has_constant='add')
            
            # Realign columns to make sure constant is first
            cols_order = ['const', 'log_p'] + competitor_cols + promo_col + list(fourier_train.columns)
            X_train = X_train[cols_order]
            X_test = X_test[cols_order]
            
            # Setup first stage matrices for 2SLS
            X_train_first_stage = None
            X_test_first_stage = None
            if model_name == "IV_2SLS":
                X_train_first_stage = X_train.copy().rename(columns={'log_p': 'log_cost'})
                X_train_first_stage['log_cost'] = train_slice['log_cost']
                
                X_test_first_stage = X_test.copy().rename(columns={'log_p': 'log_cost'})
                X_test_first_stage['log_cost'] = test_row['log_cost']
                
            # 3. Fit and predict
            try:
                fitted_model, elasticity = self.fit_model(
                    model_name, X_train, y_train, weights, X_train_first_stage
                )
                pred_log_q = self.predict_model(model_name, fitted_model, X_test, X_test_first_stage)
                pred_q = np.exp(pred_log_q)
            except Exception as e:
                # Safe fall-backs in case of convergence errors
                pred_q = np.exp(y_train.mean())
                elasticity = -1.0
                
            # Clamp elasticity to plausible bounds [-10, 2]
            elasticity = np.clip(elasticity, -10.0, 2.0)
            
            actual_quantities.append(test_row['units_sold'].values[0])
            predicted_quantities.append(pred_q)
            elasticities.append(elasticity)
            
        actuals = np.array(actual_quantities)
        preds = np.array(predicted_quantities)
        elasts = np.array(elasticities)
        
        # Calculate performance metrics
        mape = np.mean(np.abs(actuals - preds) / actuals) * 100
        wmape = (np.sum(np.abs(actuals - preds)) / np.sum(actuals)) * 100
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        
        mean_elast = np.mean(elasts)
        std_elast = np.std(elasts)
        
        # Calculate Plausibility Score (Industry target: -1.0 to -2.5)
        if -2.5 <= mean_elast <= -1.0:
            plausibility = 1.0
        elif -4.0 <= mean_elast < -1.0 or -1.0 < mean_elast <= -0.5:
            plausibility = 0.5
        else:
            plausibility = 0.0
            
        # Calculate Stability Score (normalized std, lower is better)
        # 1.0 if std < 0.5, linear decay to 0.0 at std = 2.0
        stability = np.clip(1.0 - (std_elast - 0.2) / 1.8, 0.0, 1.0)
        
        avg_k = np.mean(adaptive_k_list) if adaptive_k_list else np.nan
        
        return {
            "model": model_name,
            "seasonality": seasonality_approach,
            "decay": "WithDecay" if use_decay else "NoDecay",
            "MAPE": mape,
            "WMAPE": wmape,
            "RMSE": rmse,
            "mean_elasticity": mean_elast,
            "std_elasticity": std_elast,
            "plausibility": plausibility,
            "stability": stability,
            "avg_adaptive_k": avg_k
        }

def run_pipeline():
    data_path = "/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/mod_test/branded_rice_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found. Please run the construction script first.")
        sys.exit(1)
        
    brands = ["India_Gate", "Daawat", "Fortune"]
    model_types = ["Linear", "IV_2SLS", "Ridge", "Huber", "SVR", "RF", "GB", "MLP"]
    seasonalities = ["baseline", "fixed", "adaptive", "multi_period"]
    decay_options = [False, True]
    
    results = []
    
    print("Starting Multi-Model & Multi-Seasonality Elasticity Pipeline Backtest...")
    total_configs = len(brands) * len(model_types) * len(seasonalities) * len(decay_options)
    print(f"Total backtest tasks to evaluate: {total_configs}")
    
    task_idx = 0
    for brand in brands:
        print(f"\n--- Evaluating Brand: {brand} ---")
        pipeline = ElasticityPipeline(data_path, brand)
        
        for model in model_types:
            for seas in seasonalities:
                for decay in decay_options:
                    task_idx += 1
                    sys.stdout.write(f"\rProgress: [{task_idx}/{total_configs}] Running {model} + {seas} + {'Decay' if decay else 'NoDecay'}...   ")
                    sys.stdout.flush()
                    
                    res = pipeline.run_backtest(model, seas, decay)
                    res["brand"] = brand
                    
                    # Calculate overall Rank Score
                    # Rank Score = 0.40 * (Normalized WMAPE) + 0.35 * Plausibility + 0.25 * Stability
                    # We normalize WMAPE to a 0-1 score where WMAPE=10% is 1.0 and WMAPE=50% is 0.0
                    wmape_score = np.clip(1.0 - (res["WMAPE"] - 10.0) / 40.0, 0.0, 1.0)
                    rank_score = 0.40 * wmape_score + 0.35 * res["plausibility"] + 0.25 * res["stability"]
                    res["Rank_Score"] = rank_score
                    
                    results.append(res)
                    
    df_results = pd.DataFrame(results)
    
    # Save raw results
    out_csv = "/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/mod_test/pipeline_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n\nPipeline backtest complete! Raw results saved to: {out_csv}")
    
    # Summarize best models overall
    print("\n--- TOP 10 MODEL CONFIGURATIONS OVERALL ---")
    df_top = df_results.sort_values(by="Rank_Score", ascending=False).head(10)
    print(df_top[["brand", "model", "seasonality", "decay", "WMAPE", "MAPE", "mean_elasticity", "std_elasticity", "Rank_Score"]].to_string(index=False))
    
    # Summarize best models per brand
    for brand in brands:
        print(f"\n--- Best Model for {brand} ---")
        df_brand = df_results[df_results["brand"] == brand].sort_values(by="Rank_Score", ascending=False)
        print(df_brand[["model", "seasonality", "decay", "WMAPE", "mean_elasticity", "std_elasticity", "Rank_Score"]].head(3).to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
