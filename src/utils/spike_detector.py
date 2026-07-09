import numpy as np
import pandas as pd

class SpikeDetector:
    def __init__(self, z_threshold=2.0, wow_threshold=0.80, reversion_threshold=1.20, structural_threshold=1.80):
        self.z_threshold = z_threshold
        self.wow_threshold = wow_threshold
        self.reversion_threshold = reversion_threshold
        self.structural_threshold = structural_threshold

    def detect_and_classify_spikes(self, df_sales, df_sent=None, df_promo=None):
        """
        Detects and classifies spikes in sales demand data.
        Returns a copy of df_sales with additional columns:
          - 'is_spike' (bool)
          - 'spike_type' (str: None, 'Type A', 'Type B')
          - 'exclude_from_regression' (bool)
          - 'rebaselined_from_index' (int or None)
        """
        df = df_sales.copy()
        n = len(df)
        
        is_spike = [False] * n
        spike_type = [None] * n
        exclude_from_regression = [False] * n
        rebaseline_index = None
        
        # Merge sentiment and promo if available for rule evaluations
        df['is_promo'] = 0
        if df_promo is not None and 'is_promo' in df_promo.columns:
            df['is_promo'] = df_promo['is_promo'].values
            
        df['google_trends'] = 50.0
        if df_sent is not None and 'google_trends_score' in df_sent.columns:
            df['google_trends'] = df_sent['google_trends_score'].values

        for i in range(4, n):
            q_current = df.loc[i, 'units_sold']
            q_prev = df.loc[i-1, 'units_sold']
            
            # S1: Z-score over rolling 4 weeks
            rolling_window = df.loc[i-4:i-1, 'units_sold']
            mean_4w = rolling_window.mean()
            std_4w = rolling_window.std()
            z_score = (q_current - mean_4w) / max(0.1, std_4w) if std_4w > 0 else 0
            
            # S2: Week-on-week change
            wow_change = (q_current - q_prev) / max(0.1, q_prev)
            
            # Detect spike triggers (S1 or S2)
            s1_trigger = z_score > self.z_threshold
            s2_trigger = wow_change > self.wow_threshold
            
            if s1_trigger or s2_trigger:
                is_spike[i] = True
                
                # Compute a stable pre-spike baseline using 4-week median to ignore single-week stockouts
                pre_spike_baseline = df.loc[i-4:i-1, 'units_sold'].median()
                pre_spike_baseline = max(1.0, pre_spike_baseline)
                
                # Check for S5: Demand reverts to pre-spike baseline within 3 weeks
                reverts = False
                if i + 3 < n:
                    q_post = df.loc[i+1:i+3, 'units_sold']
                    # Reverts if any of the next 3 weeks returns near the baseline (<= 1.2 * pre_spike_baseline)
                    if (q_post <= self.reversion_threshold * pre_spike_baseline).any():
                        reverts = True
                else:
                    # If we don't have enough future data, default to reverts (Type A) as per PDF
                    reverts = True
                    
                # Check for Type B: Structural shift (holds for 4+ weeks at >= 1.8 * pre_spike_baseline)
                structural_shift = False
                if i + 4 < n:
                    q_post_4w = df.loc[i+1:i+4, 'units_sold']
                    if (q_post_4w >= self.structural_threshold * pre_spike_baseline).all():
                        structural_shift = True
                        
                if structural_shift:
                    spike_type[i] = 'Type B'
                    rebaseline_index = i
                elif reverts:
                    spike_type[i] = 'Type A'
                    exclude_from_regression[i] = True
                else:
                    # Default rule: ambiguous classified as Type A
                    spike_type[i] = 'Type A'
                    exclude_from_regression[i] = True
                    
        df['is_spike'] = is_spike
        df['spike_type'] = spike_type
        df['exclude_from_regression'] = exclude_from_regression
        
        # Apply re-baselining: if Type B structural shift occurs, we discard all pre-break data
        df['rebaseline_start'] = rebaseline_index
        
        return df
