# DATA DRIFT DETECTION
# Data drift = the distribution of input features changes over time
# This causes model performance to degrade silently!
#
# Example:
# - Model trained on 2023 data where avg_amount = $100
# - In 2024 avg_amount = $200 (inflation)
# - Model predictions become unreliable
#
# We detect drift using statistical tests:
# - KS test (Kolmogorov-Smirnov): compares distributions
# - PSI (Population Stability Index): industry standard in finance

import pandas as pd
import numpy as np
import logging
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriftDetector:

    # psi_threshold → PSI > 0.2 means significant drift
    # ks_threshold  → p-value < 0.05 means significant drift
    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_threshold  = ks_threshold
        self.reference_data = None


    # This METHOD stores reference (training) data
    def set_reference(self, df: pd.DataFrame):
        self.reference_data = df.copy()
        logger.info(f"Reference data set: {len(df)} rows")


    # This METHOD calculates PSI for a single feature
    # PSI < 0.1  → no drift
    # PSI 0.1-0.2 → slight drift
    # PSI > 0.2  → significant drift
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:

        # Create bins from reference distribution
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current,   bins=bin_edges)

        # Convert to proportions (add small epsilon to avoid log(0))
        ref_props = (ref_counts + 1e-8) / len(reference)
        cur_props = (cur_counts + 1e-8) / len(current)

        # PSI formula: sum((current - reference) * ln(current/reference))
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return round(psi, 4)


    # This METHOD runs KS test for a single feature
    # KS test checks if two samples come from the same distribution
    def ks_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> dict:

        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "statistic": round(statistic, 4),
            "p_value":   round(p_value, 4),
            "drifted":   p_value < self.ks_threshold
        }


    # This METHOD detects drift across all features
    def detect_drift(self, current_data: pd.DataFrame) -> pd.DataFrame:

        if self.reference_data is None:
            raise ValueError("Set reference data first using set_reference()")

        results = []
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue

            ref = self.reference_data[col].dropna()
            cur = current_data[col].dropna()

            psi    = self.calculate_psi(ref, cur)
            ks     = self.ks_test(ref, cur)

            # Overall drift flag
            drifted = psi > self.psi_threshold or ks["drifted"]

            results.append({
                "feature":      col,
                "psi":          psi,
                "ks_statistic": ks["statistic"],
                "ks_p_value":   ks["p_value"],
                "drifted":      drifted,
                "ref_mean":     round(ref.mean(), 4),
                "cur_mean":     round(cur.mean(), 4),
                "mean_change":  round((cur.mean() - ref.mean()) / (ref.mean() + 1e-8) * 100, 2)
            })

        df = pd.DataFrame(results)
        n_drifted = df["drifted"].sum()
        logger.info(f"Drift detected in {n_drifted}/{len(df)} features")
        return df


    # This METHOD simulates production drift
    # Adds noise to simulate data distribution changes over time
    def simulate_drift(
        self,
        df: pd.DataFrame,
        drift_magnitude: float = 0.3
    ) -> pd.DataFrame:

        drifted = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ["churned", "customer_id"]:
                continue
            # Add drift: shift mean by drift_magnitude * std
            std   = df[col].std()
            shift = drift_magnitude * std * np.random.choice([-1, 1])
            drifted[col] = df[col] + shift + np.random.normal(0, std * 0.1, len(df))

        logger.info(f"Simulated drift with magnitude {drift_magnitude}")
        return drifted