import pandas as pd
import numpy as np
import logging
from faker import Faker
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


class FeatureGenerator:

    def __init__(self, n_customers: int = 1000):
        self.n_customers = n_customers


    def generate_raw_data(self) -> pd.DataFrame:

        logger.info(f"Generating raw data for {self.n_customers} customers...")
        np.random.seed(42)
        random.seed(42)

        n = self.n_customers

        # Generate ~40% churners and ~60% non-churners
        n_churned     = int(n * 0.4)
        n_not_churned = n - n_churned

        # ── CHURNED customers ── clearly different patterns
        churned = pd.DataFrame({
            "customer_id":     range(1, n_churned + 1),
            "age":             np.random.randint(18, 75, n_churned),
            "n_transactions":  np.random.randint(1, 60, n_churned),      # low activity
            "avg_amount":      np.random.uniform(10, 350, n_churned),       # low spending
            "total_amount":    np.random.uniform(100, 20000, n_churned),
            "last_login_days": np.random.randint(30, 365, n_churned),    # inactive
            "support_tickets": np.random.randint(0, 8, n_churned),       # many complaints
            "plan":            np.random.choice(["free", "basic", "premium"], n_churned, p=[0.45, 0.35, 0.20]),
            "churned":         1
        })

        # ── NOT CHURNED customers ── clearly different patterns
        not_churned = pd.DataFrame({
            "customer_id":     range(n_churned + 1, n + 1),
            "age":             np.random.randint(18, 75, n_not_churned),
            "n_transactions":  np.random.randint(10, 100, n_not_churned),  # high activity
            "avg_amount":      np.random.uniform(30, 500, n_not_churned), # high spending
            "total_amount":    np.random.uniform(500, 50000, n_not_churned),
            "last_login_days": np.random.randint(0, 200, n_not_churned),    # very active
            "support_tickets": np.random.randint(0, 6, n_not_churned),     # few complaints
            "plan":            np.random.choice(["free", "basic", "premium"], n_not_churned, p=[0.25, 0.40, 0.35]),
            "churned":         0
        })

        # Add signup_date and country
        for df in [churned, not_churned]:
            df["signup_date"] = [
                fake.date_between(start_date="-3y", end_date="-1m").strftime("%Y-%m-%d")
                for _ in range(len(df))
            ]
            df["country"] = [fake.country_code() for _ in range(len(df))]

        # Combine and shuffle
        df = pd.concat([churned, not_churned], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["customer_id"] = range(1, len(df) + 1)

        churn_rate = df["churned"].mean()
        logger.info(f"Generated {len(df)} records | Churn rate: {churn_rate*100:.1f}%")
        return df


    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Computing ML features...")
        features = df.copy()

        features["txn_frequency_score"] = (
            features["n_transactions"] /
            features["n_transactions"].max()
        ).round(4)

        features["recency_score"] = (
            1 - features["last_login_days"] / 365
        ).clip(0, 1).round(4)

        features["avg_amount_normalized"] = (
            (features["avg_amount"] - features["avg_amount"].mean()) /
            (features["avg_amount"].std() + 1e-8)
        ).round(4)

        features["support_rate"] = (
            features["support_tickets"] /
            (features["n_transactions"] + 1)
        ).round(4)

        features["clv_proxy"] = (
            features["total_amount"] *
            features["recency_score"]
        ).round(2)

        features["engagement_score"] = (
            features["txn_frequency_score"] * 0.4 +
            features["recency_score"] * 0.4 +
            (1 - features["support_rate"].clip(0, 1)) * 0.2
        ).round(4)

        plan_map = {"free": 0, "basic": 1, "premium": 2}
        features["plan_encoded"] = features["plan"].map(plan_map)

        features["age_bucket"] = pd.cut(
            features["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        logger.info(f"Computed {len(features.columns)} features")
        return features


    FEATURE_COLUMNS = [
        "age",
        "n_transactions",
        "avg_amount",
        "last_login_days",
        "support_tickets",
        "txn_frequency_score",
        "recency_score",
        "avg_amount_normalized",
        "support_rate",
        "clv_proxy",
        "engagement_score",
        "plan_encoded",
        "age_bucket"
    ]

    TARGET_COLUMN = "churned"


if __name__ == "__main__":
    gen      = FeatureGenerator(n_customers=1000)
    raw      = gen.generate_raw_data()
    features = gen.compute_features(raw)
    print(f"Churn rate: {features['churned'].mean()*100:.1f}%")
    print(features[gen.FEATURE_COLUMNS].describe())