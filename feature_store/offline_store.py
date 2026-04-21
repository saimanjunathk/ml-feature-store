# OFFLINE FEATURE STORE
# Stores historical features for model training
# In production: this would be a data warehouse (BigQuery, Snowflake)
# Our version: SQLite database

import pandas as pd
import sqlite3
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OfflineFeatureStore:

    def __init__(self, db_path: str = "data/feature_store.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        logger.info(f"Offline store initialized: {db_path}")


    # This METHOD saves features to the offline store
    def save_features(self, df: pd.DataFrame, table_name: str = "customer_features"):
        conn = sqlite3.connect(self.db_path)
        # if_exists="replace" → overwrites existing data
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"Saved {len(df)} rows to offline store: {table_name}")


    # This METHOD loads features for model training
    def load_features(
        self,
        table_name: str = "customer_features",
        feature_columns: list = None,
        target_column: str = "churned"
    ) -> tuple:

        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])

        y = df[target_column]
        logger.info(f"Loaded {len(df)} rows | {len(X.columns)} features")
        return X, y


    # This METHOD returns feature statistics
    def get_statistics(self, table_name: str = "customer_features") -> pd.DataFrame:
        conn  = sqlite3.connect(self.db_path)
        df    = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df.describe()