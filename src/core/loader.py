import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data() -> pd.DataFrame:
    """
    Loads the Telco Churn dataset and performs initial cleaning.
    
    Returns:
        pd.DataFrame: Cleaned raw dataframe.
    """
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download it.")

        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        df.dropna(inplace=True)
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e