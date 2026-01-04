import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def process_categorical(self):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        return self

    def add_interaction(self, col1: str, col2: str):
        """Creates an interaction feature: col1 * col2."""
        if col1 in self.df.columns and col2 in self.df.columns:
            feature_name = f"{col1}_x_{col2}"
            self.df[feature_name] = self.df[col1] * self.df[col2]
        return self

    def add_polynomial(self, cols: list, degree: int = 2):
        """Adds polynomial features (squares) for numeric columns."""
        valid_cols = [c for c in cols if c in self.df.columns]
        
        if valid_cols:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = poly.fit_transform(self.df[valid_cols])
            
            new_feature_names = poly.get_feature_names_out(valid_cols)
            
            poly_df = pd.DataFrame(
                poly_data, 
                columns=new_feature_names, 
                index=self.df.index
            )
            
            poly_df = poly_df.drop(columns=valid_cols, errors='ignore')
            self.df = pd.concat([self.df, poly_df], axis=1)
            
        return self

    def bin_tenure(self):
        """Domain Feature: Groups tenure into logical cohorts."""
        if 'tenure' in self.df.columns:
            labels = ['New', 'Loyal', 'VIP']
            bins = [0, 12, 48, 999]
            self.df['Tenure_Cohort'] = pd.cut(self.df['tenure'], bins=bins, labels=labels)
            self.df = pd.get_dummies(self.df, columns=['Tenure_Cohort'], drop_first=True)
        return self

    def get_data(self) -> pd.DataFrame:
        """Returns the processed dataframe."""
        return self.df
    

    def align_features(self, reference_columns: list):
        """
        Ensures the dataframe has exactly the same columns as the training data.
        If a column is missing, it adds it with 0s.
        If there are extra columns, it drops them.
        """
        for col in reference_columns:
            if col not in self.df.columns:
                self.df[col] = 0
        
        self.df = self.df[reference_columns]
        return self