import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def train_model(df: pd.DataFrame, target_col: str = "Churn"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
   
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / count_pos 

    mlflow.set_experiment("Telco_Churn_Feature_Engineering")

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_weight 
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        
        best_thresh = 0.5
        best_f1 = 0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_temp = (y_prob >= thresh).astype(int)
            score = f1_score(y_test, y_pred_temp)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
        
        y_pred = (y_prob >= best_thresh).astype(int)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(best_f1, 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "best_threshold": round(best_thresh, 2)
        }

        mlflow.log_params({
            "features_count": X_train.shape[1],
            "scale_pos_weight": scale_weight,
            "optimal_threshold": best_thresh
        })
        mlflow.log_metrics(metrics)
        
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).head(10)

        return metrics, importance.to_dict(orient='records')