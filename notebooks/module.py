import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================
# 1. MODULE CHUẨN HÓA DỮ LIỆU
# ==========================================
def standardize_data(X_train, X_test):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

# ==========================================
# 2. MODULE KIỂM CHỨNG CHÉO 10 PHẦN (10-FOLD CV)
# ==========================================
def get_10_fold_cv():
    
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ==========================================
# 3. MODULE TÍNH TOÁN ĐỘ ĐO F-SCORE
# ==========================================
def calculate_f_score(y_true, y_pred):
    
    return f1_score(y_true, y_pred, average='weighted')