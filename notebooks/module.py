import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


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
    return f1_score(y_true, y_pred, average="weighted")


# ==========================================
# 4. MODULE HUẤN LUYỆN RANDOM FOREST + GRID SEARCH
# ==========================================
def train_rf_with_gridsearch(X_train, y_train, param_grid=None, scoring="f1_weighted"):
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],  # Số lượng cây trong rừng
            "max_depth": [None, 10, 20, 30],  # Độ sâu tối đa của cây
            # Số lượng mẫu tối thiểu để chia một node
            "min_samples_split": [2, 5, 10],
            # Số lượng mẫu tối thiểu để trở thành một node lá
            "min_samples_leaf": [1, 2, 4],
            # Số lượng đặc trưng được xem xét khi tìm kiếm điểm phân chia tốt nhất
            "max_features": ["sqrt", "log2"],
        }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = get_10_fold_cv()

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,  # Hiển thị tiến trình của GridSearchCV
    )
    grid.fit(X_train, y_train)

    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "grid_search": grid,
    }


# ==========================================


# ==========================================
# 6. MODULE ĐÁNH GIÁ MÔ HÌNH
# ==========================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_f1 = calculate_f_score(y_test, y_pred)
    return {
        "test_f1_weighted": float(test_f1),
        "y_pred": y_pred,
    }


# ==========================================
# 7. MODULE TẠO VÀ LƯU BÁO CÁO CHUẨN
# ==========================================
def build_report(
    best_params,
    best_cv_score,
    test_f1_score,
    n_train,
    n_test,
    model_name="RandomForestClassifier",
    scoring="f1_weighted",
):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "cv_strategy": "StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
        "scoring": scoring,
        "best_params": best_params,
        "best_cv_f1_weighted": float(best_cv_score),
        "test_f1_weighted": float(test_f1_score),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def save_report_json(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_report_csv(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(
        output_path, index=False, encoding="utf-8-sig")
