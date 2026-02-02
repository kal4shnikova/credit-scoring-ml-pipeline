"""
Модуль для обучения модели с использованием MLflow Tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.pipeline import create_pipeline, get_feature_lists


def load_data(data_dir: Path):
    """
    Загрузка обучающих и тестовых данных
    
    Args:
        data_dir: путь к директории с данными
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("📥 Загрузка данных...")
    
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Разделение на признаки и целевую переменную
    X_train = train_df.drop(['default', 'ID'], axis=1, errors='ignore')
    y_train = train_df['default']
    
    X_test = test_df.drop(['default', 'ID'], axis=1, errors='ignore')
    y_test = test_df['default']
    
    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"   Class balance (train): {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Вычисление всех метрик качества модели
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        y_pred_proba: вероятности предсказаний
        
    Returns:
        Словарь с метриками
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return metrics


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """
    Построение и сохранение ROC-кривой
    
    Args:
        y_true: истинные метки
        y_pred_proba: вероятности предсказаний
        save_path: путь для сохранения графика
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print(f"   📊 ROC curve saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Построение и сохранение матрицы ошибок
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        save_path: путь для сохранения графика
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print(f"   📊 Confusion matrix saved to {save_path}")


def train_model(
    model_type: str = "gradient_boosting",
    use_grid_search: bool = True,
    experiment_name: str = "credit_default_prediction"
):
    """
    Основная функция обучения модели
    
    Args:
        model_type: тип модели ("gradient_boosting" или "logistic_regression")
        use_grid_search: использовать ли GridSearchCV
        experiment_name: имя эксперимента в MLflow
    """
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / 'data' / 'processed'
    models_dir = project_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Настройка MLflow
    mlflow.set_tracking_uri("file:" + str(project_dir / "mlruns"))
    mlflow.set_experiment(experiment_name)
    
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    # Получение списков признаков
    numeric_features, categorical_features = get_feature_lists()
    
    # Старт MLflow run
    with mlflow.start_run(run_name=f"{model_type}_model"):
        
        print(f"\n🚀 Начало обучения модели: {model_type}")
        
        # Определение параметров для поиска
        if model_type == "gradient_boosting":
            if use_grid_search:
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            else:
                model_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5
                }
        else:  # logistic_regression
            if use_grid_search:
                param_grid = {
                    'classifier__C': [0.01, 0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            else:
                model_params = {
                    'C': 1.0,
                    'penalty': 'l2'
                }
        
        # Создание pipeline
        if use_grid_search:
            base_pipeline = create_pipeline(
                numeric_features,
                categorical_features,
                model_type=model_type
            )
            
            print(f"🔍 GridSearchCV с {len(param_grid)} параметрами...")
            pipeline = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:
            pipeline = create_pipeline(
                numeric_features,
                categorical_features,
                model_type=model_type,
                **model_params
            )
        
        # Обучение модели
        print("🏋️ Обучение модели...")
        pipeline.fit(X_train, y_train)
        
        # Получение лучших параметров (если использовали GridSearch)
        if use_grid_search:
            best_params = pipeline.best_params_
            print(f"✅ Лучшие параметры: {best_params}")
            mlflow.log_params(best_params)
            final_model = pipeline.best_estimator_
        else:
            mlflow.log_param("model_type", model_type)
            if model_type == "gradient_boosting":
                mlflow.log_params({
                    'n_estimators': model_params.get('n_estimators'),
                    'learning_rate': model_params.get('learning_rate'),
                    'max_depth': model_params.get('max_depth')
                })
            else:
                mlflow.log_params({
                    'C': model_params.get('C'),
                    'penalty': model_params.get('penalty')
                })
            final_model = pipeline
        
        # Предсказания
        print("🎯 Выполнение предсказаний...")
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        
        # Вычисление метрик
        print("📊 Вычисление метрик...")
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Вывод метрик
        print("\n" + "="*50)
        print("📈 МЕТРИКИ МОДЕЛИ:")
        print("="*50)
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        print("="*50 + "\n")
        
        # Логирование метрик в MLflow
        mlflow.log_metrics(metrics)
        
        # Создание и сохранение графиков
        plots_dir = models_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        roc_curve_path = plots_dir / f'roc_curve_{model_type}.png'
        plot_roc_curve(y_test, y_pred_proba, roc_curve_path)
        mlflow.log_artifact(str(roc_curve_path))
        
        cm_path = plots_dir / f'confusion_matrix_{model_type}.png'
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path))
        
        # Сохранение модели
        model_path = models_dir / f'credit_default_model_{model_type}.pkl'
        joblib.dump(final_model, model_path)
        print(f"💾 Модель сохранена: {model_path}")
        
        # Логирование модели в MLflow
        mlflow.sklearn.log_model(final_model, "model")
        
        # Classification report
        print("\n" + classification_report(y_test, y_pred))
        
        print("\n✅ Обучение завершено успешно!")
        print(f"🔗 MLflow Run ID: {mlflow.active_run().info.run_id}")


def main():
    """
    Главная функция для запуска обучения
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train credit default prediction model')
    parser.add_argument(
        '--model',
        type=str,
        default='gradient_boosting',
        choices=['gradient_boosting', 'logistic_regression'],
        help='Model type to train'
    )
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Use GridSearchCV for hyperparameter tuning'
    )
    
    args = parser.parse_args()
    
    # Запуск обучения
    train_model(
        model_type=args.model,
        use_grid_search=args.grid_search
    )


if __name__ == "__main__":
    main()
