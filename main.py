# Практическая работа №9
# Метод опорных векторов (SVM), уровень B

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Сохранение графиков без открытия отдельного окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Все результаты сохраняются в папку outputs рядом с main.py
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def print_header(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def save_decision_boundary():
    """Визуализация границ решения на двумерном наборе make_moons."""
    print_header("ЭТАП 1. ВИЗУАЛИЗАЦИЯ ГРАНИЦЫ РЕШЕНИЯ")

    X_moons, y_moons = make_moons(
        n_samples=400, noise=0.25, random_state=RANDOM_STATE
    )

    models = {
        "Линейное ядро": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1.0))
        ]),
        "RBF-ядро": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, model) in zip(axes, models.items()):
        model.fit(X_moons, y_moons)

        x_min, x_max = X_moons[:, 0].min() - 0.6, X_moons[:, 0].max() + 0.6
        y_min, y_max = X_moons[:, 1].min() - 0.6, X_moons[:, 1].max() + 0.6
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 350),
            np.linspace(y_min, y_max, 350)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(
            X_moons[:, 0], X_moons[:, 1],
            c=y_moons, cmap="coolwarm", edgecolors="black", s=25
        )
        ax.set_title(title)
        ax.set_xlabel("Признак 1")
        ax.set_ylabel("Признак 2")

    fig.suptitle("Сравнение границ решения SVM на make_moons")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_boundary.png", dpi=160)
    plt.close(fig)
    print("График сохранён: outputs/decision_boundary.png")


def load_and_split_data():
    """Загрузка основного датасета и разделение на train/test."""
    print_header("ЭТАП 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    print(f"Количество объектов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Классы: {list(dataset.target_names)}")
    print("Распределение классов:")
    print(y.value_counts().sort_index().rename(index={
        0: dataset.target_names[0],
        1: dataset.target_names[1]
    }))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]} объектов")
    print(f"Тестовая выборка: {X_test.shape[0]} объектов")
    print("Данные разделены со stratify, чтобы сохранить пропорции классов.")

    return X_train, X_test, y_train, y_test


def run_grid_search(X_train, y_train):
    """Подбор C и gamma без утечки данных: scaler находится внутри Pipeline."""
    print_header("ЭТАП 3. PIPELINE И GRIDSEARCHCV")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf"))
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [0.001, 0.01, 0.1, 1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    search_time = time.perf_counter() - start

    print("Лучшие параметры:", grid_search.best_params_)
    print(f"Лучший средний accuracy на CV: {grid_search.best_score_:.4f}")
    print(f"Время GridSearchCV: {search_time:.2f} сек.")

    results = pd.DataFrame(grid_search.cv_results_)
    results["C"] = results["param_svc__C"].astype(float)
    results["gamma"] = results["param_svc__gamma"].astype(float)
    heatmap_data = results.pivot(
        index="gamma", columns="C", values="mean_test_score"
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu",
        cbar_kws={"label": "Средняя accuracy (5-fold CV)"}
    )
    plt.title("GridSearchCV: влияние C и gamma на качество RBF-SVM")
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grid_search_heatmap.png", dpi=160)
    plt.close()

    results[[
        "C", "gamma", "mean_test_score", "std_test_score", "mean_fit_time"
    ]].sort_values("mean_test_score", ascending=False).to_csv(
        OUTPUT_DIR / "grid_search_results.csv", index=False
    )

    print("Тепловая карта сохранена: outputs/grid_search_heatmap.png")
    print("Таблица поиска сохранена: outputs/grid_search_results.csv")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, search_time


def evaluate_models(X_train, X_test, y_train, y_test, best_rbf):
    """Сравнение RBF-SVM, линейной SVM и двух ансамблевых моделей."""
    print_header("ЭТАП 4. СРАВНЕНИЕ МОДЕЛЕЙ")

    models = {
        "SVM RBF (лучшие параметры)": best_rbf,
        "SVM Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1.0))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    }

    rows = []
    reports = []

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        predictions = model.predict(X_test)
        row = {
            "Модель": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "Время обучения, сек": train_time
        }
        rows.append(row)
        reports.append(
            f"\n{name}\n" + classification_report(
                y_test, predictions, target_names=load_breast_cancer().target_names
            )
        )

        print(
            f"{name}: accuracy={row['Accuracy']:.4f}, "
            f"F1={row['F1']:.4f}, время={train_time:.4f} сек."
        )

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(reports))

    print("\nИтоговая таблица:")
    print(comparison.to_string(index=False))
    print("\nТаблица сохранена: outputs/model_comparison.csv")
    print("Подробные отчёты сохранены: outputs/classification_reports.txt")

    return comparison


def main():
    print_header("ПРАКТИЧЕСКАЯ РАБОТА №9 — SVM, УРОВЕНЬ B")
    print("Датасет: Breast Cancer Wisconsin из scikit-learn")
    print("Основная метрика подбора: accuracy")

    save_decision_boundary()

    X_train, X_test, y_train, y_test = load_and_split_data()

    best_rbf, best_params, best_cv_score, search_time = run_grid_search(
        X_train, y_train
    )

    comparison = evaluate_models(
        X_train, X_test, y_train, y_test, best_rbf
    )

    print_header("ЭТАП 5. АВТОМАТИЧЕСКИЙ КРАТКИЙ ВЫВОД")
    best_model = comparison.iloc[0]
    print(f"Лучшая модель по F1 на тестовой выборке: {best_model['Модель']}")
    print(f"Её F1: {best_model['F1']:.4f}")
    print(f"Лучшие параметры RBF-SVM: {best_params}")
    print(f"Лучший CV accuracy RBF-SVM: {best_cv_score:.4f}")
    print(f"Время полного GridSearchCV: {search_time:.2f} сек.")

    print("\nВсе результаты находятся в папке outputs/.")
    print("Важно: для сравнения именно с моделями из ваших ПР-7 и ПР-8")
    print("названия и настройки моделей при необходимости нужно заменить")
    print("на те, которые вы использовали в предыдущих практических работах.")


if __name__ == "__main__":
    main()
