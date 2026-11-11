# Практическая работа №9 — SVM (уровень B)

## Среда
Python 3.10+ и Linux. Работа запускается из терминала.

## Установка
```bash
cd ML-P9
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если интернет медленный:
```bash
pip install --default-timeout=300 --retries=10 -r requirements.txt
```

## Запуск
```bash
python main.py
```

## Что делает программа
1. Строит визуализацию границ решения линейной SVM и RBF-SVM на двумерном `make_moons`.
2. Загружает основной датасет Breast Cancer Wisconsin (569 объектов, 30 признаков).
3. Делит данные на обучающую и тестовую выборки.
4. Создаёт `Pipeline(StandardScaler, SVC)`, поэтому масштабирование выполняется внутри каждого фолда кросс-валидации и утечки данных нет.
5. Подбирает `C` и `gamma` для RBF-SVM с помощью `GridSearchCV` и 5-fold StratifiedKFold.
6. Строит тепловую карту.
7. Сравнивает лучшую RBF-SVM, линейную SVM, Random Forest и Gradient Boosting по Accuracy, Precision, Recall, F1 и времени обучения.

## Результаты
После запуска графики и таблицы появятся в `outputs/`:
- `decision_boundary.png`
- `grid_search_heatmap.png`
- `grid_search_results.csv`
- `model_comparison.csv`
- `classification_reports.txt`

## Важное замечание для сдачи
Random Forest и Gradient Boosting включены как ансамблевые базовые модели для сравнения. Если в ваших ПР-7 и ПР-8 использовались другие модели, сообщите их названия и настройки — сравнение нужно привести в соответствие с предыдущими работами. Численные результаты в отчёте берите из фактического запуска программы, а не из заранее придуманных значений.
