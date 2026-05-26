# Physics-Informed Neural Networks for 2D Heat Equation

Студенты:
Янке Анастасия (tg: @yankeeze) - математическая постановка, архитектура PINN, реализация функции потерь (PDE loss).
Пырлицану Никита (tg: @nikita0607) - реализация численного метода (FDM) для ground truth, генерация синтетических данных, визуализация, структура репозитория.

## Project Goal
Исследование PINN для моделирования температурного поля в 2D-пластине и сравнение с:
- классическим численным решением (FDM, ground truth),
- обычным MLP без физического ограничения.

## Repository Structure
```text
.
├── data
│   ├── raw                       # Сырые параметры физической среды (domain_config.json)
│   └── processed                 # Сгенерированные сетки и результаты FDM (ground truth)
├── models                        # Сохраненные веса обученных моделей
├── notebooks
│   ├── 01_eda_and_fdm.ipynb      # Генерация сетки и эталонного FDM
│   ├── 02_baseline_mlp.ipynb     # Baseline: MLP без PDE-loss (train/val/test split)
│   └── 03_experiments_pinn.ipynb # PINN и ablation study
├── presentation
├── report
├── src
│   ├── data_parsing.py           # Парсинг raw-конфига и сборка processed датасета
│   ├── preprocessing.py          # Sampling, split, нормализация, FDM dataset
│   ├── modeling.py               # MLP/PINN, функции потерь, train loop
│   └── utils.py                  # Визуализация тепловых карт и лоссов
├── fastapi
│   ├── main.py                   # REST API для предсказания температуры
│   └── Dockerfile                # Docker-образ для FastAPI сервиса
├── streamlit
│   ├── app.py                    # GUI для инференса PINN/MLP
│   └── Dockerfile                # Docker-образ для Streamlit сервиса
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Raw Data Parsing and Processed Dataset
Сырые параметры задачи задаются в `data/raw/domain_config.json`.

Сборка processed датасета:
```bash
python3 -m src.data_parsing
```

Это создаст:
- `data/processed/pinn_dataset.npz`
- `data/processed/pinn_dataset_meta.json`
- `data/processed/parsing_manifest.json`

## Correct Split for Baseline
В `notebooks/02_baseline_mlp.ipynb` используется детерминированный сплит без утечки:
- train: 70%
- validation: 15%
- test: 15%

Перемешивание индексов происходит с фиксированным `SEED=42`.

## Reproducibility
Во всех экспериментах фиксируется seed:
- `np.random.seed(42)`
- `torch.manual_seed(42)`

## Lint and Tests
```bash
make lint
make test
```

## Docker
Сборка и запуск через Docker:
```bash
make docker-build
make docker-run
```

Или напрямую:
```bash
docker compose run --rm project
```

## FastAPI Service
Сервис FastAPI поднимает HTTP API для инференса температуры в точке.

Доступные ручки:
- `GET /health`
- `GET /meta`
- `GET /predict/pinn?x=<float>&y=<float>&t=<float>`
- `GET /predict/mlp?x=<float>&y=<float>`

Особенности:
- для PINN используется время `t`, как в Streamlit;
- для MLP используется фиксированное `t=1.0`;
- веса берутся автоматически из `models/pinn_main.pt` и `models/mlp_baseline.pt`.

Запуск в Docker:
```bash
make docker-build-api
make docker-run-api
```

После запуска сервис доступен на `http://localhost:8000`.

## Streamlit Service
Streamlit-приложение предоставляет GUI для:
- точечного предсказания температуры;
- построения температурной карты;
- сравнения `t2 - t1` для PINN.

Запуск в Docker:
```bash
make docker-build-streamlit
make docker-run-streamlit
```

После запуска приложение доступно на `http://localhost:8501`.

## Experiments
1. `notebooks/01_eda_and_fdm.ipynb` - генерация и проверка датасета
2. `notebooks/02_baseline_mlp.ipynb` - supervised MLP baseline
3. `notebooks/03_experiments_pinn.ipynb` - PINN + ablation

## Report
Основной отчет: `report/main.tex`

Демонстрация работы проекта: `report/demonstration`
