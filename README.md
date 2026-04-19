# НЕФТЕКОД 2026 — Team ROCKET

![ROCKET](docks/logo.png)

Финальный репозиторий хакатонного решения команды **ROCKET**.

## Состав репозитория

- inference.ipynb
  Самодостаточный predict-only ноутбук. Он читает только исходные CSV из папки [data](/e:/Projects/Neftecode/data), сам строит признаки и через зафиксированный checkpoint воспроизводимо формирует финальный `predictions.csv`.

- train.py
  Код обучения модели. Скрипт строит component/scenario-level таблицы из `data/` и обучает historical `hierarchical model`.

- predict.py
  Код получения предсказаний. Скрипт строит те же признаки, загружает зафиксированный checkpoint модели и воспроизводимо формирует финальный [predictions.csv].

- model/hierarchical_o2_baseline.pt
  Зафиксированный checkpoint baseline-модели.

- data 
  Источник входных данных для решения:

  - daimler_component_properties.csv
  - daimler_mixtures_train.csv
  - daimler_mixtures_test.csv
- predictions.csv
  Файл, который формируется контейнерным запуском для отправки на платформу.

- docks
  Служебные материалы для сдачи:

  - logo.png
- Dockerfile
- requirements-docker.txt
- .dockerignore
- patent_data
  Папка с патентными данными и отдельным локальным pipeline для экспериментов с расширенным train:

  - daimler_mixtures_train_patent_attach.csv
  - daimler_component_properties_patent_attach.csv
  - train.py
  - predict.py
  - inference.ipynb
  - README.md

  Что это за папка:
  - отдельные скрипты для локального обучения и предсказания на объединённых `data + patent_data`.

  Важно: baseline-скрипты в корне проекта эти файлы автоматически не читают.
- tools/merge_patent_attach_into_data.py
  Вспомогательный скрипт для локального объединения патент-пакета с исходными CSV в отдельную папку `data_with_patents/`.

Папка `patent_data` не участвует в baseline напрямую. Она нужна как:
- отдельное приложение к финальному решению;
- локальный экспериментальный контур для retrain с патентными данными.

## Что делает `inference.ipynb`

Корневой ноутбук работает в predict-only режиме:

- читает исходные CSV из `data/`;
- сам строит те же признаки, что и `predict.py`;
- загружает checkpoint model/hierarchical_o2_baseline.pt;
- сохраняет итоговый `predictions.csv`.

Во время исполнения создается временная папка `_notebook_runtime_o2`. Это runtime-артефакт.

Код обучения baseline сохранен отдельно в train.py.

## Структура запуска

Ожидаемая структура проекта:

```text
Neftecode/
├── data/
│   ├── daimler_component_properties.csv
│   ├── daimler_mixtures_train.csv
│   └── daimler_mixtures_test.csv
├── docks/
│   └── logo.png
├── inference.ipynb
├── patent_data/
│   ├── daimler_component_properties_patent_attach.csv
│   ├── daimler_mixtures_train_patent_attach.csv
│   ├── inference.ipynb
│   ├── predict.py
│   ├── train.py
│   └── README.md
├── train.py
├── predict.py
├── src/
├── model/
│   └── hierarchical_o2_baseline.pt
├── predictions.csv
├── Dockerfile
└── requirements-docker.txt
```

## Запуск локально

Нужны пакеты:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`

В Docker используется CPU-only сборка PyTorch.

Обучение модели:

```powershell
python train.py
```

Скрипт обучит модель с нуля и сохранит:

- `model/trained_model.pt`
- `prediction_fresh_retrain.csv`

Получение финальных предсказаний через зафиксированный checkpoint:

```powershell
python predict.py
```

Результат:

- в корне проекта будет создан или обновлен predictions.csv

Альтернативно можно открыть inference.ipynb и выполнить ячейки сверху вниз.

Для патентного контура есть отдельный самостоятельный predict-only ноутбук:

```powershell
jupyter notebook patent_data/inference.ipynb
```

## Запуск через Docker

## Важно: у команды не получилось собрать docker без VPN, не подключается к pypi 

Сборка образа:

```bash
docker build -t neftecode-o2 .
```

Запуск в PowerShell:

```powershell
docker run --rm -v ${PWD}:/app neftecode-o2
```

Запуск в `cmd`:

```cmd
docker run --rm -v %cd%:/app neftecode-o2
```

Контейнер:

- ставит зависимости из requirements-docker.txt;
- запускает predict.py;
- воспроизводимо записывает `predictions.csv` в корень проекта.
