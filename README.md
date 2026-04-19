# НЕФТЕКОД 2026 — Team ROCKET

![ROCKET](docks/logo.png)

Финальный репозиторий хакатонного решения команды **ROCKET** для baseline-модели `O2`.

## Состав репозитория

- [inference.ipynb](/e:/Projects/Neftecode/inference.ipynb)  
  Самодостаточный ноутбук. Он читает только исходные CSV из папки [data](/e:/Projects/Neftecode/data), сам строит признаки, содержит код обучения `hierarchical model` и код получения предсказаний.

- [train.py](/e:/Projects/Neftecode/train.py)  
  Код обучения модели. Скрипт строит component/scenario-level таблицы из `data/`, добавляет `O2`-признаки и обучает historical `hierarchical model`.

- [predict.py](/e:/Projects/Neftecode/predict.py)  
  Код получения предсказаний. Скрипт строит те же признаки, загружает зафиксированный checkpoint модели и воспроизводимо формирует финальный [predictions.csv](/e:/Projects/Neftecode/predictions.csv).

- [model/hierarchical_o2_baseline.pt](/e:/Projects/Neftecode/model/hierarchical_o2_baseline.pt)  
  Зафиксированный checkpoint baseline-модели `O2`.

- [data](/e:/Projects/Neftecode/data)  
  Единственный источник входных данных для решения:
  - [daimler_component_properties.csv](/e:/Projects/Neftecode/data/daimler_component_properties.csv)
  - [daimler_mixtures_train.csv](/e:/Projects/Neftecode/data/daimler_mixtures_train.csv)
  - [daimler_mixtures_test.csv](/e:/Projects/Neftecode/data/daimler_mixtures_test.csv)

- [prediction.csv](/e:/Projects/Neftecode/prediction.csv)  
  Проверенный baseline artifact модели `O2`.

- [predictions.csv](/e:/Projects/Neftecode/predictions.csv)  
  Файл, который формируется контейнерным запуском для отправки на платформу.

- [docks](/e:/Projects/Neftecode/docks)  
  Служебные материалы для сдачи. Сейчас здесь лежит логотип команды:
  - [logo.png](/e:/Projects/Neftecode/docks/logo.png)

- [Dockerfile](/e:/Projects/Neftecode/Dockerfile)
- [requirements-docker.txt](/e:/Projects/Neftecode/requirements-docker.txt)
- [.dockerignore](/e:/Projects/Neftecode/.dockerignore)

Локальная папка [patent_extraction](/e:/Projects/Neftecode/patent_extraction) сохранена для дальнейшей работы, но не входит в финальный пакет для `main`.

## Что делает `inference.ipynb`

Ноутбук содержит то же решение, что и отдельные `.py`-файлы:

- чтение исходных CSV из `data/`;
- трансформация raw-данных в component/scenario-level представление;
- построение baseline `O2`-признаков:
  - `o2_salicylate_tbn_x_amine_ao`
  - `o2_salicylate_tbn_x_phenol_ao`
  - `o2_salicylate_tbn_x_amine_x_phenol`
  - `o2_ca_salicylate_present`
  - `o2_mg_detergent_present`
- обучение `hierarchical model`;
- получение предсказаний на тестовом наборе;
- сохранение итогового `predictions.csv`.

Во время исполнения создается временная папка `_notebook_runtime_o2`. Это runtime-артефакт, его не нужно коммитить.

Важно: нейросетевое обучение PyTorch может давать небольшие отличия между Windows и Linux/Docker даже при фиксированном seed. Поэтому для воспроизводимой контейнерной сдачи используется не retrain, а checkpoint [model/hierarchical_o2_baseline.pt](/e:/Projects/Neftecode/model/hierarchical_o2_baseline.pt). Код обучения сохранен отдельно в [train.py](/e:/Projects/Neftecode/train.py).

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
├── train.py
├── predict.py
├── src/
├── model/
│   └── hierarchical_o2_baseline.pt
├── prediction.csv
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

- `model/hierarchical_o2_trained.pt`
- `prediction_fresh_retrain.csv`

Получение финальных предсказаний через зафиксированный checkpoint:

```powershell
python predict.py
```

Результат:

- в корне проекта будет создан или обновлен [predictions.csv](/e:/Projects/Neftecode/predictions.csv)

Альтернативно можно открыть [inference.ipynb](/e:/Projects/Neftecode/inference.ipynb) и выполнить ячейки сверху вниз.

## Запуск через Docker

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

- ставит зависимости из [requirements-docker.txt](/e:/Projects/Neftecode/requirements-docker.txt);
- запускает [predict.py](/e:/Projects/Neftecode/predict.py);
- воспроизводимо записывает `predictions.csv` в корень проекта.

Проверка результата после запуска:

```powershell
Get-FileHash predictions.csv -Algorithm SHA256
```

Ожидаемый SHA256 для проверенного baseline:

```text
31D58179D347F3577F22806EE63350D6C1C5103D972C21C9EFED8AA59E02D82C
```

## Что отправлять

Для финальной сдачи нужны:

- `inference.ipynb`
- `train.py`
- `predict.py`
- папка `src/`
- папка `model/`
- `predictions.csv`
- папка `data/`
- `Dockerfile`
- `requirements-docker.txt`

`patent_extraction` в baseline `O2` не участвует и в `main` не требуется.
