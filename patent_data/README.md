# Patent Data

Эта папка содержит патентные данные и отдельный локальный pipeline для экспериментов с расширенным train.

## Что здесь лежит

- `daimler_mixtures_train_patent_attach.csv`
- `daimler_component_properties_patent_attach.csv`
- `train.py`
- `predict.py`
- `inference.ipynb`

Скрипты `train.py` и `predict.py` нужны только для локальных экспериментов.

## Что означают файлы

- `daimler_mixtures_train_patent_attach.csv`
  - 2 дополнительных train-like сценария из патента `US20250136889A1` / `EP4545621A2`
  - формат согласован с `data/daimler_mixtures_train.csv`
  - содержит состав, условия теста и целевые значения

- `daimler_component_properties_patent_attach.csv`
  - свойства компонентов для этих 2 сценариев
  - формат согласован с `data/daimler_component_properties.csv`

## Источник:
- `US20250136889A1`
- подтверждающая публикация той же патентной семьи: `EP4545621A2`

## Важные допущения

- `Дозировка катализатора, категория` выставлена в `1`.
  - В патенте катализатор задан как абсолютный уровень `100 ppm Fe`, а в хакатонных CSV используется только обезличенная категория `1/2`.
  - Для сценариев с `5%` биотоплива в исходном train чаще встречается категория `1`, поэтому для attach принято это допущение.

- Не полностью раскрытый остаток рецептуры сохранён как отдельный компонент:
  - `Прочее_patent_common_additives_1`
  - это сделано только затем, чтобы сумма массовых долей в каждом сценарии была равна `100%`

## Локальный запуск с патентными данными

В корне репозитория baseline-скрипты не подхватывают патентные данные автоматически.

Для этого в этой папке есть отдельные:
- `patent_data/train.py`
- `patent_data/predict.py`
- `patent_data/inference.ipynb`

Они делают следующее:
- читают исходные CSV из `data/`;
- добавляют патентные строки из этой папки;
- собирают объединённый набор в `patent_data/data_with_patents/`;
- строят те же признаки `O2`;
- обучают и сохраняют модель;
- строят предсказания на обычном тестовом наборе.

Команды:

```powershell
python patent_data/train.py
python patent_data/predict.py
```

Ноутбук `patent_data/inference.ipynb` работает в predict-only режиме, является самостоятельным и повторяет логику `patent_data/predict.py` без вызова этого файла изнутри ноутбука.

Артефакты локального запуска:
- `patent_data/data_with_patents/`
- `patent_data/_runtime/`
- `patent_data/model/trained_model_with_patents.pt`
- `patent_data/prediction_fresh_retrain_with_patents.csv`
- `patent_data/predictions_with_patents.csv`

## Совместимость с baseline-кодом

Baseline-скрипты в корне проекта читают только:
- `data/daimler_mixtures_train.csv`
- `data/daimler_mixtures_test.csv`
- `data/daimler_component_properties.csv`

Поэтому `patent_data/*.csv` не участвуют в baseline автоматически.

Для ручного объединения по-прежнему можно использовать вспомогательный скрипт:
- `tools/merge_patent_attach_into_data.py`
