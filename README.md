# Neftecode

Финальное решение зафиксировано в baseline `O2`.

## Что является финальным решением

- [inference.ipynb](/e:/Projects/Neftecode/inference.ipynb) — самодостаточный ноутбук
- [prediction.csv](/e:/Projects/Neftecode/prediction.csv) — baseline `O2` prediction
- `final_o2_inputs/*.csv` — exact transformed inputs, на которых baseline воспроизводится

Ноутбук:
- не обращается к внешним `.py`-скриптам;
- сам строит `O2`-признаки;
- содержит historical `hierarchical model` из baseline;
- сохраняет итоговый `prediction.csv`.

## Входные файлы

Нужны только 4 CSV из [final_o2_inputs](/e:/Projects/Neftecode/final_o2_inputs):

- `train_component_level_transformed.csv`
- `test_component_level_transformed.csv`
- `train_scenario_level_features.csv`
- `test_scenario_level_features.csv`

## Результат

Текущий [prediction.csv](/e:/Projects/Neftecode/prediction.csv) соответствует baseline `O2`-линии, восстановленной на условиях:

- transformed inputs из `aa3db981`
- historical trainer из `0dc3d75`
- baseline `O2` feature block из 5 признаков
