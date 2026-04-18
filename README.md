# Neftecode Final Submission

Репозиторий приведен к финальному baseline `O2`.

## Что осталось в репозитории

- [inference.ipynb](/e:/Projects/Neftecode/inference.ipynb)  
  Самодостаточный ноутбук с полным pipeline:
  - raw CSV -> transform,
  - добавление `O2`-признаков,
  - обучение иерархической модели,
  - сохранение `prediction.csv`.

- [prediction.csv](/e:/Projects/Neftecode/prediction.csv)  
  Итоговый baseline prediction.

- [daimler_component_properties.csv](/e:/Projects/Neftecode/daimler_component_properties.csv)
- [daimler_mixtures_train.csv](/e:/Projects/Neftecode/daimler_mixtures_train.csv)
- [daimler_mixtures_test.csv](/e:/Projects/Neftecode/daimler_mixtures_test.csv)  
  Единственные входные данные для решения.

- [patent_extraction](/e:/Projects/Neftecode/patent_extraction)  
  Оставлена полностью для дальнейшей работы.

- [Dockerfile](/e:/Projects/Neftecode/Dockerfile)
- [requirements-docker.txt](/e:/Projects/Neftecode/requirements-docker.txt)
- [.dockerignore](/e:/Projects/Neftecode/.dockerignore)

## Что удалено

Убраны старые и неиспользуемые для финальной сдачи каталоги:

- `Nikita`
- `final_o2_inputs`
- `hierarchical_model`
- `hierarchical_out`
- `interaction_out`
- `mlp`
- `mlp_out`
- `model_with_structure`
- `plsreg`

Также убран локальный runtime и исследовательский мусор:

- `_notebook_runtime_o2`
- `tmp_us2025_ocr`
- старые локальные PDF/TXT и прочие временные каталоги

## Как устроено решение

`inference.ipynb` не вызывает внешние `.py`-скрипты. Внутри ноутбука зашиты:

- логика трансформации raw-данных в component/scenario tables;
- логика построения `O2`-блока:
  - `o2_salicylate_tbn_x_amine_ao`
  - `o2_salicylate_tbn_x_phenol_ao`
  - `o2_salicylate_tbn_x_amine_x_phenol`
  - `o2_ca_salicylate_present`
  - `o2_mg_detergent_present`
- логика обучения `hierarchical model`;
- сохранение финального `prediction.csv`.

Во время исполнения ноутбук создает временную папку `_notebook_runtime_o2` с промежуточными CSV и model artifacts. Это runtime-артефакт, его не нужно коммитить.

## Запуск без Docker

Нужны зависимости Python:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`

Запуск можно сделать через Jupyter или простым исполнением кода ячеек ноутбука.

После завершения в корне будет создан или перезаписан:

- [prediction.csv](/e:/Projects/Neftecode/prediction.csv)

## Запуск через Docker

Сборка образа:

```bash
docker build -t neftecode-o2 .
```

Запуск:

```bash
docker run --rm -v %cd%:/app neftecode-o2
```

Для PowerShell:

```powershell
docker build -t neftecode-o2 .
docker run --rm -v ${PWD}:/app neftecode-o2
```

Контейнер:

- устанавливает зависимости из [requirements-docker.txt](/e:/Projects/Neftecode/requirements-docker.txt);
- исполняет все code-cells из [inference.ipynb](/e:/Projects/Neftecode/inference.ipynb);
- пишет `prediction.csv` в корень проекта.

## Что важно для сдачи

Для финальной сдачи достаточно следующего набора:

- `inference.ipynb`
- `prediction.csv`
- `daimler_component_properties.csv`
- `daimler_mixtures_train.csv`
- `daimler_mixtures_test.csv`
- `Dockerfile`
- `requirements-docker.txt`

`patent_extraction` не участвует в baseline `O2`, но сохранена в репозитории отдельно по рабочей необходимости.
