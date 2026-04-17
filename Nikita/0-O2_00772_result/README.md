Результаты экспериментов по блоку `O2` поверх `hierarchical_model`.

В пакет включены только `O2`-связанные варианты и код, который добавляет новые `O2`-фичи.

Файлы:
- `prediction_o2.csv` — полный блок `O2`, лучший по суммарному `mean_rmse`, но с ухудшением `oxidation`.
- `prediction_o2_core3.csv` — только три interaction-признака без флагов.
- `prediction_o2_core3_plus_ca.csv` — три interaction-признака + `Ca salicylate flag`, лучший по самому `oxidation`.
- `experiment_summary.csv` — сводка только по `baseline` и `O2`-вариантам.
- `run_hierarchical_o2_experiments.py` — скрипт расчета `O2`-фич и прогона `hierarchical_model`.

Какие новые `O2`-фичи считаются:
- `o2_salicylate_tbn_x_amine_ao`
- `o2_salicylate_tbn_x_phenol_ao`
- `o2_salicylate_tbn_x_amine_x_phenol`
- `o2_ca_salicylate_present`
- `o2_mg_detergent_present`

Ключевой вывод:
- если нужен самый агрессивный submission, первым стоит пробовать `prediction_o2.csv`;
- если нужен более чистый `oxidation`-ориентированный вариант, стоит пробовать `prediction_o2_core3_plus_ca.csv`.
