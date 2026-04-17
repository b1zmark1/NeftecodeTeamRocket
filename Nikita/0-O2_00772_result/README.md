Результаты экспериментов по блоку `O2` поверх `hierarchical_model`.

Файлы:
- `prediction_o2.csv` — лучший по суммарному `mean_rmse`, но с ухудшением `oxidation`.
- `prediction_o2_v1v2_combo.csv` — компромиссный вариант `O2 + V1/V2`.
- `prediction_o2_core3_plus_ca.csv` — лучший из урезанных `O2`, улучшает `oxidation`.
- `prediction_o2_core3.csv` — исследовательский вариант без `Ca salicylate flag`.
- `experiment_summary.csv` — сводка метрик всех прогонов относительно текущего hierarchical baseline.

Ключевой вывод:
- если нужен самый агрессивный submission, первым стоит пробовать `prediction_o2.csv`;
- если нужен более чистый `oxidation`-ориентированный вариант, стоит пробовать `prediction_o2_core3_plus_ca.csv`.
