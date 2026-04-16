# Roadmap

## Цель
Сильно улучшить лидербордный результат без запрещенных деревьев за счет:
- корректной валидации, которая ближе к leaderboard;
- разных feature spaces для `viscosity` и `oxidation`;
- очень коротких chemistry/residual-блоков вместо широких шумных наборов;
- target-wise ensemble, а не одной универсальной модели.

## Что уже выяснено

### 1. Validation mismatch подтвержден
Случайная CV слишком оптимистична и не отражает перенос на новые рецептуры.

Что уже проверено:
- `random_kfold`
- `dominant_base_oil` split
- `recipe_cluster` split

Ключевой результат:
- основной валидатор с этого момента: `recipe_cluster`
- `dominant_base_oil` используем как дополнительный realistic split
- `random_kfold` оставляем только как быстрый dev-check

Это значит:
- любое улучшение, которое не проходит `recipe_cluster`, не считается настоящим прогрессом;
- старые “лучшие” модели по random CV больше не являются источником истины.

### 2. Backbone уже зафиксирован
После перехода на `recipe_cluster` лучший backbone изменился.

Зафиксированные backbone:
- `viscosity backbone = compact_v3_full + MLP(relu_small)`
- `oxidation backbone = compact_v3_full + Ridge(alpha=30.0)`

Это важное решение:
- новые идеи больше не сравниваются “между собой в вакууме”;
- теперь все chemistry / residual / ensemble идеи проверяются относительно этих backbone.

### 3. Широкий `chemistry_v3` не прошел
Широкий блок химических interaction-features, собранный из `new_datasets`, не дал улучшения под `recipe_cluster`.

Что это означает:
- проблема не в том, что chemistry не нужна;
- проблема в том, что широкий merged-block слишком шумный и плохо переносится между рецептами.

Вывод:
- дальше идем не в сторону “еще больше химических фич”;
- идем в сторону **узких, target-wise, химически строго отобранных блоков**.

### 4. Широкий linear residual для viscosity не работает
Попытка прикрутить широкий линейный residual к `viscosity backbone` ухудшила качество.

Вывод:
- для `viscosity` не надо строить wide chemistry residual;
- если улучшать `viscosity`, то только через узкие interaction-блоки:
  - `antiwear-Ca/Mg`
  - `Zn-dispersant`
  - `ZDDP-detergent`
  - взаимодействия с `base oil`

### 5. Для oxidation лучший путь сейчас: ultra-compact chemistry add-on
Этап с `oxidation chemistry v4 narrow` показал, что широкий even narrow-block все еще избыточен.

Что уже подтверждено:
- `ao_core_narrow` как direct add-on улучшил backbone:
  - `RMSE 29.77` против `30.54` у backbone
- после еще более жесткого урезания лучший результат дал `chem5_core`:
  - `RMSE 29.22` против `30.54` у backbone на `recipe_cluster`
- sanity-check на `dominant_base_oil` тоже подтверждает улучшение:
  - `24.65` против `24.84`

Состав лучшего блока `chem5_core`:
- `ao_pair_x_biofuel`
- `ao_dpa_phenol_imbalance`
- `synergy_ao_phenol_x_diphenylamine_active_no`
- `ao_homo_max`
- `ao_ionization_min`

Это главный новый вывод:
- для `oxidation` лучше работает не широкий chemistry-block и не residual,
- а **очень короткий chemistry add-on** поверх устойчивого backbone.

### 6. Chemistry-only модель не заменяет backbone
На ultra-compact chemistry-only наборах:
- standalone `Ridge/PLS/MLP` стабильно хуже backbone;
- лучший standalone был около `33.97`, тогда как лучший add-on дал `29.22`.

Вывод:
- chemistry надо использовать как корректирующий блок;
- backbone по-прежнему обязателен.

### 7. Сильное сокращение числа фич реально помогает
Новый важный вывод: при `oxidation` уменьшение числа фич не просто "не мешает", а улучшает переносимость.

Что уже подтверждено:
- `chem5_core` с 5 химическими признаками оказался лучше, чем:
  - широкий `chemistry_v3`
  - `oxidation_v4_narrow`
  - более длинные chemistry-блоки на 7, 8 и 10 признаков
- лучший результат сейчас дает именно очень короткий add-on:
  - `ao_pair_x_biofuel`
  - `ao_dpa_phenol_imbalance`
  - `synergy_ao_phenol_x_diphenylamine_active_no`
  - `ao_homo_max`
  - `ao_ionization_min`

Вывод:
- aggressive feature pruning становится не вспомогательной, а основной стратегией;
- добавлять новый механизм можно только если он улучшает результат поверх уже короткого ядра.

### 8. Для каждого таргета нужен свой набор фич
Это уже не гипотеза, а зафиксированный принцип.

Для `oxidation`:
- работает `compact_v3_full backbone + ultra-compact chemistry add-on`;
- основное ядро:
  - `biofuel × AO chemistry`
  - `amine-phenol imbalance`
  - `diphenylamine-related synergy`
  - `HOMO`
  - `ionization`

Для `viscosity`:
- основной сигнал остается в:
  - `regime`
  - `base oil`
  - `antiwear`
  - узких pairwise/interactions для `Ca/Mg`, `Zn`, `ZDDP`

Вывод:
- дальше нельзя строить один и тот же feature space для обоих таргетов;
- нужно поддерживать два разных train-view:
  - `viscosity_dataset_vNext`
  - `oxidation_dataset_vNext`

### 9. MoE как основная ставка пока не подтвердился
Первый полный `Mixture of Experts` эксперимент был проведен по:
- `recipe_cluster`
- `dominant_base_oil`
- `package_type`

Проверены:
- `hard residual experts`
- `soft residual experts`
- `direct_soft experts`

Результат:
- сильного буста не получено;
- лучший глобальный `oxidation` backbone остался сильнее;
- для `viscosity` были только очень слабые сдвиги, не тянущие на production-буст.

Вывод:
- MoE пока не основной вектор;
- к нему можно вернуться позже, но только после дальнейшего укрепления target-wise feature spaces и stacking.

## Обновленная стратегия
Дальнейшая работа строится вокруг четырех параллельных линий:

1. Держать правильный режим оценки:
- главный gate = `recipe_cluster`
- sanity check = `dominant_base_oil`
- random CV только для быстрых сравнений

2. Не трогать backbone без сильной причины:
- `viscosity`: `compact_v3_full + MLP(relu_small)`
- `oxidation`: `compact_v3_full + Ridge(alpha=30.0)`

3. Пересобрать признаки только в **ультра-компактном target-wise виде**
- `oxidation ultra-compact chemistry add-on`
- `viscosity narrow residual interactions`

4. Финально собирать результат через target-wise ensemble / stacking

## План работ

### Этап 1. Validation regime
Статус: выполнен.

Что сделано:
- проверено несколько режимов валидации;
- найдено, что `recipe_cluster` лучше всего отражает domain shift по рецептурам.

Как использовать дальше:
- все новые идеи принимаются только если они улучшают `recipe_cluster`;
- `dominant_base_oil` используем как дополнительную страховку от ложных улучшений.

### Этап 2. Backbone selection
Статус: выполнен.

Зафиксировано:
- `viscosity backbone = compact_v3_full + MLP(relu_small)`
- `oxidation backbone = compact_v3_full + Ridge(alpha=30.0)`

Правило:
- все дальнейшие direct add-on, residual и pairwise-модели сравниваются именно с этими backbone.

### Этап 3. Target-wise feature engineering
Статус: частично выполнен, стратегия скорректирована.

Главное правило этапа:
- не просто разные модели на одном и том же наборе колонок;
- а **разные наборы признаков для разных таргетов**.

#### Для viscosity
Что уже понятно:
- основной устойчивый сигнал сидит в:
  - `regime`
  - `base oil`
  - `antiwear`
- широкий chemistry-блок не помогает;
- широкий линейный residual ухудшает качество.

Что делать дальше:
- оставить backbone как есть;
- улучшать только узкими interaction-блоками:
  - `antiwear-Ca/Mg`
  - `Zn-dispersant`
  - `ZDDP-detergent`
  - interaction с `dominant base oil`
- aggressively pruning тоже применять, но только внутри viscosity-specific блока;
- не возвращаться к wide chemistry residual.

#### Для oxidation
Что уже понятно:
- `oxidation` требует chemistry, но широкий `chemistry_v3` оказался слишком шумным;
- under `recipe_cluster` устойчивее всего пока работает линейный backbone;
- лучший путь сейчас: ultra-compact chemistry add-on, а не residual.

Что делать дальше:
- использовать только самые сильные химические признаки;
- явно держать отдельный oxidation-specific dataset/view;
- не возвращаться к wide merged chemistry block;
- residual для oxidation пока не использовать как основной путь.

### Этап 4. Ultra-compact chemistry add-on for oxidation
Статус: выполнен, направление зафиксировано.

Принцип:
- только признаки, которые одновременно:
  - подтверждены литературой,
  - поддержаны нашими signed pair effects / pairwise-анализом,
  - имеют понятную химическую интерпретацию.

Цель:
- не 30-50 фич,
- а 5-10 sharp features;
- chemistry используется как add-on к backbone, не как самостоятельный предиктор.

#### Что уже проверено
- `oxidation v4 narrow` оказался лучше wide chemistry, но не стал лучшим финальным блоком;
- лучший результат сейчас у `chem5_core`:
  - `ao_pair_x_biofuel`
  - `ao_dpa_phenol_imbalance`
  - `synergy_ao_phenol_x_diphenylamine_active_no`
  - `ao_homo_max`
  - `ao_ionization_min`

#### Что это означает
- amine-phenol chemistry сейчас важнее остальных механизмов;
- conditioning на `biofuel` полезен;
- квантово-химические AO-дескрипторы дают пользу в очень коротком виде;
- `AO-Mo` и `ZDDP` могут быть полезны, но только если не размывают короткое ядро.

#### Текущее правило
- сначала тестируем максимально короткий chemistry add-on;
- только потом аккуратно добавляем 1 новый механизм поверх него;
- если новый механизм не улучшает `recipe_cluster`, он исключается.

#### Новые идеи из литературы, которые надо пробовать только поверх short core
- `diphenylamine × Mg detergent`
- `diphenylamine × salicylate detergent`
- `diphenylamine × primary ZDDP`
- `Mg:Ca ratio`
- `biofuel × AO chemistry × base_oil`

Но правило жесткое:
- эти идеи не расширяют блок "вообще";
- они тестируются по одной-две поверх `chem5_core`.

### Этап 5. Signed pair effect и shortlist стабильных взаимодействий
Статус: частично выполнен.

Что уже найдено:
- `AO-Mo` ведет себя как синергистический блок;
- `ZDDP-detergent` ведет себя как антагонистический блок для oxidation;
- `Zn-dispersant` и `antiwear-Ca/Mg` сильнее для viscosity;
- `AO-AO` сигнал есть, но он слабее и менее стабилен.

Что делать дальше:
- не использовать все пары одного типа;
- собрать shortlist конкретных устойчивых пар по критериям:
  - повторяемость по сценариям
  - устойчивость знака
  - переносимость между `recipe_cluster` fold-ами

Использование:
- только top stable pairs должны попадать в новые pairwise/add-on блоки.

Дополнение:
- pairwise-сигналы теперь используются как фильтр для новых признаков, а не как повод раздувать pairwise-модель.

### Этап 6. Residual models
Статус: частично выполнен, логика скорректирована.

#### Для viscosity
Что уже показал опыт:
- residual correction может быть полезен;
- но не широкий linear chemistry residual.

Что делать дальше:
- строить только narrow residual:
  - `antiwear-Ca/Mg`
  - `Zn-dispersant`
  - `ZDDP-detergent`
  - `base-oil-conditioned pair effects`
- residual должен быть узким и chemistry-aware.

#### Для oxidation
Что уже показал опыт:
- широкий residual ухудшает качество;
- even narrow residual тоже ухудшает качество;
- backbone должен оставаться устойчивой опорой;
- chemistry для oxidation лучше работает как add-on, а не как residual.

Что делать дальше:
- residual для oxidation пока не развивать;
- развивать only direct add-on;
- следующий кандидат: `chem5_core + 1 mechanism`, а не большой residual block.

### Этап 7. Pairwise architecture
Статус: есть рабочий прототип, стратегия сужается.

Что уже показал опыт:
- общий pairwise-блок на все пары слишком шумный;
- target-wise gated pairwise работает лучше, но тоже требует жесткого отбора сигналов.

Что делать дальше:
- вместо общего pairwise использовать narrow pairwise heads:
  - `AO-Mo head`
  - `AO-AO / amine-phenol head`
  - `ZDDP-detergent head`
  - `antiwear-Ca/Mg head`
- агрегировать только эти головы, а не все возможные пары.

Но:
- pairwise как standalone/main model не приоритет;
- сначала pairwise должен доказать пользу как источник нескольких sharp features.

### Этап 8. Нелинейности
Статус: вводить точечно.

Разрешенные нелинейности:
- `log1p`
- `sqrt`
- `ratio`
- `min/max`
- `imbalance`
- `hinge` по температуре и биотопливу:
  - `max(T-150, 0)`
  - `max(T-154, 0)`
  - `max(biofuel-5, 0)`

Правило:
- нелинейности вводим только в узких chemistry-blocks;
- wide nonlinear expansion запрещен как шумный путь;
- лучшие текущие нелинейности для oxidation уже найдены внутри ultra-compact блока:
  - `ao_pair_x_biofuel`
  - `ao_dpa_phenol_imbalance`

Дополнение:
- новые нелинейности разрешены только если они заменяют старые признаки, а не раздувают размерность;
- цель не увеличить число колонок, а повысить сигнал на одну колонку.

### Этап 9. Ensemble / Stacking
Статус: финальный production-этап.

Принцип:
- итоговый сабмишн не должен быть одной моделью;
- нужен target-wise legal ensemble или stacking.

Текущее безопасное направление:
- `viscosity = backbone + narrow residual`
- `oxidation = backbone + ultra-compact chemistry add-on`

Текущее рабочее правило:
- если chemistry add-on бьет backbone и на `recipe_cluster`, и хотя бы не ломается на `dominant_base_oil`, его можно брать в production-кандидаты;
- residual для oxidation пока вес не получает вообще.

Следующий приоритет:
- собирать stacking из:
  - global backbone
  - best chemistry add-on
  - возможно лучшего narrow residual для viscosity
- а не уходить в новые большие архитектуры.

### Этап 10. Submission strategy
Нужно работать не одной “идеальной” моделью, а серией разных target-wise сабмишнов.

Первые кандидаты:
- `baseline_backbone_only`
- `viscosity_backbone + narrow residual`
- `oxidation_backbone + chem5_core add-on`
- `safe blend`
- `aggressive blend`
- `stacked_blend`

### Этап 11. Aggressive feature pruning
Статус: активный постоянный принцип.

Правило:
- если короткий набор фич работает лучше длинного, длинный удаляется из активной ветки;
- новые признаки добавляются только через ablation against short core.

Цель:
- поддерживать минимальный рабочий набор признаков для каждого таргета;
- не копить feature debt.

## Приоритеты на ближайшие 3 дня

### День 1
- validation regime уже зафиксирован;
- backbone уже зафиксирован;
- собрать ultra-compact chemistry block для oxidation;
- собрать shortlist устойчивых chemistry-блоков и stable pairs.

### День 2
- обучить `oxidation direct ultra-compact add-on`;
- проверить 1-2 аккуратных расширения сверх `chem5_core`;
- обучить `viscosity narrow residual`;
- собрать 3-4 target-wise blending / stacking-кандидата.

### День 3
- дотюнить ensemble weights;
- отрезать шумные chemistry blocks;
- отправить 3-4 финальных submission-кандидата.

## Текущее состояние проекта

### Главный валидатор
- `recipe_cluster`

### Зафиксированные backbone
- `viscosity = compact_v3_full + MLP(relu_small)`
- `oxidation = compact_v3_full + Ridge(alpha=30.0)`

### Известно плохие направления
- широкий `chemistry_v3` для oxidation
- широкий линейный residual для viscosity
- broad merged chemistry/add-on blocks без жесткого отбора
- residual для oxidation как основной путь
- chemistry-only standalone модели вместо backbone
- MoE как основная production-ставка на текущем датасете
- одинаковый feature space для обоих таргетов

### Следующий правильный шаг
- не расширять feature space дальше вширь;
- держать `chem5_core` как текущий лучший oxidation add-on;
- тестировать только very small extensions поверх `chem5_core`;
- явно поддерживать два разных набора признаков:
  - `viscosity_dataset_vNext`
  - `oxidation_dataset_vNext`
- использовать aggressive pruning как стандартный фильтр;
- собрать **narrow residual for viscosity**;
- проверять все только под `recipe_cluster`;
- финально сравнивать уже не одиночные модели, а target-wise blends / stacking.
