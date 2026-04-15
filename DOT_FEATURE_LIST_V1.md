# DOT Feature List V1

## 1. Общая структура признаков

| Блок | Что содержит | Основные фичи | Зачем нужен |
| --- | --- | --- | --- |
| Scenario-level | Условия DOT и severity | temperature_c<br>time_h<br>biofuel_pct<br>catalyst_category<br>temperature_x_time<br>temperature_x_biofuel<br>time_x_biofuel<br>biofuel_x_catalyst<br>severity_exp = time_h * exp((temperature_c - 150) / 10) | Задают жёсткость режима и служат условием для всех межкомпонентных взаимодействий. |
| Token-level | Уникальный компонент в рецептуре | scenario_id<br>component_id<br>component_family<br>batch_id<br>dose_transformed<br>dose_rank_in_scenario<br>dose_share_of_total_transformed<br>row_count_after_merge | Каждый уникальный `(component_id, batch_id)` внутри сценария становится отдельным токеном. |
| Property-level | Свойства компонента | property_value::<name><br>property_present::<name><br>property_parse_kind::<name><br>property_source::<name> | Все свойства хранятся отдельно по именам показателей, без смешивания разных единиц и смыслов. |
| Family aggregates | Агрегаты по семействам | family_total_dose::<family><br>family_component_count::<family><br>family_weighted_mean::<family>::<property><br>family_max::<family>::<property><br>family_present_count::<family>::<property> | Дают устойчивое семейно-специфичное описание пакета присадок. |
| Mechanism blocks | Физико-химические блоки | base_oil_structure::*<br>antioxidant_activity::*<br>antiwear_redox::*<br>detergency_reserve::*<br>dispersancy_polarity::*<br>polymer_rheology::*<br>foam_control::*<br>low_temp_flow::* | Группируют признаки по механизму влияния на окисление и рост вязкости. |
| Interaction block | Явные взаимодействия | ao_x_zddp<br>ao_x_mo<br>detergent_x_dispersant<br>base_oil_x_ao<br>base_oil_x_dispersant<br>biofuel_x_ao<br>biofuel_x_zddp<br>temperature_x_ao<br>temperature_x_polymer<br>catalyst_x_sulfur_or_phosphorus_or_mo | Нужны для явного моделирования синергии и антагонизма. |

## 2. Семейства компонентов и релевантные свойства

| Семейство | Свойства в V1 | Теоретическая роль |
| --- | --- | --- |
| Базовое_масло | Группа по API<br>Кинематическая вязкость, при 40°C, ASTM D445<br>Кинематическая вязкость, при 100°C, ASTM D445<br>Динамическая вязкость CCS -15/-20/-25/-30/-35°C, ASTM D5293<br>Индекс вязкости, ГОСТ 25371<br>Температура застывания, ГОСТ 20287, метод Б<br>Испаряемость по NOACK, ASTM D5800<br>Плотность при 15°С, ASTM D4052<br>Плотность при 20°С, ASTM D4052<br>Анилиновая точка<br>Содержание ароматики<br>Содержание насыщ. у/в<br>Содержание серы, мг/кг<br>Содержание серы, % масс.<br>Деаэрация | ASTM D3427<br>Деэм.вода / масло / эмульсия / время | ASTM D1401<br>Последовательность 1/2/3 | ASTM D892<br>Цвет | ASTM D1500<br>Цвет Сейболт | - | Формируют фон окисляемости, летучести, solvency и реологии. |
| Антиоксидант | Тип АО<br>Номер CAS / SMILES<br>Активный Азот / Кислород, % масс. (N или O)<br>Температура плавления, °C<br>Энергия диссоциации связи Х-Н, ккал/моль<br>Потенциал ионизации,эВ<br>Химический потенциал, Дж/моль<br>Энергия ВЗМО, эВ<br>Энергия НСМО, эВ<br>Дипольный момент, Д<br>Стерический фактор, Å3 | Описывают способность тормозить радикальную цепь и разлагать гидропероксиды. |
| Детергент | Щелочное число, ASTM D2896<br>Щелочное число, ГОСТ 11362<br>Массовая доля кальция, ASTM D6481<br>Массовая доля кальция | ASTM D6481<br>Класс субстрата<br>Содержание мыла, % масс.<br>Содержание масла, % масс.<br>Содержание MgCO3, CaCO3, % масс.<br>Отношение Мыло/Основание<br>Содержание металла (Ca/Mg), % масс.<br>Размер мицелл, нм<br>Структура УВ-радикала<br>SMILES для наиболее вероятной (средней) молекулы сульфокислоты<br>Содержание воды, % масс. | Описывают щелочной резерв и коллоидную структуру детергентного пакета. |
| Дисперсант | Класс полиамина<br>Модификация<br>Тип сукцинимида<br>Содержание Азота<br>Содержание Бора<br>Масса гидрофобного хвоста, г/моль<br>Индекс полидисперсности<br>Содержание масла<br>Общее содержание азота | ASTM D3228 | Связаны с удержанием продуктов окисления и контролем sludge/нерастворимых фракций. |
| Противоизносная_присадка | Массовая доля фосфора, ASTM D6481<br>Массовая доля фосфора | ASTM D6481<br>Массовая доля цинка, ASTM D6481<br>Массовая доля цинка | ASTM D6481<br>Массовая доля серы, ASTM D6481<br>Массовая доля серы | ASTM D6481<br>Атомное отношение P:Zn<br>Тип спиртового радикала<br>Разветвленность радикала / радикалов<br>Длина углеродной цепи<br>Степень полисульфидности<br>Массовая доля сульфатной золы, ГОСТ 12417 | Формируют redox-блок ZDDP/antiwear и влияют на антиокислительную активность. |
| Соединение_молибдена | % масс. (Mo)<br>Категория<br>Тип лиганда<br>Отношение S:Mo<br>COC (°C) | Чаще всего работают через взаимодействия с антиоксидантами и antiwear-пакетом. |
| Загуститель | Тип полимера<br>Содержание полимера<br>Средневесовая масса<br>Соотношение мономеров (EO:PO)<br>Индекс стабильности, %<br>Кинематическая вязкость, при 100°C, ASTM D445 | Критичны для таргета `Delta KV100`, так как влияют на итоговую высокотемпературную вязкость. |
| Антипенная_присадка / Депрессорная_присадка | family presence flag<br>family total dose<br>редкие специфические свойства при наличии | Оставляем как контекстные признаки с сильной регуляризацией. |

## 3. Обработка данных

| Шаг | Что делаем |
| --- | --- |
| Join | Соединяем свойства по `(Компонент, Наименование партии)`. Если свойства партии нет, берём `typical`. |
| Merge duplicates | Если один и тот же `(scenario_id, component_id, batch_id)` встречается несколько раз, суммируем transformed dose. |
| Extract family | Из `component_id` извлекаем химическое семейство по шаблону `Название_число`. |
| Parse values | Числа оставляем как есть; диапазоны заменяем серединой; `<`/`>` превращаем в значение + флаг типа записи; мусор оставляем как missing. |
| Keep masks | Для каждого свойства сохраняем `present`, `parse_kind`, `source_measured_or_typical`. |
| Scale numeric | Числовые признаки масштабируем robust-скейлером внутри train folds. |
| Build aggregates | Считаем семейные и mechanism-level агрегаты по каждому свойству отдельно. |
| Build interactions | Добавляем физически мотивированные products между условиями теста и химическими блоками. |

## 4. Типы признаков в первой версии

### Numerical

- `temperature_c`
- `time_h`
- `biofuel_pct`
- `catalyst_category`
- `temperature_x_time`
- `temperature_x_biofuel`
- `time_x_biofuel`
- `biofuel_x_catalyst`
- `severity_exp`
- `dose_transformed`
- `dose_rank_in_scenario`
- `dose_share_of_total_transformed`

### Categorical

- `component_id`
- `component_family`
- `property_parse_kind::<name>`
- `property_source::<name>`
- `AO type`
- `ligand type`
- `substrate class`
- `polymer type`
- `polyamine class`
- `succinimide type`

### Masks / quality flags

- `property_present::<name>`
- `family_present::<family>`
- `family_component_count::<family>`
- `has_unseen_property_pattern`

## 5. Что исключаем из основных признаков

- `batch_id` как прямой категориальный признак.
- Пустой показатель `''`.
- Сырые `CAS` / `SMILES` как текстовые строки.
- Глобальные средние по всем свойствам сразу.
- Редкие свойства с экстремально малым покрытием как самостоятельные dense columns.
