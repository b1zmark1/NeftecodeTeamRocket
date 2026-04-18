# Patent Extraction For Daimler DOT

Из `US20250136889A1` и `US11142719B2` извлечены данные, которые можно использовать как supplemental raw data для хакатона по `нефтекод 2026 (2).txt`.

Что сохранено:
- `patent_component_properties_raw.csv` — component-like свойства из патентов.
- `patent_daimler_scenarios_raw.csv` — scenario-like условия и результаты Daimler oxidation tests.
- `patent_mixture_components_partial.csv` — известные строки состава для патентных рецептур, где массовые доли раскрыты не полностью.

Почему это не было влито прямо в `daimler_component_properties.csv` и `daimler_mixtures_train.csv`:
- в исходном train компоненты обезличены (`Антиоксидант_5`, `Детергент_4`, ...), а в патентах компоненты открытые;
- для массовых долей в хакатоне нужен отдельный сервис преобразования raw `%` -> transformed `%`;
- часть рецептур в патентах раскрыта неполно: в таблицах есть фраза вида `each oil had the same amount of ...`, но без численных долей.

Практически полезно уже сейчас:
- использовать `patent_component_properties_raw.csv` как внешний literature prior по семействам `Антиоксидант` и `Детергент`;
- использовать `patent_daimler_scenarios_raw.csv` как внешние scenario-level наблюдения для qualitative анализа влияния состава;
- после появления маппинга на ваши anonymized component ids и сервиса трансформации долей можно собирать отдельный literature-augmented train.

Источники:
- `US20250136889A1`: https://patents.google.com/patent/US20250136889A1/en
- `US11142719B2`: https://patents.google.com/patent/US11142719
