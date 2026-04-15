from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "DOT_FEATURE_LIST_V1.md"
DOCX_PATH = ROOT / "DOT_FEATURE_LIST_V1.docx"


FEATURE_GROUPS = [
    (
        "Scenario-level",
        "Условия DOT и severity",
        [
            "temperature_c",
            "time_h",
            "biofuel_pct",
            "catalyst_category",
            "temperature_x_time",
            "temperature_x_biofuel",
            "time_x_biofuel",
            "biofuel_x_catalyst",
            "severity_exp = time_h * exp((temperature_c - 150) / 10)",
        ],
        "Задают жёсткость режима и служат условием для всех межкомпонентных взаимодействий.",
    ),
    (
        "Token-level",
        "Уникальный компонент в рецептуре",
        [
            "scenario_id",
            "component_id",
            "component_family",
            "batch_id",
            "dose_transformed",
            "dose_rank_in_scenario",
            "dose_share_of_total_transformed",
            "row_count_after_merge",
        ],
        "Каждый уникальный `(component_id, batch_id)` внутри сценария становится отдельным токеном.",
    ),
    (
        "Property-level",
        "Свойства компонента",
        [
            "property_value::<name>",
            "property_present::<name>",
            "property_parse_kind::<name>",
            "property_source::<name>",
        ],
        "Все свойства хранятся отдельно по именам показателей, без смешивания разных единиц и смыслов.",
    ),
    (
        "Family aggregates",
        "Агрегаты по семействам",
        [
            "family_total_dose::<family>",
            "family_component_count::<family>",
            "family_weighted_mean::<family>::<property>",
            "family_max::<family>::<property>",
            "family_present_count::<family>::<property>",
        ],
        "Дают устойчивое семейно-специфичное описание пакета присадок.",
    ),
    (
        "Mechanism blocks",
        "Физико-химические блоки",
        [
            "base_oil_structure::*",
            "antioxidant_activity::*",
            "antiwear_redox::*",
            "detergency_reserve::*",
            "dispersancy_polarity::*",
            "polymer_rheology::*",
            "foam_control::*",
            "low_temp_flow::*",
        ],
        "Группируют признаки по механизму влияния на окисление и рост вязкости.",
    ),
    (
        "Interaction block",
        "Явные взаимодействия",
        [
            "ao_x_zddp",
            "ao_x_mo",
            "detergent_x_dispersant",
            "base_oil_x_ao",
            "base_oil_x_dispersant",
            "biofuel_x_ao",
            "biofuel_x_zddp",
            "temperature_x_ao",
            "temperature_x_polymer",
            "catalyst_x_sulfur_or_phosphorus_or_mo",
        ],
        "Нужны для явного моделирования синергии и антагонизма.",
    ),
]


FAMILY_PROPERTIES = [
    (
        "Базовое_масло",
        [
            "Группа по API",
            "Кинематическая вязкость, при 40°C, ASTM D445",
            "Кинематическая вязкость, при 100°C, ASTM D445",
            "Динамическая вязкость CCS -15/-20/-25/-30/-35°C, ASTM D5293",
            "Индекс вязкости, ГОСТ 25371",
            "Температура застывания, ГОСТ 20287, метод Б",
            "Испаряемость по NOACK, ASTM D5800",
            "Плотность при 15°С, ASTM D4052",
            "Плотность при 20°С, ASTM D4052",
            "Анилиновая точка",
            "Содержание ароматики",
            "Содержание насыщ. у/в",
            "Содержание серы, мг/кг",
            "Содержание серы, % масс.",
            "Деаэрация | ASTM D3427",
            "Деэм.вода / масло / эмульсия / время | ASTM D1401",
            "Последовательность 1/2/3 | ASTM D892",
            "Цвет | ASTM D1500",
            "Цвет Сейболт | -",
        ],
        "Формируют фон окисляемости, летучести, solvency и реологии.",
    ),
    (
        "Антиоксидант",
        [
            "Тип АО",
            "Номер CAS / SMILES",
            "Активный Азот / Кислород, % масс. (N или O)",
            "Температура плавления, °C",
            "Энергия диссоциации связи Х-Н, ккал/моль",
            "Потенциал ионизации,эВ",
            "Химический потенциал, Дж/моль",
            "Энергия ВЗМО, эВ",
            "Энергия НСМО, эВ",
            "Дипольный момент, Д",
            "Стерический фактор, Å3",
        ],
        "Описывают способность тормозить радикальную цепь и разлагать гидропероксиды.",
    ),
    (
        "Детергент",
        [
            "Щелочное число, ASTM D2896",
            "Щелочное число, ГОСТ 11362",
            "Массовая доля кальция, ASTM D6481",
            "Массовая доля кальция | ASTM D6481",
            "Класс субстрата",
            "Содержание мыла, % масс.",
            "Содержание масла, % масс.",
            "Содержание MgCO3, CaCO3, % масс.",
            "Отношение Мыло/Основание",
            "Содержание металла (Ca/Mg), % масс.",
            "Размер мицелл, нм",
            "Структура УВ-радикала",
            "SMILES для наиболее вероятной (средней) молекулы сульфокислоты",
            "Содержание воды, % масс.",
        ],
        "Описывают щелочной резерв и коллоидную структуру детергентного пакета.",
    ),
    (
        "Дисперсант",
        [
            "Класс полиамина",
            "Модификация",
            "Тип сукцинимида",
            "Содержание Азота",
            "Содержание Бора",
            "Масса гидрофобного хвоста, г/моль",
            "Индекс полидисперсности",
            "Содержание масла",
            "Общее содержание азота | ASTM D3228",
        ],
        "Связаны с удержанием продуктов окисления и контролем sludge/нерастворимых фракций.",
    ),
    (
        "Противоизносная_присадка",
        [
            "Массовая доля фосфора, ASTM D6481",
            "Массовая доля фосфора | ASTM D6481",
            "Массовая доля цинка, ASTM D6481",
            "Массовая доля цинка | ASTM D6481",
            "Массовая доля серы, ASTM D6481",
            "Массовая доля серы | ASTM D6481",
            "Атомное отношение P:Zn",
            "Тип спиртового радикала",
            "Разветвленность радикала / радикалов",
            "Длина углеродной цепи",
            "Степень полисульфидности",
            "Массовая доля сульфатной золы, ГОСТ 12417",
        ],
        "Формируют redox-блок ZDDP/antiwear и влияют на антиокислительную активность.",
    ),
    (
        "Соединение_молибдена",
        [
            "% масс. (Mo)",
            "Категория",
            "Тип лиганда",
            "Отношение S:Mo",
            "COC (°C)",
        ],
        "Чаще всего работают через взаимодействия с антиоксидантами и antiwear-пакетом.",
    ),
    (
        "Загуститель",
        [
            "Тип полимера",
            "Содержание полимера",
            "Средневесовая масса",
            "Соотношение мономеров (EO:PO)",
            "Индекс стабильности, %",
            "Кинематическая вязкость, при 100°C, ASTM D445",
        ],
        "Критичны для таргета `Delta KV100`, так как влияют на итоговую высокотемпературную вязкость.",
    ),
    (
        "Антипенная_присадка / Депрессорная_присадка",
        [
            "family presence flag",
            "family total dose",
            "редкие специфические свойства при наличии",
        ],
        "Оставляем как контекстные признаки с сильной регуляризацией.",
    ),
]


PIPELINE_STEPS = [
    ("Join", "Соединяем свойства по `(Компонент, Наименование партии)`. Если свойства партии нет, берём `typical`."),
    ("Merge duplicates", "Если один и тот же `(scenario_id, component_id, batch_id)` встречается несколько раз, суммируем transformed dose."),
    ("Extract family", "Из `component_id` извлекаем химическое семейство по шаблону `Название_число`."),
    ("Parse values", "Числа оставляем как есть; диапазоны заменяем серединой; `<`/`>` превращаем в значение + флаг типа записи; мусор оставляем как missing."),
    ("Keep masks", "Для каждого свойства сохраняем `present`, `parse_kind`, `source_measured_or_typical`."),
    ("Scale numeric", "Числовые признаки масштабируем robust-скейлером внутри train folds."),
    ("Build aggregates", "Считаем семейные и mechanism-level агрегаты по каждому свойству отдельно."),
    ("Build interactions", "Добавляем физически мотивированные products между условиями теста и химическими блоками."),
]


NUMERICAL_COLUMNS_V1 = [
    "temperature_c",
    "time_h",
    "biofuel_pct",
    "catalyst_category",
    "temperature_x_time",
    "temperature_x_biofuel",
    "time_x_biofuel",
    "biofuel_x_catalyst",
    "severity_exp",
    "dose_transformed",
    "dose_rank_in_scenario",
    "dose_share_of_total_transformed",
]


CATEGORICAL_COLUMNS_V1 = [
    "component_id",
    "component_family",
    "property_parse_kind::<name>",
    "property_source::<name>",
    "AO type",
    "ligand type",
    "substrate class",
    "polymer type",
    "polyamine class",
    "succinimide type",
]


MASK_COLUMNS_V1 = [
    "property_present::<name>",
    "family_present::<family>",
    "family_component_count::<family>",
    "has_unseen_property_pattern",
]


def lines_to_markdown() -> str:
    lines = [
        "# DOT Feature List V1",
        "",
        "## 1. Общая структура признаков",
        "",
        "| Блок | Что содержит | Основные фичи | Зачем нужен |",
        "| --- | --- | --- | --- |",
    ]
    for block, meaning, features, goal in FEATURE_GROUPS:
        lines.append(f"| {block} | {meaning} | {'<br>'.join(features)} | {goal} |")

    lines.extend(
        [
            "",
            "## 2. Семейства компонентов и релевантные свойства",
            "",
            "| Семейство | Свойства в V1 | Теоретическая роль |",
            "| --- | --- | --- |",
        ]
    )
    for family, props, role in FAMILY_PROPERTIES:
        lines.append(f"| {family} | {'<br>'.join(props)} | {role} |")

    lines.extend(
        [
            "",
            "## 3. Обработка данных",
            "",
            "| Шаг | Что делаем |",
            "| --- | --- |",
        ]
    )
    for step, text in PIPELINE_STEPS:
        lines.append(f"| {step} | {text} |")

    lines.extend(
        [
            "",
            "## 4. Типы признаков в первой версии",
            "",
            "### Numerical",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in NUMERICAL_COLUMNS_V1])
    lines.extend(["", "### Categorical", ""])
    lines.extend([f"- `{item}`" for item in CATEGORICAL_COLUMNS_V1])
    lines.extend(["", "### Masks / quality flags", ""])
    lines.extend([f"- `{item}`" for item in MASK_COLUMNS_V1])
    lines.extend(
        [
            "",
            "## 5. Что исключаем из основных признаков",
            "",
            "- `batch_id` как прямой категориальный признак.",
            "- Пустой показатель `''`.",
            "- Сырые `CAS` / `SMILES` как текстовые строки.",
            "- Глобальные средние по всем свойствам сразу.",
            "- Редкие свойства с экстремально малым покрытием как самостоятельные dense columns.",
        ]
    )
    return "\n".join(lines) + "\n"


def xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def paragraph(text: str, bold: bool = False) -> str:
    safe = xml_text(text)
    if bold:
        return (
            "<w:p><w:r><w:rPr><w:b/></w:rPr>"
            f"<w:t xml:space=\"preserve\">{safe}</w:t>"
            "</w:r></w:p>"
        )
    return f"<w:p><w:r><w:t xml:space=\"preserve\">{safe}</w:t></w:r></w:p>"


def table(rows: list[list[str]]) -> str:
    parts = [
        "<w:tbl>",
        "<w:tblPr><w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/>"
        "</w:tblBorders></w:tblPr>",
    ]
    for row in rows:
        parts.append("<w:tr>")
        for cell in row:
            parts.append("<w:tc><w:p><w:r><w:t xml:space=\"preserve\">")
            parts.append(xml_text(cell))
            parts.append("</w:t></w:r></w:p></w:tc>")
        parts.append("</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


def build_document_xml() -> str:
    body = []
    body.append(paragraph("DOT Feature List V1", bold=True))
    body.append(paragraph("Документ фиксирует первую версию набора признаков для моделирования результатов Daimler Oxidation Test."))

    body.append(paragraph("1. Общая структура признаков", bold=True))
    rows = [["Блок", "Что содержит", "Основные фичи", "Зачем нужен"]]
    for block, meaning, features, goal in FEATURE_GROUPS:
        rows.append([block, meaning, "; ".join(features), goal])
    body.append(table(rows))

    body.append(paragraph("2. Семейства компонентов и релевантные свойства", bold=True))
    rows = [["Семейство", "Свойства в V1", "Теоретическая роль"]]
    for family, props, role in FAMILY_PROPERTIES:
        rows.append([family, "; ".join(props), role])
    body.append(table(rows))

    body.append(paragraph("3. Обработка данных", bold=True))
    rows = [["Шаг", "Что делаем"]]
    for step, text in PIPELINE_STEPS:
        rows.append([step, text])
    body.append(table(rows))

    body.append(paragraph("4. Типы признаков в первой версии", bold=True))
    body.append(paragraph("Numerical", bold=True))
    for item in NUMERICAL_COLUMNS_V1:
        body.append(paragraph(f"- {item}"))
    body.append(paragraph("Categorical", bold=True))
    for item in CATEGORICAL_COLUMNS_V1:
        body.append(paragraph(f"- {item}"))
    body.append(paragraph("Masks / quality flags", bold=True))
    for item in MASK_COLUMNS_V1:
        body.append(paragraph(f"- {item}"))

    body.append(paragraph("5. Что исключаем из основных признаков", bold=True))
    excluded = [
        "batch_id как прямой категориальный признак",
        "Пустой показатель ''",
        "Сырые CAS / SMILES как текстовые строки",
        "Глобальные средние по всем свойствам сразу",
        "Редкие свойства с экстремально малым покрытием как самостоятельные dense columns",
    ]
    for item in excluded:
        body.append(paragraph(f"- {item}"))

    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:w15=\"http://schemas.microsoft.com/office/word/2012/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 w15 wp14\">"
        "<w:body>"
        + "".join(body)
        + sect
        + "</w:body></w:document>"
    )


def write_docx(path: Path) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>DOT Feature List V1</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>
</w:styles>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        zf.writestr("word/document.xml", build_document_xml())
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", styles)


def main() -> None:
    MD_PATH.write_text(lines_to_markdown(), encoding="utf-8")
    write_docx(DOCX_PATH)
    print(MD_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
