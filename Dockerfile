FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

COPY . /app

CMD ["python", "-c", "import json; from pathlib import Path; nb=json.loads(Path('inference.ipynb').read_text(encoding='utf-8')); ns={'__name__':'__main__'}; [exec(compile(''.join(cell.get('source', [])), f'inference_cell_{i}', 'exec'), ns) for i, cell in enumerate(nb['cells']) if cell.get('cell_type') == 'code']"]
