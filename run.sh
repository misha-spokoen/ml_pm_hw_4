#!/bin/sh
# Полный пересчёт: панель, модели, разметка, эвалы, ноутбук, тесты. Около 10 минут на 8 ядрах, 8 ГБ памяти.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -q -r requirements.txt
fi
cd src
../.venv/bin/python -u labeling.py > /dev/null
../.venv/bin/python -u experiment.py all
../.venv/bin/python -u analysis.py > /dev/null
../.venv/bin/python -u evals.py
cd ..
.venv/bin/python -u notebooks/build_notebook.py
.venv/bin/python -m unittest discover -s tests
