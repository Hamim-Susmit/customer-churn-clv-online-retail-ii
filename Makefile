setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

notebook:
	. .venv/bin/activate && jupyter notebook
