# Neural Dataset Collection

### Installation

```bash
git clone https://gitlab.ai.cloud.ru/manzherok/neudc.git
cd neudc
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .
```

### Development
```python
python3 -m pip install pre-commit
pre-commit install

# run pre-commit hooks
pre-commit run --all-files
```

### Starting project
```bash
python3 neudc/main.py configs/pipeline.yaml
```

### Run pytest
> Automatically runs all written tests for core module
```bash
python3 -m pytest test
```
