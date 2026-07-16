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
python3 neudc/main.py assets/images
```

### Run pytest
Automatically runs all written tests for core module
```bash
python3 -m pytest test
```

### License

Copyright (C) 2025–2026 Sergey Sysoev

This project is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.

Note that the AGPL requires anyone who runs a modified version of this software as a
network service to make the corresponding source code available to its users.
