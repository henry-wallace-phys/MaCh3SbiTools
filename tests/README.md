# Tests
Testing is done automatically on PR to `main` using `pytest` with `.github/workflows/pytest.yml`. To perform tests locally please install with

```shell
pip install [-e] "[.dev]"
pytest .
```
This will also automatically generate a code coverage summary.  