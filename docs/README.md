# Building Docs Locally
Docs are generated automatically by `.github/workflows/docs.yml`. To build locally install with the docs option

```shell
pip install ".[docs]" 
```

Then build the docs with
```shell
sphinx-build -b html --keep-going docs docs/_build/html     
```