
python3 -m HQPINN
python3 -m HQPINN --config HQPINN/configs/dho_cc_run.json

Public entry point:
- `HQPINN.run_from_project(config)`

Runtime config:
- optional top-level `dtype` accepts aliases like `float32`, `float64`, `complex64`
- the logged JSON stays unchanged, but project code receives a validated `DtypeSpec`

Tests:
- run everything with `python3 -m unittest discover -s HQPINN/tests`
