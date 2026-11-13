PYTHON ?= python3
OUT ?= results/local

.PHONY: build test bench quick plot
build:
	$(PYTHON) build.py

test:
	$(PYTHON) check.py

bench: build
	$(PYTHON) benchmark.py --suite full --output $(OUT)

quick: build
	$(PYTHON) benchmark.py --suite quick --rounds 2 --samples 5 --min-sample-ms 20 --output $(OUT)

plot:
	$(PYTHON) plot.py $(OUT)
