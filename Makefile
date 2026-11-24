PYTHON ?= python

.PHONY: install test anonymity-check build-colab pilot-clip pilot-siglip

install:
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

test:
	$(PYTHON) -m pytest -v

anonymity-check:
	$(PYTHON) -m pytest -v tests/test_anonymity_and_clean_code.py

build-colab:
	$(PYTHON) scripts/build_pilot_f1_colab.py

pilot-clip:
	$(PYTHON) -m mnarm.runners.pilot_f1 --config configs/pilot_f1_clip_p4.yaml

pilot-siglip:
	$(PYTHON) -m mnarm.runners.pilot_f1 --config configs/pilot_f1_siglip_p4.yaml
