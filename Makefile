.PHONY: test verify

TEST ?= pytest

verify: test
	python -m scripts.verify_model_ready

test:
	$(TEST)
