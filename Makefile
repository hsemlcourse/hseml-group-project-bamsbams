.PHONY: lint test data docker-build docker-run

lint:
	flake8 src tests

data:
	python3 -m src.data_parsing

docker-build:
	docker build -t hseml-pinn:latest .

docker-run:
	docker compose run --rm project
