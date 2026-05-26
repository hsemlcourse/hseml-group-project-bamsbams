.PHONY: lint test data docker-build docker-run docker-build-api docker-run-api docker-build-streamlit docker-run-streamlit

lint:
	flake8 src tests

data:
	python3 -m src.data_parsing

docker-build:
	docker build -t hseml-pinn:latest .

docker-run:
	docker compose run --rm project

docker-build-api:
	docker compose build api

docker-run-api:
	docker compose up api

docker-build-streamlit:
	docker compose build streamlit

docker-run-streamlit:
	docker compose up streamlit
