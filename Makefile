.PHONY: test run docker-up docker-down

test:
	python run_tests.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d

docker-down:
	docker compose down
