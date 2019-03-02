build:
	docker-compose build

test:
	docker-compose run --rm app pipenv run mamba --format=documentation
