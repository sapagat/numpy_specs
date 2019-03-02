build:
	docker-compose build

test:
	docker-compose run app pipenv run mamba --format=documentation
