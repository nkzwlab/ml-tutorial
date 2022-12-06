filename ?=

init:
	docker-compose build
	docker-compose up -d

run-cmd:
	docker-compose exec cnn python -B ${filename}

down:
	docker-compose down

install:
	docker-compose exec cnn pip install -r requirements.txt