filename ?=

init:
	docker-compose build
	docker-compose up -d

run-cmd:
	docker-compose exec linear python -B ${filename}

down:
	docker-compose down

install:
	docker-compose exec linear pip install -r requirements.txt