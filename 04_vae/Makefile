filename ?=

init:
	docker-compose build
	docker-compose up -d

run-cmd:
	docker-compose exec vae python -B ${filename}

down:
	docker-compose down

install:
	docker-compose exec vae pip install -r requirements.txt