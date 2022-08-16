# dockerの起動
init:
	docker-compose build
	docker-compose up -d

# pythonの実行
run:
	docker-compose exec prepare-env python src/sample.py

# dockerの削除
down:
	docker-compose down