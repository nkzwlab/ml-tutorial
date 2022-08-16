# 01. pythonの環境構築
## 何をするのか
dockerを使ってpythonの環境構築を行います。<br>

☟dockerを使うメリット☟
- ローカル環境を汚さずに、様々なライブラリを入れられる！
- 研究テーマごとに仮想環境を作れる！
- GPU上にも同じ環境をすぐ作れる！

## 使い方
1. 01_prepare_envディレクトリに移動
```
$ cd 01_prepare_env
```
2. dockerで環境構築
```
$ make init
```
3. pythonのコマンド実行
```
$ make run

  出力結果☟
  docker-compose exec prepare-env python src/sample.py
  hello docker
```

## 使っている技術
- python
- Docker