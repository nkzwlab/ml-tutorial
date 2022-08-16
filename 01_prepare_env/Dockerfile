# pythonのイメージ（用意されている環境みたいなもの）を取得
FROM python:3

# dockerイメージ内にbuildディレクトリを作成。
RUN mkdir /build
# ローカルのコードを/buildに反映
COPY . /build/
# 作業ディレクトリを/buildに移動 -> dockerが立ち上がったら、/build/パスから始まる。
WORKDIR /build/

# Ubuntuシステムの更新系
RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# pip系のインストール
RUN pip install --upgrade pip
RUN pip install -r requirements.txt