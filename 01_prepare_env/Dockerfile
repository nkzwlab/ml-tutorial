# pythonのイメージを取得
FROM python:3

# dockerイメージ内にbuildディレクトリを作成。ローカルのコードをbuild/に反映。
RUN mkdir /build
COPY . /build/
WORKDIR /build/

# Ubuntuシステムの更新系
RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# pip系のインストール
RUN pip install --upgrade pip
RUN pip install -r requirements.txt