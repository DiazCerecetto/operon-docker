FROM continuumio/miniconda3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y \
    default-jdk \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    wget \
    vim \
    build-essential \
    jq 

RUN pip install --upgrade pip

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

RUN conda env create -f /home/app/environment.yml
SHELL ["/bin/bash", "-c"]


WORKDIR /home/app
VOLUME /home/app
COPY . .