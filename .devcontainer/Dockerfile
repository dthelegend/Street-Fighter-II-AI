FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl git

ENV PATH="/root/.local/bin:$PATH"

RUN echo "done 0" \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true


