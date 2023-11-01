################## Base Image: python-slim ##################
FROM python:3.8.16 AS base
LABEL MAINTAINER="Chowagiken"

# ENV SET    
ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive

# Update system and install relevant packages
RUN apt-get update && apt-get -yq install --no-install-recommends \
    curl \
    openjdk-17-jre-headless=17.0.6+10-1~deb11u1 \
    sudo && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
##############################################################


################## Build Image: install-deps ##################
FROM base AS builder
WORKDIR /app

# ENV SET    
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.3.2

# Install Poetry
ENV POETRY_HOME=/etc/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install project dependencies
RUN python3 -m venv .venv
COPY poetry.lock pyproject.toml ./
RUN . .venv/bin/activate && \
    # nvidia cuda toolkit
    pip install nvidia-cuda-nvrtc-cu11 && \
    poetry install \
        --without dev,docs,test,api \
        --no-interaction \
        --no-ansi \
        --no-root && \
    # install only headless version of opencv
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install opencv-python-headless==4.7.0.72

# copy code for installing
COPY hamacho ./hamacho
COPY README.md ./

# Install additional development dependencies
# and install hamacho
ADD docker/install.sh ./
ARG INSTALL_GROUPS
ENV INSTALL_GROUPS=${INSTALL_GROUPS:-prod}
ARG INSTALL_TYPE
ENV INSTALL_TYPE=${INSTALL_TYPE:-editable}
ENV VENV_PATH="/app/.venv"
RUN ./install.sh
###############################################################


################## Code Image: add needed code ##################
FROM python:3.8.16 AS code
WORKDIR /app
ENV CODE_FOLDER=code
ENV TEMP_FOLDER=temp
COPY hamacho ./${CODE_FOLDER}/hamacho
COPY api ./${TEMP_FOLDER}/api
COPY obfuscated/ ./${TEMP_FOLDER}/obfuscated
COPY tests ./${TEMP_FOLDER}/tests
ADD docker/folder_ops.sh ./

ARG INSTALL_GROUPS
ENV COPY_GROUPS=${INSTALL_GROUPS:-hamacho}
ARG INSTALL_TYPE
ENV INSTALL_TYPE=${INSTALL_TYPE:-editable}
RUN ./folder_ops.sh
###############################################################


################## Backbone Image: cache backbone ##################
FROM base AS backbone
WORKDIR /app

# Set default directory to save/load pytorch checkpoints 
ENV TORCH_HOME="/app/.torch"
ADD docker/download_torch_checkpoints.sh ./
RUN ./download_torch_checkpoints.sh

################## Final Runtime Image ##################
FROM base AS runtime
WORKDIR /app/
# Copy Poetry environment
COPY --from=builder /app/.venv ./.venv
# Update PATH
ENV PATH="/app/.venv/bin:$PATH"
# Copy backbone
ENV TORCH_HOME="/app/.torch"
COPY --from=backbone /app/.torch ./.torch
# Copy code files
COPY --from=code /app/code/ ./
###############################################################
