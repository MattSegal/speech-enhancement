FROM nvidia/cuda:9.0-base
WORKDIR /app
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED 1
ENV PYTHONIOENCODING UTF-8
ENV TIMEZONE Australia/Melbourne

# Install apt packages
RUN \
    echo "Updating apt sources." && \
    apt-get -qq update && \
    echo "Installing required packages." && \
    apt-get -qq install \
    software-properties-common \
    wget \
    curl \
    ffmpeg \
    sox


RUN add-apt-repository ppa:jonathonf/python-3.6
RUN \
    echo "Updating apt sources." && \
    apt-get -qq update && \
    echo "Installing Python 3 packages." && \
    apt-get -qq install \
    python3.6 \
    python3.6-dev \
    libpython3.6-dev

# Install Python packages
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN apt-get install -qq build-essential
COPY requirements.txt .
RUN \
    echo "Installing python packages..." && \
    pip install -r requirements.txt

RUN \
    echo "Creating Python symlink." && \
    ln -s /usr/bin/python3.6 /usr/bin/python

# Check python version
RUN  \
    echo "Verifying Python version..." && \
    python -V

# Mount the codebase
ADD . /app
