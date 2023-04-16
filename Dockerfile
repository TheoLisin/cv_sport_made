FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
ENV TZ="Europe/Tallinn"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
   python3 \
   python3-pip \
   ffmpeg libsm6 libxext6 \
   && rm -rf /var/lib/apt/lists/*

RUN mkdir /codereqs
COPY requirements.txt /codereqs/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /codereqs/requirements.txt