FROM nvcr.io/nvidia/pytorch:22.03-py3

WORKDIR /app
RUN apt-get update
RUN apt-get install -y software-properties-common tzdata
ENV TZ=Asia/Tokyo 
RUN apt-get install -y ffmpeg
RUN pip install -r requirements.txt
