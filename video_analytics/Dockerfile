From nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

WORKDIR /app

RUN apt update \
    && apt -y upgrade \
    && apt install -y python3 \
    && apt install -y python3-pip

COPY requirements.txt /app

RUN pip install -r requirements.txt 

# This dependency is for nuimages
RUN apt-get update -y \
    && apt install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-0 

COPY  main.py metric.py op.py utils.py /app

EXPOSE 12345

CMD ["python3", "main.py"]