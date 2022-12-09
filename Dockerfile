FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

#opencv deps
RUN apt-get update && \
    apt-get install -y python3-opencv

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]