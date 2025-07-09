# set a base image that includes lambda Runtime API:
# Source: https://hub.docker.com/r/amazon/aws-lambda-python
FROM amazon/aws-lambda-python:3.12

ARG DEPENDENCIES

# optional: ensure that pip is up to date
RUN /var/lang/bin/python3.12 -m pip install --upgrade pip setuptools wheel

# first we COPY only requirements.txt to ensure that later builds
# with changes to your src code will be faster due to caching of this layer
RUN dnf update && dnf -y install gcc-c++ libpq-devel
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install cython==0.29.36
RUN pip install sparse-dot-topn-for-blocks==0.3.1.post3 --no-build-isolation
# copy all your custom modules and files from the src directory
COPY src/ .
COPY $DEPENDENCIES/* data/

# specify Lambda handler that will be invoked on container start
CMD ["datapreparation.data_preparation.prepare_data"]