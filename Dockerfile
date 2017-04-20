FROM tensorflow/tensorflow:1.0.1-devel-gpu

WORKDIR /usr/src/app/

COPY requirements.txt /usr/src/app/
RUN pip --no-cache-dir install -r /usr/src/app/requirements.txt

COPY . /usr/src/app
RUN pip install -e /usr/src/app
