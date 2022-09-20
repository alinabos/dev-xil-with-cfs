FROM python:3.10-bullseye

RUN git clone https://github.com/alinabos/xil-with-cfs.git

RUN pip3 --disable-pip-version-check --no-cache-dir install -r /xil-with-cfs/requirements.txt \
    && rm -rf /pip-

ENTRYPOINT ["tail", "-f", "/dev/null"]