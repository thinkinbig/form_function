FROM python:3.10-slim-buster
LABEL authors="zeyuli"

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3490

CMD ["/bin/bash", "start.sh"]
