FROM openfabric/tee-python-cpu:latest

RUN mkdir application
WORKDIR /application
COPY . .
RUN poetry install -vvv --no-dev

RUN pip install -r requirements.txt

EXPOSE 5500
CMD ["sh","start.sh"]