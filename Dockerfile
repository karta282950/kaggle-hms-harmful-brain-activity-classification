FROM pytorch/pytorch:latest
RUN pip install -r requirements.txt
WORKDIR /models
COPY ./script ./
EXPOSE 5002
ENTRYPOINT ["bash","run.sh"]