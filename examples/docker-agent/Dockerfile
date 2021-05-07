FROM python:3.6

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install .
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
ADD ./examples/docker-agent /agent
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["A2C_run.py"]
