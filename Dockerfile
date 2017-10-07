FROM object-detection-core
MAINTAINER Josh Kurz <jkurz25@gmail.com>

# APPLICATION
ADD . /opt/object_detection/

# # expose ports
# EXPOSE 5000

RUN mkdir /opt/graph_def && \
    mkdir /opt/configs

CMD ["/usr/bin/python3", "/opt/object_detection/app.py"]