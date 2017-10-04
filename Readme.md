# Object Detection Docker
A fork of the original [object detection example](https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example) using the Tensorflow Object Detection API

### Prerequisites
* have a .pb frozen model
* have a .pbtxt data label file

### Example Usage
* update the docker-compose file to map your two files into the container
`docker-compose up`
* you are now running your model on localhost:80

