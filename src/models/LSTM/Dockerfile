FROM tensorflow/serving:latest
COPY . 

CMD [docker run -p 8501:8501 
--mount type=bind,source=/path/to/the/unzipped/model/tmp/,target=/models/fashion_mnist -e 
MODEL_NAME=fashion_mnist -t tensorflow/serving]