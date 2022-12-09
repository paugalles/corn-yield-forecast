
PROJ_NAME=corn
CONTAINER_NAME="${PROJ_NAME}-${USER}"

GEN_PORT=9099
NB_PORT=8088
MLF_PORT=5055

help:
	@echo "build -- builds the docker image"
	@echo "container -- runs a container in the background"
	@echo "nb -- launches a notebook server"
	@echo "nbstop -- Stops the notebook server"
	@echo "mlf -- launches an mlflow server"
	@echo "execsh -- raises an interactive shell docker from a running container"
	@echo "test -- run tests"
	@echo "stop -- Stop the docker container and everything that is running within it"

build:
	docker build -t ${PROJ_NAME} .

container:
	docker run --rm -itd --name $(CONTAINER_NAME) --gpus all --privileged \
	--shm-size=8gb \
	-p 9099:9099 \
	-p $(NB_PORT):$(NB_PORT) \
	-p $(MLF_PORT):$(MLF_PORT) \
	-v $(shell pwd):$(shell pwd) \
	-w $(shell pwd) \
	${PROJ_NAME} /bin/bash

nb:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab \
	--NotebookApp.token='corn' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=$(NB_PORT)

nbstop:
	docker exec -d --privileged $(CONTAINER_NAME) \
	jupyter lab stop ${NB_PORT}

mlf:
	docker exec -d $(CONTAINER_NAME) mlflow ui --host 0.0.0.0:$(MLF_PORT)

execsh:
	docker exec -it ${CONTAINER_NAME} /bin/bash

test:
	docker exec -it -w $(shell pwd) ${CONTAINER_NAME} pip3 install pytest flake8
	docker exec -it -w $(shell pwd) ${CONTAINER_NAME} pytest tests/

stop:
	docker stop $(CONTAINER_NAME)
