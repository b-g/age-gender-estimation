#!/bin/bash
DOCKER_VOLUMES+="-v $(pwd)/:/shared "
shift
docker run --runtime=nvidia -it $DOCKER_VOLUMES --workdir /shared age-gender-estimation:v0 bash
