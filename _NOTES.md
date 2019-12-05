## Install
1. `nvidia-docker build -t age-gender-estimation:v0 .`

## Running

```
./run-docker.sh
```

in docker container:
```
python3 demo.py --image_dir output/

python3 gender-filter.py --input-folder output/test/frames --output-folder output/test/


```