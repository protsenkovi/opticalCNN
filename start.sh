docker run -d --gpus "device=0" -p 9001:9001 -v $(pwd):/wd ${USER}_tensorflow
