# Refactor of opticalCNN single-layer optical correlator for QuickDraw-16 example.

Original repository https://github.com/computational-imaging/opticalCNN

## How to start
To reproduce results for single-layer optical correlator for QuickDraw-16:
```
$ ./build.sh
$ ./start.sh
$ docker logs <container_id>
```
Connect to http://ip:9001, use token from the logs. Run all cells in `main.ipynb`.