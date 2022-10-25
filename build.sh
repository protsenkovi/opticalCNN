#/bin/bash

docker build . \
	-t ${USER}_tensorflow \
	--build-arg USER=${USER} \
	--build-arg GROUP=${USER} \
	--build-arg UID=$(id -u ${USER}) \
	--build-arg GID=$(id -g ${USER})
