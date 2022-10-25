FROM tensorflow/tensorflow:2.9.1-gpu

ARG USER
ARG GROUP
ARG UID
ARG GID

RUN groupadd -g ${GID} ${GROUP}
RUN useradd -u ${UID} -g ${GROUP} -s /bin/bash -m ${USER} 

RUN mkdir /wd
RUN chown ${USER}:${GROUP} /wd
WORKDIR /wd

RUN apt update; apt install mc -y


USER ${UID}:${GID}

RUN pip install jupyterlab tqdm matplotlib scikit-learn scikit-image

EXPOSE 9001

SHELL ["/bin/bash", "--login", "-i", "-c"]
ENV SHELL=/bin/bash

CMD jupyter lab --ip 0.0.0.0 --port 9001
