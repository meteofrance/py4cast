ARG DOCKER_REGISTRY=docker.io
ARG TORCH_VERS=2.4.1
ARG CUDA_VERS=12.1

ARG INJECT_MF_CERT

###################################################
### Custom base image if a crt file is provided ###
###################################################
FROM ${DOCKER_REGISTRY}/pytorch/pytorch:${TORCH_VERS}-cuda${CUDA_VERS}-cudnn9-devel as BASE_CRT_INJECTED_1

COPY mf.crt /usr/local/share/ca-certificates/mf.crt
# The following two lines are necessary to deal with the MITM sniffing proxy we have internally.
RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"
ENV REQUESTS_CA_BUNDLE="/usr/local/share/ca-certificates/mf.crt"
ENV CURL_CA_BUNDLE="/usr/local/share/ca-certificates/mf.crt"


#####################################################
### Default base image if no crt file is provided ###
#####################################################
FROM ${DOCKER_REGISTRY}/pytorch/pytorch:${TORCH_VERS}-cuda${CUDA_VERS}-cudnn9-devel as BASE_CRT_INJECTED_0


#################################################
### Final image inherited from the base image ###
#################################################
FROM BASE_CRT_INJECTED_${INJECT_MF_CERT}

# set apt to non interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN $MY_APT update && $MY_APT install -y software-properties-common && add-apt-repository ppa:ubuntugis/ppa
RUN $MY_APT update && $MY_APT install -y curl gdal-bin libgdal-dev libgeos-dev git vim nano sudo libx11-dev tk python3-tk tk-dev libpng-dev libffi-dev dvipng texlive-latex-base texlive-latex-extra  texlive-fonts-recommended cm-super openssh-server netcat libeccodes-dev libeccodes-tools openssh-server

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN set -eux \
    && pip install --default-timeout=100 -r /app/requirements.txt
COPY requirements_lint.txt /app/requirements_lint.txt
RUN pip install --default-timeout=100 -r /app/requirements_lint.txt

ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG NODE_EXTRA_CA_CERTS

RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd \
    && mkdir /run/sshd

WORKDIR $HOME_DIR
RUN curl -fsSL https://code-server.dev/install.sh | sh
