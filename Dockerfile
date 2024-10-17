ARG DOCKER_REGISTRY=docker.io
ARG TORCH_VERS=2.2.2
ARG CUDA_VERS=12.1

FROM ${DOCKER_REGISTRY}/pytorch/pytorch:${TORCH_VERS}-cuda${CUDA_VERS}-cudnn8-devel

ARG INJECT_MF_CERT

COPY mf.crt /usr/local/share/ca-certificates/mf.crt

# The following two lines are necessary to deal with the MITM sniffing proxy we have internally.
RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"
# set apt to non interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN $MY_APT update && $MY_APT install -y software-properties-common && add-apt-repository ppa:ubuntugis/ppa
RUN $MY_APT update && $MY_APT install -y curl gdal-bin libgdal-dev libgeos-dev git vim nano sudo libx11-dev tk python3-tk tk-dev libpng-dev libffi-dev dvipng texlive-latex-base texlive-latex-extra  texlive-fonts-recommended cm-super openssh-server netcat libeccodes-dev libeccodes-tools openssh-server

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE

# Build eccodes, a recent version yields far better throughput according to our benchmarks
ARG ECCODES_VER=2.35.0
RUN curl -O https://confluence.ecmwf.int/download/attachments/45757960/eccodes-$ECCODES_VER-Source.tar.gz && tar -xzf eccodes-$ECCODES_VER-Source.tar.gz && mkdir build && cd build && cmake ../eccodes-$ECCODES_VER-Source -DENABLE_AEC=ON -DENABLE_NETCDF=ON -DENABLE_FORTRAN=OFF && make && ctest && make install && ldconfig

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
