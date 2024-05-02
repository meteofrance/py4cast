FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG INJECT_MF_CERT

COPY mf.crt /usr/local/share/ca-certificates/mf.crt

RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"

ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN $MY_APT update && $MY_APT install -y software-properties-common && add-apt-repository ppa:ubuntugis/ppa
RUN $MY_APT update && $MY_APT install -y curl gdal-bin libgdal-dev libgeos-dev git vim nano sudo libx11-dev tk python3-tk tk-dev libpng-dev libffi-dev dvipng texlive-latex-base texlive-latex-extra  texlive-fonts-recommended cm-super openssh-server netcat libeccodes-dev libeccodes-tools openssh-server

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE

RUN pip install --upgrade pip
COPY requirements.txt /root/requirements.txt
RUN set -eux && pip install --default-timeout=100 -r /root/requirements.txt

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

RUN set -eux && pip install pyg-lib==0.4.0 torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.2\
    torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

WORKDIR $HOME_DIR
RUN curl -fsSL https://code-server.dev/install.sh | sh





