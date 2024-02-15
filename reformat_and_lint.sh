#!/bin/bash

USERNAME=`whoami`
CWD=`pwd`

echo $USERNAME

set -eux
docker build -f Dockerfile.lint . -t lint-$USERNAME --build-arg USERNAME=$USERNAME --build-arg USER_UID=`id -u` --build-arg USER_GUID=`id -g` --build-arg HOME_DIR=$HOME --build-arg GROUPNAME=`id -ng`

docker run --rm -it -v $CWD:$CWD -u `id -u`:`id -g` -w $CWD lint-$USERNAME ./reformat.sh .

# Commande qui modifie l'ordre des imports et qui ne passe pas la CI
# docker run -it --rm -v /home/labia/berthomierl/monorepo4ai:/home/labia/berthomierl/monorepo4ai -w /home/labia/berthomierl/monorepo4ai lint-$USERNAME ./reformat.sh /home/labia/berthomierl/monorepo4ai/projects/pnia

docker run --rm -it -v $CWD:$CWD -u `id -u`:`id -g` -w $CWD lint-$USERNAME ./lint.sh .