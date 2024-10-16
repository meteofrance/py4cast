# Automatically retrieve the CUDA and torch version from the reference file env.yaml
CUDA_VERSION=$(cat env.yaml | grep pytorch-cuda= | sed 's/.*pytorch-cuda=\([0-9]*\.[0-9]\).*/\1/')
TORCH_VERSION=$(cat env.yaml | grep torch== | sed 's/.*torch==\([0-9]\.[0-9]\.[0-9]\).*/\1/')

# Tag the docker image with the latest commit hash on the currrent branch
TAG=$(git rev-parse --short HEAD)

# Build the py4cast docker image
docker build \
    --build-arg CUDA_VERS=${CUDA_VERSION} \
    --build-arg TORCH_VERS=${TORCH_VERSION} \
    --build-arg USERNAME=$(id -un) \
    --build-arg GROUPNAME=$(id -gn) \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GUID=$(id -g) \
    --build-arg HOME_DIR=${HOME} \
    --tag py4cast:${TAG} \
    .
