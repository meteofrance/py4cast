# Parse script options
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--builder)
      BUILDER="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [[ -z ${BUILDER} ]]; then
    echo "Error: --builder option is required, please choose in {docker, podman}."
    exit 2
elif [[ ${BUILDER} = "podman" ]]; then
    BUILDER=$(which podman)
    OPTIONS="--storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs"
elif [[ ${BUILDER} = "docker" ]]; then
    BUILDER=$(which docker)
else
    echo "Builder \"${builder}\" not supported, please choose in {docker, podman}."
    exit 3
fi

# Automatically retrieve the CUDA and torch version from env var, otherwise set them from reference file env.yaml
if [[ -z "${PY4CAST_CUDA_VERSION}" ]]; then
  PY4CAST_CUDA_VERSION=$(cat env.yaml | grep pytorch-cuda= | sed 's/.*pytorch-cuda=\([0-9]*\.[0-9]\).*/\1/')
else
  echo "PY4CAST_CUDA_VERSION version found in the environment: ${PY4CAST_CUDA_VERSION}"
fi

if [[ -z "${PY4CAST_TORCH_VERSION}" ]]; then
  PY4CAST_TORCH_VERSION=$(cat env.yaml | grep torch== | sed 's/.*torch==\([0-9]\.[0-9]\.[0-9]\).*/\1/')
else
  echo "PY4CAST_TORCH_VERSION version found in the environment: ${PY4CAST_TORCH_VERSION}"
fi

# Tag the docker image with the latest commit hash on the currrent branch
TAG=$(git rev-parse --short HEAD)

# Build the py4cast docker image
${BUILDER} build \
    ${OPTIONS} \
    --build-arg CUDA_VERS=${PY4CAST_CUDA_VERSION} \
    --build-arg TORCH_VERS=${PY4CAST_TORCH_VERSION} \
    --build-arg USERNAME=$(id -un) \
    --build-arg GROUPNAME=$(id -gn) \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GUID=$(id -g) \
    --build-arg HOME_DIR=${HOME} \
    --tag py4cast:${TAG} \
    .
