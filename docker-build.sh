# Parse script options
FORMAT="docker"
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--format)
      FORMAT="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ ${FORMAT} = "podman" ]]; then
    BUILDER=$(which podman)
    OPTIONS="--storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs"
elif [[ ${FORMAT} = "docker" ]]; then
    BUILDER=$(which docker)
else
    echo "Format \"${FORMAT}\" not supported, please choose in {docker, podman}."
    exit 2
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
