#!/bin/bash

set -eux

id
pwd
python -m isort --profile black $1
python -m black $1
