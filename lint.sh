#!/bin/bash

set -eux

id
pwd
flake8 --ignore E203,W503 --max-line-length 120 $1
isort --profile black -c $1
black --check $1
bandit -ll -r --skip B402,B321,B614 $1