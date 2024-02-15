#!/bin/bash

set -eux

id
pwd
isort --profile black $1
black $1