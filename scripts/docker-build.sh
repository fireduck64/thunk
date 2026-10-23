#!/bin/bash

docker pull debian

docker build . -f Dockerfile.jail -t thunkjail

