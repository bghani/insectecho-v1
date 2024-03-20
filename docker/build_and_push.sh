#!/bin/sh
set -eu

if [ -z ${1+x} ]; then
    echo Missing name argument
    exit 1
fi
if [ -z ${CACHE_TAG1+x} ] || [ -z ${CACHE_TAG2+x} ]; then
    echo Missing cache tag variables
    exit 2
fi
name=$1
basedir=$(dirname "$0")

docker pull "$CI_REGISTRY_IMAGE/$name:$CACHE_TAG1" || true
docker pull "$CI_REGISTRY_IMAGE/$name:$CACHE_TAG2" || true
docker build \
    --pull \
    --tag "$CI_REGISTRY_IMAGE/$name:build_$CI_COMMIT_REF_SLUG" \
    -f "$basedir/$name/Dockerfile" \
    .
docker push "$CI_REGISTRY_IMAGE/$name:build_$CI_COMMIT_REF_SLUG"
