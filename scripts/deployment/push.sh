#!/bin/bash
set -e

export COMMIT_ID=$(git rev-parse HEAD)

build_and_push() {
        aws configure set default.region eu-central-1
        aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.eu-central-1.amazonaws.com

        docker-compose -f docker-compose-$1.yml build \
            --build-arg COMMIT_ID=${COMMIT_ID} \
            --build-arg TRAVIS_BRANCH=${TRAVIS_BRANCH} \
            --build-arg AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID} --compress

        docker-compose -f docker-compose-$1.yml push

        # Get already built docker images
        images=$(cat docker-compose-$1.yml | grep 'image: ${AWS_ACCOUNT_ID' | cut -d':' -f 2 | tr -d '"')

        # Tag & push images with latest tag
        for image in $images
        do
            eval image=${image}
            docker tag ${image}:${COMMIT_ID} ${image}:latest
            docker push ${image}:latest
        done
}

if [ "${TRAVIS_BRANCH}" == "staging" -o "${TRAVIS_BRANCH}" == "production" ]; then
    build_and_push $TRAVIS_BRANCH
    exit 0
else
    echo "Skipping deploy!"
    exit 0
fi

