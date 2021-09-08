ARG AWS_ACCOUNT_ID
ARG COMMIT_ID
ARG TRAVIS_BRANCH

# FROM ${AWS_ACCOUNT_ID}.dkr.ecr.eu-central-1.amazonaws.com/evalai-${TRAVIS_BRANCH}-backend:${COMMIT_ID}
FROM 990393954645.dkr.ecr.eu-central-1.amazonaws.com/evalai-production-backend:latest

CMD ["sh", "/code/docker/prod/celery/container-start.sh"]

