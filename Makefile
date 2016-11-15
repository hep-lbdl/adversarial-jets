NAME=lukedeo/ml
BASENAME=lukedeo/ml-base
BASEVERSION=0.1
# VERSION=`git describe`
# http://victorlin.me/posts/2014/11/26/running-docker-with-aws-elastic-beanstalk
VERSION=1.0
CORE_VERSION=HEAD

all: base build

base: 
	docker build -t $(BASENAME):$(BASEVERSION) ml-base/
	docker push $(BASENAME):$(BASEVERSION)

build:
	docker build -t $(NAME):$(VERSION) .

tag_latest:
	docker tag $(NAME):$(VERSION) $(NAME):latest

test:
	nosetests -sv

push: tag_latest
	docker push $(NAME):$(VERSION)
	docker push $(NAME):latest

