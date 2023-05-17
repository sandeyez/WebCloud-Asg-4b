build:
	@brane build ./compute/container.yml
	@brane build ./visualization/container.yml

addinstance:
	@brane instance add wc.j45.nl

push:
	@brane instance select wc.j45.nl
	@brane push compute
	@brane push visualization

run:
	@brane instance select wc.j45.nl
	@brane run --remote pipeline.bs