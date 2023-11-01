.PHONY: doc-serve
doc-serve:
	docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material serve --dev-addr=0.0.0.0:8000

.PHONY: doc-build
doc-build:
	docker run --rm -it -v ${PWD}:/docs squidfunk/mkdocs-material build