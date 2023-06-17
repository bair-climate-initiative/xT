.PHONY: build
SEARCH_FOLDER=exps
build:
	mkdir -p build;
	rm -f build/*
	for f in $$(find $(SEARCH_FOLDER) -name '*.sh'); \
	do (echo "compiling $$f > build/$$(basename $$f) "; \
	  python compile.py -c $$f -o build/$$f;\
	 echo "Done"); done
