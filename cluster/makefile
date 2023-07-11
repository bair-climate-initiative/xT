
# Suggestion (Tyler): make build create the next build* folder that doesn't exist
# for better versioning instead of overwriting the same build folder. Simple to do w/ bash
.PHONY: build
SEARCH_FOLDER=rev_exps
build:
	mkdir -p build;
	rm -rf build/*
	for f in $$(find $(SEARCH_FOLDER) -name '*.sh'); \
	do (echo "compiling $$f > build/$$(basename $$f) "; \
	  python compile.py -c $$f -o build/$$f;\
	 echo "Done"); done
