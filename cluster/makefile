.PHONY: build
SEARCH_FOLDER=exps
build:
	mkdir -p build;
	for f in $$(find $(SEARCH_FOLDER) -name '*.sh'); \
	do (echo "compiling $$f > build/$$(basename $$f) "; \
	  python compile.py -c $$f >>build/$$(basename $$f);\
	 echo "Done"); done
