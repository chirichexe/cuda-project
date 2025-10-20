
NVCC     := nvcc
INCLUDES := -Iinclude
OUT_DIR  := ./bin

build:
	@if [ -z "$(file)" ]; then \
		echo "Usage: make build file=src/examples/0_hello.cu"; \
		exit 1; \
	fi
	@if [ ! -f "$(file)" ]; then \
		echo "Error: $(file) not found"; \
		exit 1; \
	fi
	@mkdir -p $(OUT_DIR)
	@echo "Compiling $(file)..."
	@$(NVCC) $(file) $(INCLUDES) -o $(OUT_DIR)/$(notdir $(file:.cu=)) || { \
		echo "*** Compilation error ***"; \
		exit 1; \
	}
	@echo "*** Build successful: $(OUT_DIR)/$(notdir $(file:.cu=)) ***"

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OUT_DIR)/*
	@echo "*** Clean complete ***"

