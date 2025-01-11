.PHONY: setup clean

# Directory paths
KERNEL_DIR = inferno/kernels/ltx_video
REPO_URL = https://github.com/KONAKONA666/q8_kernels

setup:
	@echo "Creating directories if they don't exist..."
	mkdir -p $(KERNEL_DIR)
	
	@echo "Cloning repository..."
	git clone $(REPO_URL) $(KERNEL_DIR)/q8_kernels
	
	@echo "Initializing and updating submodules..."
	cd $(KERNEL_DIR)/q8_kernels && \
	git submodule init && \
	git submodule update
	
	@echo "Installing package and dependencies..."
	cd $(KERNEL_DIR)/q8_kernels && \
	python setup.py install && \
	pip install . && \
	pip install transformers diffusers sentencepiece imageio einops
	
	@echo "Setup complete!"

clean:
	@echo "Cleaning up installation..."
	rm -rf $(KERNEL_DIR)/q8_kernels
	@echo "Cleanup complete!"