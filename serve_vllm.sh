export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-4B
		-tp 2 \