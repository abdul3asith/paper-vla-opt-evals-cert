# Does a Faster VLA Still Work? Certifying Compilation and Quantization Across GPUs

# We are achieving that by compilation, lower precision, and quantization. 

## Baseline
# 1. native (bf16)
# 2. fp32 cast

## Precision
# 1. fp16 autocast (PyTorch automatically runs safe operations (like matrix multiplies) in 16-bit and keeps sensitive ones in 32-bit)
# 2. bf16 autocast (Like fp16, but with a wider range of numbers, so it’s less likely to overflow)
# 3. fp8 

## Compilation 
# 1. compile (torch.compile torch.compile turns the model into fused GPU kernels. Same math, less overhead, so precision isn’t cut)
# 2. compile max-autotune (Same as compile, but it tries many kernel versions and keeps the fastest, and uses CUDA graphs. Slower to start, faster to run)
# 3. compile reduce-overhead (Uses CUDA graphs to cut kernel-launch overhead)

## quantization
# 1. int8 (Stores the weights as 8-bit integers instead of 16-bit. Half the memory, often faster)
# 2. int4 
# 3. int8 weights + activations(W8A8) (Quantizes the intermediate values too, not just the weights)