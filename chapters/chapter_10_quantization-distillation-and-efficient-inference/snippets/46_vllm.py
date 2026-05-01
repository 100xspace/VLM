model=model_name,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=4096,
        max_num_seqs=256,
        enforce_eager=False  # Use CUDA graphs for speedup
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
    return llm, sampling_params
def batch_inference_vllm(llm, prompts, sampling_params):
    """Run batched inference with vLLM"""
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
# Example usage
# llm, sp = deploy_with_vllm("llava-hf/llava-1.5-7b-hf", quantization="awq", tensor_parallel=2)
# prompts = ["Describe this image: <image>", "What is in this photo: <image>"]
# results = batch_inference_vllm(llm, prompts, sp)
