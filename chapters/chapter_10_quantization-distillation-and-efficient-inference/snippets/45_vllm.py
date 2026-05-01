from vllm import LLM, SamplingParams
def deploy_with_vllm(model_name, quantization="awq", tensor_parallel=1):
    """
