"""Models NVIDIA's hosted catalog lists but does not serve as chat models.

Measured 2026-10-16 by sending every model in the hosted ``/v1/models`` list a
one-token chat request from two different accounts. Listed here: models that
answered 404 Not Found or 410 Gone on both, and services that are not chat
models. A model that only timed out or returned a 5xx stays discoverable, and
a model NVIDIA adds later is discoverable until measured otherwise.
"""

UNAVAILABLE_HOSTED_NIM_MODELS = frozenset(
    {
        # Not served (404 Not Found or 410 Gone).
        "01-ai/yi-large",
        "adept/fuyu-8b",
        "ai21labs/jamba-1.5-large-instruct",
        "aisingapore/sea-lion-7b-instruct",
        "bigcode/starcoder2-15b",
        "databricks/dbrx-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "google/codegemma-1.1-7b",
        "google/codegemma-7b",
        "google/deplot",
        "google/gemma-2b",
        "google/gemma-3-12b-it",
        "google/gemma-3-4b-it",
        "google/recurrentgemma-2b",
        "ibm/granite-3.0-3b-a800m-instruct",
        "ibm/granite-3.0-8b-instruct",
        "ibm/granite-34b-code-instruct",
        "ibm/granite-8b-code-instruct",
        "meta/codellama-70b",
        "meta/llama2-70b",
        "microsoft/kosmos-2",
        "microsoft/phi-3-vision-128k-instruct",
        "microsoft/phi-3.5-moe-instruct",
        "mistralai/codestral-22b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mistral-large",
        "mistralai/mistral-large-2-instruct",
        "mistralai/mixtral-8x22b-v0.1",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k3",
        "nv-mistralai/mistral-nemo-12b-instruct",
        "nvidia/cosmos-reason2-8b",
        "nvidia/embed-qa-4",
        "nvidia/llama-3.1-nemotron-51b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/llama-3.2-nv-embedqa-1b-v1",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "nvidia/llama3-chatqa-1.5-70b",
        "nvidia/mistral-nemo-minitron-8b-8k-instruct",
        "nvidia/nemotron-3-embed-1b",
        "nvidia/nemotron-4-340b-instruct",
        "nvidia/nemotron-4-340b-reward",
        "nvidia/nemotron-nano-3-30b-a3b",
        "nvidia/neva-22b",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "nvidia/nvclip",
        "nvidia/riva-translate-4b-instruct",
        "nvidia/vila",
        "snowflake/arctic-embed-l",
        "writer/palmyra-creative-122b",
        "writer/palmyra-fin-70b-32k",
        "writer/palmyra-med-70b",
        "writer/palmyra-med-70b-32k",
        "zyphra/zamba2-7b-instruct",
        # Not chat models.
        "nvidia/ai-synthetic-video-detector",
        "nvidia/nemotron-parse",
    }
)
