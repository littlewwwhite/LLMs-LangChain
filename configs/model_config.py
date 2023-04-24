import torch.cuda
import torch.backends


embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "bert-base-chinese": "bert-base-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "/data_F/zhijian/chatglm-6b",
    "moss-moon-003-base": "fnlp/moss-moon-003-base",
    "vicuna": "lmsys/vicuna-13b-delta-v1.1"
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

