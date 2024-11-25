from importlib.metadata import version

if version("transformers") == "4.41.2":
    from .E5V_Embedder import E5V_Embedder
else:
    from .DSE_Embedder import DSE_Embedder
    from .InternVL_DescriptionGenerator import InternVL_DescriptionGenerator
    from .Qwen2_5_LLM import Qwen2_5_LLM
    from .Qwen2VL_DescriptionGenerator import Qwen2VL_DescriptionGenerator
    from .Qwen2VL_LLM import Qwen2VL_LLM
    from .Stella_Embedder import Stella_Embedder
