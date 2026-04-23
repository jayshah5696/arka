from pydantic import BaseModel, SecretStr
from src.arka.config.models import ResolvedConfig, LLMConfig, StageLLMOverride, GeneratorConfig

class Dummy(BaseModel):
    cfg: StageLLMOverride

d = Dummy(cfg=StageLLMOverride(api_key="my-secret-override-key"))
print(d.model_dump(mode="json"))
