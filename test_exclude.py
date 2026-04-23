from pydantic import BaseModel, ConfigDict, SecretStr, Field
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_encoders={SecretStr: lambda v: "***"}
    )
class StageLLMOverride(StrictModel):
    """Override top-level LLM settings for a specific stage."""
    model: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)

class Parent(StrictModel):
    override: StageLLMOverride | None = None

class Root(StrictModel):
    parent: Parent | None = None

r = Root(parent=Parent(override=StageLLMOverride(api_key="123", model="xyz")))
print(r.model_dump(mode="json"))
