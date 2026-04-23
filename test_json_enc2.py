from pydantic import BaseModel, ConfigDict, SecretStr, Field

class TestModel(BaseModel):
    model_config = ConfigDict(json_encoders={SecretStr: lambda v: "***"})
    api_key: SecretStr | None = Field(default=None, exclude=True)

t = TestModel(api_key="my-secret-key")
print(t.model_dump(mode="json"))
