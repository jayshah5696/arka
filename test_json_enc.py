from pydantic import BaseModel, ConfigDict, SecretStr

class TestModel(BaseModel):
    model_config = ConfigDict(json_encoders={SecretStr: lambda v: "***"})
    api_key: SecretStr | None = None

t = TestModel(api_key="my-secret-key")
print(t.model_dump(mode="json"))
