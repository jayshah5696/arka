from pydantic import BaseModel, SecretStr
class TestModel(BaseModel):
    api_key: SecretStr

m = TestModel(api_key="my-secret-key")
print(m.model_dump(mode="json"))
