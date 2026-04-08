from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_encoders={SecretStr: lambda v: "***"}
    )
