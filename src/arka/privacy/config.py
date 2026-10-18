from __future__ import annotations

from string import Formatter
from typing import Literal

from pydantic import Field, model_validator

from arka.common.models import StrictModel
from arka.privacy.detectors import REGEX_ENTITIES

GLINER_LANGUAGES = frozenset({"en", "fr", "es", "de", "it", "pt", "nl"})
GLINER_ENTITIES = frozenset(
    "person full_name first_name middle_name last_name date_of_birth email phone_number "
    "address street_address city state_or_region postal_code country government_id "
    "national_id_number passport_number drivers_license_number license_number tax_id "
    "tax_number bank_account account_number routing_number iban payment_card card_number "
    "card_expiry card_cvv username ip_address account_id sensitive_account_id password "
    "secret api_key access_token recovery_code sensitive_date document_date expiration_date "
    "transaction_date".split()
)


class PIIConfig(StrictModel):
    """An explicit run policy. Omit the whole section to disable protection."""

    backend: Literal["regex", "gliner2"]
    languages: list[str] = Field(min_length=1)
    entities: list[str] = Field(min_length=1)
    model_path: str | None = None
    threshold: float = Field(default=0.5, gt=0, le=1)
    placeholder_template: str = "[{entity}]"
    max_redacted_ratio: float | None = Field(default=None, ge=0, le=1)
    blocked_entities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_coverage(self):
        supported = GLINER_ENTITIES if self.backend == "gliner2" else REGEX_ENTITIES
        if set(self.entities) - supported:
            raise ValueError("PII backend does not support requested entities")
        if set(self.blocked_entities) - set(self.entities):
            raise ValueError("blocked_entities must be selected entities")
        if self.backend == "gliner2":
            if set(self.languages) - GLINER_LANGUAGES:
                raise ValueError("GLiNER2 PII supports en, fr, es, de, it, pt, nl")
            if not self.model_path:
                raise ValueError(
                    "GLiNER2 requires an explicitly provisioned model_path"
                )
        elif any(
            len(code) != 2
            or not code.isascii()
            or not code.isalpha()
            or not code.islower()
            for code in self.languages
        ):
            raise ValueError("languages must contain lowercase two-letter codes")
        try:
            fields = [
                field
                for _, field, spec, conversion in Formatter().parse(
                    self.placeholder_template
                )
                if field is not None
            ]
            if fields != ["entity"] or not self.placeholder_template.format(
                entity="email"
            ):
                raise ValueError
            prefix, _, suffix = self.placeholder_template.partition("{entity}")
            if not prefix or not suffix or not self.placeholder_template.isprintable():
                raise ValueError
            if any(
                spec or conversion
                for _, _, spec, conversion in Formatter().parse(
                    self.placeholder_template
                )
            ):
                raise ValueError
        except (ValueError, KeyError, IndexError):
            raise ValueError(
                "placeholder_template must contain exactly one plain {entity} field with printable delimiters on both sides"
            ) from None
        return self
