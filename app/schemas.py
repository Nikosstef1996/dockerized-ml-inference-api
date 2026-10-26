"""Validated API request and response schemas."""
from typing import Annotated

from pydantic import BaseModel, Field


PositiveLoad = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ForecastRequest(BaseModel):
    """Hourly demand (MWh) for the latest complete 24-hour period, oldest first."""

    last_24_hours_mwh: list[PositiveLoad] = Field(
        min_length=24,
        max_length=24,
        description="Twenty-four hourly MWh values, ordered oldest to newest.",
    )


class ForecastResponse(BaseModel):
    model_name: str
    horizon_hours: int
    forecast_next_24_hours_mwh: list[float]


from datetime import datetime, timezone
from pydantic import AwareDatetime, field_validator


class MLForecastRequest(BaseModel):
    """A 24-hour forecast issued immediately after 168 complete UTC hours."""

    last_168_hours_mwh: list[PositiveLoad] = Field(
        min_length=168, max_length=168,
        description="Exactly 168 hourly demand values, oldest to newest.",
    )
    forecast_start_utc: AwareDatetime = Field(
        description="Start of forecast day in UTC, e.g. 2026-10-27T00:00:00Z."
    )

    @field_validator("forecast_start_utc")
    @classmethod
    def requires_hour_boundary(cls, value: datetime) -> datetime:
        value = value.astimezone(timezone.utc)
        if value.minute or value.second or value.microsecond:
            raise ValueError("Forecast start must be on an exact UTC hour")
        return value
