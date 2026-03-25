"""
Pydantic schemas for API request and response validation.

FastAPI uses these schemas to:
- Validate incoming JSON automatically
- Generate the interactive Swagger docs at /docs
- Serialize outgoing responses

MLOps Concept: API contracts are the interface between ML and the rest of the product.
Versioning these schemas (v1, v2) lets you update the model without breaking clients.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input features for house price prediction.

    All features match the California Housing dataset.
    Engineered features are computed server-side — clients send raw inputs.
    """

    med_inc: float = Field(..., description="Median income in block (tens of thousands $)", gt=0)
    house_age: float = Field(..., description="Median house age in block (years)", gt=0)
    ave_rooms: float = Field(..., description="Average rooms per household", gt=0)
    ave_bedrms: float = Field(..., description="Average bedrooms per household", gt=0)
    population: float = Field(..., description="Block population", gt=0)
    ave_occup: float = Field(..., description="Average household occupancy", gt=0)
    latitude: float = Field(..., description="Block latitude")
    longitude: float = Field(..., description="Block longitude")

    model_config = {
        "json_schema_extra": {
            "example": {
                "med_inc": 3.5,
                "house_age": 20.0,
                "ave_rooms": 5.0,
                "ave_bedrms": 1.0,
                "population": 1000.0,
                "ave_occup": 3.0,
                "latitude": 37.88,
                "longitude": -122.23,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction output — price in two formats for convenience."""

    predicted_price: float = Field(..., description="Predicted price in $100,000 units")
    predicted_price_usd: float = Field(..., description="Predicted price in USD")
    unit: str = "$100,000s"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
