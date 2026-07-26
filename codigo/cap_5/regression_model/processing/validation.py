from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_settings.features
        if var
        not in config.model_settings.categorical_vars_with_na_frequent
        + config.model_settings.categorical_vars_with_na_missing
        + config.model_settings.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    # """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_settings.variables_to_rename, inplace=True)
    # input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
    input_data["MSSubClass"] = input_data["MSSubClass"].astype(str)
    relevant_data = input_data[config.model_settings.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    
    # # print("dtype:", validated_data["MSSubClass"].dtype)
    # # print(validated_data["MSSubClass"].head())
    # # print(type(validated_data["MSSubClass"].iloc[0]))
    # # print(validated_data["MSSubClass"].iloc[0])
    
    # try:
    #     # replace numpy nans so that pydantic can validate
    #     MultipleHouseDataInputs(
    #         inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
    #     )
    # except ValidationError as error:
    #     errors = error.json()


    records = validated_data.replace({np.nan: None}).to_dict(orient="records")
    
    # print("Primeiro registro:")
    # print(records[0]["MSSubClass"])
    # print(type(records[0]["MSSubClass"]))
    
    # print("Registro 1447:")
    # print(records[1447]["MSSubClass"])
    # print(type(records[1447]["MSSubClass"]))
    
    try:
        MultipleHouseDataInputs(inputs=records)
    except ValidationError as error:
        print(error)
        errors = error.json()
    return validated_data, errors


class HouseDataInputSchema(BaseModel):
    MSSubClass: str | None = None
    MSZoning: str | None = None
    LotFrontage: float | None = None
    LotShape: str | None = None
    LandContour: str | None = None
    LotConfig: str | None = None
    Neighborhood: str | None = None
    OverallQual: int | None = None
    OverallCond: int | None = None
    YearRemodAdd: int | None = None
    RoofStyle: str | None = None
    Exterior1st: str | None = None
    ExterQual: str | None = None
    Foundation: str | None = None
    BsmtQual: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    HeatingQC: str | None = None
    CentralAir: str | None = None
    FirstFlrSF: int | None = None
    SecondFlrSF: int | None = None
    GrLivArea: int | None = None
    BsmtFullBath: float | None = None
    HalfBath: int | None = None
    KitchenQual: str | None = None
    TotRmsAbvGrd: int | None = None
    Functional: str | None = None
    Fireplaces: int | None = None
    FireplaceQu: str | None = None
    GarageFinish: str | None = None
    GarageCars: float | None = None
    GarageArea: float | None = None
    PavedDrive: str | None = None
    WoodDeckSF: int | None = None
    ScreenPorch: int | None = None
    SaleCondition: str | None = None
    YrSold: int | None = None


class MultipleHouseDataInputs(BaseModel):
    inputs: List[HouseDataInputSchema]
