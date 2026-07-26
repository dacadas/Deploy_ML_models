import typing as t

import numpy as np
import pandas as pd

from regression_model import __version__ as _version
from regression_model.config.core import config
from regression_model.processing.data_manager import load_pipeline
from regression_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.model_settings.features]
        )
        results = {
            "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }
        
        
        
    # validated_data, errors = validate_inputs(input_data=data)

    # print(validated_data["MSSubClass"].dtype)
    # print(validated_data["MSSubClass"].head())
    # print(type(validated_data["MSSubClass"].iloc[0]))
    # # print("ERROS:")
    # # print(errors)
    
    # if errors:
    #     import json
    #     # print(json.dumps(json.loads(errors), indent=2))
    
    # results = {
    #     "predictions": None,
    #     "version": _version,
    #     "errors": errors,
    # }    

    return results
