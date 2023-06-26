import os
import joblib
import pandas as pd
import json

from nearest_neighbors_recommender import ItemBasedCosineNearestNeighborsRecommender


def input_fn(input_data, content_type):
    """Parse input data payload"""

    if content_type == 'application/json':
        # Load the input data
        data = json.loads(input_data)

        return data
    else:
        raise ValueError(f'Unsupported content_type: {content_type}')


def model_fn(model_dir):
    """Load the model from disk and return it."""
    with open(os.path.join(model_dir, "model.joblib"), "rb") as f:
        model = joblib.load(f)
    return model


def predict_fn(input_data, model):
    """Make a prediction with the model and return it."""
    return model.recommend(pd.DataFrame(input_data['user_ids'], columns=['user_id']), pd.DataFrame(input_data['item_ids'], columns=['item_id']), 10)


def output_fn(prediction, accept):
    """Format prediction output"""

    if accept == "application/json":
        # Convert the DataFrame to JSON
        return prediction.to_json(orient='records'), accept
    else:
        raise ValueError(f'Unsupported accept type: {accept}')
