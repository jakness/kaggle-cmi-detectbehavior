from typing import Type

from kaggle_cmi.models.model import SequenceClassifierPyTorch
from kaggle_cmi.models.conv1d_model import Conv1DSequenceClassifier
from kaggle_cmi.models.two_branch_model import TwoBranchSequenceClassifier
from kaggle_cmi.models.se_gru import SqueezeExcitationGRUSequenceClassifier


def get_model(name: str) -> Type[SequenceClassifierPyTorch]:
    models = {
        Conv1DSequenceClassifier.name: Conv1DSequenceClassifier,
        TwoBranchSequenceClassifier.name: TwoBranchSequenceClassifier,
        SqueezeExcitationGRUSequenceClassifier.name: SqueezeExcitationGRUSequenceClassifier,
    }
    model = models.get(name, None)
    if model is None:
        raise ValueError(f"Model with name '{name}' is not defined.")
    return model
