from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

def evaluate_model(model, loader, device, threshold=0.5):
    """
