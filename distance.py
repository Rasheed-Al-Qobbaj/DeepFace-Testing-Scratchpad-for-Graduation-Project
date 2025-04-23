import numpy as np

def findCosineDistance(vector1, vector2):
    """Calculate the cosine distance between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return 1 - (dot_product / (norm1 * norm2))

def findEuclideanDistance(vector1, vector2):
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def l2_normalize(vector):
    """L2 normalize a vector."""
    return vector / np.sqrt(np.sum(np.multiply(vector, vector)))

def findThreshold(model_name, distance_metric):
    """Return a default threshold for a given model and distance metric."""
    thresholds = {
        "VGG-Face": {"cosine": 0.4, "euclidean": 0.55, "euclidean_l2": 0.75},
        "Facenet": {"cosine": 0.4, "euclidean": 10, "euclidean_l2": 0.8},
        "Facenet512": {"cosine": 0.3, "euclidean": 9.4, "euclidean_l2": 0.8},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
    }
    return thresholds.get(model_name, {}).get(distance_metric, 0.5)