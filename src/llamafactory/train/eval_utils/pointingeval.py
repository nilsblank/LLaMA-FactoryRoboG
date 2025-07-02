import numpy as np
import torch
import re
import wandb
from PIL import Image
from transformers import TrainerCallback
from typing import List, Optional, Dict, Any, Union, Tuple
import os

from utils import decode_predictions

def text2pts(text: str) -> np.ndarray:
    """
    Extract points from model output text.
    Expected format: points like (x,y) or x,y
    Returns: array of [x, y] points
    """
    # Match (x,y) or x,y patterns
    pattern = r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?'
    matches = re.findall(pattern, text)
    
    if not matches:
        return np.zeros((0, 2))
    
    points = np.array([[int(x), int(y)] for x, y in matches])
    return points


def evaluate_points(
    predictions: List[Union[str, Dict[str, Any]]], 
    mask_paths: List[str],
    verbose: bool = False
) -> Tuple[float, List[float]]:
    """
    Evaluate point predictions against ground truth masks.
    
    Args:
        predictions: List of prediction strings or dicts with 'text' key
        mask_paths: List of paths to mask files
        verbose: Whether to print processing messages
        
    Returns:
        Tuple of (mean_accuracy, list_of_individual_accuracies)
    """
    accuracies = []
    
    for idx, (pred, mask_path) in enumerate(zip(predictions, mask_paths)):
        # Extract text from prediction
        if isinstance(pred, dict) and "text" in pred:
            pred_text = pred["text"]
        else:
            pred_text = pred  # Assume it's already text
        
        # Parse points from the prediction text
        try:
            points = text2pts(pred_text)
        except Exception as e:
            if verbose:
                print(f"Failed to parse points for sample {idx}: {e}")
            points = np.zeros((0, 2))
        
        # Load ground truth mask
        try:
            mask = np.array(Image.open(mask_path)) / 255.0
        except Exception as e:
            if verbose:
                print(f"Failed to load mask for sample {idx}: {e}")
            accuracies.append(0.0)
            continue
        
        # Calculate accuracy
        acc = 0.0
        if len(points) > 0:
            # Check which points are within the mask boundaries
            in_range = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
                       (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0]))
            
            # Get mask values at valid point locations
            valid_points = points[in_range]
            valid_values = np.zeros(len(points))
            
            if len(valid_points) > 0:
                # Extract mask values for valid points
                valid_values[:len(valid_points)] = mask[valid_points[:, 1], valid_points[:, 0]]
            
            # Average mask values (points in mask = 1.0, outside = 0.0)
            acc = valid_values.mean()
        
        accuracies.append(acc)
    
    # Calculate overall accuracy
    mean_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    return mean_accuracy, accuracies



class PointEvalCallback(TrainerCallback):
    """
    Evaluation callback for point prediction tasks.
    Compares predicted points against ground truth masks.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        mask_dir: str,
        name: Optional[str] = None
    ):
        """
        Initialize the callback.
        
        Args:
            trainer: The trainer instance
            tokenizer: The tokenizer for decoding predictions
            val_dataset: Validation dataset
            mask_dir: Directory containing ground truth masks
            name: Name for the metric
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.mask_dir = mask_dir
        if name is None:
            self.name = "PointAccuracy"
        else:
            self.name = name
            
    def on_evaluate(self, args, state, control, **kwargs):
        """
        Evaluate point predictions against ground truth masks.
        """
        # Get model predictions and decode them to text
        predictions = self.trainer.predict(self.val_dataset)
        pred_texts = decode_predictions(self.tokenizer, predictions)
        
        # Get mask paths for each sample
        mask_paths = []
        for idx, sample in enumerate(self.val_dataset):
            mask_path = sample.get("mask_path", os.path.join(self.mask_dir, f"{idx:02d}.jpg"))
            mask_paths.append(mask_path)
        
        # Evaluate predictions
        mean_accuracy, _ = evaluate_points(pred_texts, mask_paths)
        
        # Log the result
        self.trainer.log_metrics({self.name: mean_accuracy})
        wandb.log({f"validate/{self.name}": mean_accuracy}, step=state.global_step)
        
        return mean_accuracy