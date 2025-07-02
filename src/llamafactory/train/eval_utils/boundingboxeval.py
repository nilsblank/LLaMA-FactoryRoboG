import json
import torch
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import TrainerCallback
from typing import List, Dict, Any, Union, Optional
from utils import decode_predictions

def parse_bbox_from_text(text: str) -> List[List[float]]:
    """
    Parse bounding box coordinates from prediction text.
    Expected format: "x1,y1,x2,y2; x1,y1,x2,y2; ..."
    
    Args:
        text: String containing bounding box coordinates
        
    Returns:
        List of bounding box coordinates [x1, y1, x2, y2]
    """
    box_list = []
    if not text:
        return box_list
        
    for box_str in text.split(';'):
        box_str = box_str.strip()
        if box_str:  # Skip empty parts
            try:
                # Convert string coordinates to floats
                coords = [float(x) for x in box_str.split(',')]
                if len(coords) == 4:  # Ensure we have 4 coordinates
                    box_list.append(coords)
            except (ValueError, IndexError):
                continue
    
    return box_list


def evaluate_bounding_boxes(
    predictions: List[str], 
    ground_truths: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate bounding box predictions against ground truth.
    
    Args:
        predictions: List of predictions (strings or dictionaries with text)
        ground_truths: List of ground truth data (dictionaries with "boxes" key)
        
    Returns:
        Dictionary with MAP metrics
    """
    pred_list = []
    target_list = []
    dummy_label = 1  # Single class label for all boxes
    
    for pred, target in zip(predictions, ground_truths):
        # Process prediction
        if isinstance(pred, dict) and "text" in pred:
            pred_text = pred["text"]
        else:
            pred_text = pred  # Assume it's already text
        
        # Parse predicted boxes
        box_list = parse_bbox_from_text(pred_text) if isinstance(pred_text, str) else []
        boxes = torch.tensor(box_list) if box_list else torch.empty((0, 4))
        
        pred_dict = {
            "boxes": boxes,
            "scores": torch.ones(boxes.shape[0]),  # Assign constant score 1.0
            "labels": torch.full((boxes.shape[0],), dummy_label, dtype=torch.int)
        }
        pred_list.append(pred_dict)
        
        # Process ground truth
        gt_boxes = target.get("boxes", [])
        gt_boxes = torch.tensor(gt_boxes) if len(gt_boxes) > 0 else torch.empty((0, 4))
        
        target_dict = {
            "boxes": gt_boxes,
            "labels": torch.full((gt_boxes.shape[0],), dummy_label, dtype=torch.int)
        }
        target_list.append(target_dict)
    
    # Calculate MAP
    metric = MeanAveragePrecision()
    metric.update(pred_list, target_list)
    result = metric.compute()
    
    return {k: v.item() for k, v in result.items()}


class BoundingBoxEvalCallback(TrainerCallback):
    r"""
    Evaluation callback for bounding box detection.
    """
    
    def __init__(self, trainer, tokenizer, val_dataset, name=None):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        if name is None:
            self.name = "BoundingBoxMAP"
        else:
            self.name = name
            
    def on_evaluate(self, args, state, control, **kwargs):
        # Get model predictions and decode them
        predictions = self.trainer.predict(self.val_dataset)
        pred_texts = decode_predictions(self.tokenizer, predictions)
        
        # Evaluate predictions
        result = evaluate_bounding_boxes(pred_texts, self.val_dataset)
        
        # Log results
        map_score = result["map"]
        self.trainer.log_metrics({self.name: map_score})
        wandb.log({f"validate/{self.name}": map_score}, step=state.global_step)
        
        return result