import json
import os
import torch
import numpy as np
import re
from PIL import Image
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import List, Dict, Any, Union, Optional, Tuple


class BaseEvaluator:
    """Base evaluator class that defines the common interface."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the evaluator with a name."""
        self.name = name
    
    def load_data(self):
        """Load necessary data for evaluation."""
        pass
    
    @staticmethod
    def extract_text(prediction: Union[str, Dict[str, Any]]) -> str:
        """
        Extract text from prediction, which can be a string or dict with 'text' key.
        
        Args:
            prediction: Prediction object or string
            
        Returns:
            Extracted text
        """
        if isinstance(prediction, dict) and "text" in prediction:
            return prediction["text"]
        return prediction  # Assume it's already text
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate the predictions.
        
        Args:
            predictions: List of prediction strings or dicts
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class BoundingBoxEvaluator(BaseEvaluator):
    """Evaluator for bounding box predictions."""
    
    def __init__(
        self, 
        ground_truth_file: Optional[str] = None, 
        ground_truths: Optional[List[Dict[str, Any]]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the bounding box evaluator.
        
        Args:
            ground_truth_file: Path to ground truth file (one JSON per line)
            ground_truths: Ground truth data (alternative to file)
            name: Name for this evaluator
        """
        super().__init__(name=name or "BoundingBoxMAP")
        self.ground_truth_file = ground_truth_file
        self.ground_truths = ground_truths
    
    def load_data(self):
        """Load ground truth data if not provided directly."""
        if self.ground_truths is None and self.ground_truth_file:
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truths = [json.loads(line) for line in f]
        
        #Parse gt boxes from ground truths
        parsed = []
        for gt in self.ground_truths:
            gt_boxes = self.parse_bbox_from_text(gt)
            parsed.append(gt_boxes)
        self.ground_truths = parsed
        
            
    
    @staticmethod
    def parse_bbox_from_text(text: str) -> List[List[float]]:
        """
        Parse bounding box coordinates from prediction text Qwen Format.
        Expected format: "'```json\n[\n\t{"label": "orange", "bbox_2d": [68, 102, 97, 131]}\n]\n```'
        
        Args:
            text: String containing prediction
            
        Returns:
            List of bounding boxes as [x1, y1, x2, y2]
        """
        boxes = []
        try:
            # Remove the code block markers and parse JSON
            json_str = text.strip().replace("```json", "").replace("```", "").strip()
            bbox_data = json.loads(json_str)
            
            boxes = []
            for item in bbox_data:
                if "bbox_2d" in item:
                    box = item["bbox_2d"]
                    if len(box) == 4:
                        boxes.append(box)
            return boxes
        except json.JSONDecodeError as e:
            #fallback, try to parse from bbox_2d
            pattern = r'bbox_2d"\s*:\s*\[(.*?)\]'
            matches = re.findall(pattern, text)
            if matches:
                boxes = []
                for match in matches:
                    coords = match.split(',')
                    if len(coords) == 4:
                        try:
                            box = [int(coord.strip()) for coord in coords]
                            boxes.append(box)
                        except ValueError:
                            continue
                return boxes
            
        
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, float]:
        """
        Evaluate bounding box predictions against ground truth.
        
        Args:
            predictions: List of prediction strings or dicts with 'text' key
            
        Returns:
            Dictionary with MAP metrics
        """
        # Load ground truth data if needed
        self.load_data()
        
        if not self.ground_truths:
            raise ValueError("No ground truth data available. Provide either ground_truths or ground_truth_file.")
        
        # Ensure we have matching number of predictions and ground truths
        if len(predictions) != len(self.ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match ground truths ({len(self.ground_truths)})")
        
        # Prepare data for evaluation
        pred_list = []
        target_list = []
        dummy_label = 1  # Single class label
        
        for pred, target in zip(predictions, self.ground_truths):
            # Process prediction
            pred_text = self.extract_text(pred)
            
            # Parse predicted boxes
            box_list = self.parse_bbox_from_text(pred_text) if isinstance(pred_text, str) else []
            boxes = torch.tensor(box_list) if box_list else torch.empty((0, 4))
            
            pred_dict = {
                "boxes": boxes,
                "scores": torch.ones(boxes.shape[0]),  # Constant score
                "labels": torch.full((boxes.shape[0],), dummy_label, dtype=torch.int)
            }
            pred_list.append(pred_dict)
            
            # Process ground truth
            gt_boxes = target
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


class PointEvaluator(BaseEvaluator):
    """Evaluator for point predictions."""
    
    def __init__(
        self, 
        mask_dir: Optional[str] = None,
        mask_paths: Optional[List[str]] = None,
        name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the point evaluator.
        
        Args:
            mask_dir: Directory containing masks
            mask_paths: Specific mask paths (alternative to mask_dir)
            name: Name for this evaluator
            verbose: Whether to print processing details
        """
        super().__init__(name=name or "PointAccuracy")
        self.mask_dir = mask_dir
        self.mask_paths = mask_paths
        self.verbose = verbose
    
    def load_data(self, num_samples: Optional[int] = None):
        """Prepare mask paths if not provided directly."""
        if self.mask_paths is None and self.mask_dir:
            # Create default mask paths based on indices
            if num_samples is not None:
                self.mask_paths = [os.path.join(self.mask_dir, f"{i:02d}.jpg") for i in range(num_samples)]
    
    @staticmethod
    def text2pts(text: str) -> np.ndarray:
        """
        Extract points from model output text.
        Expected format: points like (x,y) or x,y
        
        Args:
            text: String containing point coordinates
            
        Returns:
            Array of [x, y] points
        """
        # Match (x,y) or x,y patterns
        pattern = r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?'
        matches = re.findall(pattern, text)
        
        if not matches:
            return np.zeros((0, 2))
        
        points = np.array([[int(x), int(y)] for x, y in matches])
        return points
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate point predictions against ground truth masks.
        
        Args:
            predictions: List of prediction strings or dicts with 'text' key
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Prepare mask paths if needed
        self.load_data(num_samples=len(predictions))
        
        if not self.mask_paths:
            raise ValueError("No mask paths available. Provide either mask_paths or mask_dir.")
        
        # Ensure we have matching number of predictions and masks
        if len(predictions) != len(self.mask_paths):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match masks ({len(self.mask_paths)})")
        
        # Evaluate predictions
        accuracies = []
        
        for idx, (pred, mask_path) in enumerate(zip(predictions, self.mask_paths)):
            # Extract text from prediction
            pred_text = self.extract_text(pred)
            
            # Parse points
            try:
                points = self.text2pts(pred_text)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to parse points for sample {idx}: {e}")
                points = np.zeros((0, 2))
            
            # Load mask
            try:
                mask = np.array(Image.open(mask_path)) / 255.0
            except Exception as e:
                if self.verbose:
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
        
        return {
            "accuracy": mean_accuracy,
            "individual_accuracies": accuracies
        }