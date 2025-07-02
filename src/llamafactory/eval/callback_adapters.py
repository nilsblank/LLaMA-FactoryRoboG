import wandb
from transformers import TrainerCallback
from typing import Optional, List, Dict, Any


def decode_predictions(tokenizer, predictions):
    """Decode model predictions to text"""
    prediction_text = tokenizer.batch_decode(
        predictions.predictions.argmax(axis=-1),
        skip_special_tokens=True
    )
    return prediction_text


class EvaluatorCallback(TrainerCallback):
    """
    Adapter that converts any BaseEvaluator into a TrainerCallback.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        evaluator
    ):
        """
        Initialize the adapter.
        
        Args:
            trainer: The trainer instance
            tokenizer: The tokenizer for decoding predictions
            val_dataset: Validation dataset
            evaluator: The evaluator to use for evaluation
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.evaluator = evaluator
        self.name = evaluator.name
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Run evaluation during training."""
        # Get model predictions and decode them
        predictions = self.trainer.predict(self.val_dataset)
        pred_texts = decode_predictions(self.tokenizer, predictions)
        
        # Run evaluation
        result = self.evaluator.evaluate(pred_texts)
        
        # Log main metric (use first numeric result as primary metric)
        primary_metric = None
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and primary_metric is None:
                primary_metric = (key, value)
                break
        
        if primary_metric:
            metric_name, metric_value = primary_metric
            self.trainer.log_metrics({self.name: metric_value})
            wandb.log({f"validate/{self.name}": metric_value}, step=state.global_step)
        
        return result


# Import evaluators here rather than at the top to avoid circular imports
from llamafactory.eval.evaluators import BoundingBoxEvaluator, PointEvaluator


class BoundingBoxEvaluatorCallback(EvaluatorCallback):
    """
    Callback wrapper specifically for BoundingBoxEvaluator.
    This maintains backward compatibility with the existing callback interface.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        name: Optional[str] = None
    ):
        # Create BoundingBoxEvaluator with the validation dataset
        evaluator = BoundingBoxEvaluator(
            ground_truths=val_dataset,  # Pass dataset directly as ground truth
            name=name or "BoundingBoxMAP"
        )
        
        super().__init__(trainer, tokenizer, val_dataset, evaluator)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Specialized evaluation for bounding boxes."""
        result = super().on_evaluate(args, state, control, **kwargs)
        
        # Specifically log MAP score for bounding box evaluation
        if "map" in result:
            self.trainer.log_metrics({self.name: result["map"]})
            wandb.log({f"validate/{self.name}": result["map"]}, step=state.global_step)
        
        return result


class PointEvaluatorCallback(EvaluatorCallback):
    """
    Callback wrapper specifically for PointEvaluator.
    This maintains backward compatibility with the existing callback interface.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        mask_dir: str,
        name: Optional[str] = None
    ):
        # Create mask paths from validation dataset items
        mask_paths = []
        for idx, sample in enumerate(val_dataset):
            mask_path = sample.get("mask_path", f"{mask_dir}/{idx:02d}.jpg")
            mask_paths.append(mask_path)
        
        # Create PointEvaluator with mask paths
        evaluator = PointEvaluator(
            mask_paths=mask_paths,
            name=name or "PointAccuracy"
        )
        
        super().__init__(trainer, tokenizer, val_dataset, evaluator)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Specialized evaluation for points."""
        result = super().on_evaluate(args, state, control, **kwargs)
        
        # Specifically log accuracy for point evaluation
        if "accuracy" in result:
            self.trainer.log_metrics({self.name: result["accuracy"]})
            wandb.log({f"validate/{self.name}": result["accuracy"]}, step=state.global_step)
        
        return result