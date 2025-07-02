import json
import hydra
import logging
from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="eval")
def run_eval(cfg: DictConfig) -> None:
    """Run evaluation based on Hydra configuration."""
    log.info(f"Running evaluation with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load predictions
    with open(cfg.predictions_file, 'r') as f:
        predictions = [json.loads(line) for line in f]
    
    # Instantiate evaluators
    evaluators = hydra.utils.instantiate(cfg.evaluators)
    
    # Run all evaluations
    results = {}
    for evaluator in evaluators:
        log.info(f"Running evaluation with {evaluator.name}")
        eval_result = evaluator.evaluate(predictions)
        results[evaluator.name] = eval_result
        
        # Print metrics
        log.info(f"Results for {evaluator.name}:")
        for metric, value in eval_result.items():
            if isinstance(value, (int, float)):
                log.info(f"  {metric}: {value:.4f}")
    
    # Save results if specified
    if cfg.get("output_file"):
        with open(cfg.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {cfg.output_file}")
    
    return results

if __name__ == "__main__":
    run_eval()