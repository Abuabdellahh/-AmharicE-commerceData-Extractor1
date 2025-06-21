import pandas as pd
from ner_model import AmharicNERModel
from sklearn.model_selection import train_test_split
import json
from typing import Dict, Any
import numpy as np

class ModelComparator:
    def __init__(self):
        self.models = {
            "xlm-roberta": "xlm-roberta-base",
            "distilbert": "distilbert-base-multilingual-cased",
            "mbert": "bert-base-multilingual-cased"
        }
        self.results = {}

    def load_dataset(self, dataset_path: str):
        """Load and preprocess dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokens = []
        ner_tags = []
        
        for sentence in data:
            tokens.append(sentence["tokens"])
            ner_tags.append(sentence["ner_tags"])
            
        return {
            "tokens": tokens,
            "ner_tags": ner_tags
        }

    def compare_models(self, dataset_path: str, output_dir: str):
        """Compare different NER models"""
        dataset = self.load_dataset(dataset_path)
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        for model_name, model_path in self.models.items():
            print(f"\nTraining {model_name}...")
            model = AmharicNERModel(model_path)
            
            # Train the model
            trainer = model.train(
                train_dataset=train_dataset,
                val_dataset=test_dataset,
                output_dir=f"{output_dir}/{model_name}"
            )
            
            # Get evaluation results
            metrics = trainer.evaluate()
            self.results[model_name] = metrics
            
            print(f"{model_name} results:")
            print(json.dumps(metrics, indent=2))

    def get_best_model(self) -> str:
        """Return the model with highest F1 score"""
        best_model = None
        best_f1 = 0
        
        for model_name, metrics in self.results.items():
            if metrics["eval_f1"] > best_f1:
                best_f1 = metrics["eval_f1"]
                best_model = model_name
        
        return best_model

    def save_comparison_results(self, output_path: str):
        """Save comparison results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.compare_models(
        dataset_path="path/to/your/conll/dataset.json",
        output_dir="model_comparison_results"
    )
    best_model = comparator.get_best_model()
    print(f"\nBest model: {best_model}")
    comparator.save_comparison_results("model_comparison_results.json")
