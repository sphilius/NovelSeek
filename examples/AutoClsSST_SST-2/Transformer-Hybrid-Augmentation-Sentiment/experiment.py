import os
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import time
import json
import pathlib
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    AutoTokenizer,
    AdamW
)
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import traceback
from psycholinguistic_utils import PsycholinguisticFeatures, LinguisticRules, HybridNoiseAugmentation


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    max_seq_len: int = 50
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    patience: int = 1
    max_grad_norm: float = 10.0
    warmup_ratio: float = 0.1
    model_path: str = './hug_ckpts/BERT_ckpt'
    num_labels: int = 2
    if_save_model: bool = True
    out_dir: str = './run_1'
    
    # Hybrid noise augmentation parameters
    use_hybrid_augmentation: bool = True
    sigma: float = 0.1  # Gaussian noise scaling factor
    alpha: float = 0.5  # Hybrid weight
    gamma: float = 0.1  # Attention adjustment parameter
    
    # Evaluation parameters
    evaluate_adversarial: bool = True
    adversarial_types: List[str] = field(default_factory=lambda: ['sarcasm', 'negation', 'polysemy'])

    def validate(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0.0 < self.learning_rate):
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0.0 <= self.sigma <= 1.0):
            raise ValueError("sigma must be between 0 and 1")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be between 0 and 1")


class DataPrecessForSentence(Dataset):
    def __init__(self, bert_tokenizer: AutoTokenizer, df: pd.DataFrame, max_seq_len: int = 50):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self._get_input(df)
        self.raw_texts = df['s1'].values   # Save original text for noise augmentation

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
            self.raw_texts[idx]  # Return original text
        )

    def _get_input(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences = df['s1'].values
        labels = df['similarity'].values

        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))
        result = list(map(self._truncate_and_pad, tokens_seq))

        input_ids = torch.tensor([i[0] for i in result], dtype=torch.long)
        attention_mask = torch.tensor([i[1] for i in result], dtype=torch.long)
        token_type_ids = torch.tensor([i[2] for i in result], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, labels

    def _truncate_and_pad(self, tokens_seq: List[str]) -> Tuple[List[int], List[int], List[int]]:
        tokens_seq = ['[CLS]'] + tokens_seq[:self.max_seq_len - 1]
        padding_length = self.max_seq_len - len(tokens_seq)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += [0] * padding_length
        attention_mask = [1] * len(tokens_seq) + [0] * padding_length
        token_type_ids = [0] * self.max_seq_len

        return input_ids, attention_mask, token_type_ids


class BertClassifier(nn.Module):
    def __init__(
        self, 
        model_path: str, 
        num_labels: int, 
        requires_grad: bool = True,
        use_hybrid_augmentation: bool = True,
        sigma: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.1
    ):
        super().__init__()
        try:
            self.bert = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hybrid noise augmentation settings
        self.use_hybrid_augmentation = use_hybrid_augmentation
        if use_hybrid_augmentation:
            self.hybrid_augmentation = HybridNoiseAugmentation(
                sigma=sigma,
                alpha=alpha,
                gamma=gamma
            )

        for param in self.bert.parameters():
            param.requires_grad = requires_grad
    
    def _apply_hybrid_augmentation(
        self, 
        embeddings: torch.Tensor, 
        attention_mask: torch.Tensor,
        texts: List[str]
    ) -> torch.Tensor:

        if not self.use_hybrid_augmentation:
            return embeddings
        
        # Generate hybrid embeddings
        hybrid_embeddings = self.hybrid_augmentation.generate_hybrid_embeddings(
            embeddings, texts, self.tokenizer
        )
        
        return hybrid_embeddings
    
    def _apply_attention_adjustment(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str]
    ) -> torch.Tensor:
        """Adjust attention scores"""
        if not self.use_hybrid_augmentation:
            # Standard attention calculation
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(query.size(-1))
            
            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_probs, value)
            return context_layer
        
        # Generate psycholinguistic alignment matrix
        H = self.hybrid_augmentation.generate_psycholinguistic_alignment_matrix(
            texts, query.size(2), query.device
        )
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.size(-1))
        
        # Add psycholinguistic alignment
        gamma = self.hybrid_augmentation.gamma
        attention_scores = attention_scores + gamma * H.unsqueeze(1)  # Add dimension for multi-head attention
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value)
        return context_layer

    def forward(
            self,
            batch_seqs: torch.Tensor,
            batch_seq_masks: torch.Tensor,
            batch_seq_segments: torch.Tensor,
            labels: torch.Tensor,
            texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If hybrid noise augmentation is enabled but no texts provided, use standard forward pass
        if self.use_hybrid_augmentation and texts is None:
            logger.warning("Hybrid augmentation enabled but no texts provided. Using standard forward pass.")
            self.use_hybrid_augmentation = False
        
        # Standard BERT forward pass
        outputs = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
            output_hidden_states=self.use_hybrid_augmentation  # Need hidden states if using augmentation
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # If hybrid noise augmentation is enabled, apply to hidden states
        if self.use_hybrid_augmentation and texts:
            # Get the last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            
            # Apply hybrid noise augmentation
            augmented_hidden_states = self._apply_hybrid_augmentation(
                hidden_states, batch_seq_masks, texts
            )
            
            # Recalculate classifier output using augmented hidden states
            pooled_output = augmented_hidden_states[:, 0]  # Use [CLS] token representation
            logits = self.bert.classifier(pooled_output)
            
            # Recalculate loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
        
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities



class BertTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.validate()
        self.model = BertClassifier(
            config.model_path, 
            config.num_labels,
            use_hybrid_augmentation=config.use_hybrid_augmentation,
            sigma=config.sigma,
            alpha=config.alpha,
            gamma=config.gamma
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_data(
            self,
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data = DataPrecessForSentence(
            self.model.tokenizer,
            train_df,
            max_seq_len=self.config.max_seq_len
        )
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        dev_data = DataPrecessForSentence(
            self.model.tokenizer,
            dev_df,
            max_seq_len=self.config.max_seq_len
        )
        dev_loader = DataLoader(
            dev_data,
            shuffle=False,
            batch_size=self.config.batch_size
        )

        test_data = DataPrecessForSentence(
            self.model.tokenizer,
            test_df,
            max_seq_len=self.config.max_seq_len
        )
        test_loader = DataLoader(
            test_data,
            shuffle=False,
            batch_size=self.config.batch_size
        )

        return train_loader, dev_loader, test_loader

    def _prepare_optimizer(self, num_training_steps: int) -> Tuple[AdamW, Any]:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def _initialize_training_stats(self) -> Dict[str, List]:
        return {
            'epochs_count': [],
            'train_losses': [],
            'train_accuracies': [],
            'valid_losses': [],
            'valid_accuracies': [],
            'valid_aucs': []
        }

    def _update_training_stats(
            self,
            training_stats: Dict[str, List],
            epoch: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float]
    ) -> None:
        training_stats['epochs_count'].append(epoch)
        training_stats['train_losses'].append(train_metrics['loss'])
        training_stats['train_accuracies'].append(train_metrics['accuracy'])
        training_stats['valid_losses'].append(val_metrics['loss'])
        training_stats['valid_accuracies'].append(val_metrics['accuracy'])
        training_stats['valid_aucs'].append(val_metrics['auc'])

        logger.info(
            f"Training - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy'] * 100:.2f}%"
        )
        logger.info(
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy'] * 100:.2f}%, "
            f"AUC: {val_metrics['auc']:.4f}"
        )

    def _save_checkpoint(
            self,
            target_dir: str,
            epoch: int,
            optimizer: AdamW,
            best_score: float,
            training_stats: Dict[str, List]
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            **training_stats
        }
        torch.save(
            checkpoint,
            os.path.join(target_dir, "best.pth.tar")
        )
        logger.info("Model saved successfully")

    def _load_checkpoint(
            self,
            checkpoint_path: str,
            optimizer: AdamW,
            training_stats: Dict[str, List]
    ) -> float:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for key in training_stats:
            training_stats[key] = checkpoint[key]
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["best_score"]

    def _train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: AdamW,
            scheduler: Any
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct_preds = 0

        for batch in tqdm(train_loader, desc="Training"):
            # Process batch containing texts
            input_ids, attention_mask, token_type_ids, labels, texts = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            loss, _, probabilities = self.model(
                input_ids, 
                attention_mask, 
                token_type_ids, 
                labels,
                texts  # Pass original texts for noise augmentation
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct_preds += (probabilities.argmax(dim=1) == labels).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_preds / len(train_loader.dataset)
        }

    def _validate_epoch(self, dev_loader: DataLoader) -> Tuple[Dict[str, float], List[float]]:
        self.model.eval()
        total_loss = 0
        correct_preds = 0
        all_probs = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                
                input_ids, attention_mask, token_type_ids, labels, texts = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                loss, _, probabilities = self.model(
                    input_ids, 
                    attention_mask, 
                    token_type_ids, 
                    labels,
                    texts  
                )

                total_loss += loss.item()
                predictions = probabilities.argmax(dim=1)
                correct_preds += (predictions == labels).sum().item()
                all_probs.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        metrics = {
            'loss': total_loss / len(dev_loader),
            'accuracy': correct_preds / len(dev_loader.dataset),
            'auc': roc_auc_score(all_labels, all_probs),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }

        return metrics, all_probs

    def _evaluate_test_set(
            self,
            test_loader: DataLoader,
            target_dir: str,
            epoch: int
    ) -> Dict[str, float]:
        test_metrics, all_probs = self._validate_epoch(test_loader)
        logger.info(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Test F1 score: {test_metrics['f1'] * 100:.2f}%")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")

        test_prediction = pd.DataFrame({'prob_1': all_probs})
        test_prediction['prob_0'] = 1 - test_prediction['prob_1']
        test_prediction['prediction'] = test_prediction.apply(
            lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1,
            axis=1
        )

        output_path = os.path.join(target_dir, f"test_prediction_epoch_{epoch}.csv")
        test_prediction.to_csv(output_path, index=False)
        logger.info(f"Test predictions saved to {output_path}")
        
        if self.config.evaluate_adversarial:
            self._evaluate_adversarial_robustness(test_loader, target_dir, epoch)
        
        return test_metrics
    
    def _evaluate_adversarial_robustness(
            self,
            test_loader: DataLoader,
            target_dir: str,
            epoch: int
    ) -> None:
        """Evaluate model robustness across different linguistic phenomena"""
        logger.info("Evaluating adversarial robustness...")
        
        linguistic_rules = LinguisticRules()
        
        phenomenon_results = {
            'sarcasm': {'correct': 0, 'total': 0},
            'negation': {'correct': 0, 'total': 0},
            'polysemy': {'correct': 0, 'total': 0}
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Adversarial Evaluation"):
                input_ids, attention_mask, token_type_ids, labels, texts = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions
                _, _, probabilities = self.model(
                    input_ids, attention_mask, token_type_ids, labels, texts
                )
                predictions = probabilities.argmax(dim=1)
                
                # Check linguistic phenomena for each sample
                for i, text in enumerate(texts):
                    # Check for sarcasm
                    if linguistic_rules.detect_sarcasm(text):
                        phenomenon_results['sarcasm']['total'] += 1
                        if predictions[i] == labels[i]:
                            phenomenon_results['sarcasm']['correct'] += 1
                    
                    # Check for negation
                    if linguistic_rules.detect_negation(text):
                        phenomenon_results['negation']['total'] += 1
                        if predictions[i] == labels[i]:
                            phenomenon_results['negation']['correct'] += 1
                    
                    # Check for polysemy
                    if linguistic_rules.find_polysemy_words(text):
                        phenomenon_results['polysemy']['total'] += 1
                        if predictions[i] == labels[i]:
                            phenomenon_results['polysemy']['correct'] += 1
        
        phenomenon_accuracy = {}
        for phenomenon, results in phenomenon_results.items():
            if results['total'] > 0:
                accuracy = results['correct'] / results['total']
                phenomenon_accuracy[phenomenon] = accuracy
                logger.info(f"Accuracy on {phenomenon}: {accuracy * 100:.2f}% ({results['correct']}/{results['total']})")
            else:
                phenomenon_accuracy[phenomenon] = 0.0
                logger.info(f"No samples found for {phenomenon}")
        
        with open(os.path.join(target_dir, f"adversarial_results_epoch_{epoch}.json"), "w") as f:
            json.dump(phenomenon_accuracy, f)

    def train_and_evaluate(
            self,
            train_df: pd.DataFrame,
            dev_df: pd.DataFrame,
            test_df: pd.DataFrame,
            target_dir: str,
            checkpoint: Optional[str] = None
    ) -> Dict[str, float]:
        try:
            os.makedirs(target_dir, exist_ok=True)

            train_loader, dev_loader, test_loader = self._prepare_data(
                train_df, dev_df, test_df
            )

            optimizer, scheduler = self._prepare_optimizer(
                len(train_loader) * self.config.epochs
            )

            training_stats = self._initialize_training_stats()
            best_score = 0.0
            patience_counter = 0
            best_test_metrics = None

            if checkpoint:
                best_score = self._load_checkpoint(checkpoint, optimizer, training_stats)

            for epoch in range(1, self.config.epochs + 1):
                logger.info(f"Training epoch {epoch}")

                # Train
                train_metrics = self._train_epoch(train_loader, optimizer, scheduler)

                # Val
                val_metrics, _ = self._validate_epoch(dev_loader)

                self._update_training_stats(training_stats, epoch, train_metrics, val_metrics)

                # Saving / Early stopping
                if val_metrics['accuracy'] > best_score:
                    best_score = val_metrics['accuracy']
                    patience_counter = 0
                    if self.config.if_save_model:
                        self._save_checkpoint(
                            target_dir,
                            epoch,
                            optimizer,
                            best_score,
                            training_stats
                        )
                    best_test_metrics = self._evaluate_test_set(test_loader, target_dir, epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info("Early stopping triggered")
                        break

            if best_test_metrics is None:
                best_test_metrics = self._evaluate_test_set(test_loader, target_dir, epoch)

            return best_test_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    try:
        config = TrainingConfig(out_dir=args.out_dir)
        pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(config.out_dir, "config.json"), "w") as f:
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
            json.dump(config_dict, f, indent=2)

        train_df = pd.read_csv(
            os.path.join(args.data_path, "train.tsv"),
            sep='\t',
            header=None,
            names=['similarity', 's1']
        )
        dev_df = pd.read_csv(
            os.path.join(args.data_path, "dev.tsv"),
            sep='\t',
            header=None,
            names=['similarity', 's1']
        )
        test_df = pd.read_csv(
            os.path.join(args.data_path, "test.tsv"),
            sep='\t',
            header=None,
            names=['similarity', 's1']
        )

        set_seed(2024)

        logger.info(f"Starting training with hybrid augmentation: {config.use_hybrid_augmentation}")
        if config.use_hybrid_augmentation:
            logger.info(f"Augmentation parameters - sigma: {config.sigma}, alpha: {config.alpha}, gamma: {config.gamma}")

        trainer = BertTrainer(config)
        test_metrics = trainer.train_and_evaluate(train_df, dev_df, test_df, os.path.join(config.out_dir, "output"))

        final_infos = {
            "sentiment": {
                "means": {
                    "best_acc": test_metrics['accuracy'],
                    "best_f1": test_metrics['f1'],
                    "best_auc": test_metrics['auc']
                }
            }
        }

        with open(os.path.join(config.out_dir, "final_info.json"), "w") as f:
            json.dump(final_infos, f, indent=2)

        logger.info(f"Training completed successfully. Results saved to {config.out_dir}")

    except Exception as e:
        logger.error(f"Program failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./run_1")
    parser.add_argument("--data_path", type=str, default="./datasets/SST-2/")
    args = parser.parse_args()
    try: 
        main(args)
    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
