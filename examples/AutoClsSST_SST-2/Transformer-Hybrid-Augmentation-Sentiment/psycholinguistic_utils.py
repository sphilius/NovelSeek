import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Union, Optional
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
def ensure_nltk_resources():
    """Ensure necessary NLTK resources are downloaded"""
    resources = ['punkt', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource {resource} already exists")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource {resource}")
                nltk.download(resource, quiet=False)
                logger.info(f"NLTK resource {resource} downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
    
    # Try to download punkt_tab resource
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            logger.info("Downloading NLTK resource punkt_tab")
            nltk.download('punkt_tab', quiet=False)
            logger.info("NLTK resource punkt_tab downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download NLTK resource punkt_tab: {str(e)}")
            logger.info("Will use alternative tokenization method")

# Try to download resources when module is imported
ensure_nltk_resources()

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Simple tokenization function, not dependent on NLTK
def simple_tokenize(text):
    """Simple tokenization function using regular expressions"""
    if not isinstance(text, str):
        return []
    # Convert text to lowercase
    text = text.lower()
    # Use regular expressions for tokenization, preserving letters, numbers, and some basic punctuation
    import re
    tokens = re.findall(r'\b\w+\b|[!?,.]', text)
    return tokens

# Add more robust tokenization processing
def safe_tokenize(text):
    """Safe tokenization function, uses simple tokenization method when NLTK tokenization fails"""
    if not isinstance(text, str):
        return []
    
    # First try using NLTK's word_tokenize
    punkt_available = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        punkt_available = False
    
    if punkt_available:
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
    
    # If NLTK tokenization is not available or fails, use simple tokenization method
    return simple_tokenize(text)

# Load psycholinguistic dictionary (simulated - should use real data in actual applications)
class PsycholinguisticFeatures:
    def __init__(self, lexicon_path: Optional[str] = None):
        """
        Initialize psycholinguistic feature extractor
        
        Args:
            lexicon_path: Path to psycholinguistic lexicon, uses simulated data if None
        """
        # If no lexicon is provided, create a simple simulated dictionary
        if lexicon_path and os.path.exists(lexicon_path):
            self.lexicon = pd.read_csv(lexicon_path)
            self.word_to_scores = {
                row['word']: {
                    'valence': row['valence'],
                    'arousal': row['arousal'],
                    'dominance': row['dominance']
                } for _, row in self.lexicon.iterrows()
            }
        else:
            # Create simulated dictionary
            self.word_to_scores = {}
            # Sentiment vocabulary
            positive_words = ['good', 'great', 'excellent', 'happy', 'joy', 'love', 'nice', 'wonderful', 'amazing', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate', 'poor', 'horrible', 'disappointing', 'worst', 'negative']
            neutral_words = ['the', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'and', 'or', 'but', 'if', 'while', 'when']
            
            # Assign high values to positive words
            for word in positive_words:
                self.word_to_scores[word] = {
                    'valence': np.random.uniform(0.7, 0.9),
                    'arousal': np.random.uniform(0.5, 0.8),
                    'dominance': np.random.uniform(0.6, 0.9)
                }
            
            # Assign low values to negative words
            for word in negative_words:
                self.word_to_scores[word] = {
                    'valence': np.random.uniform(0.1, 0.3),
                    'arousal': np.random.uniform(0.5, 0.8),
                    'dominance': np.random.uniform(0.1, 0.4)
                }
            
            # Assign medium values to neutral words
            for word in neutral_words:
                self.word_to_scores[word] = {
                    'valence': np.random.uniform(0.4, 0.6),
                    'arousal': np.random.uniform(0.3, 0.5),
                    'dominance': np.random.uniform(0.4, 0.6)
                }
    
    def get_token_scores(self, token: str) -> Dict[str, float]:
        """Get psycholinguistic scores for a single token"""
        token = token.lower()
        if token in self.word_to_scores:
            return self.word_to_scores[token]
        else:
            # Return medium values for unknown words
            return {
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }
    
    def get_importance_score(self, token: str) -> float:
        """Calculate importance score for a token"""
        scores = self.get_token_scores(token)
        # Importance score is a weighted combination of valence, arousal, and dominance
        # Here we give valence a higher weight because it is more relevant to sentiment analysis
        importance = 0.6 * abs(scores['valence'] - 0.5) + 0.2 * scores['arousal'] + 0.2 * scores['dominance']
        return importance
    
    def compute_scores_for_text(self, text: str) -> List[Dict[str, float]]:
        """Calculate psycholinguistic scores for each token in the text"""
        tokens = safe_tokenize(text)
        return [self.get_token_scores(token) for token in tokens]
    
    def compute_importance_for_text(self, text: str) -> List[float]:
        """Calculate importance scores for each token in the text"""
        tokens = safe_tokenize(text)
        return [self.get_importance_score(token) for token in tokens]


class LinguisticRules:
    def __init__(self):
        """Initialize linguistic rules processor"""
        # Regular expressions for sarcasm patterns
        self.sarcasm_patterns = [
            r'(so|really|very|totally) (great|nice|good|wonderful|fantastic)',
            r'(yeah|sure|right),? (like|as if)',
            r'(oh|ah),? (great|wonderful|fantastic|perfect)'
        ]
        
        # List of negation words
        self.negation_words = [
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere',
            "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"
        ]
        
        # Polysemous words and their possible substitutes
        self.polysemy_words = {
            'fine': ['good', 'acceptable', 'penalty', 'delicate'],
            'right': ['correct', 'appropriate', 'conservative', 'direction'],
            'like': ['enjoy', 'similar', 'such as', 'want'],
            'mean': ['signify', 'unkind', 'average', 'intend'],
            'kind': ['type', 'benevolent', 'sort', 'sympathetic'],
            'fair': ['just', 'pale', 'average', 'exhibition'],
            'light': ['illumination', 'lightweight', 'pale', 'ignite'],
            'hard': ['difficult', 'solid', 'harsh', 'diligent'],
            'sound': ['noise', 'healthy', 'logical', 'measure'],
            'bright': ['intelligent', 'luminous', 'vivid', 'promising']
        }
    
    def detect_sarcasm(self, text: str) -> bool:
        """Detect if sarcasm patterns exist in the text"""
        text = text.lower()
        for pattern in self.sarcasm_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def detect_negation(self, text: str) -> List[int]:
        """Detect positions of negation words in the text"""
        tokens = safe_tokenize(text)
        negation_positions = []
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                negation_positions.append(i)
        return negation_positions
    
    def find_polysemy_words(self, text: str) -> Dict[int, List[str]]:
        """Find polysemous words in the text and their possible substitutes"""
        tokens = safe_tokenize(text)
        polysemy_positions = {}
        for i, token in enumerate(tokens):
            if token in self.polysemy_words:
                polysemy_positions[i] = self.polysemy_words[token]
        return polysemy_positions
    
    def get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet"""
        synonyms = []
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return list(set(synonyms))
    
    def apply_rule_transformations(self, token_embeddings: torch.Tensor, text: str, tokenizer) -> torch.Tensor:
        """
        Apply rule-based transformations to token embeddings
        
        Args:
            token_embeddings: Original token embeddings [batch_size, seq_len, hidden_dim]
            text: Original text
            tokenizer: Tokenizer
        
        Returns:
            Transformed token embeddings
        """
        # Clone original embeddings
        transformed_embeddings = token_embeddings.clone()
        
        try:
            # Detect sarcasm
            if self.detect_sarcasm(text):
                # For sarcasm, we reverse sentiment-related embedding dimensions
                # This is a simplified implementation, more complex transformations may be needed in real applications
                sentiment_dims = torch.randperm(token_embeddings.shape[-1])[:token_embeddings.shape[-1]//10]
                transformed_embeddings[:, :, sentiment_dims] = -transformed_embeddings[:, :, sentiment_dims]
            
            # Handle negation
            negation_positions = self.detect_negation(text)
            if negation_positions:
                # For words following negation words, reverse their sentiment-related embedding dimensions
                try:
                    tokens = tokenizer.tokenize(text)
                except Exception as e:
                    logger.warning(f"Tokenization failed: {str(e)}, using alternative tokenization")
                    tokens = safe_tokenize(text)
                
                for pos in negation_positions:
                    if pos + 1 < len(tokens):  # Ensure there's a word after the negation
                        # Find the position of the token after negation in the embeddings
                        # Simplified handling, actual applications should consider tokenization differences
                        sentiment_dims = torch.randperm(token_embeddings.shape[-1])[:token_embeddings.shape[-1]//10]
                        if pos + 1 < token_embeddings.shape[1]:  # Ensure not exceeding embedding dimensions
                            transformed_embeddings[:, pos+1, sentiment_dims] = -transformed_embeddings[:, pos+1, sentiment_dims]
            
            # Handle polysemy
            polysemy_positions = self.find_polysemy_words(text)
            if polysemy_positions:
                # For polysemous words, add some noise to simulate semantic ambiguity
                for pos in polysemy_positions:
                    if pos < token_embeddings.shape[1]:  # Ensure not exceeding embedding dimensions
                        noise = torch.randn_like(transformed_embeddings[:, pos, :]) * 0.1
                        transformed_embeddings[:, pos, :] += noise
        except Exception as e:
            logger.error(f"Error applying rule transformations: {str(e)}")
            # Return original embeddings in case of error
        
        return transformed_embeddings


class HybridNoiseAugmentation:
    def __init__(
        self, 
        sigma: float = 0.1, 
        alpha: float = 0.5,
        gamma: float = 0.1,
        psycholinguistic_features: Optional[PsycholinguisticFeatures] = None,
        linguistic_rules: Optional[LinguisticRules] = None
    ):
        """
        Initialize hybrid noise augmentation
        
        Args:
            sigma: Scaling factor for Gaussian noise
            alpha: Mixing weight parameter
            gamma: Adjustment parameter in attention mechanism
            psycholinguistic_features: Psycholinguistic feature extractor
            linguistic_rules: Linguistic rules processor
        """
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.psycholinguistic_features = psycholinguistic_features or PsycholinguisticFeatures()
        self.linguistic_rules = linguistic_rules or LinguisticRules()
    
    def apply_psycholinguistic_noise(
        self, 
        token_embeddings: torch.Tensor, 
        texts: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Apply psycholinguistic-based noise
        
        Args:
            token_embeddings: Original token embeddings [batch_size, seq_len, hidden_dim]
            texts: List of original texts
            tokenizer: Tokenizer
        
        Returns:
            Token embeddings with applied noise
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        noised_embeddings = token_embeddings.clone()
        
        for i, text in enumerate(texts):
            try:
                # Calculate importance scores for each token
                importance_scores = self.psycholinguistic_features.compute_importance_for_text(text)
                
                # Tokenize the text to match the model's tokenization
                try:
                    model_tokens = tokenizer.tokenize(text)
                except Exception as e:
                    logger.warning(f"Model tokenization failed: {str(e)}, using alternative tokenization")
                    model_tokens = safe_tokenize(text)
                
                # Assign importance scores to each token (simplified handling)
                token_scores = torch.ones(seq_len, device=token_embeddings.device) * 0.5
                for j, token in enumerate(model_tokens[:seq_len-2]):  # Exclude [CLS] and [SEP]
                    if j < len(importance_scores):
                        token_scores[j+1] = importance_scores[j]  # +1 is for [CLS]
                
                # Scale noise according to importance scores
                noise = torch.randn_like(token_embeddings[i]) * self.sigma
                scaled_noise = noise * token_scores.unsqueeze(1)
                
                # Apply noise
                noised_embeddings[i] = token_embeddings[i] + scaled_noise
            except Exception as e:
                logger.error(f"Error processing text {i}: {str(e)}")
                # Use original embeddings in case of error
                continue
        
        return noised_embeddings
    
    def apply_rule_based_perturbation(
        self, 
        token_embeddings: torch.Tensor, 
        texts: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Apply rule-based perturbation
        
        Args:
            token_embeddings: Original token embeddings [batch_size, seq_len, hidden_dim]
            texts: List of original texts
            tokenizer: Tokenizer
        
        Returns:
            Token embeddings with applied perturbation
        """
        batch_size = token_embeddings.shape[0]
        perturbed_embeddings = token_embeddings.clone()
        
        for i, text in enumerate(texts):
            try:
                # Apply rule transformations
                perturbed_embeddings[i:i+1] = self.linguistic_rules.apply_rule_transformations(
                    token_embeddings[i:i+1], text, tokenizer
                )
            except Exception as e:
                logger.error(f"Error applying rule transformations to text {i}: {str(e)}")
                # Keep original embeddings in case of error
                continue
        
        return perturbed_embeddings
    
    def generate_hybrid_embeddings(
        self, 
        token_embeddings: torch.Tensor, 
        texts: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Generate hybrid embeddings
        
        Args:
            token_embeddings: Original token embeddings [batch_size, seq_len, hidden_dim]
            texts: List of original texts
            tokenizer: Tokenizer
        
        Returns:
            Hybrid embeddings
        """
        # Apply psycholinguistic noise
        psycholinguistic_embeddings = self.apply_psycholinguistic_noise(token_embeddings, texts, tokenizer)
        
        # Apply rule-based perturbation
        rule_based_embeddings = self.apply_rule_based_perturbation(token_embeddings, texts, tokenizer)
        
        # Mix the two types of embeddings
        hybrid_embeddings = (
            self.alpha * psycholinguistic_embeddings + 
            (1 - self.alpha) * rule_based_embeddings
        )
        
        return hybrid_embeddings
    
    def generate_psycholinguistic_alignment_matrix(
        self, 
        texts: List[str], 
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate psycholinguistic alignment matrix
        
        Args:
            texts: List of original texts
            seq_len: Sequence length
            device: Computation device
        
        Returns:
            Psycholinguistic alignment matrix [batch_size, seq_len, seq_len]
        """
        batch_size = len(texts)
        H = torch.zeros((batch_size, seq_len, seq_len), device=device)
        
        for i, text in enumerate(texts):
            try:
                # Calculate importance scores for each token
                importance_scores = self.psycholinguistic_features.compute_importance_for_text(text)
                
                # Pad to sequence length
                padded_scores = importance_scores + [0.5] * (seq_len - len(importance_scores))
                padded_scores = padded_scores[:seq_len]
                
                # Create alignment matrix
                scores_tensor = torch.tensor(padded_scores, device=device)
                # Use outer product to create matrix, emphasizing relationships between important tokens
                H[i] = torch.outer(scores_tensor, scores_tensor)
            except Exception as e:
                logger.error(f"Error generating alignment matrix for text {i}: {str(e)}")
                # Use default values in case of error
                H[i] = torch.eye(seq_len, device=device) * 0.5
        
        return H