"""
Multi-Task NLP Model for AI Predictive Intelligence Platform.

Architecture:
    DeBERTa-v3-base Encoder
            │
    ┌───────┴───────┐
    │  Source Emb(16)│   ← Which platform the text came from
    │  Temporal Emb  │   ← Day-of-week + Month embeddings
    └───────┬───────┘
            │
    Shared Encoder Output (768-D)
            │
    ┌───────┼───────────────┬──────────────┐
    │       │               │              │
  Sentiment  Event Detection  Topic Head   Entity Head
   Head        Head                        
    │       │               │              │
  3-class   N-class         K-topic       Named Entity
  (pos/neg  (policy/crash   probabilities  counts
   /neutral) /launch/...)
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "microsoft/deberta-v3-base"
HIDDEN_DIM = 768
SOURCE_EMBEDDING_DIM = 16
NUM_SOURCES = 10          # see dataset_loader.SOURCE_TO_ID
NUM_DAYS = 7              # Monday=0 ... Sunday=6
NUM_MONTHS = 12           # Jan=0 ... Dec=11
DAY_EMBEDDING_DIM = 8
MONTH_EMBEDDING_DIM = 8

# Task head dimensions
NUM_SENTIMENT_CLASSES = 3     # positive, negative, neutral
NUM_EVENT_CLASSES = 8         # policy_change, tech_breakthrough, supply_chain, market_crash, product_launch, regulation, economic_policy, none
NUM_TOPICS = 10               # AI, EV, Semiconductors, Crypto, Climate, Energy, Healthcare, Finance, Geopolitics, Other
NUM_ENTITY_TYPES = 5          # Company, Person, Location, Product, Other


class MultiTaskNLPModel(nn.Module):
    """
    Shared DeBERTa Encoder + Source/Temporal Embeddings + Multi-Task Heads.
    
    Inputs:
        input_ids:      [batch, seq_len]
        attention_mask: [batch, seq_len]
        source_ids:     [batch]           (integer source ID)
        day_of_week:    [batch]           (0-6)
        month:          [batch]           (0-11)
    
    Outputs (dict):
        'sentiment':    [batch, 3]        (logits)
        'events':       [batch, 8]        (logits)
        'topics':       [batch, 10]       (logits)
        'entities':     [batch, 5]        (logits)
        'embedding':    [batch, 768]      (dense representation for fusion)
    """

    def __init__(self, model_name: str = MODEL_NAME, freeze_encoder_layers: int = 6):
        super().__init__()

        # ── Shared Encoder ──
        logging.info(f"Loading transformer encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze the bottom N layers to speed up training and prevent catastrophic forgetting
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer[:freeze_encoder_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
            logging.info(f"Froze bottom {freeze_encoder_layers} encoder layers")

        # ── Source & Temporal Embeddings ──
        self.source_embedding = nn.Embedding(NUM_SOURCES, SOURCE_EMBEDDING_DIM)
        self.day_embedding = nn.Embedding(NUM_DAYS, DAY_EMBEDDING_DIM)
        self.month_embedding = nn.Embedding(NUM_MONTHS, MONTH_EMBEDDING_DIM)

        # Combined context dimension
        context_dim = HIDDEN_DIM + SOURCE_EMBEDDING_DIM + DAY_EMBEDDING_DIM + MONTH_EMBEDDING_DIM
        # = 768 + 16 + 8 + 8 = 800

        # ── Projection to standard embedding dimension ──
        self.projection = nn.Sequential(
            nn.Linear(context_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── Task Heads ──
        self.sentiment_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_SENTIMENT_CLASSES),
        )

        self.event_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_EVENT_CLASSES),
        )

        self.topic_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_TOPICS),
        )

        self.entity_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_ENTITY_TYPES),
        )

    def forward(self, input_ids, attention_mask, source_ids, day_of_week, month):
        # ── Encode text ──
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        cls_output = encoder_output.last_hidden_state[:, 0, :]  # [batch, 768]

        # ── Get context embeddings ──
        src_emb = self.source_embedding(source_ids)       # [batch, 16]
        day_emb = self.day_embedding(day_of_week)          # [batch, 8]
        month_emb = self.month_embedding(month)            # [batch, 8]

        # ── Concatenate and project ──
        combined = torch.cat([cls_output, src_emb, day_emb, month_emb], dim=-1)  # [batch, 800]
        embedding = self.projection(combined)  # [batch, 768]

        # ── Task heads ──
        sentiment_logits = self.sentiment_head(embedding)
        event_logits = self.event_head(embedding)
        topic_logits = self.topic_head(embedding)
        entity_logits = self.entity_head(embedding)

        return {
            'sentiment': sentiment_logits,
            'events': event_logits,
            'topics': topic_logits,
            'entities': entity_logits,
            'embedding': embedding,  # 768-D dense vector for fusion later
        }

    def get_embedding(self, input_ids, attention_mask, source_ids, day_of_week, month):
        """Extract only the 768-D embedding without computing task heads (for inference speed)."""
        with torch.no_grad():
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = encoder_output.last_hidden_state[:, 0, :]
            src_emb = self.source_embedding(source_ids)
            day_emb = self.day_embedding(day_of_week)
            month_emb = self.month_embedding(month)
            combined = torch.cat([cls_output, src_emb, day_emb, month_emb], dim=-1)
            embedding = self.projection(combined)
        return embedding


# ── Weak Supervision Label Generators ────────────────────────────────────────

def generate_weak_sentiment_labels(texts: list) -> torch.Tensor:
    """
    Generate weak sentiment labels using expanded keyword-based heuristic.
    Phase 11 — 3× expanded keyword pools with financial domain terms.
    
    Returns tensor of shape [batch] with values in {0=positive, 1=negative, 2=neutral}
    """
    positive_keywords = {
        'surge', 'rally', 'gain', 'profit', 'growth', 'bullish', 'soar', 'jump',
        'high', 'record', 'boom', 'up', 'rise', 'strong', 'upgrade', 'outperform',
        'beat', 'exceed', 'optimistic', 'recovery', 'momentum', 'breakout', 'buy',
        'accumulate', 'dividend', 'revenue growth', 'earnings beat', 'all-time high',
        'expanding', 'accelerating', 'robust', 'healthy', 'impressive', 'blowout',
        'upside', 'upbeat', 'confidence', 'tailwind', 'catalyst', 'opportunity'
    }
    negative_keywords = {
        'crash', 'fall', 'drop', 'loss', 'decline', 'bearish', 'plunge', 'sink',
        'low', 'fear', 'recession', 'down', 'weak', 'sell', 'downgrade', 'underperform',
        'miss', 'disappoint', 'pessimistic', 'collapse', 'slump', 'selloff', 'short',
        'default', 'bankruptcy', 'layoff', 'earnings miss', 'guidance cut', 'warning',
        'headwind', 'risk', 'concern', 'uncertainty', 'volatile', 'correction',
        'contraction', 'slowdown', 'overvalued', 'bubble', 'deficit', 'inflation'
    }

    labels = []
    for text in texts:
        text_lower = text.lower()
        words = set(text_lower.split())
        # Word-level matching
        pos_count = len(words & positive_keywords)
        neg_count = len(words & negative_keywords)
        # Phrase-level matching (multi-word)
        for kw in positive_keywords:
            if ' ' in kw and kw in text_lower:
                pos_count += 1
        for kw in negative_keywords:
            if ' ' in kw and kw in text_lower:
                neg_count += 1

        total_matches = pos_count + neg_count
        if total_matches == 0:
            labels.append(2)  # neutral
            continue
            
        # Confidence logic: require a margin between pos and neg to be confident
        confidence = abs(pos_count - neg_count) / total_matches
        if confidence < 0.65:
            labels.append(2)  # neutral (gated)
        elif pos_count > neg_count:
            labels.append(0)  # positive
        else:
            labels.append(1)  # negative

    return torch.tensor(labels, dtype=torch.long)


def generate_weak_event_labels(texts: list) -> torch.Tensor:
    """
    Generate weak event labels using expanded keyword matching.
    Phase 11 — Expanded to cover financial jargon and multi-word event phrases.
    Returns tensor of shape [batch] with event class index.
    """
    event_patterns = {
        0: {'policy', 'regulation', 'government', 'law', 'legislation', 'ban', 'mandate',
            'executive order', 'stimulus', 'subsidy', 'tax reform', 'trade policy'},
        1: {'breakthrough', 'innovation', 'patent', 'discover', 'invent', 'advance',
            'achievement', 'milestone', 'first-ever', 'state-of-the-art', 'cutting-edge'},
        2: {'supply chain', 'shortage', 'logistics', 'shipping', 'port', 'tariff',
            'inventory', 'backlog', 'bottleneck', 'disruption', 'raw material'},
        3: {'crash', 'crisis', 'collapse', 'panic', 'bubble', 'meltdown',
            'flash crash', 'black swan', 'contagion', 'liquidity crisis', 'bank run'},
        4: {'launch', 'release', 'unveil', 'announce', 'debut', 'introduce',
            'rollout', 'preview', 'showcase', 'keynote', 'reveal'},
        5: {'compliance', 'SEC', 'FDA', 'antitrust', 'investigation', 'probe',
            'subpoena', 'fine', 'enforcement', 'audit', 'whistleblower'},
        6: {'interest rate', 'inflation', 'GDP', 'unemployment', 'fiscal', 'monetary',
            'fed', 'central bank', 'rate hike', 'rate cut', 'cpi', 'ppi', 'fomc'},
    }
    none_class = 7

    labels = []
    for text in texts:
        text_lower = text.lower()
        found = none_class
        max_matches = 0
        total_matches = 0
        
        for class_id, keywords in event_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                total_matches += matches
                if matches > max_matches:
                    max_matches = matches
                    found = class_id
        
        # Confidence gating logic
        if total_matches == 0:
            labels.append(none_class)
        else:
            confidence = max_matches / total_matches
            if confidence >= 0.65:
                labels.append(found)
            else:
                labels.append(none_class)

    return torch.tensor(labels, dtype=torch.long)


def generate_weak_topic_labels(texts: list) -> torch.Tensor:
    """
    Generate weak multi-label topic probabilities using keyword matching.
    Returns tensor of shape [batch, NUM_TOPICS] with soft labels (0 or 1).
    """
    topic_keywords = {
        0: {'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'gpt', 'llm'},
        1: {'ev', 'electric vehicle', 'tesla', 'battery', 'charging'},
        2: {'semiconductor', 'chip', 'nvidia', 'amd', 'intel', 'tsmc', 'fab'},
        3: {'crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft'},
        4: {'climate', 'carbon', 'emissions', 'renewable', 'solar', 'wind'},
        5: {'energy', 'oil', 'gas', 'opec', 'crude', 'pipeline'},
        6: {'health', 'pharma', 'drug', 'vaccine', 'biotech', 'fda'},
        7: {'stock', 'market', 'trading', 'investor', 'earnings', 'dividend'},
        8: {'geopolitics', 'war', 'sanctions', 'nato', 'china', 'russia', 'trade war'},
        9: set(),  # "Other" — fallback
    }

    labels = []
    for text in texts:
        text_lower = text.lower()
        row = [0] * NUM_TOPICS
        any_match = False
        for topic_id, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                row[topic_id] = 1
                any_match = True
        if not any_match:
            row[9] = 1  # Other
        labels.append(row)

    return torch.tensor(labels, dtype=torch.float32)


def generate_weak_entity_labels(texts: list) -> torch.Tensor:
    """
    Generate weak multi-label entity type probabilities using keyword matching.
    Returns tensor of shape [batch, NUM_ENTITY_TYPES] with soft labels (0 or 1).
    Entity types: Company, Person, Location, Product, Other
    """
    entity_keywords = {
        0: {'apple', 'google', 'microsoft', 'amazon', 'meta', 'nvidia', 'tesla', 'openai',
             'netflix', 'intel', 'amd', 'tsmc', 'samsung', 'alibaba', 'tencent', 'jpmorgan',
             'goldman', 'blackrock', 'berkshire', 'boeing', 'spacex', 'pfizer', 'moderna'},
        1: {'elon musk', 'musk', 'bezos', 'zuckerberg', 'altman', 'nadella', 'cook', 'pichai',
             'dimon', 'buffett', 'powell', 'yellen', 'biden', 'trump', 'xi jinping', 'lagarde'},
        2: {'china', 'usa', 'europe', 'japan', 'india', 'russia', 'ukraine', 'taiwan',
             'wall street', 'silicon valley', 'washington', 'beijing', 'london', 'frankfurt'},
        3: {'iphone', 'chatgpt', 'gpt-4', 'pixel', 'windows', 'android', 'ios',
             'model', 'chip', 'gpu', 'a100', 'h100', 'vaccine', 'drug'},
    }

    labels = []
    for text in texts:
        text_lower = text.lower()
        row = [0] * NUM_ENTITY_TYPES
        any_match = False
        for entity_id, keywords in entity_keywords.items():
            if any(kw in text_lower for kw in keywords):
                row[entity_id] = 1
                any_match = True
        if not any_match:
            row[4] = 1  # Other
        labels.append(row)

    return torch.tensor(labels, dtype=torch.float32)

def validate_weak_labels(labels_dict: dict) -> dict:
    """
    Phase 11 — Label Quality Validation.
    Measures distribution balance and entropy for each task's weak labels.
    Warns if labels are severely imbalanced (>90% one class).
    """
    import collections
    results = {}
    
    for task_name, labels in labels_dict.items():
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 1:
                counts = collections.Counter(labels.numpy().tolist())
                total = len(labels)
            else:
                # Multi-label: count columns
                counts = {i: int(labels[:, i].sum()) for i in range(labels.shape[1])}
                total = len(labels)
        else:
            counts = collections.Counter(labels)
            total = len(labels)
        
        # Calculate entropy
        probs = np.array([v / total for v in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(counts))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        # Check for severe imbalance
        max_ratio = max(counts.values()) / total
        is_balanced = max_ratio < 0.9
        
        results[task_name] = {
            'distribution': dict(counts),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'max_class_ratio': float(max_ratio),
            'is_balanced': is_balanced
        }
        
        if not is_balanced:
            logging.warning(f"⚠️ LABEL QUALITY: {task_name} is severely imbalanced! "
                          f"Max class ratio: {max_ratio:.2%}")
    
    return results


if __name__ == "__main__":
    # Quick architecture verification
    model = MultiTaskNLPModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
