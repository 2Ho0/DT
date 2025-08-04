"""
Prioritized Experience Replay Buffer implementation
"""

import torch as t
import numpy as np
from collections import deque


class PERBuffer:
    """Prioritized Experience Replay Buffer with embedding similarity"""
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.embeddings = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience, embedding, priority=1.0):
        """Add experience with its embedding and priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.embeddings.append(embedding)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.embeddings[self.position] = embedding
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
    
    def compute_similarity_priority(self, new_embedding):
        """Compute priority based on embedding similarity"""
        if len(self.embeddings) == 0:
            return 1.0
            
        similarities = []
        for stored_embedding in self.embeddings:
            # Cosine similarity
            cos_sim = t.nn.functional.cosine_similarity(
                new_embedding.flatten().unsqueeze(0),
                stored_embedding.flatten().unsqueeze(0)
            )
            similarities.append(cos_sim.item())
        
        # Higher priority for more novel (less similar) experiences
        max_similarity = max(similarities)
        priority = 1.0 - max_similarity
        return max(0.1, priority)  # Minimum priority of 0.1
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) == 0:
            return [], []
            
        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        return batch, weights