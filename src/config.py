from dataclasses import dataclass
import argparse
from typing import Optional

@dataclass
class Config:
    """Configuration for TriLens probe training."""
    
    # Model parameters
    input_dim: int = 32
    hidden_dim: int = 128
    
    # Training parameters
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    lr_factor: float = 0.5
    lr_patience: int = 5
    
    # Data parameters
    test_size: float = 0.2
    dataset_weight: bool = True
    
    # Paths
    data_dir: str = None
    save_dir: str = None
    
    @classmethod
    def from_args(cls):
        """Create config from command line arguments."""
        parser = argparse.ArgumentParser()
        # Add arguments
        parser.add_argument('--data_dir', required=True)
        parser.add_argument('--save_dir', required=True)
        
        args = parser.parse_args()
        return cls(**vars(args))
