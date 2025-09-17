"""
Configuration management for the airplane price prediction pipeline.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_data_path: str = "Airplane_Dataset/Data_Train.xlsx"
    test_data_path: str = "Airplane_Dataset/Test_set.xlsx"
    processed_data_path: str = "data/processed"
    output_path: str = "data/output"
    log_path: str = "logs"
    
    #Data quality thresholds
    max_missing_percentage: float = 0.1
    outlier_threshold: float = 3.0
    min_price: float = 100.0
    max_price: float = 100000.0
    
    #Feature engineering settings
    categorical_columns: List[str] = None
    numerical_columns: List[str] = None
    target_column: str = "Price"
    
    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = ['Route', 'Destination', 'Source', 'Airline']
        if self.numerical_columns is None:
            self.numerical_columns = [
                'Total_Stops', 'Journey_Day', 'Journey_Month', 
                'Duration (min)', 'Arrival_Hour', 'Arrival_Minute',
                'Dep_Time_Since_Midnight', 'Dep_Cos', 'Dep_Sin', 
                'Arrival_Since_Midnight', 'Arr_Cos', 'Arr_Sin', 'Arrive_Next_Day'
            ]

@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_type: str = "RandomForestRegressor"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    #Hyperparameter search space
    param_grid: Dict = None
    
    def __post_init__(self):
        if self.param_grid is None:
            self.param_grid = {
                "regressor__n_estimators": [100, 200, 300],
                "regressor__max_depth": [10, 20, None]
            }

@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    enable_data_quality_checks: bool = True
    enable_feature_engineering: bool = True
    enable_model_training: bool = True
    enable_model_evaluation: bool = True
    save_artifacts: bool = True
    
    #Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_config() -> tuple[DataConfig, ModelConfig, PipelineConfig]:
    """Get configuration objects."""
    return DataConfig(), ModelConfig(), PipelineConfig()

def create_directories(config: DataConfig):
    """Create necessary directories."""
    directories = [
        config.processed_data_path,
        config.output_path,
        config.log_path
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
