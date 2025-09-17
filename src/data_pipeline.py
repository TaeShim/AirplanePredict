"""
Modular data pipeline for airplane price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

from .data_quality import DataQualityValidator, DataProfiler
from .config import DataConfig, ModelConfig, PipelineConfig

class DataExtractor:
    """Data extraction and loading component."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_data(self, file_path: str) -> pd.DataFrame:
        """Extract data from various sources."""
        self.logger.info(f"Extracting data from {file_path}")
        
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Successfully extracted {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting data from {file_path}: {str(e)}")
            raise
    
    def extract_training_data(self) -> pd.DataFrame:
        """Extract training data."""
        return self.extract_data(self.config.raw_data_path)
    
    def extract_test_data(self) -> pd.DataFrame:
        """Extract test data."""
        return self.extract_data(self.config.test_data_path)

class DataTransformer:
    """Data transformation and feature engineering component."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_mappings = {}
        self.scalers = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data."""
        self.logger.info("Starting data cleaning process")
        
        #Create a copy to avoid modifying original
        df_clean = df.copy()
        
        #Remove problematic rows (like the one at index 9039)
        if len(df_clean) > 9039:
            df_clean = df_clean.drop(index=9039).reset_index(drop=True)
            self.logger.info("Removed problematic row at index 9039")
        
        #Remove Additional_Info column if it exists
        if 'Additional_Info' in df_clean.columns:
            df_clean = df_clean.drop('Additional_Info', axis=1)
            self.logger.info("Removed Additional_Info column")
        
        return df_clean
    
    def transform_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features."""
        self.logger.info("Transforming categorical features")
        df_transformed = df.copy()
        
        #Transform Total_Stops
        if 'Total_Stops' in df_transformed.columns:
            stops_mapping = {
                'non-stop': 0,
                '1 stop': 1,
                '2 stops': 2,
                '3 stops': 3,
                '4 stops': 4
            }
            df_transformed['Total_Stops'] = df_transformed['Total_Stops'].map(stops_mapping)
            self.feature_mappings['Total_Stops'] = stops_mapping
            self.logger.info("Transformed Total_Stops to numeric")
        
        return df_transformed
    
    def transform_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform temporal features."""
        self.logger.info("Transforming temporal features")
        df_transformed = df.copy()
        
        #Transform Date_of_Journey
        if 'Date_of_Journey' in df_transformed.columns:
            df_transformed['Date_of_Journey'] = pd.to_datetime(
                df_transformed['Date_of_Journey'], dayfirst=True
            )
            df_transformed['Journey_Day'] = df_transformed['Date_of_Journey'].dt.day
            df_transformed['Journey_Month'] = df_transformed['Date_of_Journey'].dt.month
            df_transformed.drop(columns="Date_of_Journey", axis=1, inplace=True)
            self.logger.info("Extracted day and month from Date_of_Journey")
        
        #Transform Duration
        if 'Duration' in df_transformed.columns:
            df_transformed['Duration'] = df_transformed['Duration'].apply(self._convert_duration)
            df_transformed.rename(columns={'Duration': 'Duration (min)'}, inplace=True)
            self.logger.info("Converted Duration to minutes")
        
        #Transform Dep_Time
        if 'Dep_Time' in df_transformed.columns:
            df_transformed['Dep_Time'] = pd.to_datetime(df_transformed['Dep_Time'], format='%H:%M')
            df_transformed['Dep_Hour'] = df_transformed['Dep_Time'].dt.hour
            df_transformed['Dep_Minute'] = df_transformed['Dep_Time'].dt.minute
            df_transformed.drop(columns='Dep_Time', inplace=True)
            self.logger.info("Extracted hour and minute from Dep_Time")
        
        #Transform Arrival_Time
        if 'Arrival_Time' in df_transformed.columns:
            df_transformed = self._transform_arrival_time(df_transformed)
            self.logger.info("Transformed Arrival_Time")
        
        return df_transformed
    
    def _convert_duration(self, x: str) -> int:
        """Convert duration string to minutes."""
        minutes = 0
        hours = 0
        x = x.strip()
        
        if 'h' in x:
            hours = int(x.split('h')[0].strip())
            x = x.split('h')[1]
        if 'm' in x:
            minutes = int(x.split('m')[0].strip())
        
        return hours * 60 + minutes
    
    def _transform_arrival_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform arrival time features."""
        def process_arrival_time(row):
            cell_length = row.split(' ')
            split_time = row.split(' ')[0].strip()
            
            if len(cell_length) >= 3:
                next_day = 1
            else:
                next_day = 0
            
            hour, minute = map(int, split_time.split(':'))
            return hour, minute, next_day
        
        #Extract arrival components
        df['Arrival_Hour'], df['Arrival_Minute'], df['Arrive_Next_Day'] = zip(
            *df['Arrival_Time'].apply(process_arrival_time)
        )
        
        #Create cyclical encoding for arrival time
        df['Arrival_Since_Midnight'] = df['Arrival_Hour'] * 60 + df['Arrival_Minute']
        df['Arr_Cos'] = np.cos(2 * np.pi * df['Arrival_Since_Midnight'] / 1440)
        df['Arr_Sin'] = np.sin(2 * np.pi * df['Arrival_Since_Midnight'] / 1440)
        
        #Create cyclical encoding for departure time
        df['Dep_Time_Since_Midnight'] = df['Dep_Hour'] * 60 + df['Dep_Minute']
        df['Dep_Cos'] = np.cos(2 * np.pi * df['Dep_Time_Since_Midnight'] / 1440)
        df['Dep_Sin'] = np.sin(2 * np.pi * df['Dep_Time_Since_Midnight'] / 1440)
        
        #Drop original arrival time
        df.drop(columns='Arrival_Time', inplace=True)
        
        return df
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        self.logger.info("Starting feature transformation pipeline")
        
        #Clean data
        df_transformed = self.clean_data(df)
        
        #Transform categorical features
        df_transformed = self.transform_categorical_features(df_transformed)
        
        #Transform temporal features
        df_transformed = self.transform_temporal_features(df_transformed)
        
        self.logger.info("Feature transformation completed")
        return df_transformed
    
    def save_transformation_artifacts(self, output_path: str):
        """Save transformation artifacts for later use."""
        artifacts = {
            'feature_mappings': self.feature_mappings,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{output_path}/transformation_artifacts.json", 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        self.logger.info(f"Transformation artifacts saved to {output_path}")

class DataLoader:
    """Data loading and persistence component."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, output_path: str):
        """Save processed data to various formats."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        #Save as parquet (recommended for data engineering)
        parquet_path = f"{output_path}/{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Data saved as parquet: {parquet_path}")
        
        #Save as CSV for compatibility
        csv_path = f"{output_path}/{filename}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Data saved as CSV: {csv_path}")
        
        #Save metadata
        metadata = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = f"{output_path}/{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved: {metadata_path}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data."""
        if filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, data_config: DataConfig, pipeline_config: PipelineConfig):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        self.logger = logging.getLogger(__name__)
        
        #Initialize components
        self.extractor = DataExtractor(data_config)
        self.transformer = DataTransformer(data_config)
        self.loader = DataLoader(data_config)
        self.quality_validator = DataQualityValidator(data_config)
        self.profiler = DataProfiler()
        
        #Create output directories
        Path(data_config.processed_data_path).mkdir(parents=True, exist_ok=True)
        Path(data_config.output_path).mkdir(parents=True, exist_ok=True)
        Path(data_config.log_path).mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(self, dataset_type: str = "train") -> Tuple[pd.DataFrame, Dict]:
        """Run the complete data pipeline."""
        self.logger.info(f"Starting data pipeline for {dataset_type} dataset")
        
        pipeline_metadata = {
            'start_time': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'config': {
                'data_config': self.data_config.__dict__,
                'pipeline_config': self.pipeline_config.__dict__
            }
        }
        
        try:
            #Extract data
            if dataset_type == "train":
                df = self.extractor.extract_training_data()
            else:
                df = self.extractor.extract_test_data()
            
            #Data quality validation
            if self.pipeline_config.enable_data_quality_checks:
                quality_report = self.quality_validator.validate_dataset(df, f"{dataset_type}_raw")
                self._log_quality_report(quality_report)
                pipeline_metadata['quality_report'] = quality_report.__dict__
            
            #Data profiling
            if self.pipeline_config.enable_data_quality_checks:
                profile = self.profiler.profile_dataset(df)
                pipeline_metadata['data_profile'] = profile
            
            #Transform data
            if self.pipeline_config.enable_feature_engineering:
                df_transformed = self.transformer.transform_features(df)
                
                #Validate transformed data
                if self.pipeline_config.enable_data_quality_checks:
                    quality_report_transformed = self.quality_validator.validate_dataset(
                        df_transformed, f"{dataset_type}_transformed"
                    )
                    self._log_quality_report(quality_report_transformed)
                    pipeline_metadata['quality_report_transformed'] = quality_report_transformed.__dict__
            else:
                df_transformed = df
            
            #Save processed data
            if self.pipeline_config.save_artifacts:
                self.loader.save_processed_data(
                    df_transformed, 
                    f"{dataset_type}_processed", 
                    self.data_config.processed_data_path
                )
                
                #Save transformation artifacts
                self.transformer.save_transformation_artifacts(self.data_config.output_path)
            
            pipeline_metadata['end_time'] = datetime.now().isoformat()
            pipeline_metadata['status'] = 'success'
            pipeline_metadata['output_shape'] = df_transformed.shape
            
            self.logger.info(f"Pipeline completed successfully for {dataset_type} dataset")
            return df_transformed, pipeline_metadata
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            pipeline_metadata['end_time'] = datetime.now().isoformat()
            pipeline_metadata['status'] = 'failed'
            pipeline_metadata['error'] = str(e)
            raise
    
    def _log_quality_report(self, report):
        """Log quality report to file."""
        report_text = self.quality_validator.generate_quality_report(report)
        
        log_file = f"{self.data_config.log_path}/quality_report_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Quality report saved to {log_file}")
        
        #Log quality score
        if report.quality_score < 70:
            self.logger.warning(f"Low data quality score: {report.quality_score:.1f}/100")
        else:
            self.logger.info(f"Data quality score: {report.quality_score:.1f}/100")
    
    def get_pipeline_summary(self) -> Dict:
        """Get pipeline summary and statistics."""
        return {
            'data_config': self.data_config.__dict__,
            'pipeline_config': self.pipeline_config.__dict__,
            'components': [
                'DataExtractor',
                'DataTransformer', 
                'DataLoader',
                'DataQualityValidator',
                'DataProfiler'
            ]
        }
