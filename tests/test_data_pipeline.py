"""
Tests for data pipeline components.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

#Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_pipeline import DataExtractor, DataTransformer, DataLoader, DataPipeline
from src.config import DataConfig, PipelineConfig

class TestDataExtractor:
    """Test cases for DataExtractor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DataConfig()
    
    @pytest.fixture
    def extractor(self, config):
        """Create extractor instance."""
        return DataExtractor(config)
    
    def test_extract_data_csv(self, extractor, tmp_path):
        """Test CSV data extraction."""
        #Create test CSV file
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_file = tmp_path / "test.csv"
        test_data.to_csv(csv_file, index=False)
        
        #Extract data
        result = extractor.extract_data(str(csv_file))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['col1', 'col2']
    
    def test_extract_data_unsupported_format(self, extractor, tmp_path):
        """Test unsupported file format."""
        #Create test file with unsupported extension
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            extractor.extract_data(str(test_file))

class TestDataTransformer:
    """Test cases for DataTransformer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DataConfig()
    
    @pytest.fixture
    def transformer(self, config):
        """Create transformer instance."""
        return DataTransformer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'Price': [1000, 2000, 3000],
            'Total_Stops': ['non-stop', '1 stop', '2 stops'],
            'Date_of_Journey': ['15/01/2019', '16/01/2019', '17/01/2019'],
            'Duration': ['2h 30m', '1h 45m', '3h 15m'],
            'Dep_Time': ['10:30', '14:15', '08:45'],
            'Arrival_Time': ['13:00', '16:00', '12:00'],
            'Route': ['DEL-BOM', 'BOM-DEL', 'BLR-DEL'],
            'Airline': ['Air India', 'IndiGo', 'SpiceJet'],
            'Source': ['Delhi', 'Mumbai', 'Bangalore'],
            'Destination': ['Mumbai', 'Delhi', 'Delhi']
        })
    
    def test_clean_data(self, transformer, sample_data):
        """Test data cleaning."""
        #Add problematic row
        sample_data.loc[9039] = sample_data.iloc[0]
        sample_data['Additional_Info'] = 'test'
        
        result = transformer.clean_data(sample_data)
        
        assert 'Additional_Info' not in result.columns
        assert len(result) == 3  # Original 3 rows
    
    def test_transform_categorical_features(self, transformer, sample_data):
        """Test categorical feature transformation."""
        result = transformer.transform_categorical_features(sample_data)
        
        assert result['Total_Stops'].dtype in ['int64', 'int32']
        assert result['Total_Stops'].iloc[0] == 0  # non-stop -> 0
        assert result['Total_Stops'].iloc[1] == 1  # 1 stop -> 1
        assert result['Total_Stops'].iloc[2] == 2  # 2 stops -> 2
    
    def test_convert_duration(self, transformer):
        """Test duration conversion."""
        assert transformer._convert_duration('2h 30m') == 150  # 2*60 + 30
        assert transformer._convert_duration('1h 45m') == 105  # 1*60 + 45
        assert transformer._convert_duration('30m') == 30     # 0*60 + 30
        assert transformer._convert_duration('2h') == 120     # 2*60 + 0
    
    def test_transform_temporal_features(self, transformer, sample_data):
        """Test temporal feature transformation."""
        result = transformer.transform_temporal_features(sample_data)
        
        #Check date features
        assert 'Journey_Day' in result.columns
        assert 'Journey_Month' in result.columns
        assert 'Date_of_Journey' not in result.columns
        
        #Check duration
        assert 'Duration (min)' in result.columns
        assert 'Duration' not in result.columns
        
        #Check departure time features
        assert 'Dep_Hour' in result.columns
        assert 'Dep_Minute' in result.columns
        assert 'Dep_Time' not in result.columns
        
        #Check arrival time features
        assert 'Arrival_Hour' in result.columns
        assert 'Arrival_Minute' in result.columns
        assert 'Arrive_Next_Day' in result.columns
        assert 'Arrival_Time' not in result.columns

class TestDataLoader:
    """Test cases for DataLoader."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DataConfig()
    
    @pytest.fixture
    def loader(self, config):
        """Create loader instance."""
        return DataLoader(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def test_save_processed_data(self, loader, sample_data, tmp_path):
        """Test saving processed data."""
        output_path = str(tmp_path)
        
        loader.save_processed_data(sample_data, "test_data", output_path)
        
        #Check if files were created
        assert (tmp_path / "test_data.parquet").exists()
        assert (tmp_path / "test_data.csv").exists()
        assert (tmp_path / "test_data_metadata.json").exists()
    
    def test_load_processed_data(self, loader, sample_data, tmp_path):
        """Test loading processed data."""
        output_path = str(tmp_path)
        
        #Save data first
        loader.save_processed_data(sample_data, "test_data", output_path)
        
        #Load data
        loaded_data = loader.load_processed_data(str(tmp_path / "test_data.parquet"))
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ['col1', 'col2']

class TestDataPipeline:
    """Test cases for DataPipeline."""
    
    @pytest.fixture
    def configs(self):
        """Create test configurations."""
        data_config = DataConfig()
        pipeline_config = PipelineConfig()
        return data_config, pipeline_config
    
    @pytest.fixture
    def pipeline(self, configs):
        """Create pipeline instance."""
        data_config, pipeline_config = configs
        return DataPipeline(data_config, pipeline_config)
    
    @patch('src.data_pipeline.DataExtractor.extract_training_data')
    def test_run_pipeline_mock(self, mock_extract, pipeline):
        """Test pipeline execution with mocked data extraction."""
        #Mock the data extraction
        mock_data = pd.DataFrame({
            'Price': [1000, 2000, 3000],
            'Total_Stops': ['non-stop', '1 stop', '2 stops'],
            'Date_of_Journey': ['15/01/2019', '16/01/2019', '17/01/2019'],
            'Duration': ['2h 30m', '1h 45m', '3h 15m'],
            'Dep_Time': ['10:30', '14:15', '08:45'],
            'Arrival_Time': ['13:00', '16:00', '12:00'],
            'Route': ['DEL-BOM', 'BOM-DEL', 'BLR-DEL'],
            'Airline': ['Air India', 'IndiGo', 'SpiceJet'],
            'Source': ['Delhi', 'Mumbai', 'Bangalore'],
            'Destination': ['Mumbai', 'Delhi', 'Delhi']
        })
        mock_extract.return_value = mock_data
        
        #Run pipeline
        result_data, metadata = pipeline.run_pipeline("train")
        
        #Verify results
        assert isinstance(result_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata['status'] == 'success'
        assert 'output_shape' in metadata
