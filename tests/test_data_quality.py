"""
Tests for data quality validation framework.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

#Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_quality import DataQualityValidator, DataProfiler, DataQualityReport
from src.config import DataConfig

class TestDataQualityValidator:
    """Test cases for DataQualityValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DataConfig()
    
    @pytest.fixture
    def validator(self, config):
        """Create validator instance."""
        return DataQualityValidator(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'Price': [1000, 2000, 3000, 4000, 5000],
            'Airline': ['Air India', 'IndiGo', 'SpiceJet', 'Air India', 'IndiGo'],
            'Source': ['Delhi', 'Mumbai', 'Bangalore', 'Delhi', 'Mumbai'],
            'Destination': ['Mumbai', 'Delhi', 'Delhi', 'Bangalore', 'Bangalore'],
            'Duration': ['2h 30m', '1h 45m', '3h 15m', '2h 0m', '1h 30m'],
            'Date_of_Journey': ['15/01/2019', '16/01/2019', '17/01/2019', '18/01/2019', '19/01/2019']
        })
    
    def test_validate_dataset_basic(self, validator, sample_data):
        """Test basic dataset validation."""
        report = validator.validate_dataset(sample_data, "test_dataset")
        
        assert isinstance(report, DataQualityReport)
        assert report.dataset_name == "test_dataset"
        assert report.total_rows == 5
        assert report.total_columns == 6
        assert 0 <= report.quality_score <= 100
    
    def test_missing_data_analysis(self, validator, sample_data):
        """Test missing data analysis."""
        #Add some missing values
        sample_data.loc[0, 'Price'] = np.nan
        sample_data.loc[1, 'Airline'] = np.nan
        
        report = validator.validate_dataset(sample_data, "test_dataset")
        
        assert 'Price' in report.missing_data
        assert 'Airline' in report.missing_data
        assert report.missing_data['Price'] == 20.0  # 1 out of 5 = 20%
        assert report.missing_data['Airline'] == 20.0
    
    def test_duplicate_analysis(self, validator, sample_data):
        """Test duplicate analysis."""
        #Add duplicate row
        duplicate_row = sample_data.iloc[0].copy()
        sample_data = pd.concat([sample_data, duplicate_row.to_frame().T], ignore_index=True)
        
        report = validator.validate_dataset(sample_data, "test_dataset")
        
        assert report.duplicates == 1
    
    def test_business_rule_validation(self, validator, sample_data):
        """Test business rule validation."""
        #Add negative price
        sample_data.loc[0, 'Price'] = -100
        
        report = validator.validate_dataset(sample_data, "test_dataset")
        
        assert len(report.issues) > 0
        assert any("negative prices" in issue.lower() for issue in report.issues)
    
    def test_quality_score_calculation(self, validator, sample_data):
        """Test quality score calculation."""
        #Perfect data should have high score
        report = validator.validate_dataset(sample_data, "test_dataset")
        assert report.quality_score > 80
        
        #Add issues to lower score
        sample_data.loc[0, 'Price'] = np.nan
        sample_data.loc[1, 'Price'] = -100
        sample_data = pd.concat([sample_data, sample_data.iloc[0].to_frame().T], ignore_index=True)
        
        report_with_issues = validator.validate_dataset(sample_data, "test_dataset")
        assert report_with_issues.quality_score < report.quality_score

class TestDataProfiler:
    """Test cases for DataProfiler."""
    
    @pytest.fixture
    def profiler(self):
        """Create profiler instance."""
        return DataProfiler()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data with mixed types."""
        return pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'price': [100, 200, 300, 400, 500]
        })
    
    def test_profile_dataset(self, profiler, sample_data):
        """Test dataset profiling."""
        profile = profiler.profile_dataset(sample_data)
        
        assert 'basic_info' in profile
        assert 'numerical_summary' in profile
        assert 'categorical_summary' in profile
        assert 'correlation_analysis' in profile
        assert 'distribution_analysis' in profile
    
    def test_basic_info(self, profiler, sample_data):
        """Test basic info extraction."""
        profile = profiler.profile_dataset(sample_data)
        basic_info = profile['basic_info']
        
        assert basic_info['shape'] == (5, 3)
        assert 'memory_usage' in basic_info
        assert 'columns' in basic_info
        assert 'dtypes' in basic_info
    
    def test_numerical_summary(self, profiler, sample_data):
        """Test numerical summary."""
        profile = profiler.profile_dataset(sample_data)
        numerical_summary = profile['numerical_summary']
        
        assert 'numeric_col' in numerical_summary
        assert 'price' in numerical_summary
        assert 'count' in numerical_summary['numeric_col']
        assert 'mean' in numerical_summary['numeric_col']
    
    def test_categorical_summary(self, profiler, sample_data):
        """Test categorical summary."""
        profile = profiler.profile_dataset(sample_data)
        categorical_summary = profile['categorical_summary']
        
        assert 'categorical_col' in categorical_summary
        assert 'unique_count' in categorical_summary['categorical_col']
        assert 'most_frequent' in categorical_summary['categorical_col']
        assert 'frequency' in categorical_summary['categorical_col']

class TestDataQualityReport:
    """Test cases for DataQualityReport."""
    
    def test_report_creation(self):
        """Test report creation."""
        report = DataQualityReport(
            timestamp=datetime.now(),
            dataset_name="test",
            total_rows=100,
            total_columns=10,
            missing_data={'col1': 5.0},
            data_types={'col1': 'int64'},
            duplicates=2,
            outliers={'col1': 3},
            quality_score=85.5,
            issues=['Issue 1'],
            recommendations=['Rec 1']
        )
        
        assert report.dataset_name == "test"
        assert report.total_rows == 100
        assert report.quality_score == 85.5
        assert len(report.issues) == 1
        assert len(report.recommendations) == 1
