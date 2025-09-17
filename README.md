Airplane Price Prediction

### Core Components

1. **Data Extraction** (`DataExtractor`)
   - Multi-format data loading (Excel, CSV, JSON)
   - Source data validation
   - Error handling and logging

2. **Data Transformation** (`DataTransformer`)
   - Feature engineering
   - Categorical encoding
   - Temporal feature extraction
   - Cyclical encoding for time features

3. **Data Quality Validation** (`DataQualityValidator`)
   - Missing data analysis
   - Outlier detection
   - Business rule validation
   - Data quality scoring

4. **Data Profiling** (`DataProfiler`)
   - Statistical analysis
   - Distribution analysis
   - Correlation analysis
   - Normality testing

5. **Data Loading** (`DataLoader`)
   - Multi-format data persistence
   - Metadata tracking
   - Artifact management

## 🚀 Quick Start

### Installation

```bash
#Install dependencies
pip install -r requirements.txt

#Create necessary directories
mkdir -p data/processed data/output logs
```

### Running the Pipeline

```bash
#Process both training and test datasets
python run_pipeline.py

#Process only training data
python run_pipeline.py --dataset train

#Skip data quality checks
python run_pipeline.py --skip-quality-checks

#Skip feature engineering
python run_pipeline.py --skip-feature-engineering
```

### Demo

```bash
#Run the demo script
python examples/pipeline_demo.py
```

## 📊 Data Quality Framework

### Quality Metrics

- **Missing Data Analysis**: Tracks missing values per column
- **Outlier Detection**: Identifies statistical outliers
- **Business Rule Validation**: Validates domain-specific rules
- **Data Type Validation**: Ensures correct data types
- **Duplicate Detection**: Identifies duplicate records

### Quality Scoring

The pipeline generates a quality score (0-100) based on:
- Missing data percentage
- Duplicate records
- Outlier frequency
- Business rule violations

### Quality Reports

Automated quality reports include:
- Data quality score
- Issues found
- Recommendations
- Statistical summaries

## 🔧 Configuration

### Data Configuration (`DataConfig`)

```python
@dataclass
class DataConfig:
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
```

### Pipeline Configuration (`PipelineConfig`)

```python
@dataclass
class PipelineConfig:
    enable_data_quality_checks: bool = True
    enable_feature_engineering: bool = True
    enable_model_training: bool = True
    enable_model_evaluation: bool = True
    save_artifacts: bool = True
```

## 🔍 Data Quality Features

### Automated Validation

- **Missing Data**: Configurable thresholds for missing values
- **Outliers**: Statistical outlier detection using IQR method
- **Business Rules**: Domain-specific validation rules
- **Data Types**: Automatic data type validation
- **Duplicates**: Duplicate record detection

### Monitoring & Alerting

- **Quality Scoring**: Automated quality assessment
- **Issue Tracking**: Detailed issue identification
- **Recommendations**: Actionable improvement suggestions
- **Logging**: Comprehensive pipeline logging

### Data Profiling

- **Statistical Analysis**: Descriptive statistics
- **Distribution Analysis**: Normality testing
- **Correlation Analysis**: Feature correlation matrices
- **Categorical Analysis**: Frequency analysis

## 🛠️ Usage Examples

### Basic Pipeline Execution

```python
from src.config import get_config
from src.data_pipeline import DataPipeline

#Get configuration
data_config, model_config, pipeline_config = get_config()

#Initialize pipeline
pipeline = DataPipeline(data_config, pipeline_config)

#Run pipeline
train_data, metadata = pipeline.run_pipeline("train")
```

### Data Quality Validation

```python
from src.data_quality import DataQualityValidator

#Initialize validator
validator = DataQualityValidator(data_config)

#Validate dataset
report = validator.validate_dataset(df, "my_dataset")

#Generate report
report_text = validator.generate_quality_report(report)
print(report_text)
```

### Data Profiling

```python
from src.data_quality import DataProfiler

#Initialize profiler
profiler = DataProfiler()

#Profile dataset
profile = profiler.profile_dataset(df)

#Access profile components
print(profile['basic_info'])
print(profile['numerical_summary'])
print(profile['categorical_summary'])
```

## 📈 Output Files

### Processed Data
- `data/processed/train_processed.parquet` - Training data
- `data/processed/test_processed.parquet` - Test data
- `data/processed/*_metadata.json` - Data metadata

### Quality Reports
- `logs/quality_report_*.txt` - Detailed quality reports
- `logs/pipeline_*.log` - Pipeline execution logs

### Artifacts
- `data/output/transformation_artifacts.json` - Feature mappings
- `data/output/pipeline_metadata.json` - Pipeline metadata

## 🔧 Customization

### Adding New Quality Checks

```python
def _validate_custom_rules(self, df: pd.DataFrame):
    """Add custom business rules."""
    #Your custom validation logic here
    pass
```

### Adding New Transformations

```python
def custom_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add custom feature engineering."""
    #Your transformation logic here
    return df
```

## 📊 Monitoring

The pipeline provides comprehensive monitoring through:

- **Quality Metrics**: Automated quality scoring
- **Performance Metrics**: Processing time tracking
- **Error Tracking**: Detailed error logging
- **Data Lineage**: Track data transformations

