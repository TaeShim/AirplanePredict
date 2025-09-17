"""
Complete demonstration of the data engineering pipeline.
This script shows the full capabilities of the data pipeline architecture.
"""
import sys
import logging
from pathlib import Path
import pandas as pd

#Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, create_directories
from src.data_pipeline import DataPipeline
from src.data_quality import DataQualityValidator, DataProfiler

def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def demonstrate_data_quality():
    """Demonstrate data quality validation capabilities."""
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION DEMONSTRATION")
    print("="*60)
    
    #Get configuration
    data_config, model_config, pipeline_config = get_config()
    
    #Create sample data with various quality issues
    sample_data = pd.DataFrame({
        'Price': [1000, 2000, -100, 5000, None],  # Negative price and missing value
        'Airline': ['Air India', 'IndiGo', 'SpiceJet', 'Air India', 'IndiGo'],
        'Source': ['Delhi', 'Mumbai', 'Bangalore', 'Delhi', 'Mumbai'],
        'Destination': ['Mumbai', 'Delhi', 'Delhi', 'Bangalore', 'Bangalore'],
        'Total_Stops': ['non-stop', '1 stop', '2 stops', 'non-stop', '1 stop'],
        'Date_of_Journey': ['15/01/2019', '16/01/2019', '17/01/2019', '18/01/2019', '19/01/2019'],
        'Duration': ['2h 30m', '1h 45m', '3h 15m', '2h 0m', '1h 30m'],
        'Dep_Time': ['10:30', '14:15', '08:45', '12:00', '16:30'],
        'Arrival_Time': ['13:00', '16:00', '12:00', '14:00', '18:00']
    })
    
    #Add duplicate row
    sample_data = pd.concat([sample_data, sample_data.iloc[0].to_frame().T], ignore_index=True)
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    #Initialize validator
    validator = DataQualityValidator(data_config)
    
    #Validate data
    report = validator.validate_dataset(sample_data, "demo_dataset")
    
    #Generate and display report
    report_text = validator.generate_quality_report(report)
    print("\n" + report_text)
    
    return report

def demonstrate_data_profiling():
    """Demonstrate data profiling capabilities."""
    print("\n" + "="*60)
    print("DATA PROFILING DEMONSTRATION")
    print("="*60)
    
    #Create sample data
    sample_data = pd.DataFrame({
        'Price': [1000, 2000, 3000, 4000, 5000, 1500, 2500, 3500, 4500, 5500],
        'Airline': ['Air India', 'IndiGo', 'SpiceJet', 'Air India', 'IndiGo', 
                   'SpiceJet', 'Air India', 'IndiGo', 'SpiceJet', 'Air India'],
        'Duration_Minutes': [150, 105, 195, 120, 90, 135, 165, 180, 210, 75],
        'Total_Stops': [0, 1, 2, 0, 1, 0, 2, 1, 2, 0]
    })
    
    #Initialize profiler
    profiler = DataProfiler()
    
    #Profile data
    profile = profiler.profile_dataset(sample_data)
    
    print("Basic Information:")
    print(f"  Shape: {profile['basic_info']['shape']}")
    print(f"  Memory Usage: {profile['basic_info']['memory_usage']} bytes")
    print(f"  Columns: {profile['basic_info']['columns']}")
    
    print("\nNumerical Summary:")
    for col, stats in profile['numerical_summary'].items():
        print(f"  {col}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Std: {stats['std']:.2f}")
        print(f"    Min: {stats['min']:.2f}")
        print(f"    Max: {stats['max']:.2f}")
    
    print("\nCategorical Summary:")
    for col, stats in profile['categorical_summary'].items():
        print(f"  {col}:")
        print(f"    Unique Count: {stats['unique_count']}")
        print(f"    Most Frequent: {stats['most_frequent']}")
        print(f"    Top Values: {list(stats['frequency'].keys())[:3]}")
    
    return profile

def demonstrate_pipeline_execution():
    """Demonstrate full pipeline execution."""
    print("\n" + "="*60)
    print("FULL PIPELINE EXECUTION DEMONSTRATION")
    print("="*60)
    
    #Get configuration
    data_config, model_config, pipeline_config = get_config()
    
    #Create directories
    create_directories(data_config)
    
    #Initialize pipeline
    pipeline = DataPipeline(data_config, pipeline_config)
    
    print("Pipeline Configuration:")
    print(f"  Data Quality Checks: {pipeline_config.enable_data_quality_checks}")
    print(f"  Feature Engineering: {pipeline_config.enable_feature_engineering}")
    print(f"  Save Artifacts: {pipeline_config.save_artifacts}")
    
    print("\nPipeline Components:")
    summary = pipeline.get_pipeline_summary()
    for component in summary['components']:
        print(f"  - {component}")
    
    #Note: In a real scenario, this would process actual data files
    print("\nNote: This demo shows the pipeline structure.")
    print("To run with actual data, ensure your data files are in the correct location:")
    print(f"  Training data: {data_config.raw_data_path}")
    print(f"  Test data: {data_config.test_data_path}")
    
    return pipeline

def main():
    """Main demonstration function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 AIRPLANE PRICE PREDICTION - DATA ENGINEERING PIPELINE DEMO")
    print("="*80)
    
    try:
        #Demonstrate data quality validation
        quality_report = demonstrate_data_quality()
        
        #Demonstrate data profiling
        data_profile = demonstrate_data_profiling()
        
        #Demonstrate pipeline execution
        pipeline = demonstrate_pipeline_execution()
        
        print("\n" + "="*80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nKey Features Demonstrated:")
        print("✅ Modular Data Pipeline Architecture")
        print("✅ Comprehensive Data Quality Validation")
        print("✅ Advanced Data Profiling")
        print("✅ Configurable Pipeline Components")
        print("✅ Automated Quality Scoring")
        print("✅ Business Rule Validation")
        print("✅ Statistical Analysis")
        print("✅ Error Handling and Logging")
        
        print(f"\nData Quality Score: {quality_report.quality_score:.1f}/100")
        print(f"Issues Found: {len(quality_report.issues)}")
        print(f"Recommendations: {len(quality_report.recommendations)}")
        
        print("\nNext Steps:")
        print("1. Run 'python run_pipeline.py' to process your actual data")
        print("2. Check the 'logs/' directory for detailed quality reports")
        print("3. Review processed data in 'data/processed/' directory")
        print("4. Examine pipeline artifacts in 'data/output/' directory")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
