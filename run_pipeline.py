import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime

#Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, create_directories
from src.data_pipeline import DataPipeline

def setup_logging(data_config, pipeline_config):
    """Setup logging configuration."""
    log_format = pipeline_config.log_format
    log_level = getattr(logging, pipeline_config.log_level.upper())
    
    #Create logs directory
    Path(data_config.log_path).mkdir(parents=True, exist_ok=True)
    
    #Setup logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(f"{data_config.log_path}/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description="Airplane Price Prediction Data Pipeline")
    parser.add_argument("--dataset", choices=["train", "test", "both"], default="both",
                       help="Dataset to process (default: both)")
    parser.add_argument("--skip-quality-checks", action="store_true",
                       help="Skip data quality validation")
    parser.add_argument("--skip-feature-engineering", action="store_true",
                       help="Skip feature engineering")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save processed data")
    
    args = parser.parse_args()
    
    #Get configuration
    data_config, model_config, pipeline_config = get_config()
    
    #Override config based on arguments
    if args.skip_quality_checks:
        pipeline_config.enable_data_quality_checks = False
    if args.skip_feature_engineering:
        pipeline_config.enable_feature_engineering = False
    if args.no_save:
        pipeline_config.save_artifacts = False
    
    #Create directories
    create_directories(data_config)
    
    #Setup logging
    setup_logging(data_config, pipeline_config)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("AIRPLANE PRICE PREDICTION DATA PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Pipeline started at: {datetime.now()}")
    logger.info(f"Dataset(s) to process: {args.dataset}")
    logger.info(f"Data quality checks: {pipeline_config.enable_data_quality_checks}")
    logger.info(f"Feature engineering: {pipeline_config.enable_feature_engineering}")
    logger.info(f"Save artifacts: {pipeline_config.save_artifacts}")
    
    try:
        #Initialize pipeline
        pipeline = DataPipeline(data_config, pipeline_config)
        
        #Process datasets
        results = {}
        
        if args.dataset in ["train", "both"]:
            logger.info("Processing training dataset...")
            train_data, train_metadata = pipeline.run_pipeline("train")
            results["train"] = {
                "data": train_data,
                "metadata": train_metadata
            }
            logger.info(f"Training dataset processed: {train_data.shape}")
        
        if args.dataset in ["test", "both"]:
            logger.info("Processing test dataset...")
            test_data, test_metadata = pipeline.run_pipeline("test")
            results["test"] = {
                "data": test_data,
                "metadata": test_metadata
            }
            logger.info(f"Test dataset processed: {test_data.shape}")
        
        #Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        for dataset_name, result in results.items():
            metadata = result["metadata"]
            logger.info(f"{dataset_name.upper()} DATASET:")
            logger.info(f"  - Status: {metadata['status']}")
            logger.info(f"  - Shape: {metadata['output_shape']}")
            logger.info(f"  - Processing time: {metadata['end_time']}")
            
            if 'quality_report' in metadata:
                quality_score = metadata['quality_report']['quality_score']
                logger.info(f"  - Quality score: {quality_score:.1f}/100")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error("=" * 60)
        raise

if __name__ == "__main__":
    main()
