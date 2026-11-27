"""
Main script to create the EVSE Vulnerability Vector Database
"""

import argparse
from loguru import logger
import sys

from pipeline import VulnerabilityDBPipeline
from config import config

def setup_logging(log_file: str = "pipeline.log"):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Create EVSE Vulnerability Vector Database for RAG"
    )
    
    parser.add_argument(
        "--nvd-start-date",
        type=str,
        default="2022-01-01",
        help="Start date for NVD CVE collection (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--nvd-max-results",
        type=int,
        default=100,
        help="Maximum number of CVEs to collect from NVD"
    )
    
    parser.add_argument(
        "--cicevse-dataset",
        type=str,
        default=None,
        help="Path to CICEVSE2024 dataset CSV file"
    )
    
    parser.add_argument(
        "--skip-nvd",
        action="store_true",
        help="Skip NVD CVE collection"
    )
    
    parser.add_argument(
        "--skip-mitre",
        action="store_true",
        help="Skip MITRE ATT&CK collection"
    )
    
    parser.add_argument(
        "--skip-stride",
        action="store_true",
        help="Skip STRIDE pattern creation"
    )
    
    parser.add_argument(
        "--skip-mitre-stride",
        action="store_true",
        help="Skip MITRE-STRIDE mapping"
    )
    
    parser.add_argument(
        "--skip-cicevse",
        action="store_true",
        help="Skip CICEVSE2024 processing"
    )
    
    parser.add_argument(
        "--export-db",
        type=str,
        default=None,
        help="Export database to JSON file"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="pipeline.log",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    logger.info("Starting EVSE Vulnerability Vector Database Creation")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        pipeline = VulnerabilityDBPipeline()
        
        pipeline.run_full_pipeline(
            nvd_start_date=args.nvd_start_date,
            nvd_max_results=args.nvd_max_results,
            cicevse_dataset_path=args.cicevse_dataset,
            skip_nvd=args.skip_nvd,
            skip_mitre=args.skip_mitre,
            skip_stride=args.skip_stride,
            skip_mitre_stride=args.skip_mitre_stride,
            skip_cicevse=args.skip_cicevse
        )
        
        stats = pipeline.get_database_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 80)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        if args.export_db:
            logger.info(f"\nExporting database to {args.export_db}")
            pipeline.export_database(args.export_db)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
