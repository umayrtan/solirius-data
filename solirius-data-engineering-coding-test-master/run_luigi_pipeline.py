#!/usr/bin/env python3
"""
Luigi Pipeline Runner
Simple script to run the Luigi pandas pipeline
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_luigi_pipeline():
    """Run the complete Luigi pipeline"""
    try:
        import luigi
        from luigi_pandas_pipeline import FilmPipeline
        
        logger.info("🚀 Starting Luigi Pandas Pipeline...")
        
        # Set up paths
        csv_path = "resources/csv/allFilms.csv"
        schema_path = "resources/json/allFilesSchema.json"
        
        # Verify input files exist
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV file not found: {csv_path}")
            return False
        
        if not os.path.exists(schema_path):
            logger.error(f"❌ Schema file not found: {schema_path}")
            return False
        
        logger.info(f"📁 Input CSV: {csv_path}")
        logger.info(f"📁 Schema: {schema_path}")
        
        # Run the pipeline
        pipeline = FilmPipeline(csv_path=csv_path, schema_path=schema_path)
        
        # Build the pipeline
        success = luigi.build([pipeline], local_scheduler=True, workers=1)
        
        if success:
            logger.info("✅ Luigi pipeline completed successfully!")
            return True
        else:
            logger.error("❌ Luigi pipeline failed!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Install Luigi with: pip install luigi")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        return False

def run_luigi_queries():
    """Run Luigi query tasks"""
    try:
        import luigi
        from luigi_query_system import QueryPipeline
        
        logger.info("🔍 Running Luigi Query Pipeline...")
        
        # Set up paths
        csv_path = "resources/csv/allFilms.csv"
        schema_path = "resources/json/allFilesSchema.json"
        
        # Verify input files exist
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV file not found: {csv_path}")
            return False
        
        if not os.path.exists(schema_path):
            logger.error(f"❌ Schema file not found: {schema_path}")
            return False
        
        # Run query pipeline with parameters
        success = luigi.build([QueryPipeline(csv_path=csv_path, schema_path=schema_path)], local_scheduler=True, workers=1)
        
        if success:
            logger.info("✅ Luigi query pipeline completed successfully!")
            return True
        else:
            logger.error("❌ Luigi query pipeline failed!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Query pipeline error: {e}")
        return False

def run_luigi_similarity():
    """Run Luigi similarity tasks"""
    try:
        import luigi
        from luigi_similarity_system import SimilarityPipeline
        
        logger.info("🎯 Running Luigi Similarity Pipeline...")
        
        # Set up paths
        csv_path = "resources/csv/allFilms.csv"
        schema_path = "resources/json/allFilesSchema.json"
        
        # Verify input files exist
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV file not found: {csv_path}")
            return False
        
        if not os.path.exists(schema_path):
            logger.error(f"❌ Schema file not found: {schema_path}")
            return False
        
        # Run similarity pipeline with parameters
        success = luigi.build([SimilarityPipeline(csv_path=csv_path, schema_path=schema_path)], local_scheduler=True, workers=1)
        
        if success:
            logger.info("✅ Luigi similarity pipeline completed successfully!")
            return True
        else:
            logger.error("❌ Luigi similarity pipeline failed!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Similarity pipeline error: {e}")
        return False

def main():
    """Main function to run different Luigi pipelines"""
    if len(sys.argv) < 2:
        print("Usage: python run_luigi_pipeline.py [etl|queries|similarity|all]")
        print("  etl        - Run the main ETL pipeline")
        print("  queries    - Run query analysis tasks")
        print("  similarity - Run similarity analysis tasks")
        print("  all        - Run all pipelines")
        return
    
    command = sys.argv[1].lower()
    
    if command == "etl":
        success = run_luigi_pipeline()
    elif command == "queries":
        success = run_luigi_queries()
    elif command == "similarity":
        success = run_luigi_similarity()
    elif command == "all":
        logger.info("🔄 Running all Luigi pipelines...")
        success1 = run_luigi_pipeline()
        success2 = run_luigi_queries()
        success3 = run_luigi_similarity()
        success = success1 and success2 and success3
    else:
        logger.error(f"❌ Unknown command: {command}")
        return
    
    if success:
        logger.info("🎉 All tasks completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Some tasks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 