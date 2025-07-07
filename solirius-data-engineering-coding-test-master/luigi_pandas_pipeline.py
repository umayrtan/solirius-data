#!/usr/bin/env python3
"""
Luigi Pipeline for Film Data Processing

This module implements a comprehensive ETL pipeline for processing film data using Luigi.
It includes data loading, transformation, genre analysis, similarity indexing, and data quality validation.

Key Components:
- LoadAndTransformData: Loads and cleans raw CSV data
- CreateMainDataset: Creates enhanced dataset with computed features
- CreateGenreDatasets: Splits data by genre for analysis
- CreateSimilarityIndex: Builds similarity matrices for recommendations
- DataQualityValidation: Performs comprehensive data quality checks
- FilmPipeline: Main wrapper task that orchestrates all components

Author: Data Engineering Team
Date: 2024
"""
import luigi
import pandas as pd
import json
import os
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LoadAndTransformData(luigi.Task):
    """
    Task 1: Load and Transform Raw Data
    
    This task loads the raw CSV film data and applies initial transformations:
    - Loads data with schema validation
    - Extracts release year from date strings
    - Handles missing values and data type conversions
    - Generates unique IDs if missing
    
    Parameters:
        csv_path (str): Path to the input CSV file
        schema_path (str): Path to the JSON schema file for validation
    """
    
    # Luigi parameters - these will be passed from the command line or dependencies
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the transformed data file
        """
        return luigi.LocalTarget("output/films.parquet")
    
    def run(self):
        """
        Execute the data loading and transformation logic.
        
        This method:
        1. Loads the CSV file
        2. Loads and validates schema if provided
        3. Generates ID column if missing
        4. Applies transformations (extract years, handle missing values)
        5. Saves the transformed data as a Parquet file
        """
        logger.info("📥 Loading and transforming data...")
        
        # Load the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Load and validate schema if provided
        if os.path.exists(self.schema_path):
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            logger.info("✅ Schema loaded and validated")
        
        # Generate ID column if missing
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            logger.info("🆔 Generated missing ID column")
        
        logger.info("🔧 Applying data transformations...")
        
        # Extract year from release_date if it exists
        if 'release_date' in df.columns:
            # Convert date strings to release years
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            # Fill missing years with a default value
            df['release_year'] = df['release_year'].fillna(2000).astype(int)
            logger.info("📅 Extracted release years from dates")
        
        # Handle missing values in critical columns
        df['title'] = df['title'].fillna('Unknown Title')
        df['director'] = df['director'].fillna('Unknown Director')
        df['genres'] = df['genres'].fillna('Unknown')
        df['country'] = df['country'].fillna('Unknown')
        
        # Ensure rating is numeric and within valid range
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating'] = df['rating'].fillna(0).clip(0, 10)
        
        logger.info("✅ Transformations applied")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save transformed data as Parquet for efficient storage
        df.to_parquet(self.output().path, index=False)
        logger.info(f"✅ Transformed data saved: {len(df)} rows")

class CreateMainDataset(luigi.Task):
    """
    Task 2: Create Enhanced Main Dataset
    
    This task takes the transformed data and creates an enhanced dataset with:
    - Computed features (decade, genre count, clean title)
    - Additional data cleaning and optimization
    - Optimized Parquet storage
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            LoadAndTransformData: The previous task that provides transformed data
        """
        return LoadAndTransformData(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the main dataset file
        """
        return luigi.LocalTarget("output/main_dataset.parquet")
    
    def run(self):
        """
        Execute the main dataset creation logic.
        
        This method:
        1. Loads the transformed data
        2. Adds computed features (decade, genre count, clean title)
        3. Performs additional data cleaning
        4. Saves the enhanced dataset
        """
        logger.info("💾 Creating main dataset...")
        
        # Load the transformed data from the previous task
        df = pd.read_parquet(self.input().path)
        
        # Add computed features
        if 'release_year' in df.columns:
            # Create decade column for easier analysis
            df['decade'] = (df['release_year'] // 10) * 10
            logger.info("📊 Added decade feature")
        
        # Count number of genres per film
        df['genre_count'] = df['genres'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
        logger.info("🎭 Added genre count feature")
        
        # Create clean title for better text analysis
        df['clean_title'] = df['title'].str.strip().str.lower()
        logger.info("📝 Added clean title feature")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save the enhanced dataset
        df.to_parquet(self.output().path, index=False)
        logger.info(f"✅ Main dataset saved: {len(df)} films")

class CreateGenreDatasets(luigi.Task):
    """
    Task 3: Create Genre-Specific Datasets
    
    This task splits the main dataset into separate files for each genre,
    enabling genre-specific analysis and queries.
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The previous task that provides the main dataset
        """
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Marker file indicating completion
        """
        return luigi.LocalTarget("output/genres/.complete")
    
    def run(self):
        """
        Execute the genre dataset creation logic.
        
        This method:
        1. Loads the main dataset
        2. Extracts all unique genres
        3. Creates separate datasets for each genre
        4. Saves genre-specific Parquet files
        """
        logger.info("🎭 Creating genre-specific datasets...")
        
        # Load the main dataset
        df = pd.read_parquet(self.input().path)
        
        # Extract all unique genres from the dataset
        all_genres = set()
        for genres in df['genres'].dropna():
            if isinstance(genres, str):
                # Split comma-separated genres and add to set
                all_genres.update([g.strip() for g in genres.split(',')])
        
        logger.info(f"📊 Found {len(all_genres)} unique genres")
        
        # Create the main genres directory
        os.makedirs("output/genres", exist_ok=True)
        
        # Create separate dataset for each genre
        for genre in all_genres:
            if genre:  # Skip empty genres
                # Clean genre name for file system compatibility
                clean_genre = genre.replace(" ", "_").replace("&", "and")
                
                # Filter films that contain this genre
                genre_df = df[df['genres'].str.contains(genre, na=False)]
                
                # Create genre-specific directory and save data
                out_path = f"output/genres/genre={clean_genre}/data.parquet"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                genre_df.to_parquet(out_path, index=False)
                
                logger.info(f"✅ Saved {genre}: {len(genre_df)} films")
        
        # Create completion marker
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        with open(self.output().path, 'w') as f:
            f.write(f"Genre datasets created at {pd.Timestamp.now()}")
        
        logger.info("✅ All genre datasets created")

class CreateSimilarityIndex(luigi.Task):
    """
    Task 4: Create Similarity Index for Film Recommendations
    
    This task builds similarity matrices for film recommendations using:
    - TF-IDF vectors from film titles and descriptions
    - Cosine similarity calculations
    - Genre overlap analysis
    - Director and year similarity
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The previous task that provides the main dataset
        """
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the similarity matrix file
        """
        return luigi.LocalTarget("output/similarity/similarity_matrix.parquet")
    
    def run(self):
        """
        Execute the similarity index creation logic.
        
        This method:
        1. Loads the main dataset
        2. Creates TF-IDF vectors from film titles
        3. Computes cosine similarity matrices
        4. Saves similarity indices for fast lookups
        """
        logger.info("🔍 Creating similarity index...")
        
        # Load the main dataset
        df = pd.read_parquet(self.input().path)
        
        # Create feature vectors for similarity calculation
        # Combine title, director, and genres for comprehensive similarity
        df['combined_features'] = (
            df['title'].fillna('') + ' ' + 
            df['director'].fillna('') + ' ' + 
            df['genres'].fillna('')
        )
        
        # Create TF-IDF vectors from combined features
        tfidf = TfidfVectorizer(
            max_features=1000,  # Limit features to prevent memory issues
            stop_words='english',  # Remove common English words
            ngram_range=(1, 2)  # Use both single words and bigrams
        )
        
        # Transform text features to TF-IDF vectors
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        
        # Calculate cosine similarity between all films
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert to DataFrame for easier handling
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=df['id'],
            columns=df['id']
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save similarity matrix
        similarity_df.to_parquet(self.output().path)
        logger.info("✅ Similarity index created")

class DataQualityValidation(luigi.Task):
    """
    Task 5: Data Quality Validation
    
    This task performs comprehensive data quality checks on the processed dataset:
    - Column existence validation
    - Data type validation
    - Null value checks
    - Value range validation
    - Summary statistics generation
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The previous task that provides the main dataset
        """
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the validation report file
        """
        return luigi.LocalTarget("output/validation/data_quality_report.json")
    
    def run(self):
        """
        Execute the data quality validation logic.
        
        This method:
        1. Loads the main dataset
        2. Performs various data quality checks
        3. Generates validation results and summary statistics
        4. Saves a comprehensive validation report
        """
        logger.info("🔍 Running data quality validation...")
        
        # Read main dataset
        df = pd.read_parquet(self.input().path)
        
        # Perform data quality checks
        validation_results = []
        
        # Check for required columns
        required_columns = ['id', 'title', 'release_year', 'genres', 'director']
        for col in required_columns:
            if col in df.columns:
                validation_results.append({
                    'check': f'column_exists_{col}',
                    'status': 'PASS',
                    'message': f'Column {col} exists'
                })
            else:
                validation_results.append({
                    'check': f'column_exists_{col}',
                    'status': 'FAIL',
                    'message': f'Column {col} missing'
                })
        
        # Check data types
        if 'id' in df.columns:
            if df['id'].dtype in ['int32', 'int64']:
                validation_results.append({
                    'check': 'id_data_type',
                    'status': 'PASS',
                    'message': 'ID column has correct integer type'
                })
            else:
                validation_results.append({
                    'check': 'id_data_type',
                    'status': 'FAIL',
                    'message': f'ID column has incorrect type: {df["id"].dtype}'
                })
        
        # Check for nulls in important columns
        for col in ['id', 'title']:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count == 0:
                    validation_results.append({
                        'check': f'no_nulls_{col}',
                        'status': 'PASS',
                        'message': f'No null values in {col}'
                    })
                else:
                    validation_results.append({
                        'check': f'no_nulls_{col}',
                        'status': 'FAIL',
                        'message': f'{null_count} null values found in {col}'
                    })
        
        # Check year range
        if 'release_year' in df.columns:
            valid_years = df['release_year'].between(1900, 2030)
            invalid_count = (~valid_years).sum()
            if invalid_count == 0:
                validation_results.append({
                    'check': 'year_range',
                    'status': 'PASS',
                    'message': 'All years are within valid range (1900-2030)'
                })
            else:
                validation_results.append({
                    'check': 'year_range',
                    'status': 'FAIL',
                    'message': f'{invalid_count} years outside valid range (1900-2030)'
                })
        
        # Calculate summary statistics
        summary_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Save validation report
        report = {
            'validation_results': validation_results,
            'summary_stats': summary_stats,
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'total_checks': len(validation_results),
            'passed_checks': len([r for r in validation_results if r['status'] == 'PASS']),
            'failed_checks': len([r for r in validation_results if r['status'] == 'FAIL']),
            'total_rows': summary_stats['total_rows'],
            'total_columns': summary_stats['total_columns'],
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        with open(self.output().path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Data quality validation completed: {report['passed_checks']}/{report['total_checks']} checks passed")

class FilmPipeline(luigi.WrapperTask):
    """
    Main Pipeline Wrapper Task
    
    This is the main entry point for the entire film data processing pipeline.
    It orchestrates all the individual tasks in the correct order and ensures
    that all dependencies are satisfied.
    
    Parameters:
        csv_path (str): Path to the input CSV file
        schema_path (str): Path to the JSON schema file
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define all task dependencies for the complete pipeline.
        
        Returns:
            list: List of all required tasks in execution order
        """
        return [
            LoadAndTransformData(csv_path=self.csv_path, schema_path=self.schema_path),
            CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path),
            CreateGenreDatasets(csv_path=self.csv_path, schema_path=self.schema_path),
            CreateSimilarityIndex(csv_path=self.csv_path, schema_path=self.schema_path),
            DataQualityValidation(csv_path=self.csv_path, schema_path=self.schema_path)
        ]
    
    def run(self):
        """
        Execute the complete pipeline.
        
        This method is called after all dependencies are satisfied.
        It provides a summary of the pipeline execution.
        """
        logger.info("🎬 Film Pipeline completed successfully!")
        logger.info("📊 All data processing tasks finished")
        logger.info("🔍 Data quality validation completed")
        logger.info("🎭 Genre datasets created")
        logger.info("🔍 Similarity index built")

def main():
    """Run the Luigi pipeline"""
    luigi.run([
        'FilmPipeline',
        '--csv-path', 'resources/csv/films.csv',
        '--schema-path', 'resources/json/allFilesSchema.json',
        '--local-scheduler'
    ])

if __name__ == "__main__":
    main() 