#!/usr/bin/env python3
"""
Luigi Query System for Film Data Analysis

This module implements a flexible query system for analyzing film data using Luigi.
It provides various query types including genre analysis, director analysis, 
year range queries, and custom query functionality.

Key Components:
- GenreAnalysisQuery: Analyzes films by genre
- DirectorAnalysisQuery: Analyzes films by director  
- YearRangeQuery: Queries films within year ranges
- CustomQueryTask: Flexible querying with custom conditions
- QueryPipeline: Main wrapper task that orchestrates all query types

Author: Data Engineering Team
Date: 2024
"""
import luigi
import pandas as pd
import json
import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BaseQueryTask(luigi.Task):
    """Base class for query tasks"""
    
    def requires(self):
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path="resources/csv/allFilms.csv", schema_path="resources/json/allFilesSchema.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)

class GenreAnalysisQuery(BaseQueryTask):
    """
    Query Task: Genre-Based Film Analysis
    
    This task analyzes films by genre, providing statistics such as:
    - Number of films per genre
    - Average ratings by genre
    - Genre distribution analysis
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        genre (str, optional): Specific genre to filter by
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    genre = luigi.Parameter(default="", description="Specific genre to analyze")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The task that provides the main dataset
        """
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the genre analysis results file
        """
        genre_suffix = f"_{self.genre}" if self.genre else ""
        return luigi.LocalTarget(f"output/queries/genre_analysis{genre_suffix}.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)
    
    def run(self):
        """
        Execute the genre analysis logic.
        
        This method:
        1. Loads the main dataset
        2. Filters by genre if specified
        3. Calculates genre statistics
        4. Saves analysis results
        """
        logger.info(f"🎭 Analyzing genre: {self.genre if self.genre else 'All'}")
        
        # Load the main dataset
        df = self._load_data()
        
        # Filter by specific genre if provided
        if self.genre:
            df = df[df['genres'].str.contains(self.genre, na=False)]
            logger.info(f"📊 Filtered to {len(df)} films in genre '{self.genre}'")
        
        # Extract all genres and count films per genre
        genre_counts = {}
        
        for _, row in df.iterrows():
            if pd.notna(row['genres']):
                # Split comma-separated genres
                genres = [g.strip() for g in str(row['genres']).split(',')]
                
                for genre in genres:
                    if genre:
                        # Count films per genre
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Prepare analysis results
        analysis_results = {
            'total_films': len(df),
            'genre_distribution': genre_counts,
            'top_genres': sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save analysis results
        with open(self.output().path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"✅ Genre analysis saved: {len(df)} films")

class DirectorAnalysisQuery(BaseQueryTask):
    """
    Query Task: Director-Based Film Analysis
    
    This task analyzes films by director, providing statistics such as:
    - Number of films per director
    - Average ratings by director
    - Director filmography analysis
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        director (str, optional): Specific director to filter by
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    director = luigi.Parameter(default="", description="Specific director to analyze")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The task that provides the main dataset
        """
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the director analysis results file
        """
        director_suffix = f"_{self.director}" if self.director else ""
        return luigi.LocalTarget(f"output/queries/director_analysis{director_suffix}.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)
    
    def run(self):
        """
        Execute the director analysis logic.
        
        This method:
        1. Loads the main dataset
        2. Filters by director if specified
        3. Calculates director statistics
        4. Saves analysis results
        """
        logger.info(f"🎬 Analyzing director: {self.director if self.director else 'All'}")
        
        # Load the main dataset
        df = self._load_data()
        
        # Filter by specific director if provided
        if self.director:
            df = df[df['director'].str.contains(self.director, na=False)]
            logger.info(f"📊 Filtered to {len(df)} films by director '{self.director}'")
        
        # Group by director and calculate statistics
        director_stats = df.groupby('director').agg({
            'id': 'count',  # Number of films
            'release_year': ['min', 'max']  # Year range
        }).round(2)
        
        # Flatten column names
        director_stats.columns = ['film_count', 'earliest_year', 'latest_year']
        
        # Convert to dictionary for JSON serialization
        director_analysis = {
            'total_directors': len(director_stats),
            'total_films': len(df),
            'director_statistics': director_stats.to_dict('index'),
            'top_directors_by_film_count': director_stats.nlargest(10, 'film_count').to_dict('index'),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save analysis results
        with open(self.output().path, 'w') as f:
            json.dump(director_analysis, f, indent=2, default=str)
        
        logger.info(f"✅ Director analysis saved: {len(df)} films")

class YearRangeQuery(BaseQueryTask):
    """
    Query Task: Year Range Film Analysis
    
    This task analyzes films within specified year ranges, providing:
    - Films released in specific time periods
    - Year-based statistics and trends
    - Temporal analysis of film data
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        start_year (int): Start year for the range
        end_year (int): End year for the range
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    start_year = luigi.IntParameter(default=1900, description="Start year for range")
    end_year = luigi.IntParameter(default=2030, description="End year for range")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The task that provides the main dataset
        """
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the year range analysis results file
        """
        return luigi.LocalTarget(f"output/queries/year_analysis_{self.start_year}_{self.end_year}.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)
    
    def run(self):
        """
        Execute the year range analysis logic.
        
        This method:
        1. Loads the main dataset
        2. Filters by year range
        3. Calculates year-based statistics
        4. Saves analysis results
        """
        logger.info(f"📅 Analyzing years {self.start_year} to {self.end_year}")
        
        # Load the main dataset
        df = self._load_data()
        
        # Handle release_year column - it might be datetime or integer
        if df['release_year'].dtype == 'object':
            # Try to convert to datetime first
            try:
                df['release_year'] = pd.to_datetime(df['release_year'], errors='coerce')
                df['year'] = df['release_year'].dt.year
            except:
                # If datetime conversion fails, try to extract year from string
                df['year'] = pd.to_numeric(df['release_year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
        elif pd.api.types.is_datetime64_any_dtype(df['release_year']):
            # Already datetime
            df['year'] = df['release_year'].dt.year
        else:
            # Assume it's already numeric year
            df['year'] = df['release_year']
        
        # Filter by year range
        year_filter = (df['year'] >= self.start_year) & (df['year'] <= self.end_year)
        filtered_df = df[year_filter].copy()
        
        logger.info(f"📊 Found {len(filtered_df)} films in year range {self.start_year}-{self.end_year}")
        
        # Calculate year-based statistics
        year_stats = filtered_df.groupby('year').agg({
            'id': 'count',  # Number of films per year
        }).rename(columns={'id': 'film_count'})
        
        # Genre distribution in the time period
        genre_counts = {}
        for _, row in filtered_df.iterrows():
            if pd.notna(row['genres']):
                genres = [g.strip() for g in str(row['genres']).split(',')]
                for genre in genres:
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Prepare analysis results
        analysis_results = {
            'year_range': {'start': self.start_year, 'end': self.end_year},
            'total_films': len(filtered_df),
            'films_per_year': year_stats.to_dict('index'),
            'genre_distribution': genre_counts,
            'top_genres': sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save analysis results
        with open(self.output().path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"✅ Year range analysis saved: {len(filtered_df)} films")

class CustomQueryTask(BaseQueryTask):
    """
    Query Task: Custom Flexible Querying
    
    This task provides flexible querying capabilities with custom conditions.
    Users can specify any number of columns with various operators to filter films.
    
    Supported operators:
    - equals: Exact match
    - in: Value in list
    - between: Value between range
    - greater_than: Value greater than
    - less_than: Value less than
    - contains: String contains
    - not_contains: String does not contain
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        query_conditions (List[Dict]): List of query conditions
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    query_conditions = luigi.ListParameter(default=[], description="List of query conditions")
    
    def requires(self):
        """
        Define task dependencies.
        
        Returns:
            CreateMainDataset: The task that provides the main dataset
        """
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the custom query results file
        """
        return luigi.LocalTarget("output/queries/custom_query_results.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)
    
    def run(self):
        """
        Execute the custom query logic.
        
        This method:
        1. Loads the main dataset
        2. Applies custom query conditions
        3. Filters the dataset accordingly
        4. Saves query results
        """
        logger.info("🔍 Running custom query...")
        
        # Load the main dataset
        df = self._load_data()
        
        # Apply query conditions
        filtered_df = df.copy()
        
        for condition in self.query_conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if column not in df.columns:
                logger.warning(f"⚠️ Column '{column}' not found in dataset, skipping condition")
                continue
            
            # Apply filter based on operator
            if operator == 'equals':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == 'in':
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif operator == 'between':
                filtered_df = filtered_df[(filtered_df[column] >= value[0]) & (filtered_df[column] <= value[1])]
            elif operator == 'greater_than':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == 'less_than':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == 'contains':
                filtered_df = filtered_df[filtered_df[column].str.contains(str(value), na=False)]
            elif operator == 'not_contains':
                filtered_df = filtered_df[~filtered_df[column].str.contains(str(value), na=False)]
            else:
                logger.warning(f"⚠️ Unknown operator '{operator}', skipping condition")
        
        # Prepare query results
        query_results = {
            'query_conditions': self.query_conditions,
            'total_films_found': len(filtered_df),
            'films': filtered_df.to_dict('records'),
            'query_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save query results
        with open(self.output().path, 'w') as f:
            json.dump(query_results, f, indent=2, default=str)
        
        logger.info(f"✅ Custom query completed: {len(filtered_df)} films found")

class QueryPipeline(luigi.WrapperTask):
    """
    Main Query Pipeline Wrapper Task
    
    This is the main entry point for all query operations. It orchestrates
    different types of queries and ensures all dependencies are satisfied.
    
    Parameters:
        csv_path (str): Path to the input CSV file
        schema_path (str): Path to the JSON schema file
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define all query task dependencies.
        
        Returns:
            list: List of all required query tasks
        """
        return [
            # Genre analysis queries
            GenreAnalysisQuery(csv_path=self.csv_path, schema_path=self.schema_path),
            GenreAnalysisQuery(csv_path=self.csv_path, schema_path=self.schema_path, genre="Action"),
            GenreAnalysisQuery(csv_path=self.csv_path, schema_path=self.schema_path, genre="Drama"),
            
            # Director analysis queries
            DirectorAnalysisQuery(csv_path=self.csv_path, schema_path=self.schema_path),
            DirectorAnalysisQuery(csv_path=self.csv_path, schema_path=self.schema_path, director="Christopher Nolan"),
            
            # Year range queries
            YearRangeQuery(csv_path=self.csv_path, schema_path=self.schema_path, start_year=2000, end_year=2010),
            YearRangeQuery(csv_path=self.csv_path, schema_path=self.schema_path, start_year=1990, end_year=2000),
            
            # Custom query example
            CustomQueryTask(
                csv_path=self.csv_path, 
                schema_path=self.schema_path,
                query_conditions=[
                    {"column": "rating", "operator": "greater_than", "value": 8.0},
                    {"column": "release_year", "operator": "between", "value": [2000, 2010]}
                ]
            )
        ]
    
    def run(self):
        """
        Execute the complete query pipeline.
        
        This method is called after all dependencies are satisfied.
        It provides a summary of the query execution.
        """
        logger.info("🔍 Query Pipeline completed successfully!")
        logger.info("📊 All query tasks finished")
        logger.info("🎭 Genre analysis completed")
        logger.info("🎬 Director analysis completed")
        logger.info("📅 Year range analysis completed")
        logger.info("🔍 Custom queries completed")

def main():
    """Run the query pipeline"""
    luigi.run([
        'QueryPipeline',
        '--local-scheduler'
    ])

if __name__ == "__main__":
    main() 