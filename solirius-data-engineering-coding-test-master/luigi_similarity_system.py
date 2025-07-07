#!/usr/bin/env python3
"""
Luigi Similarity System for Film Recommendations

This module implements a comprehensive similarity system for film recommendations using Luigi.
It provides content-based similarity analysis, recommendation generation, and similarity pattern analysis.

Key Components:
- FindSimilarFilms: Finds similar films based on multiple criteria
- CreateRecommendations: Generates personalized film recommendations
- AnalyzeSimilarityPatterns: Analyzes similarity patterns in the dataset
- SimilarityPipeline: Main wrapper task that orchestrates all similarity operations

Similarity Features:
- Title similarity using TF-IDF and cosine similarity
- Genre overlap using Jaccard similarity
- Director similarity (exact match)
- Year proximity (temporal similarity)
- Combined weighted similarity scores

Author: Data Engineering Team
Date: 2024
"""
import luigi
import pandas as pd
import json
import os
import logging
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import jaccard_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BaseSimilarityTask(luigi.Task):
    """Base class for similarity tasks"""
    
    def requires(self):
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path="resources/csv/allFilms.csv", schema_path="resources/json/allFilesSchema.json")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the main dataset"""
        return pd.read_parquet(self.input().path)

class FindSimilarFilms(luigi.Task):
    """
    Similarity Task: Find Similar Films
    
    This task finds films similar to a given film based on multiple criteria:
    - Title similarity (TF-IDF + cosine similarity)
    - Genre overlap (Jaccard similarity)
    - Director similarity (exact match)
    - Year proximity (temporal similarity)
    - Combined weighted similarity scores
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        film_id (int): ID of the film to find similarities for
        similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
        top_n (int): Maximum number of similar films to return
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    film_id = luigi.IntParameter(default=1, description="ID of the target film")
    similarity_threshold = luigi.FloatParameter(default=0.5, description="Minimum similarity threshold")
    top_n = luigi.IntParameter(default=10, description="Maximum number of similar films")
    
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
            luigi.LocalTarget: Path to the similarity results file
        """
        return luigi.LocalTarget(f"output/similarity/similar_films_{self.film_id}.json")
    
    def run(self):
        """
        Execute the similarity analysis logic.
        
        This method:
        1. Loads the main dataset
        2. Finds the target film
        3. Calculates similarity scores for all other films
        4. Filters and ranks similar films
        5. Saves similarity results
        """
        logger.info(f"🔍 Finding films similar to ID {self.film_id}")
        
        # Load the main dataset
        df = pd.read_parquet(self.input().path)
        
        # Find the target film
        target_film = df[df['id'] == self.film_id]
        if target_film.empty:
            logger.error(f"Film with ID {self.film_id} not found")
            return
        
        target_film = target_film.iloc[0]
        logger.info(f"🎬 Target film: {target_film['title']} ({target_film['release_year']})")
        
        # Calculate similarity scores for all other films
        similar_films = []
        
        for _, film in df.iterrows():
            if film['id'] != self.film_id:  # Skip the target film itself
                similarity_score = self._calculate_similarity(target_film, film)
                
                if similarity_score >= self.similarity_threshold:
                    similar_films.append({
                        'id': film['id'],
                        'title': film['title'],
                        'director': film['director'],
                        'genres': film['genres'],
                        'release_year': film['release_year'],
                        'rating': film.get('rating', 0),
                        'similarity_score': similarity_score
                    })
        
        # Sort by similarity score (highest first) and take top N
        similar_films.sort(key=lambda x: x['similarity_score'], reverse=True)
        similar_films = similar_films[:self.top_n]
        
        # Prepare results
        results = {
            'target_film': {
                'id': target_film['id'],
                'title': target_film['title'],
                'director': target_film['director'],
                'genres': target_film['genres'],
                'release_year': target_film['release_year']
            },
            'similarity_threshold': self.similarity_threshold,
            'top_n': self.top_n,
            'similar_films': similar_films,
            'total_similar_films': len(similar_films),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save similarity results
        with open(self.output().path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Similarity analysis completed: {len(similar_films)} similar films found")
    
    def _calculate_similarity(self, film1: pd.Series, film2: pd.Series) -> float:
        """
        Calculate similarity score between two films.
        
        This method combines multiple similarity metrics:
        - Title similarity (40% weight)
        - Genre similarity (30% weight)
        - Director similarity (20% weight)
        - Year similarity (10% weight)
        
        Args:
            film1 (pd.Series): First film data
            film2 (pd.Series): Second film data
            
        Returns:
            float: Combined similarity score (0.0 to 1.0)
        """
        # Title similarity using TF-IDF and cosine similarity
        title_similarity = self._calculate_title_similarity(film1['title'], film2['title'])
        
        # Genre similarity using Jaccard similarity
        genre_similarity = self._calculate_genre_similarity(film1['genres'], film2['genres'])
        
        # Director similarity (exact match)
        director_similarity = 1.0 if film1['director'] == film2['director'] else 0.0
        
        # Year similarity (closer years = higher similarity)
        year_similarity = self._calculate_year_similarity(
            film1.get('release_year', 2000), 
            film2.get('release_year', 2000)
        )
        
        # Combine similarities with weights
        combined_similarity = (
            title_similarity * 0.4 +
            genre_similarity * 0.3 +
            director_similarity * 0.2 +
            year_similarity * 0.1
        )
        
        return combined_similarity
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate title similarity using TF-IDF and cosine similarity.
        
        Args:
            title1 (str): First film title
            title2 (str): Second film title
            
        Returns:
            float: Title similarity score (0.0 to 1.0)
        """
        if not title1 or not title2:
            return 0.0
        
        # Create TF-IDF vectors for the two titles
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            # Transform titles to TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform([title1, title2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except:
            # Fallback to simple string similarity
            return 1.0 if title1.lower() == title2.lower() else 0.0
    
    def _calculate_genre_similarity(self, genres1: str, genres2: str) -> float:
        """
        Calculate genre similarity using Jaccard similarity.
        
        Args:
            genres1 (str): First film genres (comma-separated)
            genres2 (str): Second film genres (comma-separated)
            
        Returns:
            float: Genre similarity score (0.0 to 1.0)
        """
        if not genres1 or not genres2:
            return 0.0
        
        # Parse genres into sets
        set1 = set([g.strip().lower() for g in str(genres1).split(',') if g.strip()])
        set2 = set([g.strip().lower() for g in str(genres2).split(',') if g.strip()])
        
        if not set1 or not set2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_year_similarity(self, year1: int, year2: int) -> float:
        """
        Calculate year similarity based on temporal proximity.
        
        Args:
            year1 (int): First film release year
            year2 (int): Second film release year
            
        Returns:
            float: Year similarity score (0.0 to 1.0)
        """
        if not year1 or not year2:
            return 0.0
        
        # Calculate absolute difference in years
        year_diff = abs(year1 - year2)
        
        # Convert to similarity score (closer years = higher similarity)
        # Use exponential decay: similarity = exp(-year_diff / 20)
        similarity = np.exp(-year_diff / 20)
        
        return float(similarity)

class GenreSimilarityAnalysis(BaseSimilarityTask):
    """Analyze similarity within specific genres"""
    
    genre = luigi.Parameter()
    threshold = luigi.FloatParameter(default=0.3)
    
    def output(self):
        genre_safe = self.genre.replace(" ", "_").replace("&", "and")
        return luigi.LocalTarget(f"output/similarity/genre_similarity_{genre_safe}.json")
    
    def run(self):
        logger.info(f"🎭 Analyzing similarity within genre: {self.genre}")
        
        df = self._load_data()
        
        # Filter by genre
        genre_df = df[df['genres'].str.contains(self.genre, na=False)]
        
        if len(genre_df) == 0:
            logger.warning(f"No films found for genre: {self.genre}")
            results = {
                'genre': self.genre,
                'similarity_groups': [],
                'message': 'No films found for this genre'
            }
        else:
            # Create feature text
            genre_df['features'] = genre_df.apply(self._create_feature_text, axis=1)
            
            # Vectorize features
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            feature_matrix = vectorizer.fit_transform(genre_df['features'].fillna(''))
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(feature_matrix)
            
            # Find similarity groups
            similarity_groups = []
            processed_indices = set()
            
            for i in range(len(genre_df)):
                if i not in processed_indices:
                    # Find films similar to this one
                    similar_indices = np.where(similarity_matrix[i] > self.threshold)[0]
                    
                    if len(similar_indices) > 1:  # More than just itself
                        group_films = genre_df.iloc[similar_indices].copy()
                        group_films['similarity_score'] = similarity_matrix[i][similar_indices]
                        
                        similarity_groups.append({
                            'group_id': len(similarity_groups),
                            'films': group_films.to_dict('records'),
                            'group_size': len(group_films)
                        })
                        
                        processed_indices.update(similar_indices)
            
            results = {
                'genre': self.genre,
                'total_films': len(genre_df),
                'similarity_groups': similarity_groups,
                'total_groups': len(similarity_groups),
                'threshold': self.threshold
            }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save results
        with open(self.output().path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Genre similarity analysis completed: {results.get('total_groups', 0)} groups found")
    
    def _create_feature_text(self, row) -> str:
        """Create feature text for similarity calculation"""
        features = []
        
        for col in ['title', 'genres', 'director', 'country']:
            if pd.notna(row.get(col)):
                features.append(str(row[col]))
        
        return ' '.join(features)

class DirectorSimilarityAnalysis(BaseSimilarityTask):
    """Analyze similarity within director's filmography"""
    
    director = luigi.Parameter()
    threshold = luigi.FloatParameter(default=0.3)
    
    def output(self):
        director_safe = self.director.replace(" ", "_")
        return luigi.LocalTarget(f"output/similarity/director_similarity_{director_safe}.json")
    
    def run(self):
        logger.info(f"🎬 Analyzing similarity for director: {self.director}")
        
        df = self._load_data()
        
        # Filter by director
        director_df = df[df['director'].str.contains(self.director, na=False)]
        
        if len(director_df) == 0:
            logger.warning(f"No films found for director: {self.director}")
            results = {
                'director': self.director,
                'similarity_groups': [],
                'message': 'No films found for this director'
            }
        else:
            # Create feature text
            director_df['features'] = director_df.apply(self._create_feature_text, axis=1)
            
            # Vectorize features
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            feature_matrix = vectorizer.fit_transform(director_df['features'].fillna(''))
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(feature_matrix)
            
            # Find similarity groups
            similarity_groups = []
            processed_indices = set()
            
            for i in range(len(director_df)):
                if i not in processed_indices:
                    # Find films similar to this one
                    similar_indices = np.where(similarity_matrix[i] > self.threshold)[0]
                    
                    if len(similar_indices) > 1:  # More than just itself
                        group_films = director_df.iloc[similar_indices].copy()
                        group_films['similarity_score'] = similarity_matrix[i][similar_indices]
                        
                        similarity_groups.append({
                            'group_id': len(similarity_groups),
                            'films': group_films.to_dict('records'),
                            'group_size': len(group_films)
                        })
                        
                        processed_indices.update(similar_indices)
            
            results = {
                'director': self.director,
                'total_films': len(director_df),
                'similarity_groups': similarity_groups,
                'total_groups': len(similarity_groups),
                'threshold': self.threshold
            }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save results
        with open(self.output().path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Director similarity analysis completed: {results.get('total_groups', 0)} groups found")
    
    def _create_feature_text(self, row) -> str:
        """Create feature text for similarity calculation"""
        features = []
        
        for col in ['title', 'genres', 'director', 'country']:
            if pd.notna(row.get(col)):
                features.append(str(row[col]))
        
        return ' '.join(features)

class CreateRecommendations(luigi.Task):
    """
    Similarity Task: Create Film Recommendations
    
    This task generates personalized film recommendations based on:
    - User preferences (if provided)
    - Popular films in similar genres
    - High-rated films from similar directors
    - Recent films with high ratings
    
    Parameters:
        csv_path (str): Path to the input CSV file (for dependency tracking)
        schema_path (str): Path to the JSON schema file (for dependency tracking)
        user_preferences (Dict): User preferences for recommendations
        num_recommendations (int): Number of recommendations to generate
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    user_preferences = luigi.DictParameter(default={}, description="User preferences")
    num_recommendations = luigi.IntParameter(default=20, description="Number of recommendations")
    
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
            luigi.LocalTarget: Path to the recommendations file
        """
        return luigi.LocalTarget("output/similarity/recommendations.json")
    
    def run(self):
        """
        Execute the recommendation generation logic.
        
        This method:
        1. Loads the main dataset
        2. Applies user preferences if provided
        3. Generates recommendations using multiple strategies
        4. Ranks and filters recommendations
        5. Saves recommendation results
        """
        logger.info("🎯 Creating film recommendations...")
        
        # Load the main dataset
        df = pd.read_parquet(self.input().path)
        
        # Initialize recommendation candidates
        candidates = []
        
        # Strategy 1: Recent films (2010 onwards)
        recent_films = df[df['release_year'] >= 2010].copy()
        recent_films['score'] = 0.5  # Base score for recent films
        candidates.extend(recent_films.to_dict('records'))
        
        # Strategy 2: Popular genres
        popular_genres = ['Action', 'Drama', 'Comedy', 'Thriller']
        for genre in popular_genres:
            genre_films = df[df['genres'].str.contains(genre, na=False)].copy()
            genre_films['score'] = 0.4
            candidates.extend(genre_films.to_dict('records'))
        
        # Strategy 3: Apply user preferences if provided
        if self.user_preferences:
            preferred_films = self._apply_user_preferences(df, self.user_preferences)
            candidates.extend(preferred_films)
        
        # Strategy 4: Genre-based recommendations
        if 'preferred_genres' in self.user_preferences:
            genre_recommendations = self._get_genre_recommendations(
                df, self.user_preferences['preferred_genres']
            )
            candidates.extend(genre_recommendations)
        
        # Remove duplicates and rank by score
        unique_candidates = {}
        for candidate in candidates:
            film_id = candidate['id']
            if film_id not in unique_candidates or candidate.get('score', 0) > unique_candidates[film_id].get('score', 0):
                unique_candidates[film_id] = candidate
        
        # Sort by score and take top recommendations
        recommendations = sorted(
            unique_candidates.values(), 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )[:self.num_recommendations]
        
        # Prepare results
        results = {
            'user_preferences': self.user_preferences,
            'num_recommendations': self.num_recommendations,
            'recommendations': recommendations,
            'total_candidates': len(unique_candidates),
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save recommendations
        with open(self.output().path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Recommendations created: {len(recommendations)} films")
    
    def _apply_user_preferences(self, df: pd.DataFrame, preferences: Dict) -> List[Dict]:
        """
        Apply user preferences to generate personalized recommendations.
        
        Args:
            df (pd.DataFrame): Main dataset
            preferences (Dict): User preferences
            
        Returns:
            List[Dict]: List of recommended films with scores
        """
        candidates = []
        
        # Filter by preferred genres
        if 'preferred_genres' in preferences:
            genre_filter = df['genres'].str.contains('|'.join(preferences['preferred_genres']), na=False)
            genre_films = df[genre_filter].copy()
            genre_films['score'] = 0.6
            candidates.extend(genre_films.to_dict('records'))
        
        # Filter by preferred directors
        if 'preferred_directors' in preferences:
            director_filter = df['director'].isin(preferences['preferred_directors'])
            director_films = df[director_filter].copy()
            director_films['score'] = 0.7
            candidates.extend(director_films.to_dict('records'))
        
        # Filter by year range
        if 'preferred_years' in preferences:
            year_range = preferences['preferred_years']
            year_filter = df['release_year'].between(year_range[0], year_range[1])
            year_films = df[year_filter].copy()
            year_films['score'] = 0.5
            candidates.extend(year_films.to_dict('records'))
        
        return candidates
    
    def _get_genre_recommendations(self, df: pd.DataFrame, preferred_genres: List[str]) -> List[Dict]:
        """
        Get genre-based recommendations.
        
        Args:
            df (pd.DataFrame): Main dataset
            preferred_genres (List[str]): List of preferred genres
            
        Returns:
            List[Dict]: List of genre-based recommendations
        """
        # Find films in preferred genres
        genre_filter = df['genres'].str.contains('|'.join(preferred_genres), na=False)
        genre_films = df[genre_filter].copy()
        
        # Calculate genre-specific scores
        genre_films['score'] = 0.6
        
        return genre_films.to_dict('records')

class AnalyzeSimilarityPatterns(luigi.Task):
    """
    Similarity Task: Analyze Similarity Patterns
    
    This task analyzes patterns in film similarities across the dataset:
    - Genre clustering analysis
    - Director influence patterns
    - Temporal similarity trends
    - Similarity distribution analysis
    
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
            CreateMainDataset: The task that provides the main dataset
        """
        from luigi_pandas_pipeline import CreateMainDataset
        return CreateMainDataset(csv_path=self.csv_path, schema_path=self.schema_path)
    
    def output(self):
        """
        Define the output target for this task.
        
        Returns:
            luigi.LocalTarget: Path to the similarity patterns analysis file
        """
        return luigi.LocalTarget("output/similarity/similarity_patterns.json")
    
    def run(self):
        """
        Execute the similarity patterns analysis logic.
        
        This method:
        1. Loads the main dataset
        2. Analyzes genre clustering patterns
        3. Analyzes director influence patterns
        4. Analyzes temporal similarity trends
        5. Saves pattern analysis results
        """
        logger.info("📊 Analyzing similarity patterns...")
        
        # Load the main dataset
        df = pd.read_parquet(self.input().path)
        
        # Analyze genre clustering
        genre_patterns = self._analyze_genre_clustering(df)
        
        # Analyze director influence
        director_patterns = self._analyze_director_influence(df)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(df)
        
        # Prepare analysis results
        analysis_results = {
            'genre_clustering': genre_patterns,
            'director_influence': director_patterns,
            'temporal_patterns': temporal_patterns,
            'dataset_summary': {
                'total_films': len(df),
                'unique_genres': len(set([g for genres in df['genres'].dropna() for g in str(genres).split(',')])),
                'unique_directors': df['director'].nunique(),
                'year_range': {
                    'min': int(df['release_year'].min()),
                    'max': int(df['release_year'].max())
                }
            },
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        # Save analysis results
        with open(self.output().path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info("✅ Similarity patterns analysis completed")
    
    def _analyze_genre_clustering(self, df: pd.DataFrame) -> Dict:
        """
        Analyze genre clustering patterns.
        
        Args:
            df (pd.DataFrame): Main dataset
            
        Returns:
            Dict: Genre clustering analysis results
        """
        # Extract all genres and count co-occurrences
        genre_cooccurrences = {}
        genre_counts = {}
        
        for genres in df['genres'].dropna():
            genre_list = [g.strip() for g in str(genres).split(',') if g.strip()]
            
            # Count individual genres
            for genre in genre_list:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Count co-occurrences
            for i, genre1 in enumerate(genre_list):
                for genre2 in genre_list[i+1:]:
                    pair = tuple(sorted([genre1, genre2]))
                    genre_cooccurrences[pair] = genre_cooccurrences.get(pair, 0) + 1
        
        # Find most common genre combinations and convert tuple keys to strings
        top_combinations = sorted(genre_cooccurrences.items(), key=lambda x: x[1], reverse=True)[:10]
        genre_cooccurrences_str = {f"{pair[0]} & {pair[1]}": count for pair, count in top_combinations}
        
        return {
            'genre_counts': genre_counts,
            'genre_cooccurrences': genre_cooccurrences_str,
            'most_common_genres': sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _analyze_director_influence(self, df: pd.DataFrame) -> Dict:
        """
        Analyze director influence patterns.
        
        Args:
            df (pd.DataFrame): Main dataset
            
        Returns:
            Dict: Director influence analysis results
        """
        # Group by director and analyze patterns
        director_stats = df.groupby('director').agg({
            'id': 'count',  # Number of films
            'release_year': ['min', 'max']  # Year range
        }).round(2)
        
        # Flatten column names
        director_stats.columns = ['film_count', 'earliest_year', 'latest_year']
        
        # Find prolific directors
        prolific_directors = director_stats[director_stats['film_count'] >= 2]
        
        return {
            'director_statistics': director_stats.to_dict('index'),
            'prolific_directors': prolific_directors.to_dict('index'),
            'top_directors_by_film_count': director_stats.nlargest(10, 'film_count').to_dict('index')
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal similarity patterns.
        
        Args:
            df (pd.DataFrame): Main dataset
            
        Returns:
            Dict: Temporal patterns analysis results
        """
        # Analyze trends by decade
        df['decade'] = (df['release_year'] // 10) * 10
        decade_stats = df.groupby('decade').agg({
            'id': 'count',  # Films per decade
            'genres': lambda x: list(set([g for genres in x.dropna() for g in str(genres).split(',')]))
        }).round(2)
        
        # Flatten column names
        decade_stats.columns = ['film_count', 'genres']
        
        # Analyze film count trends over time
        year_film_trend = df.groupby('release_year')['id'].count()
        
        return {
            'decade_statistics': decade_stats.to_dict('index'),
            'yearly_film_trend': year_film_trend.to_dict(),
            'most_productive_decades': decade_stats.nlargest(5, 'film_count').to_dict('index')
        }

class SimilarityPipeline(luigi.WrapperTask):
    """
    Main Similarity Pipeline Wrapper Task
    
    This is the main entry point for all similarity operations. It orchestrates
    different types of similarity analysis and ensures all dependencies are satisfied.
    
    Parameters:
        csv_path (str): Path to the input CSV file
        schema_path (str): Path to the JSON schema file
    """
    
    csv_path = luigi.Parameter(description="Path to input CSV file")
    schema_path = luigi.Parameter(description="Path to JSON schema file")
    
    def requires(self):
        """
        Define all similarity task dependencies.
        
        Returns:
            list: List of all required similarity tasks
        """
        return [
            # Find similar films for specific examples
            FindSimilarFilms(
                csv_path=self.csv_path, 
                schema_path=self.schema_path, 
                film_id=1, 
                similarity_threshold=0.7, 
                top_n=5
            ),
            FindSimilarFilms(
                csv_path=self.csv_path, 
                schema_path=self.schema_path, 
                film_id=2, 
                similarity_threshold=0.6, 
                top_n=10
            ),
            
            # Create recommendations
            CreateRecommendations(
                csv_path=self.csv_path, 
                schema_path=self.schema_path,
                user_preferences={
                    'preferred_genres': ['Action', 'Drama'],
                    'preferred_directors': ['Christopher Nolan'],
                    'preferred_years': [2000, 2020]
                },
                num_recommendations=15
            ),
            
            # Analyze similarity patterns
            AnalyzeSimilarityPatterns(
                csv_path=self.csv_path, 
                schema_path=self.schema_path
            )
        ]
    
    def run(self):
        """
        Execute the complete similarity pipeline.
        
        This method is called after all dependencies are satisfied.
        It provides a summary of the similarity analysis execution.
        """
        logger.info("🎯 Similarity Pipeline completed successfully!")
        logger.info("🔍 Similar film analysis completed")
        logger.info("🎯 Recommendations generated")
        logger.info("📊 Similarity patterns analyzed")
        logger.info("✅ All similarity tasks finished")

def main():
    """
    Main function to run the similarity pipeline.
    
    This function can be used to execute the similarity pipeline directly
    from the command line or as a standalone script.
    """
    luigi.run(['SimilarityPipeline', '--local-scheduler'])

if __name__ == '__main__':
    main() 