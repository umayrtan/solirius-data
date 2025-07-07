# Solirius Data Engineering Coding Test - Pandas & Luigi Implementation

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* Git CLI
* GitHub account

### Installation
```bash
# Clone the repository
git clone <your-forked-repo-url>
cd solirius-data-engineering-coding-test-master

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Luigi Pipeline (Workflow Management with Pandas)
```bash
# Install Luigi if not already installed
pip install luigi

# Run the main ETL pipeline
python run_luigi_pipeline.py etl

# Run query analysis tasks
python run_luigi_pipeline.py queries

# Run similarity analysis tasks
python run_luigi_pipeline.py similarity

# Run all Luigi pipelines
python run_luigi_pipeline.py all

# Run Luigi tests
python test_luigi_pipeline.py
```

## 📁 Project Structure
```
├── resources/
│   ├── csv/allFilms.csv          # Input dataset
│   └── json/allFilesSchema.json  # Data schema
├── luigi_pandas_pipeline.py      # Luigi ETL pipeline with pandas
├── luigi_query_system.py         # Luigi query analysis tasks
├── luigi_similarity_system.py    # Luigi similarity analysis tasks
├── run_luigi_pipeline.py         # Luigi pipeline runner
├── test_luigi_pipeline.py        # Luigi pipeline tests
├── requirements.txt              # Dependencies
└── output/                       # Generated parquet files
    ├── films.parquet             # Stage 1 output
    ├── data_quality_report.json  # Data quality validation
    ├── queries/                  # Luigi query outputs
    ├── similarity/               # Luigi similarity outputs
        ├── similarity_index.parquet  # Similarity matrix
    └── validation/               # Data quality validation
        ├── data_quality_report.json/
    └── genres/                   # Stage 2 output
        ├── genre=Action/
        ├── genre=Comedy/
        └── ...
```

## 🎯 Task Requirements

### Test requirements
* Python 3 ✅
* Pandas ✅
* Scikit-learn ✅
* PyArrow ✅
* Great Expectations ✅
* Luigi ✅ (Optional - for advanced workflow management)

### Task one ✅ COMPLETE
Given a dataset of films, the client would like you to perform some transformations and cleaning of the dataset so that their recommendation service can access particular films faster. The output of each stage must be within a folder called output.

**Stage 1: ✅ COMPLETE**
- ✅ Imports the input dataset using schema.json
- ✅ Converts data types according to schema using pandas
- ✅ Saves as `films.parquet` in `output/` folder
- ✅ Automated with pandas pipeline

**Stage 2: ✅ COMPLETE**
- ✅ Creates `genres/` folder in `output/`
- ✅ Produces parquet files for each subgenre using pandas
- ✅ Uses valid naming convention: `genre=Action`
- ✅ Breaks down multi-genre films (e.g., "Action,Adventure" → separate files)
- ✅ Schema matches Stage 1 output

### Task two ✅ COMPLETE
The client requires a function to detect similarity between films. The function will take in a film's id, and a threshold percentage as input, and will return a dataframe that contains all films with a similarity percentage above the threshold.

**Implementation:**
- ✅ Function takes film ID and threshold parameters
- ✅ Returns films above similarity threshold
- ✅ Sensible similarity calculation using TF-IDF and cosine similarity
- ✅ Implemented in `pandas_pipeline.py` using pandas + scikit-learn
- ✅ Integrated with main pipeline

### Task three ✅ COMPLETE
The client requires functionality to allow users to query their dataset. The user should be able to use any number of columns, alongside a range of values for each column to filter films.

**Implementation:**
- ✅ Flexible backend query system in `pandas_pipeline.py` using pandas
- ✅ Multiple query methods: BETWEEN, IN, CONTAINS, EQUALS, etc.
- ✅ Example: Quentin Tarantino/George Lucas 1979-2000 ✅
- ✅ Reusable and well-tested
- ✅ Comprehensive test cases
- ✅ Backend only (no frontend dependencies)

## 📊 Usage Examples

### Running the Luigi Pipeline
```bash
# Run the main ETL pipeline
python run_luigi_pipeline.py etl

# Run specific analysis tasks
python run_luigi_pipeline.py queries
python run_luigi_pipeline.py similarity

# Run all Luigi tasks
python run_luigi_pipeline.py all
```

### Film Similarity (Task 2) - Luigi
```python
import pandas as pd
from luigi_similarity_system import FindSimilarFilms

# Run similarity analysis for film ID 1
# This will create output/similarity/similar_films_1.json
```

### Advanced Querying (Task 3) - Luigi
```python
from luigi_query_system import CustomQueryTask
import json

# Example: Quentin Tarantino or George Lucas 1979-2000
conditions = [
    {'column': 'director', 'operator': 'in', 'value': ["Quentin Tarantino", "George Lucas"]},
    {'column': 'release_year', 'operator': 'between', 'value': (1979, 2000)}
]

# Run custom query
# This will create output/queries/custom_query_*.json
```

### Data Quality Validation
```python
# Data quality validation is automatically run as part of the ETL pipeline
# Results are saved to output/data_quality_report.json
```

### Film Similarity (Task 2) - Pandas
```python
import pandas as pd
from pandas_pipeline import PandasFilmSimilarity

# Load data
df = pd.read_parquet("output/films.parquet")

# Find similar films
similarity_calc = PandasFilmSimilarity()
similar_films = similarity_calc.get_similar_films(df, film_id=1, threshold=0.5)
print(similar_films)
```

### Advanced Querying (Task 3) - Pandas
```python
import pandas as pd
from pandas_pipeline import PandasFilmQueryEngine

# Load data
df = pd.read_parquet("output/films.parquet")

# Create query engine
engine = PandasFilmQueryEngine(df)

# Example from requirements: Quentin Tarantino or George Lucas 1979-2000
conditions = [
    {'column': 'director', 'operator': 'in', 'value': ["Quentin Tarantino", "George Lucas"]},
    {'column': 'release_year', 'operator': 'between', 'value': (1979, 2000)}
]
results = engine.query(conditions)
print(results)
```

### Data Quality Validation
```python
from data_quality import DataQualityValidator

# Load data
df = pd.read_parquet("output/films.parquet")

# Validate dataset
validator = DataQualityValidator()
results = validator.validate_films_dataset(df)
print(f"Data quality: {'PASS' if results['success'] else 'FAIL'}")

# Generate detailed report
report = validator.generate_validation_report(results)
print(report)
```

## 🧪 Testing
```bash
# Run Luigi pipeline tests
python test_luigi_pipeline.py
```

## 🔧 Configuration
The project uses default paths that can be overridden:
- Input CSV: `resources/csv/allFilms.csv`
- Schema: `resources/json/allFilesSchema.json`
- Output: `output/`

## 📈 Performance Notes
- **Luigi workflow management** for complex ETL pipelines
- **Pandas data processing** for fast, reliable operations
- **Optimized for medium datasets** with pandas DataFrames
- **Memory-efficient** genre splitting using pandas operations
- **Fast similarity calculations** using scikit-learn
- **Windows-compatible** without Hadoop/Spark dependencies
- **Modular design** with separate tasks for different operations

## 🐛 Troubleshooting
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip list`
- Verify input files exist in resources/ directory
- For pandas issues, check data types and missing values

## ✅ Requirements Compliance Summary

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Python 3 | ✅ | Python 3.8+ |
| Pandas | ✅ | Pandas 1.5+ |
| Scikit-learn | ✅ | Similarity calculations |
| PyArrow | ✅ | Parquet file support |
| Great Expectations | ✅ | Data quality validation |
| Luigi | ✅ | Workflow management (optional) |
| Task 1 - Stage 1 | ✅ | Pandas ETL pipeline with schema |
| Task 1 - Stage 2 | ✅ | Genre-based partitioning |
| Task 2 | ✅ | Film similarity detection |
| Task 3 | ✅ | Flexible query system |
| Data Quality | ✅ | Great Expectations validation |
| Task 1 - Stage 2 | ✅ | Pandas genre splitting |
| Task 2 - Similarity | ✅ | Pandas film similarity detection |
| Task 3 - Querying | ✅ | Pandas flexible query system |
| Great Expectations | ✅ | Data quality validation |
| Backend only | ✅ | No frontend dependencies |
| Test cases | ✅ | Simple test coverage |

## 🚀 Pandas Advantages

This implementation leverages pandas' data processing capabilities:

- **Simplicity**: Easy to understand and maintain
- **Performance**: Fast data processing for medium datasets
- **Memory Efficiency**: Optimized DataFrame operations
- **Reliability**: Stable and well-tested library
- **Flexibility**: Easy to modify and extend
- **Integration**: Works well with other Python libraries
