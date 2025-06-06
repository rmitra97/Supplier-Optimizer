# ESG Analysis and Supplier Optimization Pipeline

A comprehensive ESG (Environmental, Social, and Governance) analysis pipeline that combines document processing, semantic search, scoring, and optimization capabilities. This system helps organizations analyze ESG metrics, optimize supplier selection, and make data-driven sustainability decisions.

## Key Features

### 1. Document Processing
- PDF document ingestion and processing
- Intelligent text chunking with semantic boundaries
- Support for multiple document formats
- Automated metadata extraction

### 2. Semantic Search
- Vector-based semantic search using Pinecone
- Category-based filtering
- Real-time search results
- Confidence scoring for matches

### 3. ESG Scoring
- Multi-dimensional ESG metric analysis
- Confidence-weighted scoring
- Category-specific normalization
- Comparative analysis capabilities

### 4. Supplier Optimization
- Multi-criteria decision making
- ESG-integrated supplier selection
- Weighted scoring system
- Cost-quality-sustainability balance

### 5. Web Interface
- Interactive dashboard
- Real-time data visualization
- Document upload and processing
- Search and filter capabilities

## Architecture

### Core Components

1. **Document Processor** (`extract_chunk.py`)
   - PDF text extraction
   - Semantic chunking
   - Metadata preservation
   - Output: Structured CSV data

2. **Vector Engine** (`embedding.py`, `search_pinecone.py`)
   - Text to vector conversion
   - Semantic similarity search
   - Category filtering
   - Real-time query processing

3. **Scoring Engine** (`calculate_scores.py`)
   - Metric normalization
   - Confidence weighting
   - Category scoring
   - Final score computation

4. **Optimization Engine** (`optimization.py`)
   - Supplier evaluation
   - Multi-criteria optimization
   - ESG integration
   - Decision support

5. **Web Interface** (`app.py`)
   - Streamlit-based dashboard
   - Interactive visualizations
   - User-friendly controls
   - Real-time updates

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kopalbhatnagar05/Supplier-Optimizer.git
cd Supplier-Optimizer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create pinecone.env
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=your_environment_here
```

### Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload ESG documents through the interface

4. Use the search functionality to query ESG information

5. View and analyze ESG scores

6. Optimize supplier selection based on ESG metrics

## Data Flow

1. **Input Processing**
   ```
   PDF Document → Chunking → Vector Embedding → Storage
   ```

2. **Search Flow**
   ```
   Query → Vector Conversion → Semantic Search → Results
   ```

3. **Analysis Flow**
   ```
   Raw Data → Normalization → Scoring → Optimization
   ```

## Configuration

### Environment Variables
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment name

### Model Configuration
- Default embedding model: `all-MiniLM-L6-v2`
- Customizable chunk size and overlap
- Adjustable scoring weights

## Performance

- Distributed processing with PySpark
- Efficient vector search with Pinecone
- Optimized chunking strategy
- Cached embeddings for faster retrieval

## Security

- Environment variable management
- API key protection
- Secure file handling
- Input validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sentence Transformers for text embeddings
- Pinecone for vector database
- Streamlit for web interface
- PySpark for distributed processing

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Updates

Stay tuned for future updates including:
- Enhanced document processing
- Advanced analytics
- Improved visualization
- Additional optimization algorithms 