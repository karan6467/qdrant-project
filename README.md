# CLIP Embeddings API

A Flask-based REST API for generating and searching image/text embeddings using OpenAI's CLIP model with Qdrant vector database integration.

## Features
- Generate embeddings for images and text
- Store embeddings in Qdrant vector database
- Search for similar images using text queries
- Batch processing support
- Debug and analytics endpoints

### Installation Commands for Users:
```bash
# Clone your repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Run the application
python app.py