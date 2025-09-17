from flask import Flask, request, jsonify
from flask_cors import CORS
import clip
import torch
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging
import traceback
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPEmbeddings:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logger.info(f"Loaded CLIP model: {model_name}")

    def download_image(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image.convert('RGB')
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None

    def get_image_embedding(self, image_url):
        image = self.download_image(image_url)
        if image is None:
            return None
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def get_text_embedding(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()[0]

# Initialize CLIP model globally
clip_model = None

def initialize_clip():
    global clip_model
    if clip_model is None:
        clip_model = CLIPEmbeddings("ViT-B/32")
    return clip_model

def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        port=os.getenv("QDRANT_PORT") or None,
        api_key=os.getenv("QDRANT_API_KEY"),
        check_compatibility=False,
        timeout=30
    )

@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint with API documentation"""
    return jsonify({
        "message": "CLIP Embeddings API",
        "version": "1.0.0",
        "status": "running",
        "available_endpoints": {
            "GET /": "This welcome message",
            "GET /health": "Health check",
            "POST /embed-image": "Generate embedding for single image",
            "POST /embed-batch-images": "Generate embeddings for multiple images",
            "POST /embed-text": "Generate embedding for text",
            "POST /search": "Search similar vectors",
            "GET /collections": "List all collections"
        },
        "usage_example": {
            "embed_image": {
                "url": "/embed-image",
                "method": "POST",
                "body": {
                    "image_url": "https://example.com/image.jpg",
                    "store_in_qdrant": False
                }
            },
            "search": {
                "url": "/search",
                "method": "POST", 
                "body": {
                    "query": "red car",
                    "limit": 5
                }
            }
        }
    })

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/embed-image', methods=['POST'])
def embed_image():
    """
    Embed a single image and optionally store in Qdrant
    Expected JSON payload:
    {
        "image_url": "https://example.com/image.jpg",
        "store_in_qdrant": true/false (optional, default: false),
        "collection_name": "collection_name" (optional, default: "karan"),
        "metadata": {
            "filename": "image.jpg",
            "tags": ["tag1", "tag2"],
            "custom_field": "custom_value"
        } (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "image_url is required"}), 400

        image_url = data['image_url']
        store_in_qdrant = data.get('store_in_qdrant', False)
        collection_name = data.get('collection_name', 'karan')
        metadata = data.get('metadata', {})

        # Initialize CLIP model
        model = initialize_clip()
        
        # Get embedding
        logger.info(f"Processing image: {image_url}")
        embedding = model.get_image_embedding(image_url)
        
        if embedding is None:
            return jsonify({"error": "Failed to generate embedding for the image"}), 400

        result = {
            "success": True,
            "image_url": image_url,
            "embedding_dimension": len(embedding),
            "embedding": embedding.tolist()
        }

        # Optionally store in Qdrant
        if store_in_qdrant:
            try:
                client = get_qdrant_client()
                
                # Ensure collection exists
                existing_collections = [c.name for c in client.get_collections().collections]
                if collection_name not in existing_collections:
                    logger.info(f"Creating collection '{collection_name}'...")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=512,  # CLIP ViT-B/32 embedding size
                            distance=Distance.COSINE
                        )
                    )

                # Generate unique ID based on timestamp and URL hash
                point_id = abs(hash(f"{image_url}_{datetime.now().isoformat()}")) % (10**8)
                
                # Prepare payload
                payload = {
                    "image_url": image_url,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )

                client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                
                result["stored_in_qdrant"] = True
                result["collection_name"] = collection_name
                result["point_id"] = point_id
                
                logger.info("Image embedding stored in Qdrant successfully")
                
            except Exception as e:
                logger.error(f"Error storing in Qdrant: {e}")
                result["qdrant_error"] = str(e)
                result["stored_in_qdrant"] = False

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in embed_image: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed-batch-images', methods=['POST'])
def embed_batch_images():
    """
    Embed multiple images and optionally store in Qdrant
    Expected JSON payload:
    {
        "image_urls": ["url1", "url2", "url3"],
        "store_in_qdrant": true/false (optional, default: false),
        "collection_name": "collection_name" (optional, default: "karan"),
        "metadata": {
            "batch_id": "batch_123",
            "tags": ["tag1", "tag2"]
        } (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'image_urls' not in data:
            return jsonify({"error": "image_urls array is required"}), 400

        image_urls = data['image_urls']
        store_in_qdrant = data.get('store_in_qdrant', False)
        collection_name = data.get('collection_name', 'karan')
        metadata = data.get('metadata', {})

        if not isinstance(image_urls, list):
            return jsonify({"error": "image_urls must be an array"}), 400

        # Initialize CLIP model
        model = initialize_clip()
        
        results = []
        points_to_upsert = []
        client = None

        # Initialize Qdrant client if needed
        if store_in_qdrant:
            try:
                client = get_qdrant_client()
                existing_collections = [c.name for c in client.get_collections().collections]
                if collection_name not in existing_collections:
                    logger.info(f"Creating collection '{collection_name}'...")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=512,
                            distance=Distance.COSINE
                        )
                    )
            except Exception as e:
                logger.error(f"Error initializing Qdrant: {e}")
                return jsonify({"error": f"Qdrant initialization failed: {str(e)}"}), 500

        # Process each image
        for i, image_url in enumerate(image_urls):
            logger.info(f"Processing image {i+1}/{len(image_urls)}: {image_url}")
            embedding = model.get_image_embedding(image_url)
            
            result = {
                "image_url": image_url,
                "index": i
            }
            
            if embedding is None:
                result["success"] = False
                result["error"] = "Failed to generate embedding"
            else:
                result["success"] = True
                result["embedding_dimension"] = len(embedding)
                result["embedding"] = embedding.tolist()
                
                # Prepare for Qdrant if needed
                if store_in_qdrant and client:
                    point_id = abs(hash(f"{image_url}_{datetime.now().isoformat()}_{i}")) % (10**8)
                    payload = {
                        
                        "image_url": image_url,
                        "timestamp": datetime.now().isoformat(),
                        "batch_index": i,
                        **metadata
                    }
                    
                    points_to_upsert.append(PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    ))
                    result["point_id"] = point_id
            
            results.append(result)

        # Batch upsert to Qdrant
        if store_in_qdrant and points_to_upsert and client:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert
                )
                logger.info(f"Batch upserted {len(points_to_upsert)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error batch upserting to Qdrant: {e}")
                return jsonify({
                    "error": f"Batch upsert failed: {str(e)}",
                    "partial_results": results
                }), 500

        return jsonify({
            "success": True,
            "total_images": len(image_urls),
            "successful_embeddings": sum(1 for r in results if r.get("success")),
            "stored_in_qdrant": store_in_qdrant,
            "collection_name": collection_name if store_in_qdrant else None,
            "results": results
        })

    except Exception as e:
        logger.error(f"Error in embed_batch_images: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed-text', methods=['POST'])
def embed_text():
    """
    Embed text using CLIP
    Expected JSON payload:
    {
        "text": "description of image or query",
        "store_in_qdrant": true/false (optional, default: false),
        "collection_name": "collection_name" (optional, default: "karan"),
        "metadata": {
            "type": "query",
            "source": "user_input"
        } (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "text is required"}), 400

        text = data['text']
        store_in_qdrant = data.get('store_in_qdrant', False)
        collection_name = data.get('collection_name', 'karan')
        metadata = data.get('metadata', {})

        # Initialize CLIP model
        model = initialize_clip()
        
        # Get embedding
        logger.info(f"Processing text: {text}")
        embedding = model.get_text_embedding(text)
        
        result = {
            "success": True,
            "text": text,
            "embedding_dimension": len(embedding),
            "embedding": embedding.tolist()
        }

        # Optionally store in Qdrant
        if store_in_qdrant:
            try:
                client = get_qdrant_client()
                
                # Ensure collection exists
                existing_collections = [c.name for c in client.get_collections().collections]
                if collection_name not in existing_collections:
                    logger.info(f"Creating collection '{collection_name}'...")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=512,
                            distance=Distance.COSINE
                        )
                    )

                point_id = abs(hash(f"{text}_{datetime.now().isoformat()}")) % (10**8)
                
                payload = {
                    "text": text,
                    "timestamp": datetime.now().isoformat(),
                    "type": "text",
                    **metadata
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )

                client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                
                result["stored_in_qdrant"] = True
                result["collection_name"] = collection_name
                result["point_id"] = point_id
                
            except Exception as e:
                logger.error(f"Error storing in Qdrant: {e}")
                result["qdrant_error"] = str(e)
                result["stored_in_qdrant"] = False

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in embed_text: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_vectors():
    """
    Search for similar vectors in Qdrant collection
    Expected JSON payload:
    {
        "query": "text query" OR "query_vector": [0.1, 0.2, ...],
        "collection_name": "collection_name" (optional, default: "karan"),
        "limit": 5 (optional, default: 5),
        "score_threshold": 0.7 (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON payload is required"}), 400

        collection_name = data.get('collection_name', 'karan')
        limit = data.get('limit', 5)
        score_threshold = data.get('score_threshold')

        query_vector = None
        
        # Get query vector
        if 'query_vector' in data:
            query_vector = data['query_vector']
        elif 'query' in data:
            # Convert text query to vector
            model = initialize_clip()
            query_text = data['query']
            query_vector = model.get_text_embedding(query_text).tolist()
        else:
            return jsonify({"error": "Either 'query' (text) or 'query_vector' is required"}), 400

        # Search in Qdrant
        client = get_qdrant_client()
        
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit
        }
        
        if score_threshold:
            search_params["score_threshold"] = score_threshold

        search_results = client.search(**search_params)
        
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return jsonify({
            "success": True,
            "collection_name": collection_name,
            "query_used": data.get('query', 'vector provided'),
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(f"Error in search_vectors: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# @app.route('/collections', methods=['GET'])
# def list_collections():
#     """List all available Qdrant collections"""
#     try:
#         client = get_qdrant_client()
#         collections = client.get_collections()
        
#         collection_info = []
#         for collection in collections.collections:
#             try:
#                 info = client.get_collection(collection.name)
#                 collection_info.append({
#                     "name": collection.name,
#                     "vectors_count": info.vectors_count,
#                     "status": info.status
#                 })
#             except Exception as e:
#                 collection_info.append({
#                     "name": collection.name,
#                     "error": str(e)
#                 })

#         return jsonify({
#             "success": True,
#             "collections": collection_info
#         })

#     except Exception as e:
#         logger.error(f"Error listing collections: {traceback.format_exc()}")
#         return jsonify({"error": str(e)}), 500

@app.route('/collections', methods=['GET'])
def list_collections():
    """List all available Qdrant collections with actual data check"""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        collection_info = []
        for collection in collections.collections:
            try:
                # Get basic info
                info = client.get_collection(collection.name)
                
                # Try to count actual points by scrolling
                actual_count = 0
                try:
                    scroll_result = client.scroll(
                        collection_name=collection.name,
                        limit=1,
                        with_vectors=False,
                        with_payload=False
                    )
                    if scroll_result[0]:  # If there are points
                        # Try to get all points to count them
                        all_points = client.scroll(
                            collection_name=collection.name,
                            limit=10000,  # Large number to get all
                            with_vectors=False,
                            with_payload=False
                        )
                        actual_count = len(all_points[0])
                    else:
                        actual_count = 0
                except Exception as count_error:
                    actual_count = f"Count error: {str(count_error)}"
                
                collection_info.append({
                    "name": collection.name,
                    "vectors_count": info.vectors_count,  # Might be null
                    "actual_points_count": actual_count,  # Our manual count
                    "status": info.status,
                    "has_data": actual_count > 0 if isinstance(actual_count, int) else "unknown"
                })
                
            except Exception as e:
                collection_info.append({
                    "name": collection.name,
                    "error": str(e),
                    "vectors_count": "error",
                    "actual_points_count": "error"
                })

        return jsonify({
            "success": True,
            "collections": collection_info
        })

    except Exception as e:
        logger.error(f"Error listing collections: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug-collection/<collection_name>', methods=['GET'])
def debug_collection(collection_name):
    """Debug what's actually in a collection"""
    try:
        client = get_qdrant_client()
        
        # Get all points in collection
        all_points = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_vectors=False,
            with_payload=True
        )
        
        points_data = []
        for point in all_points[0]:
            points_data.append({
                "id": point.id,
                "payload": point.payload
            })
        
        return jsonify({
            "success": True,
            "collection_name": collection_name,
            "total_points": len(points_data),
            "points": points_data
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "collection_name": collection_name
        }), 500
@app.route('/view-embeddings/<collection_name>', methods=['GET'])
def view_embeddings(collection_name):
    """View stored embeddings with their vectors"""
    try:
        client = get_qdrant_client()
        
        # Get limit from query parameter (default 5)
        limit = request.args.get('limit', 5, type=int)
        
        # Get points WITH vectors
        points = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=True,  # This includes the actual embeddings
            with_payload=True
        )
        
        embeddings_data = []
        for point in points[0]:
            embeddings_data.append({
                "id": point.id,
                "payload": point.payload,
                "vector": point.vector,  # The actual 512 numbers
                "vector_length": len(point.vector) if point.vector else 0,
                "first_5_values": point.vector[:5] if point.vector else [],
                "last_5_values": point.vector[-5:] if point.vector else []
            })
        
        return jsonify({
            "success": True,
            "collection_name": collection_name,
            "total_returned": len(embeddings_data),
            "embeddings": embeddings_data
        })
        
    except Exception as e:
        logger.error(f"Error viewing embeddings: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare-embeddings/<collection_name>', methods=['POST'])
def compare_embeddings(collection_name):
    """Compare a text query embedding with stored image embeddings"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "query text is required"}), 400
        
        query_text = data['query']
        limit = data.get('limit', 3)
        
        # Generate query embedding
        model = initialize_clip()
        query_embedding = model.get_text_embedding(query_text)
        
        # Get stored embeddings
        client = get_qdrant_client()
        points = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_vectors=True,
            with_payload=True
        )
        
        # Calculate similarities manually
        comparisons = []
        for point in points[0]:
            if point.vector:
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(query_embedding, point.vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(point.vector)
                )
                
                comparisons.append({
                    "id": point.id,
                    "payload": point.payload,
                    "similarity_score": float(similarity),
                    "image_url": point.payload.get('image_url', 'N/A'),
                    "query_embedding_sample": query_embedding[:5].tolist(),
                    "stored_embedding_sample": point.vector[:5]
                })
        
        # Sort by similarity
        comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            "success": True,
            "query": query_text,
            "collection_name": collection_name,
            "query_embedding_length": len(query_embedding),
            "comparisons": comparisons[:limit]
        })
        
    except Exception as e:
        logger.error(f"Error comparing embeddings: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/embedding-stats/<collection_name>', methods=['GET'])
def embedding_stats(collection_name):
    """Get statistics about stored embeddings"""
    try:
        client = get_qdrant_client()
        
        # Get all points with vectors
        points = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_vectors=True,
            with_payload=True
        )
        
        if not points[0]:
            return jsonify({
                "success": True,
                "collection_name": collection_name,
                "message": "No embeddings found"
            })
        
        # Calculate statistics
        import numpy as np
        
        vectors = [point.vector for point in points[0] if point.vector]
        if not vectors:
            return jsonify({
                "success": True,
                "collection_name": collection_name,
                "message": "No valid vectors found"
            })
        
        vectors_array = np.array(vectors)
        
        # Calculate stats
        vector_stats = {
            "total_embeddings": len(vectors),
            "vector_dimension": len(vectors[0]) if vectors else 0,
            "mean_values": np.mean(vectors_array, axis=0)[:10].tolist(),  # First 10 dimensions
            "std_values": np.std(vectors_array, axis=0)[:10].tolist(),    # First 10 dimensions
            "min_value": float(np.min(vectors_array)),
            "max_value": float(np.max(vectors_array)),
            "sample_payloads": [
                point.payload for point in points[0][:3]
            ]
        }
        
        return jsonify({
            "success": True,
            "collection_name": collection_name,
            "stats": vector_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding stats: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500 

if __name__ == '__main__':
    # Initialize CLIP model on startup
    initialize_clip()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=False      # Set to True for development
    )