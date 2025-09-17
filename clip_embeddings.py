import clip
import torch
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class CLIPEmbeddings:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"Loaded CLIP model: {model_name}")

    def download_image(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image.convert('RGB')
        except Exception as e:
            print(f"Error downloading image: {e}")
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

if __name__ == "__main__":
    # Initialize CLIP
    clip_model = CLIPEmbeddings("ViT-B/32")

    # Your image URLs
    image_urls = [
        "https://media.licdn.com/dms/image/v2/D4E22AQF8_kxle_D-2A/feedshare-shrink_1280/B4EZk3w4rHGcAs-/0/1757577188674?e=1760572800&v=beta&t=65aRcAEILVpbL5ZqKk6eXDDrOoJrvhub2U3hXg8Nk58"
    ]

    # Qdrant client with API key and compatibility check disabled
    client = QdrantClient(
        url="https://fae242d8-a289-47b1-81de-7581be0601e4.us-east-1-1.aws.cloud.qdrant.io",  # Replace with your Qdrant endpoint URL
        port=None,                    # Avoid double specifying port if included in URL
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.46Ud_HT9wZc45jrn00oMrvqScuJY0KzG_VwTKY-91z4",  # Replace with your actual API key
        check_compatibility=False,
        timeout=30   # Skip version check to avoid forbidden error
    )

    collection_name = "karan"

    # Check existing collections
    try:
        existing_collections = [c.name for c in client.get_collections().collections]
        print("Existing collections:", existing_collections)
    except Exception as e:
        print("Error fetching collections:", e)
        existing_collections = []

    # Create collection if missing using recommended methods (avoid deprecated recreate_collection)
    if collection_name not in existing_collections:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=512,  # CLIP ViT-B/32 embedding size
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Process each image and upsert into Qdrant
    for i, image_url in enumerate(image_urls):
        print(f"\nProcessing image {i+1}: {image_url}")
        embedding = clip_model.get_image_embedding(image_url)
        if embedding is None:
            print("Failed to generate embedding for this image.")
            continue

        vector = embedding.tolist()
        print(f"Embedding dimension: {len(vector)}")

        point = PointStruct(
            id=i+1,
            vector=vector,
            payload={
                "filename": f"image_{i+1}.jpg",
                "tags": ["product", "demo"],
                "image_url":image_urls
            }
        )

        try:
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            print("✅ Image embedding stored in Qdrant.")
        except Exception as e:
            print("Error upserting vector:", e)
