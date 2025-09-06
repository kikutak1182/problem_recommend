#!/usr/bin/env python3
"""
ONNX-based Sentence Embedding API
PyTorch-free inference using ONNXRuntime + tokenizers
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ONNX Runtime and tokenizers
import onnxruntime as ort
from tokenizers import Tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class EmbedRequest(BaseModel):
    """Request model for batch embedding"""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)

class EmbedResponse(BaseModel):
    """Response model for embedding"""
    vectors: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")

class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool = Field(..., description="Service status")
    model_path: str = Field(..., description="Currently loaded model path")
    dimension: Optional[int] = Field(None, description="Embedding dimension")

class EmbeddingService:
    """ONNX-based embedding service"""
    
    def __init__(self, model_path: str = "model_int8.onnx", tokenizer_path: str = "tokenizer/", max_length: int = 256):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.session = None
        self.tokenizer = None
        self.embedding_dim = None
        
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            if not os.path.exists(self.model_path):
                # Try fallback to FP32 model
                fallback_path = "model_fp32.onnx"
                if os.path.exists(fallback_path):
                    logger.warning(f"Model {self.model_path} not found, using fallback: {fallback_path}")
                    self.model_path = fallback_path
                else:
                    raise FileNotFoundError(f"No model found at {self.model_path} or {fallback_path}")
            
            # Create ONNX Runtime session with optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use CPU provider (optimized for Cloud Run)
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get embedding dimension from model output shape
            output_shape = self.session.get_outputs()[0].shape
            if len(output_shape) >= 2:
                self.embedding_dim = output_shape[1]
            
            logger.info(f"✅ ONNX model loaded: {self.model_path}")
            logger.info(f"📐 Embedding dimension: {self.embedding_dim}")
            
            # Log input/output info
            inputs = [(inp.name, inp.shape, inp.type) for inp in self.session.get_inputs()]
            outputs = [(out.name, out.shape, out.type) for out in self.session.get_outputs()]
            logger.info(f"🔧 Model inputs: {inputs}")
            logger.info(f"📤 Model outputs: {outputs}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer from saved files"""
        try:
            tokenizer_file = os.path.join(self.tokenizer_path, "tokenizer.json")
            
            if not os.path.exists(tokenizer_file):
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
            
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            logger.info(f"✅ Tokenizer loaded: {tokenizer_file}")
            
            # Get special tokens
            vocab = self.tokenizer.get_vocab()
            self.cls_token_id = vocab.get("[CLS]", vocab.get("<cls>", 101))  # Default BERT CLS
            self.sep_token_id = vocab.get("[SEP]", vocab.get("<sep>", 102))  # Default BERT SEP
            self.pad_token_id = vocab.get("[PAD]", vocab.get("<pad>", 0))    # Default PAD
            
            logger.info(f"🔤 Special tokens - CLS: {self.cls_token_id}, SEP: {self.sep_token_id}, PAD: {self.pad_token_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
    
    def _tokenize_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize texts with BERT format: [CLS] text [SEP]"""
        try:
            batch_input_ids = []
            batch_attention_mask = []
            
            for text in texts:
                # Tokenize the text (without special tokens first)
                encoding = self.tokenizer.encode(text, add_special_tokens=False)
                token_ids = encoding.ids
                
                # Manually add [CLS] and [SEP] tokens
                # Format: [CLS] + tokens + [SEP]
                full_token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
                
                # Truncate if too long
                if len(full_token_ids) > self.max_length:
                    full_token_ids = full_token_ids[:self.max_length-1] + [self.sep_token_id]
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(full_token_ids)
                
                # Pad to max_length
                padding_length = self.max_length - len(full_token_ids)
                if padding_length > 0:
                    full_token_ids.extend([self.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
                
                batch_input_ids.append(full_token_ids)
                batch_attention_mask.append(attention_mask)
            
            return {
                'input_ids': np.array(batch_input_ids, dtype=np.int64),
                'attention_mask': np.array(batch_attention_mask, dtype=np.int64)
            }
            
        except Exception as e:
            logger.error(f"❌ Tokenization failed: {e}")
            raise
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            # Tokenize inputs
            inputs = self._tokenize_texts(texts)
            
            # Run ONNX inference
            outputs = self.session.run(None, inputs)
            
            # Get embeddings (first output)
            embeddings = outputs[0]
            
            # Ensure L2 normalization (the model should already output normalized vectors)
            # But we can double-check by re-normalizing
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise

# Initialize service
MODEL_PATH = os.getenv("MODEL_PATH", "model_int8.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer/")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

logger.info(f"🚀 Initializing Embedding Service...")
logger.info(f"📍 Model path: {MODEL_PATH}")
logger.info(f"🔤 Tokenizer path: {TOKENIZER_PATH}")
logger.info(f"📏 Max length: {MAX_LENGTH}")

try:
    embedding_service = EmbeddingService(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        max_length=MAX_LENGTH
    )
    logger.info("✅ Embedding service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize embedding service: {e}")
    raise

# FastAPI app
app = FastAPI(
    title="ONNX Sentence Embedding API",
    description="High-performance sentence embedding API using ONNX Runtime",
    version="1.0.0",
    docs_url="/docs"
)

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        ok=True,
        model_path=embedding_service.model_path,
        dimension=embedding_service.embedding_dim
    )

@app.post("/embed", response_model=EmbedResponse)
async def embed_batch(request: EmbedRequest):
    """Batch embedding endpoint"""
    try:
        embeddings = embedding_service.embed(request.texts)
        
        return EmbedResponse(
            vectors=embeddings.tolist(),
            dimension=embeddings.shape[1],
            count=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/embed")
async def embed_single(q: str = Query(..., description="Text to embed")):
    """Single text embedding endpoint"""
    try:
        embeddings = embedding_service.embed([q])
        vector = embeddings[0].tolist()
        
        return {
            "vector": vector,
            "dimension": len(vector),
            "text": q
        }
        
    except Exception as e:
        logger.error(f"Single embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "ONNX Sentence Embedding API",
        "version": "1.0.0",
        "model": embedding_service.model_path,
        "dimension": embedding_service.embedding_dim,
        "endpoints": {
            "health": "/healthz",
            "batch_embed": "POST /embed",
            "single_embed": "GET /embed?q=<text>",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )