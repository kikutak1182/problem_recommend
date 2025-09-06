#!/usr/bin/env python3
"""
ONNX Export Script for paraphrase-multilingual-MiniLM-L12-v2
Converts Sentence-Transformers model to ONNX with mean pooling + L2 normalization
"""

import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import onnx
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerONNX(nn.Module):
    """Wrapper to include mean pooling + L2 normalization in ONNX export"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.sentence_transformer = SentenceTransformer(model_name)
        self.tokenizer = self.sentence_transformer.tokenizer
        
        # Get the transformer model
        self.transformer = self.sentence_transformer._modules['0'].auto_model
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with mean pooling + L2 normalization"""
        # Get token embeddings from transformer
        model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply mean pooling
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        
        # L2 normalization
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings

def export_onnx_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Export sentence transformer model to ONNX format"""
    
    logger.info(f"Starting ONNX export for model: {model_name}")
    
    # Initialize model wrapper and force to CPU for ONNX export
    model_wrapper = SentenceTransformerONNX(model_name)
    model_wrapper.eval()
    model_wrapper.to('cpu')  # Force model to CPU for ONNX export
    
    # Save tokenizer
    tokenizer_path = "tokenizer/"
    os.makedirs(tokenizer_path, exist_ok=True)
    model_wrapper.tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Create dummy inputs for export (ensure they're on CPU)
    max_length = 256
    batch_size = 1
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length), dtype=torch.long, device='cpu')
    dummy_attention_mask = torch.ones((batch_size, max_length), dtype=torch.long, device='cpu')
    
    # Define input names and dynamic axes
    input_names = ['input_ids', 'attention_mask']
    output_names = ['sentence_embedding']
    
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'sentence_embedding': {0: 'batch_size'}
    }
    
    # Export to ONNX
    output_path = "model_fp32.onnx"
    
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        model_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"✅ ONNX model exported successfully: {output_path}")
        
        # Get model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"📊 Model size: {file_size:.1f} MB")
        
        # Test with actual inference
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        # Test inference
        test_inputs = {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        }
        
        outputs = session.run(None, test_inputs)
        embedding_dim = outputs[0].shape[1]
        logger.info(f"🎯 Output embedding dimension: {embedding_dim}")
        logger.info("🔍 ONNX inference test passed!")
        
    except Exception as e:
        logger.error(f"❌ ONNX model verification failed: {e}")
        raise
    
    logger.info("🎉 ONNX export completed successfully!")
    return output_path

if __name__ == "__main__":
    try:
        export_onnx_model()
    except Exception as e:
        logger.error(f"Export failed: {e}")
        exit(1)