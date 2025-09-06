#!/usr/bin/env python3
"""
Benchmark script for ONNX model accuracy validation
Compares FP32 vs INT8 quantized model performance
"""

import os
import logging
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
from tokenizers import Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXEmbeddingBenchmark:
    """Benchmark ONNX embedding models"""
    
    def __init__(self, tokenizer_path: str = "tokenizer/", max_length: int = 256):
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer_file = os.path.join(self.tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        
        # Get special tokens
        vocab = self.tokenizer.get_vocab()
        self.cls_token_id = vocab.get("[CLS]", vocab.get("<cls>", 101))
        self.sep_token_id = vocab.get("[SEP]", vocab.get("<sep>", 102))
        self.pad_token_id = vocab.get("[PAD]", vocab.get("<pad>", 0))
    
    def _tokenize_texts(self, texts: List[str]) -> dict:
        """Tokenize texts with BERT format"""
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = encoding.ids
            
            # Add [CLS] and [SEP]
            full_token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
            
            # Truncate if needed
            if len(full_token_ids) > self.max_length:
                full_token_ids = full_token_ids[:self.max_length-1] + [self.sep_token_id]
            
            # Attention mask
            attention_mask = [1] * len(full_token_ids)
            
            # Padding
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
    
    def get_embeddings(self, model_path: str, texts: List[str]) -> np.ndarray:
        """Get embeddings from ONNX model"""
        # Load model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=session_options)
        
        # Tokenize
        inputs = self._tokenize_texts(texts)
        
        # Inference
        outputs = session.run(None, inputs)
        embeddings = outputs[0]
        
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
        
        return embeddings
    
    def compare_models(self, fp32_path: str, int8_path: str, test_texts: List[str]) -> dict:
        """Compare FP32 and INT8 models"""
        logger.info(f"🔍 Comparing models:")
        logger.info(f"  FP32: {fp32_path}")
        logger.info(f"  INT8: {int8_path}")
        logger.info(f"  Test texts: {len(test_texts)}")
        
        # Check model files
        for path in [fp32_path, int8_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")
        
        try:
            # Get embeddings from both models
            logger.info("Getting FP32 embeddings...")
            fp32_embeddings = self.get_embeddings(fp32_path, test_texts)
            
            logger.info("Getting INT8 embeddings...")
            int8_embeddings = self.get_embeddings(int8_path, test_texts)
            
            # Calculate cosine similarities
            similarities = []
            for i in range(len(test_texts)):
                sim = cosine_similarity([fp32_embeddings[i]], [int8_embeddings[i]])[0][0]
                similarities.append(sim)
            
            # Statistics
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            # Calculate difference (1 - similarity)
            differences = [1 - sim for sim in similarities]
            mean_difference = np.mean(differences)
            
            results = {
                'test_count': len(test_texts),
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'mean_difference': mean_difference,
                'similarities': similarities,
                'differences': differences,
                'test_texts': test_texts
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {e}")
            raise
    
    def print_results(self, results: dict, threshold: float = 0.02):
        """Print benchmark results"""
        logger.info("=" * 60)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Test samples: {results['test_count']}")
        logger.info(f"Mean cosine similarity: {results['mean_similarity']:.4f}")
        logger.info(f"Std cosine similarity:  {results['std_similarity']:.4f}")
        logger.info(f"Min cosine similarity:  {results['min_similarity']:.4f}")
        logger.info(f"Max cosine similarity:  {results['max_similarity']:.4f}")
        logger.info(f"Mean difference:        {results['mean_difference']:.4f}")
        
        # Check threshold
        passed = results['mean_difference'] < threshold
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"Threshold check (<{threshold}): {status}")
        
        logger.info("-" * 60)
        logger.info("Per-sample results:")
        for i, (text, sim, diff) in enumerate(zip(results['test_texts'], results['similarities'], results['differences'])):
            logger.info(f"  {i+1:2d}. {text[:30]:30s} | Sim: {sim:.4f} | Diff: {diff:.4f}")
        
        logger.info("=" * 60)
        
        return passed

def main():
    """Run benchmark"""
    # Test texts (Japanese programming terms)
    test_texts = [
        "動的計画法",
        "グラフ理論", 
        "深さ優先探索",
        "幅優先探索",
        "ベルマンフォード法",
        "ワーシャルフロイド法",
        "セグメント木",
        "Union-Find",
        "最大流問題",
        "二分探索"
    ]
    
    # Model paths
    fp32_model = "model_fp32.onnx"
    int8_model = "model_int8.onnx"
    
    try:
        # Initialize benchmark
        benchmark = ONNXEmbeddingBenchmark()
        
        # Run comparison
        results = benchmark.compare_models(fp32_model, int8_model, test_texts)
        
        # Print results
        passed = benchmark.print_results(results, threshold=0.02)
        
        if passed:
            logger.info("🎉 Benchmark PASSED! Quantization quality is acceptable.")
            return 0
        else:
            logger.error("💥 Benchmark FAILED! Consider using FP32 model for higher accuracy.")
            return 1
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())