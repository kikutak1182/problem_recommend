#!/usr/bin/env python3
"""
INT8 Quantization Script for ONNX Models
Converts FP32 ONNX model to INT8 using dynamic quantization
"""

import os
import logging
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_info(model_path: str) -> dict:
    """Get model information (size, nodes, etc.)"""
    try:
        model = onnx.load(model_path)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        num_nodes = len(model.graph.node)
        
        return {
            'file_size_mb': file_size,
            'num_nodes': num_nodes,
            'path': model_path
        }
    except Exception as e:
        logger.error(f"Failed to get model info for {model_path}: {e}")
        return {}

def validate_onnx_model(model_path: str) -> bool:
    """Validate ONNX model using onnx.checker"""
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        logger.info(f"✅ ONNX model validation passed: {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ ONNX model validation failed for {model_path}: {e}")
        return False

def test_inference(model_path: str) -> bool:
    """Test inference with dummy data"""
    try:
        session = ort.InferenceSession(model_path)
        
        # Get input info
        input_names = [inp.name for inp in session.get_inputs()]
        logger.info(f"Model inputs: {input_names}")
        
        # Create dummy inputs based on input shapes
        import numpy as np
        dummy_inputs = {}
        
        for inp in session.get_inputs():
            if inp.name == 'input_ids':
                # Create dummy token IDs
                dummy_inputs[inp.name] = np.random.randint(0, 1000, (1, 128), dtype=np.int64)
            elif inp.name == 'attention_mask':
                # Create dummy attention mask
                dummy_inputs[inp.name] = np.ones((1, 128), dtype=np.int64)
        
        # Run inference
        outputs = session.run(None, dummy_inputs)
        
        if outputs and len(outputs) > 0:
            output_shape = outputs[0].shape
            logger.info(f"✅ Inference test passed. Output shape: {output_shape}")
            return True
        else:
            logger.error("❌ No outputs from inference")
            return False
            
    except Exception as e:
        logger.error(f"❌ Inference test failed for {model_path}: {e}")
        return False

def quantize_model(
    input_model: str = "model_fp32.onnx",
    output_model: str = "model_int8.onnx"
) -> bool:
    """Quantize ONNX model from FP32 to INT8"""
    
    logger.info(f"🚀 Starting INT8 quantization...")
    logger.info(f"Input: {input_model} → Output: {output_model}")
    
    # Check input model exists
    if not os.path.exists(input_model):
        logger.error(f"❌ Input model not found: {input_model}")
        logger.info("💡 Run 'python export_onnx.py' first to create the FP32 model")
        return False
    
    # Get pre-quantization info
    logger.info("📊 Analyzing original model...")
    original_info = get_model_info(input_model)
    if original_info:
        logger.info(f"Original model: {original_info['file_size_mb']:.1f} MB, {original_info['num_nodes']} nodes")
    
    # Validate original model
    if not validate_onnx_model(input_model):
        return False
    
    # Test original model inference
    logger.info("🔍 Testing original model inference...")
    if not test_inference(input_model):
        logger.warning("⚠️  Original model inference test failed, but continuing...")
    
    try:
        # Perform dynamic quantization
        logger.info("⚙️  Performing INT8 dynamic quantization...")
        
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8,  # INT8 quantization
            extra_options={
                'EnableSubgraph': True,    # Enable subgraph optimization
                'ForceQuantizeNoInput': True  # Force quantization even without calibration data
            }
        )
        
        logger.info(f"✅ Quantization completed: {output_model}")
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        return False
    
    # Validate quantized model
    logger.info("🔍 Validating quantized model...")
    if not validate_onnx_model(output_model):
        return False
    
    # Test quantized model inference
    logger.info("🔍 Testing quantized model inference...")
    if not test_inference(output_model):
        logger.warning("⚠️  Quantized model inference test failed")
        return False
    
    # Get post-quantization info and compare
    logger.info("📊 Analyzing quantized model...")
    quantized_info = get_model_info(output_model)
    
    if original_info and quantized_info:
        size_reduction = ((original_info['file_size_mb'] - quantized_info['file_size_mb']) / original_info['file_size_mb']) * 100
        
        logger.info("=" * 50)
        logger.info("📈 QUANTIZATION REPORT")
        logger.info("=" * 50)
        logger.info(f"Original model:  {original_info['file_size_mb']:.1f} MB ({original_info['num_nodes']} nodes)")
        logger.info(f"Quantized model: {quantized_info['file_size_mb']:.1f} MB ({quantized_info['num_nodes']} nodes)")
        logger.info(f"Size reduction:  {size_reduction:.1f}% ({original_info['file_size_mb'] - quantized_info['file_size_mb']:.1f} MB saved)")
        logger.info("=" * 50)
    
    logger.info("🎉 INT8 quantization completed successfully!")
    logger.info(f"💡 Use MODEL_PATH={output_model} to run with quantized model")
    
    return True

if __name__ == "__main__":
    try:
        success = quantize_model()
        if not success:
            logger.error("❌ Quantization process failed")
            exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)