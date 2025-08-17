#!/usr/bin/env python3
"""
Confidence System for Tag Inference

Implements a composite confidence score with multiple components:
- self_conf: Model's self-reported confidence with reasoning
- verifier: Separate verification scoring 
- embed_sim: Embedding similarity score
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import openai

from tag_inference_config import inference_config
from embedding_tag_filter import EmbeddingTagFilter

@dataclass
class ConfidenceWeights:
    """Weights for composite confidence calculation"""
    self_conf: float = 0.4
    verifier: float = 0.3
    embed_sim: float = 0.3
    
    def __post_init__(self):
        total = self.self_conf + self.verifier + self.embed_sim
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

class ConfidenceSystem:
    """Comprehensive confidence scoring system"""
    
    def __init__(self, weights: Optional[ConfidenceWeights] = None):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.embedding_filter = EmbeddingTagFilter()
        self.weights = weights or ConfidenceWeights()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for confidence system"""
        logger = logging.getLogger('confidence_system')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_self_confidence_prompt(self, problem_title: str, editorial_text: str, 
                                    candidate_tags: List[Dict]) -> Tuple[str, str]:
        """Create system and user messages for self-confidence assessment"""
        
        system_message = """あなたは競技プログラミングのタグ付け専門家です。
問題と解説を読んで、候補タグから3つ選択し、各タグの確信度(0-1)と根拠を必ず示してください。"""
        
        tag_list = []
        allowed_ids = []
        for tag in candidate_tags:
            tag_list.append(f"{tag['id']}: {tag['name']}")
            allowed_ids.append(tag['id'])
        
        tags_info = "\\n".join(tag_list)
        
        user_message = f"""[Problem]
{problem_title}

[Editorial]
{editorial_text[:1000]}

[Candidate Tags]
{tags_info}

上記から3つのタグを選び、各タグの確信度(0-1)と根拠を示してください。"""
        
        return system_message, user_message
    
    def create_self_confidence_schema(self, allowed_tag_ids: List[str]) -> Dict:
        """Create JSON schema for self-confidence assessment"""
        return {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": allowed_tag_ids
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "reasoning": {
                                "type": "string",
                                "minLength": 10
                            }
                        },
                        "required": ["id", "confidence", "reasoning"]
                    },
                    "minItems": 3,
                    "maxItems": 3
                }
            },
            "required": ["tags"],
            "additionalProperties": False
        }
    
    def get_self_confidence(self, problem_title: str, editorial_text: str, 
                          candidate_tags: List[Dict]) -> Optional[Dict]:
        """Get model's self-reported confidence with reasoning"""
        
        try:
            system_msg, user_msg = self.create_self_confidence_prompt(
                problem_title, editorial_text, candidate_tags
            )
            
            allowed_ids = [tag['id'] for tag in candidate_tags]
            schema = self.create_self_confidence_schema(allowed_ids)
            
            response = self.client.chat.completions.create(
                model=inference_config.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": system_msg,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {"role": "user", "content": user_msg}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tag_confidence",
                        "schema": schema
                    }
                }
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get self-confidence: {e}")
            return None
    
    def create_verifier_prompt(self, problem_title: str, editorial_text: str, 
                             tag_name: str, tag_description: str) -> Tuple[str, str]:
        """Create prompt for tag verification"""
        
        system_message = """あなたは競技プログラミングの専門家です。
問題と解説を読んで、指定されたタグが適切かを0-1のスコアで判定してください。
根拠が明確でない場合は低いスコアをつけてください。"""
        
        user_message = f"""[Problem]
{problem_title}

[Editorial]
{editorial_text[:1000]}

[Tag to Verify]
{tag_name}: {tag_description}

この問題に「{tag_name}」タグは適切ですか？
0-1のスコアと簡潔な理由を答えてください。"""
        
        return system_message, user_message
    
    def create_verifier_schema(self) -> Dict:
        """Create JSON schema for verification scoring"""
        return {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "reason": {
                    "type": "string",
                    "minLength": 5
                }
            },
            "required": ["score", "reason"],
            "additionalProperties": False
        }
    
    def get_verifier_score(self, problem_title: str, editorial_text: str,
                          tag_id: str, tag_name: str, tag_description: str) -> float:
        """Get verification score from separate prompt"""
        
        try:
            system_msg, user_msg = self.create_verifier_prompt(
                problem_title, editorial_text, tag_name, tag_description
            )
            
            schema = self.create_verifier_schema()
            
            response = self.client.chat.completions.create(
                model=inference_config.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": system_msg,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {"role": "user", "content": user_msg}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verification",
                        "schema": schema
                    }
                }
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            score = result.get("score", 0.0)
            reason = result.get("reason", "")
            
            self.logger.debug(f"Verifier for {tag_name}: {score:.3f} - {reason}")
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Failed to get verifier score for {tag_name}: {e}")
            return 0.0
    
    def get_embedding_similarity(self, problem_title: str, editorial_text: str,
                                tag_id: str) -> float:
        """Get normalized embedding similarity score"""
        
        try:
            # Get problem embedding
            problem_embedding = self.embedding_filter.get_problem_embedding(
                problem_title, editorial_text
            )
            
            # Get tag embedding
            tag_embedding = self.embedding_filter.tag_embeddings.get(tag_id)
            if tag_embedding is None:
                self.logger.warning(f"No embedding found for tag {tag_id}")
                return 0.0
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                problem_embedding.reshape(1, -1),
                tag_embedding.reshape(1, -1)
            )[0][0]
            
            # Normalize to 0-1 range (cosine similarity ranges from -1 to 1)
            normalized = (similarity + 1) / 2
            return float(normalized)
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding similarity for {tag_id}: {e}")
            return 0.0
    
    def calculate_composite_confidence(self, problem_title: str, editorial_text: str,
                                     candidate_tags: List[Dict]) -> Dict[str, Dict]:
        """Calculate composite confidence for all tags"""
        
        self.logger.info("Starting composite confidence calculation...")
        
        # Get self-confidence scores
        self_conf_result = self.get_self_confidence(
            problem_title, editorial_text, candidate_tags
        )
        
        if not self_conf_result:
            self.logger.error("Failed to get self-confidence scores")
            return {}
        
        results = {}
        
        for tag_data in self_conf_result["tags"]:
            tag_id = tag_data["id"]
            tag_info = self.embedding_filter.get_tag_info_by_id(tag_id)
            
            if not tag_info:
                continue
            
            tag_name = tag_info["name"]
            tag_description = tag_info["description"]
            
            self.logger.info(f"Processing confidence for: {tag_name}")
            
            # Component 1: Self-confidence
            self_conf = tag_data["confidence"]
            reasoning = tag_data["reasoning"]
            
            # Component 2: Verifier score
            verifier_score = self.get_verifier_score(
                problem_title, editorial_text, tag_id, tag_name, tag_description
            )
            
            # Component 3: Embedding similarity
            embed_sim = self.get_embedding_similarity(
                problem_title, editorial_text, tag_id
            )
            
            # Calculate composite score
            composite_score = (
                self.weights.self_conf * self_conf +
                self.weights.verifier * verifier_score +
                self.weights.embed_sim * embed_sim
            )
            
            results[tag_id] = {
                "tag_name": tag_name,
                "composite_confidence": float(composite_score),
                "components": {
                    "self_confidence": float(self_conf),
                    "verifier_score": float(verifier_score),
                    "embedding_similarity": float(embed_sim)
                },
                "reasoning": reasoning,
                "weights_used": {
                    "self_conf": self.weights.self_conf,
                    "verifier": self.weights.verifier,
                    "embed_sim": self.weights.embed_sim
                }
            }
            
            self.logger.info(f"✓ {tag_name}: composite={composite_score:.3f} "
                           f"(self={self_conf:.3f}, verif={verifier_score:.3f}, embed={embed_sim:.3f})")
        
        return results

if __name__ == "__main__":
    # Test the confidence system
    confidence_system = ConfidenceSystem()
    
    # Test problem
    test_problem = "最短路問題"
    test_editorial = "この問題はDijkstraのアルゴリズムを使います。正の重みグラフで単一始点最短路を求めます。"
    
    # Get candidate tags first
    candidates = confidence_system.embedding_filter.filter_candidate_tags(
        test_problem, test_editorial
    )
    
    print("Testing composite confidence system...")
    results = confidence_system.calculate_composite_confidence(
        test_problem, test_editorial, candidates[:5]  # Top 5 candidates
    )
    
    print("\\n=== Confidence Results ===")
    for tag_id, data in results.items():
        print(f"{data['tag_name']}: {data['composite_confidence']:.3f}")
        print(f"  Components: self={data['components']['self_confidence']:.3f}, "
              f"verif={data['components']['verifier_score']:.3f}, "
              f"embed={data['components']['embedding_similarity']:.3f}")
        print(f"  Reasoning: {data['reasoning']}")
        print()