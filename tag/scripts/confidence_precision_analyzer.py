#!/usr/bin/env python3
"""
Confidence Precision Analyzer

Analyzes the precision and distribution of confidence scores from batch results.
"""

import json
import glob
from collections import Counter

def analyze_confidence_precision():
    """Analyze confidence value precision across all batch results"""
    
    all_confidence_values = []
    
    # Collect all confidence values
    for file in glob.glob("data/batch_results_batch_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        if "response" in result and result["response"].get("body"):
                            content = result["response"]["body"]["choices"][0]["message"]["content"]
                            parsed = json.loads(content)
                            for tag in parsed.get("selected_tags", []):
                                conf = tag.get("confidence", 0)
                                all_confidence_values.append(conf)
                    except:
                        pass
    
    print(f"全confidence値の総数: {len(all_confidence_values)}")
    
    # 精度別分析
    decimal_analysis = {}
    
    for val in all_confidence_values:
        val_str = str(val)
        if "." in val_str:
            decimal_part = val_str.split(".")[1]
            precision = len(decimal_part)
            decimal_analysis[precision] = decimal_analysis.get(precision, 0) + 1
        else:
            decimal_analysis[0] = decimal_analysis.get(0, 0) + 1
    
    print(f"精度分布: {decimal_analysis}")
    
    # ユニークな値の詳細
    unique_values = sorted(set(all_confidence_values))
    print(f"ユニーク値の総数: {len(unique_values)}")
    print(f"全ユニーク値: {unique_values}")
    
    # 小数点第2位まである値
    two_decimal_values = [v for v in unique_values if "." in str(v) and len(str(v).split(".")[1]) == 2]
    print(f"小数点第2位まである値: {two_decimal_values}")
    
    # 小数点第1位のみの値
    one_decimal_values = [v for v in unique_values if "." in str(v) and len(str(v).split(".")[1]) == 1]
    print(f"小数点第1位のみの値: {one_decimal_values}")
    
    # 整数値
    integer_values = [v for v in unique_values if "." not in str(v) or str(v).endswith(".0")]
    print(f"整数値: {integer_values}")
    
    # 統計情報
    print(f"\n=== 統計情報 ===")
    print(f"最小値: {min(all_confidence_values)}")
    print(f"最大値: {max(all_confidence_values)}")
    print(f"平均値: {sum(all_confidence_values) / len(all_confidence_values):.3f}")
    
    # 分布
    value_counts = Counter(all_confidence_values)
    print(f"\n=== 頻出値 TOP 10 ===")
    for value, count in value_counts.most_common(10):
        percentage = count / len(all_confidence_values) * 100
        print(f"{value}: {count}回 ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_confidence_precision()