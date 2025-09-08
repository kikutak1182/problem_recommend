#!/bin/bash
# 段階的な依存関係インストールスクリプト

echo "=== AtCoder Problem Recommendation App - Dependencies Installation ==="

# Step 1: pip更新
echo "Step 1: Updating pip..."
pip install --upgrade pip

# Step 2: 基本パッケージをインストール
echo "Step 2: Installing basic packages..."
pip install -r requirements.txt

# Step 3: PyTorch CPU版を専用インデックスでインストール
echo "Step 3: Installing PyTorch CPU version..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== Installation completed! ==="
echo "Next steps:"
echo "1. Run: python3 app/main.py"
echo "2. Test: curl http://127.0.0.1:8000/"