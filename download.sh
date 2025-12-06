#!/bin/bash


set -e


GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Dependencies kontrol ediliyor...${NC}"
pip install huggingface_hub --upgrade --quiet

# ---------------------------------------------------------
# 1. VACE 
# ---------------------------------------------------------
TARGET_VACE="./src/vace_lib/models/Wan2.1-VACE-1.3B"
echo -e "${GREEN}Wan2.1-VACE-1.3B indiriliyor... Hedef: ${TARGET_VACE}${NC}"

# Klasörü oluştur
mkdir -p "$TARGET_VACE"

# İndirme komutu (Symlink kullanma, gerçek dosyaları indir)
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
    --local-dir "$TARGET_VACE" \
    --local-dir-use-symlinks False \
    --exclude "*.git*" "README.md" # Gereksiz dosyaları hariç tutabiliriz

# ---------------------------------------------------------
# 2. Qwen 
# ---------------------------------------------------------
TARGET_QWEN="./src/qwen_checkpoints/LAMP-Qwen-2.5-VL"
echo -e "${GREEN}Qwen2.5-VL-7B-Instruct indiriliyor... Hedef: ${TARGET_QWEN}${NC}"

# Klasörü oluştur
mkdir -p "$TARGET_QWEN"

huggingface-cli download burakkizil/LAMP-Qwen-2.5-VL \
    --local-dir "$TARGET_QWEN" \
    --local-dir-use-symlinks False \
    --exclude "*.git*"
