#!/usr/bin/env bash
# ============================================================
# download_arabic_ocr.sh
# 下载 HuggingFace 上主要的阿拉伯 OCR 数据集
#
# 用法:
#   chmod +x download_arabic_ocr.sh
#   ./download_arabic_ocr.sh
#   ./download_arabic_ocr.sh /custom/output/dir
#
# 依赖:
#   pip install huggingface_hub
#   可选: 登录私有数据集 → huggingface-cli login
# ============================================================

set -euo pipefail

BASE_DIR="${1:-./arabic_ocr_datasets}"
LOG_FILE="$BASE_DIR/download_log.txt"
SUMMARY_FILE="$BASE_DIR/download_summary.txt"

mkdir -p "$BASE_DIR"

# ── 颜色输出 ────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log() { echo -e "$*" | tee -a "$LOG_FILE"; }
ok()  { log "${GREEN}  ✓ $*${RESET}"; }
warn(){ log "${YELLOW}  ⚠ $*${RESET}"; }
err() { log "${RED}  ✗ $*${RESET}"; }
sep() { log "\n${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; }

# ── 数据集列表 ───────────────────────────────────────────────
# 格式: "repo_id|描述|预估大小|优先级"
# 优先级: high=必下  medium=建议  low=可选
declare -a DATASETS=(
    # ---- 高质量综合数据集 ------------------------------------
    "mssqpi/Arabic-OCR-Dataset|综合阿拉伯 OCR 数据集，1454 下载，parquet 格式，含图像|~1.9GB|high"
    "TheRealOKAI/arabic_ocr_merged_dataset|合并版阿拉伯 OCR，202 下载|~500MB|high"
    "aallail/arabic_ocr_synth_2|合成阿拉伯 OCR 数据 v2，814 下载|~1GB|high"
    "aallail/arabic_ocr_synth_3|合成阿拉伯 OCR 数据 v3，126 下载|~500MB|high"

    # ---- 场景文字 / 自然场景 ---------------------------------
    "Melaraby/EvArEST-dataset-for-Arabic-scene-text-recognition|阿拉伯场景文字识别，123 下载|~2GB|high"
    "Melaraby/EvArEST-dataset-for-Arabic-scene-text-detection|阿拉伯场景文字检测，19 下载|~1GB|medium"

    # ---- 合成扫描 / 文档 ------------------------------------
    "loay/arabic-ocr-synthetic-scans-faker-300k|30万合成扫描文档，96 下载|~5GB|medium"
    "FatimahEmadEldin/Gutenberg-Arabic-OCR-HTML-Pages|Gutenberg 阿拉伯 HTML 页面 OCR，89 下载|~300MB|medium"

    # ---- 波斯语/阿拉伯语通用 --------------------------------
    "mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Medium|波斯/阿拉伯文本行 OCR medium，411 下载|~2GB|medium"
    "mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Small|波斯/阿拉伯文本行 OCR small，222 下载|~500MB|medium"

    # ---- 评测基准 -------------------------------------------
    "ahmedheakl/arabic_ocrisi|Arabic OCR-ISI 基准，30 下载|~100MB|low"
    "ahmedheakl/arocrbench_arabicocr|ArocBench 评测集，30 下载|~50MB|low"

    # ---- 其他 -----------------------------------------------
    "JayanthMuthu/arabic-ocr|197 下载，文本+图像|~800MB|medium"
    "Omar-youssef/arabic-ocr-dataset|12 下载，3 likes|~50MB|low"
    "Nexdata/10020_Images_of_Arabic_Natural_Scene_OCR_Data|10020 张自然场景|~300MB|low"
)

# ── 下载函数 ─────────────────────────────────────────────────
download_dataset() {
    local repo_id="$1"
    local desc="$2"
    local size="$3"
    local priority="$4"

    local safe_name="${repo_id//\//__}"
    local dst="$BASE_DIR/$safe_name"

    log "\n${BOLD}[$priority] $repo_id${RESET}"
    log "  描述: $desc"
    log "  大小: $size"
    log "  目标: $dst"

    # 已下载则跳过
    if [[ -d "$dst" ]] && [[ -n "$(ls -A "$dst" 2>/dev/null)" ]]; then
        warn "已存在，跳过 (删除目录可重新下载: rm -rf $dst)"
        echo "SKIP: $repo_id" >> "$SUMMARY_FILE"
        return 0
    fi

    mkdir -p "$dst"

    # 用 Python huggingface_hub 下载（比 CLI 更稳定）
    python3 - <<PYEOF
import sys
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="$repo_id",
        repo_type="dataset",
        local_dir="$dst",
        ignore_patterns=["*.md", "*.txt", "*.py"],
    )
    print(f"  Downloaded to: {path}")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    local status=$?
    if [[ $status -eq 0 ]]; then
        local size_actual
        size_actual=$(du -sh "$dst" 2>/dev/null | cut -f1)
        ok "下载成功 (实际大小: $size_actual)"
        echo "OK: $repo_id (${size_actual})" >> "$SUMMARY_FILE"
    else
        err "下载失败: $repo_id"
        rm -rf "$dst"   # 清理不完整的下载
        echo "FAIL: $repo_id" >> "$SUMMARY_FILE"
        return 1
    fi
}

# ── 主流程 ───────────────────────────────────────────────────
sep
log "${BOLD}阿拉伯 OCR 数据集批量下载${RESET}"
log "输出目录: ${CYAN}$BASE_DIR${RESET}"
log "时间: $(date)"
sep

# 检查依赖
if ! python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    err "缺少 huggingface_hub，运行: pip install huggingface_hub"
    exit 1
fi

# 解析优先级参数
PRIORITY_FILTER="${2:-all}"    # all / high / medium / low
log "下载优先级: ${YELLOW}$PRIORITY_FILTER${RESET}"

# 统计
total=0
success=0
failed=0
skipped=0

: > "$SUMMARY_FILE"   # 清空 summary

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r repo_id desc size priority <<< "$entry"

    # 过滤优先级
    if [[ "$PRIORITY_FILTER" != "all" ]] && [[ "$priority" != "$PRIORITY_FILTER" ]]; then
        continue
    fi

    ((total++)) || true

    if download_dataset "$repo_id" "$desc" "$size" "$priority"; then
        local_status=$(grep "^OK:\|^SKIP:" "$SUMMARY_FILE" | tail -1 | cut -d: -f1)
        if [[ "$local_status" == "SKIP" ]]; then
            ((skipped++)) || true
        else
            ((success++)) || true
        fi
    else
        ((failed++)) || true
    fi
done

# ── 结果汇总 ─────────────────────────────────────────────────
sep
log "${BOLD}下载完成${RESET}"
log "  总计: $total"
log "  ${GREEN}成功: $success${RESET}"
log "  ${YELLOW}跳过: $skipped${RESET}"
log "  ${RED}失败: $failed${RESET}"
log ""
log "目录结构:"
du -sh "$BASE_DIR"/*/  2>/dev/null | sort -h | while read -r size path; do
    log "  $size  $(basename "$path")"
done
log ""
log "${BOLD}下一步:${RESET}"
log "  autotrainer data --path $BASE_DIR/"
log ""
log "详细日志: $LOG_FILE"
log "摘要:     $SUMMARY_FILE"
sep
