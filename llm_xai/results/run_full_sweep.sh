#!/bin/bash
# Full narrative evaluation sweep in four resumable stages. Progress goes to sweep_status.txt and
# sweep_full.log; SWEEP_DONE or SWEEP_FAILED marks the end. Run from anywhere.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
LOG=llm_xai/results/sweep_full.log
STATUS=llm_xai/results/sweep_status.txt
rm -f llm_xai/results/SWEEP_DONE llm_xai/results/SWEEP_FAILED

echo "STAGE=1/4 llama31_8b/c" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE 1/4: llama31_8b variant c (about 2 h) ===" >> "$LOG"
python -m llm_xai.tools.evaluate_llm_paper --variant c --models llama31_8b --resume >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED=1" >> "$STATUS"; touch llm_xai/results/SWEEP_FAILED; exit 1; fi

echo "STAGE=2/4 biomistral/a,b,c" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE 2/4: biomistral variant a,b,c (about 6 h) ===" >> "$LOG"
python -m llm_xai.tools.evaluate_llm_paper --variant a,b,c --models biomistral --resume >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED=1" >> "$STATUS"; touch llm_xai/results/SWEEP_FAILED; exit 1; fi

echo "STAGE=3/4 mistral7b/c" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE 3/4: mistral7b variant c (about 2 h) ===" >> "$LOG"
python -m llm_xai.tools.evaluate_llm_paper --variant c --models mistral7b --resume >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED=1" >> "$STATUS"; touch llm_xai/results/SWEEP_FAILED; exit 1; fi

echo "STAGE=4/4 qwen25_14b/c" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE 4/4: qwen25_14b variant c (about 4 h) ===" >> "$LOG"
python -m llm_xai.tools.evaluate_llm_paper --variant c --models qwen25_14b --resume >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "FAILED=1" >> "$STATUS"; touch llm_xai/results/SWEEP_FAILED; exit 1; fi

echo "STAGE=DONE" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] SWEEP COMPLETE ===" >> "$LOG"
touch llm_xai/results/SWEEP_DONE
