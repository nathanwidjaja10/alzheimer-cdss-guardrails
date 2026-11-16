#!/bin/bash
# Q&A benchmark evaluation (resumable). Progress goes to qa_eval_status.txt and qa_eval.log;
# QA_EVAL_DONE or QA_EVAL_FAILED marks the end.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
LOG=llm_xai/results/qa_eval.log
STATUS=llm_xai/results/qa_eval_status.txt
rm -f llm_xai/results/QA_EVAL_DONE llm_xai/results/QA_EVAL_FAILED

echo "STAGE=running" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] evaluate_qa full run (n_per_bucket=5, 675 cases) ===" >> "$LOG"
python -m llm_xai.tools.evaluate_qa --resume >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "STAGE=FAILED" > "$STATUS"
  touch llm_xai/results/QA_EVAL_FAILED
  exit 1
fi

echo "STAGE=DONE" > "$STATUS"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] evaluate_qa COMPLETE ===" >> "$LOG"
touch llm_xai/results/QA_EVAL_DONE
