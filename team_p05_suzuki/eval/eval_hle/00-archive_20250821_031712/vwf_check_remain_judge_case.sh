#!/bin/bash

python -c "
import json
with open('predictions/hle_Qwen3-235B-A22B.json') as f: pred = json.load(f)
with open('judged/judged_hle_Qwen3-235B-A22B.json') as f: judged = json.load(f)
print(f'予測: {len(pred)}, Judge済み: {len(judged)}, 残り: {len(pred)-len(judged)}')
"
