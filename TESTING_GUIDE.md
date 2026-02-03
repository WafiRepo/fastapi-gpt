# Testing Guide - Generate Inquiry Samples

Panduan untuk testing generation dengan berbagai konfigurasi.

## Quick Test (3 samples per stage)

### Test dengan 1 strategy (zero_shot)
```bash
python generate_and_evaluate_complete.py \
    --model gpt-4 \
    --count-per-stage 3 \
    --strategies zero_shot \
    --no-eval \
    --output-dir data
```

**Hasil:**
- 3 stages × 2 languages × 3 items × 1 strategy = **18 items**
- Estimasi waktu: ~2-3 menit
- Estimasi biaya: ~$0.20-0.30

### Test dengan semua strategies
```bash
python generate_and_evaluate_complete.py \
    --model gpt-4 \
    --count-per-stage 3 \
    --strategies zero_shot few_shot cot draft_critique_revise \
    --no-eval \
    --output-dir data
```

**Hasil:**
- 3 stages × 2 languages × 3 items × 4 strategies = **72 items**
- Estimasi waktu: ~8-12 menit
- Estimasi biaya: ~$0.80-1.20

## Minimal Test (1 sample per stage)

### Test cepat dengan 1 item per stage
```bash
python generate_and_evaluate_complete.py \
    --model gpt-4 \
    --count-per-stage 1 \
    --strategies zero_shot \
    --no-eval \
    --output-dir data
```

**Hasil:**
- 3 stages × 2 languages × 1 item × 1 strategy = **6 items**
- Estimasi waktu: ~30 detik
- Estimasi biaya: ~$0.05-0.10

## Production Test (10 samples per stage)

### Full test dengan semua strategies
```bash
python generate_and_evaluate_complete.py \
    --model gpt-4 \
    --count-per-stage 10 \
    --strategies zero_shot few_shot cot draft_critique_revise \
    --output-dir data \
    --plots-dir eval_plots
```

**Hasil:**
- 3 stages × 2 languages × 10 items × 4 strategies = **240 items**
- Estimasi waktu: ~30-45 menit
- Estimasi biaya: ~$3-5

## Test dengan Evaluation

### Generate + Evaluate (3 samples)
```bash
python generate_and_evaluate_complete.py \
    --model gpt-4 \
    --count-per-stage 3 \
    --strategies zero_shot \
    --output-dir data \
    --plots-dir eval_plots \
    --bleu-scale 0-100
```

Ini akan:
1. Generate 18 items
2. Evaluate dengan semua metrics
3. Generate comparison report
4. Generate plots

## Verifikasi Hasil

### Cek jumlah items
```bash
python -c "import json; items = [json.loads(line) for line in open('data/inquiry_samples_all_strategies.jsonl', 'r', encoding='utf-8')]; print(f'Total: {len(items)} items')"
```

### Cek breakdown
```bash
python -c "import json; from collections import Counter; items = [json.loads(line) for line in open('data/inquiry_samples_all_strategies.jsonl', 'r', encoding='utf-8')]; stages = Counter([item['stage'] for item in items]); langs = Counter([item['lang'] for item in items]); print('By stage:'); [print(f'  {s}: {c}') for s, c in stages.items()]; print('By language:'); [print(f'  {l}: {c}') for l, c in langs.items()]"
```

### Lihat sample item
```bash
python -c "import json; items = [json.loads(line) for line in open('data/inquiry_samples_all_strategies.jsonl', 'r', encoding='utf-8')]; print(json.dumps(items[0], indent=2, ensure_ascii=False))"
```

## Troubleshooting

### Error: OPENAI_API_KEY not set
```bash
# Set di PowerShell
$env:OPENAI_API_KEY="your-key-here"

# Atau buat file .env
echo OPENAI_API_KEY=your-key-here > .env
```

### Error: Template not found
Pastikan folder `data/prompts/` ada dan berisi:
- `zero_shot.md`
- `few_shot.md`
- `cot.md`
- `draft_critique_revise.md`

### Error: Encoding issues
Script sudah handle encoding untuk Windows. Jika masih error, set:
```bash
$env:PYTHONIOENCODING="utf-8"
```

## Recommended Testing Flow

1. **Minimal Test** (1 sample) - Verify setup works
2. **Quick Test** (3 samples, 1 strategy) - Verify templates work
3. **Strategy Test** (3 samples, all strategies) - Compare strategies
4. **Production** (10 samples, all strategies) - Full dataset

## Output Files

Setelah generation, file berikut akan dibuat:

- `data/inquiry_samples_all_strategies.jsonl` - Generated samples
- `data/evaluation_results_all.csv` - Evaluation results (jika --no-eval tidak digunakan)
- `data/evaluation_results_all.jsonl` - Evaluation results JSONL
- `data/prompt_strategy_comparison.json` - Strategy comparison
- `eval_plots/*.png` - Metric plots
