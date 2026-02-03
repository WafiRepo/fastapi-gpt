# Prompt Templates

Folder ini berisi template prompt untuk berbagai strategi generation.

## File Templates

- `zero_shot.md` - Zero-shot prompt (tanpa contoh)
- `few_shot.md` - Few-shot prompt (dengan contoh)
- `cot.md` - Chain of Thought prompt (dengan silent reasoning)
- `draft_critique_revise.md` - Draft-Critique-Revise prompt (self-improvement)

## Format Template

Setiap template menggunakan placeholder:
- `{STAGE}` - Stage pembelajaran (problem_finding, conceptual, application)
- `{TOPIC}` - Topik (e.g., centripetal acceleration)
- `{AUDIENCE}` - Target audience (e.g., Grade 11)
- `{INPUT_TEXT}` - Context/input text untuk generation

## Penggunaan

Templates ini dibaca oleh `generate_and_evaluate_complete.py` untuk membangun prompt yang dikirim ke GPT.

## Memodifikasi Template

Untuk mengubah template:
1. Edit file `.md` yang sesuai
2. Pastikan placeholder `{STAGE}`, `{TOPIC}`, `{AUDIENCE}`, `{INPUT_TEXT}` tetap ada
3. Restart script generation jika sedang berjalan
