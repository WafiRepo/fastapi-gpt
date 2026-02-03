# Struktur Prompt Templates

## Overview

Template prompt sekarang dipisahkan menjadi file-file terpisah untuk setiap method, memudahkan maintenance dan modifikasi.

## Struktur File

```
data/prompts/
├── README.md                    # Dokumentasi umum
├── STRUCTURE.md                 # File ini
├── zero_shot.md                 # Template zero-shot prompt
├── few_shot.md                  # Template few-shot prompt
├── cot.md                       # Template Chain of Thought prompt
└── draft_critique_revise.md     # Template draft-critique-revise prompt
```

## Format Template

Setiap template menggunakan placeholder yang akan di-replace saat runtime:

- `{STAGE}` - Stage pembelajaran (problem_finding, conceptual, application)
- `{TOPIC}` - Topik pembelajaran (e.g., centripetal acceleration)
- `{AUDIENCE}` - Target audience (e.g., Grade 11)
- `{INPUT_TEXT}` - Context/input text untuk generation

## Cara Menggunakan

### 1. Edit Template

Edit file `.md` yang sesuai di folder `data/prompts/`:

```bash
# Edit zero-shot template
notepad data/prompts/zero_shot.md

# Edit few-shot template
notepad data/prompts/few_shot.md
```

### 2. Template Otomatis Dibaca

Script `generate_and_evaluate_complete.py` akan otomatis membaca template dari file saat runtime. Tidak perlu restart atau recompile.

### 3. Validasi Template

Pastikan placeholder tetap ada:
- `{STAGE}` harus ada (kecuali few_shot yang tidak perlu AUDIENCE)
- `{TOPIC}` harus ada
- `{AUDIENCE}` harus ada (kecuali few_shot)
- `{INPUT_TEXT}` harus ada

## Contoh Modifikasi

### Menambah Contoh di Few-shot Template

Edit `data/prompts/few_shot.md` dan tambahkan contoh baru:

```markdown
Stage: application
Topic: centripetal acceleration
Input:
<<<
Designing a safe carousel ride
>>>
Output:
{"inquiry":"How would you determine the maximum safe rotation speed for a carousel?"}
```

### Mengubah Requirements di Zero-shot

Edit `data/prompts/zero_shot.md` dan modifikasi bagian Requirements sesuai kebutuhan.

## Testing

Test template loading:

```python
from generate_and_evaluate_complete import load_prompt_template

# Test load
template = load_prompt_template("zero_shot")
print(template)
```

## Best Practices

1. **Backup sebelum edit**: Simpan versi lama sebelum modifikasi besar
2. **Test setelah edit**: Jalankan test generation dengan count kecil
3. **Konsisten format**: Gunakan format yang sama untuk semua template
4. **Dokumentasi**: Tambahkan komentar di template jika perlu

## Troubleshooting

### Error: Template not found
- Pastikan file ada di `data/prompts/`
- Pastikan nama file sesuai: `zero_shot.md`, `few_shot.md`, dll

### Error: Placeholder not replaced
- Pastikan placeholder menggunakan format `{PLACEHOLDER}` (huruf besar, underscore)
- Pastikan tidak ada typo di nama placeholder

### Template tidak ter-update
- Restart script generation
- Clear Python cache jika perlu: `find . -name "*.pyc" -delete`
