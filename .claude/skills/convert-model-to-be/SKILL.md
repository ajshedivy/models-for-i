---
name: convert-model-to-be
description: Convert a GGUF model to big-endian format for IBM i / PowerPC systems and publish it to the huggingface.co/models-for-i collection. Use when the user asks to convert a GGUF model to big-endian (BE), add a model to the models-for-i HF collection, or prepare a HuggingFace model for use on IBM i with Llama.cpp. Triggers on phrases like "convert X to big endian", "upload X to models-for-i", "add this model to the collection".
---

# Convert Model To Big-Endian

Workflow for taking a little-endian GGUF on HuggingFace, converting it to big-endian for IBM i, and publishing it under `huggingface.co/models-for-i`. Assumes the project root (containing `gguf_convert_endian.py`) is the working directory.

## Workflow

1. **Verify the source repo and pick a quantization**
2. **Check tensor-type compatibility (dry-run)**
3. **Download the file**
4. **Convert in place to big-endian**
5. **Rename with `-be` suffix and write the README**
6. **Create the HF repo and upload** (with xet disabled)

Execute steps sequentially; do not skip the compatibility check — it saves a wasted multi-GB download when the converter doesn't yet support a tensor type used by the model.

## 1. Pick a quantization

Unsloth and similar publishers ship many quants per model. The converter supports only a subset of GGML tensor types: **F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4**. Any other type (Q2_K, Q3_K, Q5_0, Q5_1, IQ*, UD-*) will fail.

**Default: `Q4_K_M`** — matches the precedent set by `Ministral-3-8B-Instruct-2512` already in the collection and produces tensors in {Q4_K, Q6_K, F32}. Only deviate if the user names a specific quant or Q4_K_M is unavailable.

List what's available:

```bash
uv run python -c "
from huggingface_hub import HfApi
for s in HfApi().repo_info('<owner>/<repo>', files_metadata=True).siblings:
    if s.rfilename.endswith('.gguf'):
        print(f'{s.rfilename:55s} {s.size/(1024*1024):8.1f} MB')
"
```

## 2. Compatibility check

The converter script's own `--dry-run` validates types and aborts on the first unsupported tensor. Since `gguf.GGUFReader` must read the whole file, you have to download first (step 3) before you can dry-run. But also enumerate tensor types up front so you can foresee failures and plan a fix before committing to a huge download:

```bash
uv run python -c "
import gguf
r = gguf.GGUFReader('models/<name>/<file>.gguf', 'r')
from collections import Counter
print(Counter(t.tensor_type.name for t in r.tensors))
supported = {'F32','F16','BF16','Q4_0','Q8_0','Q4_K','Q5_K','Q6_K','MXFP4'}
bad = [(t.name, t.tensor_type.name) for t in r.tensors if t.tensor_type.name not in supported]
for n,k in bad: print('UNSUPPORTED:', n, k)
"
```

If any unsupported tensor types appear, extend `gguf_convert_endian.py` before running the conversion. See [references/extending_converter.md](references/extending_converter.md) for block layouts and the pattern.

## 3. Download

```bash
mkdir -p models/<name>
hf download <owner>/<repo> <quant-file>.gguf --local-dir models/<name>
```

## 4. Convert

The converter modifies the file in-place and prompts `YES, I am sure>` before writing. Pipe `YES` in:

```bash
echo "YES" | uv run gguf_convert_endian.py models/<name>/<quant-file>.gguf big
```

Verify:

```bash
uv run python -c "import gguf; print(gguf.GGUFReader('models/<name>/<quant-file>.gguf','r').endianess.name)"
# → BIG
```

## 5. Rename and write README

Add the `-be` suffix matching the collection convention:

```bash
mv models/<name>/<quant-file>.gguf models/<name>/<quant-file>-be.gguf
```

Write `models/<name>/README.md` using this template (fill `<license>`, `<base-model>`, `<name>`, `<quant>`):

```markdown
---
license: <license>
base_model:
  - <base-model>
tags:
- BE
---

Big Endian GGUF <name> (<quant>), converted from [<source-owner>/<source-repo>](https://huggingface.co/<source-owner>/<source-repo>) for use on IBM i with Llama.cpp.
```

`<base-model>` is the original upstream model (e.g. `google/gemma-4-E2B-it`), NOT the GGUF republisher. If uncertain, inspect the source repo's own README for its `base_model` frontmatter.

## 6. Create repo and upload

```bash
uv run python -c "
from huggingface_hub import HfApi
print(HfApi().create_repo('models-for-i/<name>', repo_type='model', exist_ok=True))
"
```

Then upload with **Xet disabled**:

```bash
HF_HUB_DISABLE_XET=1 hf upload models-for-i/<name> models/<name> \
  --repo-type model \
  --commit-message "Add big-endian <quant> converted from <source-owner>/<source-repo>" \
  --exclude ".cache/*"
```

**Why `HF_HUB_DISABLE_XET=1`:** the default Xet uploader hangs after partial uploads against this collection (sockets stall in CLOSE_WAIT, ~72% in; observed twice). Plain HTTPS/LFS uploads cleanly at 7-9 MB/s with no intervention needed. Always use the flag until the upstream xet-core bug is fixed.

**Verify the commit:**

```bash
uv run python -c "
from huggingface_hub import HfApi
info = HfApi().repo_info('models-for-i/<name>', files_metadata=True)
for s in info.siblings: print(f'{s.rfilename:45s} {s.size}')
"
```

## Troubleshooting

**`ValueError: Cannot handle type <QTYPE> for tensor '<name>'`** — the file uses a tensor type the converter doesn't support. Extend `gguf_convert_endian.py` per [references/extending_converter.md](references/extending_converter.md). The file is untouched at this point (the check runs before writes).

**Upload appears stuck at N% for 5+ minutes with xet enabled** — kill the process (`kill $(pgrep -f "python3.*hf upload")`) and retry with `HF_HUB_DISABLE_XET=1`. The xet cache dedups already-uploaded chunks, but the LFS path is more reliable here.

**Auth:** `hf auth whoami` must show `models-for-i` in `orgs`. If not, run `hf auth login` with a write-scope token.
