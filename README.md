# Verifying results in "Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs"

This project checks the results reported by [Will DePue](https://unemploymentlabs.com/checkpoint-forensics) ([study repository](https://github.com/macrogrants/checkpoint-forensics)). We verify the following results from the original study:
- **The published weights match the weights used in the study.** We downloaded the same rows used in the study from public model checkpoints on Hugging Face. All of these samples matched the hashes recorded in the study.
- **We reproduced the reported starting-weight patterns for three of the eleven models from the study.** We reproduced the study’s embedding matches for DeepSeek-V3-Base on a H200 and DeepSeek-V4-Flash on a B200. We also reproduced the same embedding correlations for MiniMax-M3 on B200. Changing the seed or GPU layout removed these matches/correlations, as expected. We did not obtain full coverage of the models used in the study. Seven of the remaining eight models lacked public samples or settings needed to replay the study on GPUs. Fully testing the one remaining model (Kimi K3) required a 152-SM GPU, which we lacked.
- **The starting-weight pattern is still detectable in the trained weights of DeepSeek-V4-Flash.** Across 42 V4-Flash layers, the correlation of the averaged weights across all weight positions was *r* = 0.065 on B200 and 0.0033 on H200. The study scored a subset of positions and reported r = 0.066 on Blackwell and 0.001 on Hopper.

## Summarized results

### DeepSeek

| Model | Released | Reported GPU layout | Do the published weights match? | Do the starting-weight patterns reproduce on the reported GPU? | Is the match unique to this GPU and seed? | Does the pattern persist in the trained weights? |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V3-Base | Dec 2024 | Hopper · 132 SMs | **✓** [Hash matched](#checkpoint-bytes) (8 rows) | **✓** [H200](evidence/h200.jsonl): 8/8 bit-exact | **✓** [H200, different seed](evidence/h200.jsonl): 0/8 · **✓** [B200](evidence/b200.jsonl): 0/8 | N/A[^trained] |
| DeepSeek-V4-Flash | Apr 2026 | B200/B300 · 148 SMs | **✓** [Hash matched](#checkpoint-bytes) (469 rows) | **✓** [B200](evidence/b200.jsonl): 445/469 exact, 461/469 with r > 0.99 | **✓** [H200](evidence/h200.jsonl): 0/469 · **✓** [B200, different seed](evidence/b200.jsonl): 0/469 | **✓** [B200 r = 0.06498, H200 r = 0.00334](#deepseek-v4-flash-trained-weights) |
| DeepSeek-V4-Pro | Apr 2026 | GB200/GB300 · 152 SMs | N/A[^untested] | N/A[^untested] | N/A[^untested] | N/A[^trained] |

### Moonshot AI

| Model | Released | Reported GPU layout | Do the published weights match? | Do the starting-weight patterns reproduce on the reported GPU? | Is the match unique to this GPU and seed? | Does the pattern persist in the trained weights? |
| --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.5 | Jan 2026 | Hopper · 132 SMs | N/A[^untested] | N/A[^untested] | N/A[^untested] | N/A[^trained] |
| Kimi K3 | Jul 2026 | GB200/GB300 · 152 SMs | **✓** [Hash matched](#checkpoint-bytes) (13 rows) | N/A[^untested] | **✓** [H200](evidence/h200.jsonl): 0/13 · **✓** [B200](evidence/b200.jsonl): 0/13 | N/A[^trained] |
| Kimi K3.1-Pro | Nov 2026 | GB200/GB300 · 152 SMs | N/A[^untested] | N/A[^untested] | N/A[^untested] | N/A[^trained] |

### MiniMax

| Model | Released | Reported GPU layout | Do the published weights match? | Do the starting-weight patterns reproduce on the reported GPU? | Is the match unique to this GPU and seed? | Does the pattern persist in the trained weights? |
| --- | --- | --- | --- | --- | --- | --- |
| MiniMax-M2.7 | Apr 2026 | Hopper · 132 SMs | N/A[^untested] | N/A[^untested] | N/A[^untested] | N/A[^trained] |
| MiniMax-M3 | Jun 2026 | B200/B300 · 148 SMs | **✓** [Hash matched](#checkpoint-bytes) (17 rows) | **✓** [B200](evidence/b200.jsonl): 17/17 with r > 0.99, 0/17 exact | **✓** [H200](evidence/h200.jsonl): 0/17 · **✓** [B200, different seed](evidence/b200.jsonl): 0/17 | N/A[^trained] |

### Ant Group

| Model | Released | Reported GPU layout | Do the published weights match? | Do the starting-weight patterns reproduce on the reported GPU? | Is the match unique to this GPU and seed? | Does the pattern persist in the trained weights? |
| --- | --- | --- | --- | --- | --- | --- |
| Ling-3.0-flash | Aug 2026 | Hopper · 132 SMs | N/A[^untested] | N/A[^untested] | N/A[^untested] | N/A[^trained] |
| Ling-3.1-flash-base | Oct 2026 | GB200/GB300 · 152 SMs | **✓** [Hash matched](#checkpoint-bytes) (384 rows) | N/A[^untested] | N/A[^untested] | N/A[^trained] |

## How the original study works
Model weights are numbers that training adjusts. During training, weights are initialized using a "seed", which is a number used to generate random values for the weights. These weights are then divided across the GPU's individual processing units, called streaming multiprocessors (SMs). GPUs (e.g. B200 vs H200) often have different numbers of SMs. This means different GPUs place the same initial weight values in different positions. This pattern can be used to determine the GPU that was used to train a given set of model weights, using only the weights themsleves. The study uses the following procedure:

1. **Find weights that are unlikely to change much in training.** These weights can be used to figure out what the starting seed was. For example, weights corresponding to rarely-used words or symbols likely receive few updates during training. However, it is unclear from the original study how these weights were chosen..
2. **Search for matches between these weights and particular seeds/GPU layouts.**  For each combination of seed/GPU layout, predict the starting values at the weight positions identified by (1), and compare them with the values for the published weights for the model.
3. **Test each match on other weights in the model.** Keeping the matched seed and GPU layout fixed, test on the other weights of the model. These weights will have updated more during training, so the study looks for a statistically relevant match. Matches indicate that this seed and GPU layout were very likely those used to train the model.

## What we check

Using H200 and B200 GPUs, we run the following checks:

- **Did the study use the real published model weights?** We download the developers’ checkpoints from Hugging Face. We then compare their digital fingerprints (SHA-256 hashes) with the weights reported in the study.
- **When weights are loaded onto the GPUs from the study, do the weights positions match those reported by the study?** On H200 and B200 GPUs, we generate fresh starting weight values from the reported seed and settings. We then confirm that these match the corresponding values in the study.
- **Are these matches unique to each particular GPU and seed?** We test whether the same seed produces a match on a GPU with a different number of processing units. We also test whether different seeds on the same GPU produces a match.
- **Does this pattern persist in the trained weights?** We regenerate starting values for DeepSeek-V4-Flash’s (used because the study names the trained weights and seed used) shared-expert layers and measure how closely they vary with the published weights on B200 and H200.

## Full results

### Checkpoint bytes

| Checkpoint | Pinned revision | Tensor | Selected rows | SHA-256 |
| --- | --- | --- | ---: | --- |
| DeepSeek-V3-Base | [`afb92e1`](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/afb92e1fa402c2be2a9eb085312bb02e0384d6c7) | `model.embed_tokens.weight` | 8 | `bdf7bd8a2838851d88ba47e29713752003cdff02d7dfc50e22725bbbc49a6ee9` |
| DeepSeek-V4-Flash | [`60d8d70`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1) | `embed.weight` | 469 | `3ed8cd8456e71888ad9806bf82cc9cd8f6eca611d5032541f3f5b3801e99f116` |
| Kimi-K3 | [`f831ab6`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | `language_model.model.embed_tokens.weight` | 13 | `286fbc06716270f53ce7deb34d7c6ef1e8dd554fe50ab80bc9d85b6c1d95d117` |
| MiniMax-M3 | [`d95e30c`](https://huggingface.co/MiniMaxAI/MiniMax-M3/tree/d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9) | `language_model.model.embed_tokens.weight` | 17 | `b7b0bb3f28da67df928f45311908125b3913be78421a238dfc507231c7a351f7` |
| Ling-3.1-flash-base | [`c2cad79`](https://huggingface.co/inclusionAI/Ling-3.1-flash-base/tree/c2cad79e46fcb7b46d96f9cf876ce1f2418b6e64) | `model.word_embeddings.weight` | 384 | `19a20bd00b6ad971abda0b2abb76f8b89da3e5518696b8f8e7bf7e8d2186ed9c` |

### GPU replay

The table counts rows that match exactly and rows with Pearson correlation above 0.99.

| Run | Checkpoint | Seed | Bit-exact rows | Rows with Pearson r > 0.99 | Median r |
| --- | --- | ---: | ---: | ---: | ---: |
| H200 NVL, 132 SMs | DeepSeek-V3-Base | 953049583 | 8/8 | 8/8 | +1.000000 |
| H200 NVL, 132 SMs | DeepSeek-V3-Base | 953049584 | 0/8 | 0/8 | -0.003800 |
| H200 NVL, 132 SMs | DeepSeek-V4-Flash | 953049583 | 0/469 | 0/469 | -0.000432 |
| H200 NVL, 132 SMs | DeepSeek-V4-Flash | 953049584 | 0/469 | 0/469 | +0.000064 |
| H200 NVL, 132 SMs | Kimi-K3 | 20256586 | 0/13 | 0/13 | -0.002129 |
| H200 NVL, 132 SMs | Kimi-K3 | 20256587 | 0/13 | 0/13 | -0.001367 |
| H200 NVL, 132 SMs | MiniMax-M3 | 2760 | 0/17 | 0/17 | -0.009525 |
| H200 NVL, 132 SMs | MiniMax-M3 | 2761 | 0/17 | 0/17 | +0.005365 |
| H200 NVL, 132 SMs | Ling-3.1-flash-base | — | Skipped | — | — |
| B200, 148 SMs | DeepSeek-V3-Base | 953049583 | 0/8 | 0/8 | +0.004358 |
| B200, 148 SMs | DeepSeek-V3-Base | 953049584 | 0/8 | 0/8 | +0.004390 |
| B200, 148 SMs | DeepSeek-V4-Flash | 953049583 | 445/469 | 461/469 | +0.999999 |
| B200, 148 SMs | DeepSeek-V4-Flash | 953049584 | 0/469 | 0/469 | +0.001582 |
| B200, 148 SMs | Kimi-K3 | 20256586 | 0/13 | 0/13 | +0.008490 |
| B200, 148 SMs | Kimi-K3 | 20256587 | 0/13 | 0/13 | +0.002480 |
| B200, 148 SMs | MiniMax-M3 | 2760 | 0/17 | 17/17 | +0.999990 |
| B200, 148 SMs | MiniMax-M3 | 2761 | 0/17 | 0/17 | +0.001724 |
| B200, 148 SMs | Ling-3.1-flash-base | — | Skipped | — | — |

The B200 replay of DeepSeek-V4-Flash matched the study’s B300 counts of 445 exact rows and 461 rows with r > 0.99. Both verification runs used PyTorch 2.9.1 with CUDA 12.8.

[H200 replay log](evidence/h200.jsonl) · [B200 replay log](evidence/b200.jsonl) · [Original GPU results](https://unemploymentlabs.com/checkpoint-forensics#verifying-the-findings)

### DeepSeek-V4-Flash trained weights

We tested the shared-expert `w1` weights in all 42 layers plotted in Figure 5 using the study’s reported seed, `2133857753`. We downloaded the quantized tensors and their scales from the public checkpoint, dequantized the weights, and scored every position. Figure 5 scores a subset of positions. Its correlations for the weights averaged across layers were 0.066 on Blackwell and 0.001 on Hopper. [Study Figure 5](https://unemploymentlabs.com/checkpoint-forensics#figure-5)

| GPU | Pearson r for weights averaged across layers | Range of individual-layer r | Largest absolute r for weights averaged across layers with a different seed |
| --- | ---: | ---: | ---: |
| H200, 132 SMs | 0.003345 | −0.000237 to 0.001274 | 0.000664 |
| B200, 148 SMs | 0.064976 | 0.003637 to 0.017780 | 0.000748 |

The table uses FP32 initialization. BF16 gave nearly identical results. The reported seed scored higher on B200 than H200 in all 42 layers. H200 showed a small positive correlation above all 21 alternate seeds, so the results show a much stronger match on B200 and a smaller signal on H200. The alternate seeds were the reported seed plus one and 20 fixed random seeds.

[Checkpoint fetch log](evidence/v4-trained-fetch.jsonl) · [H200 log](evidence/v4-trained-h200.jsonl) · [B200 log](evidence/v4-trained-b200.jsonl)

## Run the verification

Use Python 3.11 or 3.12 on a machine with an NVIDIA GPU and a driver compatible with CUDA 12.8. [uv installation](https://docs.astral.sh/uv/getting-started/installation/) · [PyTorch wheels](https://docs.pytorch.org/get-started/previous-versions/)

```bash
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
uv pip install numpy
```

Fetch the checkpoint samples once into a cache accessible to both GPUs. This step runs without a GPU. Use a separate output filename for each replay.

```bash
.venv/bin/python verify_models.py fetch > checkpoint-fetch.jsonl
```

Run the same verification command on H200 and B200. The default covers every supported embedding and trained-weight check. For a machine without a shared cache, CUDA mode downloads missing samples.

```bash
# On H200
.venv/bin/python verify_models.py cuda > replay-h200.jsonl
# On B200
.venv/bin/python verify_models.py cuda > replay-b200.jsonl
```

Use `--check embeddings` or `--check trained` to run either check on its own. The evidence logs preserve the output of the original runs. The current verifier runs both checks through the same entry point. [Verification script](verify_models.py) · [Checkpoint manifest](vendor/models.json) · [Upstream license](vendor/LICENSE)

[^untested]: N/A means we did not run that check. For DeepSeek-V4-Pro, the public replay package omits the weight sample and test settings, and reproducing the reported match requires a 152-SM GPU. For Kimi K2.5, MiniMax-M2.7, Ling-3.0-flash, and Kimi K3.1-Pro, the package omits the row lists and full replay settings. A matching replay for Kimi K3.1-Pro and Kimi K3 also requires a 152-SM GPU. For Ling-3.1-flash-base, CUDA replay and controls need initializer settings. [Published replay package](https://github.com/macrogrants/checkpoint-forensics)

[^trained]: The trained-weight check covers only DeepSeek-V4-Flash. The [method](#what-we-check) explains the inputs and GPU layouts needed for this check.
