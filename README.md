# HoVer Inference Engine

*HoVer*, an inference scheduling system for multimodal LLM serving.

## Included

- Modality-aware Partitioner for soft boundary-aligned prefill segments.
- Resident Expert Prefetcher with history-aware demand prediction,
  locality-aware resident selection, and route-exact transfer overlap.
- Horizontal Scheduler for request-side PD batch selection.
- Vertical Scheduler for static stage grouping and dynamic stage advancement.
- DeepSeek-VL2 and Kimi-VL model adapters.
- CUDA kernels and the minimal HTTP inference server.

The Resident Expert Prefetcher keeps history-aware prediction,
locality-aware selection, and route-exact transfer overlap in one module.

## Requirements

- Linux with NVIDIA CUDA 12.8.
- Python 3.10 or newer.
- PyTorch 2.8.0.
- A compatible NVIDIA GPU and CUDA compiler toolchain.

The bundled `flash-attention.patch` targets PyTorch 2.8.0, CUDA 12.8,
and `vllm-project/flash-attention` commit `d9e577e`. Use it only for a source
build:

```bash
git clone https://github.com/vllm-project/flash-attention.git
cd flash-attention
git checkout d9e577e
patch -p0 < /absolute/path/to/HoVer/flash-attention.patch
```

`requirements.txt` otherwise installs the packaged `vllm-flash-attn` build.

## DeepSeek-VL2 dependency

The DeepSeek adapter uses the upstream `deepseek_vl2` Python package. Install
the official DeepSeek-VL2 source tree and expose its repository root:

```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
export DEEPSEEK_VL2_PATH=/absolute/path/to/DeepSeek-VL2
```

The variable is only required when the selected model needs the external
DeepSeek-VL2 processor or vision implementation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Running

The default server profile matches the paper's DeepSeek-VL2-Small and
VisionArena base experiment: a 4,096-token shared batch budget, 40 active
sequences, four vertical stages, 32 logical-persistent expert-cache slots,
tensor parallelism of two, and HoVer route-exact transfer overlap.

```bash
python -m nanovllm.entrypoints.api_server \
  --model /absolute/path/to/model \
  --trust-remote-code
```

Run `python -m nanovllm.entrypoints.api_server --help` for the complete runtime
configuration. The HTTP server is a research interface and is not hardened for
production deployment.

## Paper evaluation profiles

All reported configurations use BF16, four warmup requests followed by 200
measured requests, a maximum of 128 output tokens, a shared 4,096-token batch
budget, tensor parallelism of two, data seed 43, arrival seed 1043, a 50%
urgent-request ratio, 512-token base prefill chunks, and deterministic manifest
replay.  The urgent and regular TTFT SLOs are attached to individual benchmark
requests; `--ttft-slo-ms` supplies the server-side fallback.

| Evaluation | Model and workload | RPS | TTFT SLOs | TBT SLO | Cache | Max seqs | Stages | Max length |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | DeepSeek-VL2-Small, VisionArena | 1.0 | 10/25 s | 1000 ms | 32 | 40 | 4 | 4096 |
| Multi-rate | DeepSeek-VL2-Small, VisionArena | 0.5/1.5/2.0 | 10/25 s | 1000 ms | 32 | 40 | 4 | 4096 |
| Dataset | DeepSeek-VL2-Small, Mantis-Eval | 1.0 | 18/34 s | 1000 ms | 32 | 32 | 12 | 4096 |
| Model | Kimi-VL-A3B-Instruct, MMLongBench-Doc | 0.2 | 20/45 s | 1500 ms | 40 | 32 | 12 | 8192 |

The base and multi-rate profiles use `alpha=0.5`, `KH=4`, a pinned-resident
ratio of 0.875, and 16 prefill candidates.  The Mantis-Eval and Kimi profiles
use `alpha=0.8`, `KH=5`, a pinned-resident ratio of 0.5, and one prefill
candidate.  `theta=5`, `TTLmax=5`, expert-transition decay 0.95, and `Jmax=0`
are shared by all profiles.  Route-exact transfer overlap is a HoVer component
and is not enabled for comparison schedulers.

Expert-cache capacities in this prototype bound the logical-persistent set that
can be reused across forwards.  They do not describe a compact physical tensor
containing only 32 or 40 experts.

## License

The project is distributed under AGPL-3.0. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
