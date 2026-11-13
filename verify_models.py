"""Replay published initialization recipes against independently fetched checkpoint rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import random
import struct
import tempfile
import time
from pathlib import Path
import urllib.request

MANIFEST = Path(__file__).with_name("vendor") / "models.json"

V4_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
V4_BASE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/{V4_REVISION}/"
V4_TRAINED_SEED = 2133857753
V4_LAYERS = range(1, 43)
V4_SHAPE = (2048, 4096)
V4_SCALE_SHAPE = (16, 32)
V4_COUNT = V4_SHAPE[0] * V4_SHAPE[1]
V4_SIZE = V4_COUNT + V4_SCALE_SHAPE[0] * V4_SCALE_SHAPE[1]


def get(url, start=None, end=None):
    headers = {"Accept-Encoding": "identity", "User-Agent": "checkpoint-forensics-verification/1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
        # Distinct URLs prevent intermediaries from replaying cached ranges.
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}replay_range={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        if start is None:
            return response.read()
        expected = f"bytes {start}-{end}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError(f"Unexpected response for {expected}: {response.status}, {response.headers.get('Content-Range')}")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError(f"Unexpected byte count for {expected}")
        return data


def fetch(model, cache):
    name = model["name"]
    revision = model["revision"]
    tensor = model["tensor"]
    height, width = model["shape"]
    rows = model["rows"]
    digest = model["sha256"]
    path = cache / f"{name}-{revision}.bf16" if cache is not None else None
    if path is not None and path.exists():
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise RuntimeError(f"{path}: checkpoint row hash differs: {actual}")
        return data, {"model": name, "revision": revision, "tensor": tensor,
                      "rows": len(rows), "sha256": actual}
    repo = model["repo"]
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    shard = json.loads(get(base + "model.safetensors.index.json"))["weight_map"][tensor]
    url = base + shard
    header_size = struct.unpack("<Q", get(url, 0, 7))[0]
    if not 0 < header_size <= 100_000_000:
        raise RuntimeError(f"Invalid safetensors header size: {header_size}")
    entry = json.loads(get(url, 8, 7 + header_size))[tensor]
    if entry["dtype"] != "BF16" or entry["shape"] != [height, width]:
        raise RuntimeError(f"Unexpected tensor metadata: {entry}")
    start, end = entry["data_offsets"]
    stride = width * 2
    if start < 0 or end - start != height * stride:
        raise RuntimeError(f"Unexpected tensor offsets: {entry}")
    data_start = 8 + header_size + start
    ranges = []
    for _, group in itertools.groupby(enumerate(rows), key=lambda item: item[1] - item[0]):
        span = [row for _, row in group]
        ranges.append((data_start + span[0] * stride, data_start + (span[-1] + 1) * stride - 1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        data = b"".join(pool.map(lambda bounds: get(url, *bounds), ranges))
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{name}: checkpoint row hash differs: {actual}")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, {"model": name, "revision": revision, "shard": shard,
                  "tensor": tensor, "rows": len(rows), "sha256": actual}


def shards(model):
    height, _ = model["shape"]
    shard_height = (height + model.get("tp", 1) - 1) // model.get("tp", 1)
    by_rank = {}
    for i, row in enumerate(model["rows"]):
        rank, local_row = divmod(row, shard_height)
        by_rank.setdefault(rank, []).append((i, local_row))
    for rank, positions in by_rank.items():
        output_indices, local_rows = zip(*positions)
        yield (output_indices, local_rows, min(shard_height, height - rank * shard_height),
               model["seed"] + rank * model.get("seed_per_rank", 0))


def summarize(np, model, observed, predicted, shift, sms, exact=None):
    x = predicted.astype(np.float64)
    y = observed.astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    r = (x * y).sum(axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    threshold = model.get("thr", 0.99)
    result = {
        "layout_sms": sms,
        "base_seed": model["seed"] + shift,
        "threshold": threshold,
        "rows_above_threshold": int((r > threshold).sum()),
        "median_r": float(np.median(r)),
    }
    if exact is not None:
        result["bit_exact_rows"] = int(exact.sum())
        result["exact_row_indices"] = [row for row, match in zip(model["rows"], exact) if match]
    return result


def cuda_replay(np, torch, device, sms, model, observed, bits):
    width = model["shape"][1]
    dtype = getattr(torch, model["dtype"])
    results = []
    for shift in (0, 1):
        predicted = np.empty_like(observed)
        exact = np.zeros(len(observed), dtype=bool)
        for indices, local_rows, height, seed in shards(model):
            generator = torch.Generator(device=device).manual_seed(seed + shift)
            generator.set_offset(model.get("cpu_offset", 0))
            tensor = torch.empty((height, width), device=device, dtype=dtype)
            tensor.normal_(0, model.get("std", 1), generator=generator)
            selected = tensor.index_select(0, torch.tensor(local_rows, device=device)).cpu()
            del tensor
            predicted[list(indices)] = selected.float().numpy()
            generated_bits = selected.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            exact[list(indices)] = (generated_bits == bits[list(indices)]).all(axis=1)
        results.append(summarize(np, model, observed, predicted, shift, sms, exact))
    return results


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def fetch_trained_weights(cache):
    index = json.loads(get(V4_BASE + "model.safetensors.index.json"))["weight_map"]
    cache.mkdir(parents=True, exist_ok=True)

    def fetch_layer(layer):
        name = f"layers.{layer}.ffn.shared_experts.w1"
        shard = index[name + ".weight"]
        if index[name + ".scale"] != shard:
            raise ValueError(f"Weight and scale for {name} are in different shards")
        url = V4_BASE + shard
        size = struct.unpack("<Q", get(url, 0, 7))[0]
        if not 0 < size <= 100_000_000:
            raise ValueError(f"Unexpected safetensors header size for {shard}")
        header = json.loads(get(url, 8, 7 + size))
        parts = []
        for suffix, dtype, shape in (
            ("weight", "F8_E4M3", V4_SHAPE), ("scale", "F8_E8M0", V4_SCALE_SHAPE)
        ):
            entry = header[f"{name}.{suffix}"]
            start, end = entry["data_offsets"]
            if (entry["dtype"], tuple(entry["shape"]), end - start) != (dtype, shape, shape[0] * shape[1]):
                raise ValueError(f"Unexpected tensor metadata for {name}.{suffix}: {entry}")
            parts.append(get(url, 8 + size + start, 7 + size + end))
        payload = b"".join(parts)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
            output.write(payload)
        Path(output.name).replace(cache / f"{layer}.bin")
        return dict(layer=layer, tensor=name, shard=shard, revision=V4_REVISION,
                    sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_layer, V4_LAYERS))
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as output:
        json.dump(records, output)
    Path(output.name).replace(cache / "manifest.json")
    return records


def dequantize(torch, payload, device):
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
    weight = raw[:V4_COUNT].view(torch.float8_e4m3fn).float().reshape(16, 128, 32, 128)
    scale = torch.exp2(raw[V4_COUNT:].float().reshape(V4_SCALE_SHAPE) - 127)
    return (weight * scale[:, None, :, None]).flatten()


def read_trained_payload(cache, record):
    layer = record["layer"]
    payload = (cache / f"{layer}.bin").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if (len(payload), digest, record["revision"]) != (V4_SIZE, record["sha256"], V4_REVISION):
        raise ValueError(f"Cache validation failed for layer {layer}")
    return payload


def replay_trained_weights(torch, device, cache, records):
    started = time.monotonic()
    controls = random.Random(20261113)
    seeds = [V4_TRAINED_SEED, V4_TRAINED_SEED + 1]
    while len(seeds) < 22:
        candidate = controls.randrange(2**32)
        if candidate not in seeds:
            seeds.append(candidate)
    emit(dict(check="trained_weights", revision=V4_REVISION, layers=list(V4_LAYERS),
              seeds=seeds, shape=V4_SHAPE, offset=0, score="Pearson r on dequantized weights"))

    observed = torch.empty((len(V4_LAYERS) + 1, V4_COUNT), device=device, dtype=torch.float64)
    for row, record in zip(observed, records):
        row.copy_(dequantize(torch, read_trained_payload(cache, record), device))
    observed[-1] = observed[:-1].mean(dim=0)
    observed -= observed.mean(dim=1, keepdim=True)
    observed /= torch.linalg.vector_norm(observed, dim=1, keepdim=True)

    for dtype in (torch.float32, torch.bfloat16):
        for seed in seeds:
            rng = torch.Generator(device=device).manual_seed(seed)
            sample = torch.empty(V4_SHAPE, device=device, dtype=dtype).normal_(generator=rng)
            sample = sample.double().flatten()
            sample -= sample.mean()
            sample /= torch.linalg.vector_norm(sample)
            scores = (observed @ sample).tolist()
            emit(dict(check="trained_weights", dtype=str(dtype), seed=seed, reported_seed=seed == V4_TRAINED_SEED,
                 layer_r=dict(zip(V4_LAYERS, scores[:-1])), layer_average_r=scores[-1]))
    emit(dict(check="trained_weights", elapsed_s=round(time.monotonic() - started, 3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "cuda"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".rows"))
    parser.add_argument("--device", type=int, default=0)
    models = json.loads(MANIFEST.read_text())
    parser.add_argument("--model", action="append", choices=[m["name"] for m in models])
    parser.add_argument("--check", choices=("all", "embeddings", "trained"), default="all",
                        help="Select checks to run (default: all)")
    args = parser.parse_args()
    if args.check == "trained" and args.model and args.model != ["DeepSeek-V4-Flash"]:
        parser.error("The trained-weight check supports only DeepSeek-V4-Flash")

    if args.mode == "cuda":
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA PyTorch and an assigned GPU are required")
        device = torch.device("cuda", args.device)
        props = torch.cuda.get_device_properties(device)
        emit({"backend": "stock_pytorch_cuda", "gpu": props.name,
              "sm_count": props.multi_processor_count, "torch": str(torch.__version__),
              "cuda": torch.version.cuda,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    embedding_models = models if args.check != "trained" else []
    for model in embedding_models:
        if args.model and model["name"] not in args.model:
            continue
        data, evidence = fetch(model, args.cache_dir)
        evidence["repo"] = model["repo"]
        evidence["expected_sms"] = model["sms"]
        if args.mode != "fetch":
            bits = np.frombuffer(data, dtype="<u2").reshape(len(model["rows"]), model["shape"][1])
            observed = (bits.astype(np.uint32) << 16).view(np.float32)
            if model["dist"] == "normal":
                evidence["replays"] = cuda_replay(np, torch, device, props.multi_processor_count,
                                                   model, observed, bits)
            else:
                evidence["skipped"] = "Stock CUDA initializer recipe unavailable"
        emit(evidence)

    if args.check != "embeddings" and (not args.model or "DeepSeek-V4-Flash" in args.model):
        cache = args.cache_dir / "v4-trained" / V4_REVISION
        manifest = cache / "manifest.json"
        records = (json.loads(manifest.read_text()) if manifest.exists()
                   else fetch_trained_weights(cache))
        if [record["layer"] for record in records] != list(V4_LAYERS):
            raise ValueError("Unexpected layer list in cache manifest")
        for record in records:
            if args.mode == "fetch":
                read_trained_payload(cache, record)
            emit({"check": "trained_weights", **record})
        if args.mode == "cuda":
            replay_trained_weights(torch, device, cache, records)


if __name__ == "__main__":
    main()
