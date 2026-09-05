"""
Synthetic optimizer-step benchmark. Needs no dataset.

Times one full optimizer step over TOTAL_BATCH_SIZE tokens, exactly the way
train.py does it, so device batch size / gradient accumulation / compile can be
compared in seconds instead of waiting out a 5-minute run.

    uv run bench/micro.py                      # sweep device batch 16 and 32
    uv run bench/micro.py --batch 32 --steps 20
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

sys.path.insert(0, ".")

import train as T
from prepare import MAX_SEQ_LEN, VOCAB_SIZE


def build(depth, batch, accum, seed=0):
    mx.random.seed(seed)
    model_dim = ((depth * T.ASPECT_RATIO + T.HEAD_DIM - 1) // T.HEAD_DIM) * T.HEAD_DIM
    config = T.GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n_layer=depth,
        n_head=model_dim // T.HEAD_DIM,
        n_kv_head=model_dim // T.HEAD_DIM,
        n_embd=model_dim,
        window_pattern=T.WINDOW_PATTERN,
    )
    model = T.GPT(config)
    model.init_weights()
    mx.eval(model.parameters())

    optimizer = T.AdamW(
        model,
        unembedding_lr=T.UNEMBEDDING_LR,
        embedding_lr=T.EMBEDDING_LR,
        matrix_lr=T.MATRIX_LR,
        weight_decay=T.WEIGHT_DECAY,
        adam_betas=T.ADAM_BETAS,
        scalar_lr=T.SCALAR_LR,
    )
    xs = [mx.random.randint(0, VOCAB_SIZE, (batch, MAX_SEQ_LEN)).astype(mx.int32) for _ in range(accum)]
    ys = [mx.random.randint(0, VOCAB_SIZE, (batch, MAX_SEQ_LEN)).astype(mx.int32) for _ in range(accum)]
    mx.eval(xs, ys)
    return model, optimizer, xs, ys


def run(depth, batch, accum, compiled, steps, warmup=3):
    model, optimizer, xs, ys = build(depth, batch, accum)
    loss_grad_fn = nn.value_and_grad(model, lambda m, i, t: m(i, targets=t))
    state = [model.state, *optimizer.state]

    def full_step(xs, ys):
        loss = None
        accum_grads = None
        for i in range(accum):
            loss, grads = loss_grad_fn(model, xs[i], ys[i])
            accum_grads = grads if accum_grads is None else tree_map(lambda a, g: a + g, accum_grads, grads)
        if accum > 1:
            accum_grads = tree_map(lambda g: g * (1.0 / accum), accum_grads)
        optimizer.update(model, accum_grads)
        return loss

    step = mx.compile(full_step, inputs=state, outputs=state) if compiled else full_step

    times = []
    losses = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        loss = step(xs, ys)
        mx.eval(loss, state)
        dt = time.perf_counter() - t0
        losses.append(float(loss.item()))
        if i >= warmup:
            times.append(dt)

    mx.clear_cache()
    peak_gb = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()
    times.sort()
    median = times[len(times) // 2]
    tokens = batch * accum * MAX_SEQ_LEN
    return {"median_s": median, "tok_per_s": tokens / median, "peak_gb": peak_gb, "loss_last": losses[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=T.DEPTH)
    parser.add_argument("--batch", type=int, default=None, help="single device batch; default sweeps 16 and 32")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--total-batch", type=int, default=T.TOTAL_BATCH_SIZE)
    args = parser.parse_args()

    info = mx.device_info()
    print(f"{info['device_name']}  mlx {mx.__version__}  "
          f"working_set {info['max_recommended_working_set_size'] / 1024**3:.1f} GB  "
          f"max_buffer {info['max_buffer_length'] / 1024**3:.1f} GB")
    print(f"depth={args.depth}  seq={MAX_SEQ_LEN}  total_batch={args.total_batch} tokens/step  "
          f"median of {args.steps} steps\n")

    batches = [args.batch] if args.batch else [16, 32]
    header = (f"{'dev batch':>10} {'accum':>6} {'mode':>9} {'s/step':>9} "
              f"{'tok/s':>12} {'peak GB':>9} {'loss[-1]':>10}")
    print(header)
    print("-" * len(header))

    results = {}
    for batch in batches:
        tokens_per_fwdbwd = batch * MAX_SEQ_LEN
        if args.total_batch % tokens_per_fwdbwd != 0:
            continue
        accum = args.total_batch // tokens_per_fwdbwd
        for compiled in (False, True):
            r = run(args.depth, batch, accum, compiled, args.steps)
            mode = "compiled" if compiled else "eager"
            print(f"{batch:>10} {accum:>6} {mode:>9} {r['median_s']:>9.4f} {r['tok_per_s']:>12,.0f} "
                  f"{r['peak_gb']:>9.2f} {r['loss_last']:>10.4f}")
            results[(batch, compiled)] = r
        print()

    ref = results.get((16, False))
    if ref:
        best_key, best = max(results.items(), key=lambda kv: kv[1]["tok_per_s"])
        print(f"best: device batch {best_key[0]} {'compiled' if best_key[1] else 'eager'} "
              f"-> {best['tok_per_s'] / ref['tok_per_s']:.2f}x tok/s vs the upstream default "
              f"(device batch 16, eager)")


if __name__ == "__main__":
    main()
