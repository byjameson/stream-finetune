"""Raw-boundary binding: no separator token, learned statement ends.

Statements are `name has (big)*k color object` and the query is
`who has color object name color <eos>`, concatenated without any separator.
A learned write gate marks the last object token of a statement; a learned read
gate marks the query object. The readout is factorized color x object attention,
plus a query-color copy for the second answer token.

    python research/spiking_binding_raw_boundary.py --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from research.spiking_binding import COLORS, NAMES, OBJECTS
from streamft.spiking import SpikingConfig, SpikingLM

ROOT = Path(__file__).resolve().parents[1]
NAMES8 = NAMES + ["zoe", "max", "ben", "ava"]
WORDS = NAMES8 + COLORS + OBJECTS + ["has", "who", "<eos>", "big", "<pad>"]
W2I = {w: i for i, w in enumerate(WORDS)}
VOCAB = len(WORDS)
N_STMT = 8
MAX_LEN = N_STMT * 7 + 7  # 63
PAD = W2I["<pad>"]


def sample_story(allowed, rng, filler):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES8}
    for _ in range(2000):
        names = rng.sample(NAMES8, N_STMT)
        if any(not by_name[n] for n in names):
            continue
        stmts = [rng.choice(by_name[n]) for n in names]
        rng.shuffle(stmts)
        q = rng.choice(stmts)
        if any(s != q and s[1] == q[1] and s[2] == q[2] for s in stmts):
            continue
        toks = []
        for n, c, o in stmts:
            toks += [n, "has"] + ["big"] * filler + [c, o]
        toks += ["who", "has", q[1], q[2], q[0], q[1], "<eos>"]
        assert len(toks) <= MAX_LEN
        toks += ["<pad>"] * (MAX_LEN - len(toks))
        return [W2I[t] for t in toks]
    raise RuntimeError("could not build relaxed story")


def make_data(n, allowed, seed, filler):
    rng = random.Random(seed)
    return torch.tensor([sample_story(allowed, rng, filler)
                         for _ in range(n)], dtype=torch.long)


def permute_relaxed(test):
    out = test.clone()
    for i in range(len(test)):
        row = test[i].tolist()
        valid = [t for t in row if t != PAD]
        who = valid.index(W2I["who"])
        block = who // N_STMT
        stmts = [valid[j * block:(j + 1) * block] for j in range(N_STMT)]
        query = valid[N_STMT * block:]
        new = [t for b in reversed(stmts) for t in b] + query
        new = new[:MAX_LEN] + [PAD] * max(0, MAX_LEN - len(new))
        out[i] = torch.tensor(new, dtype=torch.long)
    return out


def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.to(device))
        who = (data.to(device) == W2I["who"]).float()
        idx = torch.arange(MAX_LEN, device=device).float()
        who_pos = (who * idx).max(dim=1).values.long()
        qpos = who_pos + 3
        block = who_pos // N_STMT
        ar = torch.arange(len(data), device=device)
        pred_name = out[ar, qpos].argmax(-1)
        pred_color = out[ar, qpos + 1].argmax(-1)
        tgt_name = data.to(device)[ar, qpos + 1]
        tgt_color = data.to(device)[ar, qpos + 2]
        name_acc = (pred_name == tgt_name).float().mean().item()
        color_acc = (pred_color == tgt_color).float().mean().item()
        exact = (pred_name == tgt_name) & (pred_color == tgt_color)
        acc = exact.float().mean().item()
        by_pos = {}
        for i in range(N_STMT):
            mask = torch.zeros(len(data), dtype=torch.bool, device=device)
            for j in range(len(data)):
                row = data[j].tolist()
                b = int(block[j])
                starts = [k * b for k in range(N_STMT)]
                names = [row[s] for s in starts]
                if int(tgt_name[j]) in names:
                    mask[j] = names.index(int(tgt_name[j])) == i
            if int(mask.sum()) > 0:
                by_pos[f"answer_at_stmt{i}"] = {
                    "n": int(mask.sum()),
                    "acc": float(exact[mask].float().mean()),
                }
        counts = pred_name.cpu().bincount(minlength=VOCAB).tolist()
    model.train()
    return {"accuracy": acc, "name_acc": name_acc, "color_acc": color_acc,
            "by_answer_pos": by_pos,
            "pred_counts": {WORDS[i]: c for i, c in enumerate(counts) if c}}


def eval_set(model, data_by_filler, device):
    rows = {str(f): evaluate(model, data, device)
            for f, data in data_by_filler.items()}
    mean = sum(r["accuracy"] for r in rows.values()) / max(len(rows), 1)
    return {"by_filler": rows, "mean": mean}


def train_phase(model, new_data, old_data, steps, batch, lr, device, amp,
                replay=0.0, fillers=(0, 1, 2)):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    n_old = int(round(batch * replay))
    n_new = batch - n_old
    for _ in range(steps):
        f = random.choice(list(fillers))
        idx = torch.randint(0, len(new_data[f]), (n_new,))
        parts = [new_data[f][idx].to(device)]
        if n_old:
            idx_old = torch.randint(0, len(old_data[f]), (n_old,))
            parts.append(old_data[f][idx_old].to(device))
        x = torch.cat(parts, dim=0)
        x = x[torch.randperm(len(x), device=device)]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            mask = x[:, 1:] != PAD
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                 x[:, 1:].reshape(-1), reduction="none")
            ce = ce.reshape(x.size(0), -1)
            loss = (ce * mask).sum() / mask.sum().clamp_min(1)
            who = (x == W2I["who"]).float()
            idx = torch.arange(x.size(1), device=device).float()
            who_pos = (who * idx).max(dim=1).values.long()
            qpos = who_pos + 3
            block = who_pos // N_STMT
            ar = torch.arange(x.size(0), device=device)
            if model.last_copy_logits is not None:
                target = x[ar, qpos + 1]
                loss = loss + 0.5 * F.cross_entropy(model.last_copy_logits,
                                                    target)
            if model.last_color_copy_logits is not None:
                loss = loss + 0.5 * F.cross_entropy(
                    model.last_color_copy_logits, x[ar, qpos + 2])
            if model.last_write_logits is not None:
                step_i = torch.arange(N_STMT, device=device)
                ends = (who_pos.unsqueeze(1)
                        - (N_STMT - 1 - step_i).unsqueeze(0)
                        * block.unsqueeze(1) - 1)
                write_t = torch.zeros_like(who)
                write_t[ar.unsqueeze(1), ends] = 1.0
                write_t[ar, qpos] = 1.0  # candidate; the read gate suppresses it
                read_t = torch.zeros_like(who)
                read_t[ar, qpos] = 1.0
                wlog = model.last_write_logits.squeeze(-1)
                rlog = model.last_read_logits.squeeze(-1)
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                    wlog, write_t)
                obj_mask = (write_t + read_t).bool()
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                    rlog[obj_mask], read_t[obj_mask],
                    pos_weight=torch.tensor(4.0, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def run_arm(name, args, train_triples, old_triples, new_triples,
            held_triples, device):
    torch.manual_seed(args.seed)
    model = SpikingLM(SpikingConfig(
        vocab=VOCAB, dim=args.dim, layers=args.layers, mode="spike",
        adapt=args.adapt, memory=False, slots=N_STMT,
        slot_heads=args.heads, slot_attr_keys=True,
        slot_conj=args.slot_conj,
        slot_factorized=args.slot_factorized,
        query_color_copy=args.query_color_copy,
        color_copy_init=3.0,
        learned_boundaries=True,
        raw_boundaries=True,
        write_token=-1, read_token=-1,
        mem_read_pos=-1)).to(device)

    old_train = {f: make_data(args.n_train, old_triples, args.seed + 11 + f, f)
                 for f in (0, 1, 2)}
    new_train = {f: make_data(args.n_train, new_triples, args.seed + 21 + f, f)
                 for f in (0, 1, 2)}
    sets = {
        "old": {f: make_data(args.n_test, old_triples, args.seed + 31 + f, f)
                for f in (0, 1, 2)},
        "new": {f: make_data(args.n_test, new_triples, args.seed + 41 + f, f)
                for f in (0, 1, 2)},
        "held": {f: make_data(args.n_test, held_triples, args.seed + 51 + f, f)
                 for f in (0, 1, 2)},
    }
    stage_rows = []
    if args.curriculum:
        total_sec = 0.0
        for f, steps in ((0, args.steps_0), (1, args.steps_1),
                         (2, args.steps_2)):
            losses, sec = train_phase(model, old_train, old_train, steps,
                                      args.batch, args.lr, device, args.amp,
                                      fillers=(f,))
            total_sec += sec
            stage_rows.append({"filler": f, "steps": steps,
                               "loss": losses[-1], "sec": round(sec, 1)})
            print(json.dumps({"event": "stage", "filler": f,
                              "steps": steps, "sec": round(sec, 1)}),
                  flush=True)
        losses_a, sec = train_phase(model, old_train, old_train, args.steps_mix,
                                    args.batch, args.lr, device, args.amp,
                                    fillers=(0, 1, 2))
        total_sec += sec
        stage_rows.append({"filler": "mixed", "steps": args.steps_mix,
                           "loss": losses_a[-1], "sec": round(sec, 1)})
        sec_a = total_sec
    else:
        sec_a = 0.0
        if args.old_curriculum:
            warm_triples = [(n, COLORS[0], o) for n in NAMES8 for o in OBJECTS]
            warm = {f: make_data(args.n_train, warm_triples,
                                 args.seed + 61 + f, f) for f in (0, 1, 2)}
            losses_w, sec_w = train_phase(model, warm, warm, args.steps_warm,
                                          args.batch, args.lr, device, args.amp)
            sec_a += sec_w
            print(json.dumps({"event": "warm", "filler": "red-only",
                              "sec": round(sec_w, 1),
                              "loss": losses_w[-1]}), flush=True)
        losses_a, sec_new = train_phase(model, old_train, old_train,
                                        args.steps_a, args.batch, args.lr,
                                        device, args.amp)
        sec_a += sec_new
    before = {k: eval_set(model, v, device) for k, v in sets.items()}
    print(json.dumps({"event": "after_a", "seed": args.seed,
                      "sec": round(sec_a, 1), "loss": losses_a[-1],
                      "mean": {k: round(v["mean"], 4) for k, v in before.items()}}),
          flush=True)

    replay = args.replay if name == "replay_20" else 0.0
    losses_b, sec_b = train_phase(
        model, new_train, old_train, args.steps_b, args.batch, args.lr,
        device, args.amp, replay=replay)
    after = {k: eval_set(model, v, device) for k, v in sets.items()}
    perm = permute_relaxed(sets["old"][1])
    perm_ev = evaluate(model, perm, device)
    row = {"arm": name, "seed": args.seed,
           "replay": replay, "sec": round(sec_a + sec_b, 1),
           "train_loss": losses_b[-1], "stages": stage_rows,
           "before": before, "after": after,
           "old_perm": perm_ev["accuracy"]}
    print(json.dumps({"event": "arm", "arm": name, "seed": args.seed,
                      "old": round(after["old"]["mean"], 4),
                      "new": round(after["new"]["mean"], 4),
                      "held": round(after["held"]["mean"], 4),
                      "old_perm": round(perm_ev["accuracy"], 4)}), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=5000)
    ap.add_argument("--old-curriculum", action="store_true")
    ap.add_argument("--steps-warm", type=int, default=3000)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--steps-0", type=int, default=2000)
    ap.add_argument("--steps-1", type=int, default=2000)
    ap.add_argument("--steps-2", type=int, default=2000)
    ap.add_argument("--steps-mix", type=int, default=3000)
    ap.add_argument("--steps-b", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--slot-conj", action="store_true")
    ap.add_argument("--slot-factorized", action="store_true")
    ap.add_argument("--query-color-copy", action="store_true")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--arms", default="naive,replay_20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_binding_raw_boundary.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    old_triples = [(n, c, o) for n in NAMES8 for c in COLORS[:2]
                   for o in OBJECTS]
    new_triples = [(n, c, o) for n in NAMES8 for c in COLORS[2:]
                   for o in OBJECTS[:2]]
    held_triples = [(n, c, o) for n in NAMES8 for c in COLORS[2:]
                    for o in OBJECTS[2:]]
    rows = [run_arm(name.strip(), args, None, old_triples, new_triples,
                    held_triples, device)
            for name in args.arms.split(",") if name.strip()]
    payload = {"question": "Does binding survive variable statement lengths?",
               "config": vars(args),
               "max_len": MAX_LEN,
               "results": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
