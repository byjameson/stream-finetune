import json
from argparse import Namespace
from pathlib import Path

from streamft.spiking_watch import process_file, scan_once


def make_args(tmp_path, state_path, promote_to=""):
    return Namespace(
        inbox=str(tmp_path / "inbox"),
        model=str(tmp_path / "model"),
        old_data="old.pt",
        old_valid="old_valid.pt",
        tokenizer="tokenizer.json",
        out=str(tmp_path / "out"),
        state=str(state_path),
        promote_to=promote_to,
        once=True,
        poll=0.2,
        max_files=0,
        steps=1,
        batch=1,
        seq=8,
        lr=1e-4,
        replay_fraction=0.2,
        max_old_regression=0.02,
        min_new_gain=0.01,
        rollback_fraction=0.1,
        fisher_batches=1,
        eval_every=1,
        eval_batches=1,
        eval_batch_size=1,
        amp=False,
        keep_rejected=False,
        seed=0,
    )


def test_scan_promotes_only_accepted(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "accept.txt").write_text("new data", encoding="utf-8")
    (inbox / "reject.txt").write_text("other data", encoding="utf-8")
    out = tmp_path / "out"
    accepted = out / "versions" / "v0001"
    accepted.mkdir(parents=True)
    (accepted / "marker.txt").write_text("model", encoding="utf-8")
    calls = []

    def fake_ingest(ns):
        calls.append(ns)
        name = Path(ns.text).name
        if name == "accept.txt":
            return {"accepted": True, "version": "v0001",
                    "path": str(accepted), "before": {}, "after": {}}
        return {"accepted": False, "version": "v0002",
                "path": None, "before": {}, "after": {}}

    args = make_args(tmp_path, out / "watch_state.json",
                     promote_to=str(tmp_path / "promoted"))
    state = {"active_model": args.model, "active_version": None,
             "processed": {}}
    entries = scan_once(args, state, ingest_fn=fake_ingest)
    assert len(entries) == 2
    assert state["active_version"] == "v0001"
    assert (tmp_path / "promoted" / "marker.txt").exists()
    active = json.loads((out / "active.json").read_text(encoding="utf-8"))
    assert active["version"] == "v0001"
    assert state["processed"][entries[0]["id"]]["status"] == "accepted"
    assert state["processed"][entries[1]["id"]]["status"] == "rejected"
    assert not (inbox / "accept.txt").exists()
    assert not (inbox / "reject.txt").exists()
    assert list((out / "processed").iterdir())
    assert list((out / "rejected").iterdir())


def test_duplicate_is_archived_without_second_ingest(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text("same", encoding="utf-8")
    out = tmp_path / "out"
    accepted = out / "versions" / "v0001"
    accepted.mkdir(parents=True)
    (accepted / "x").write_text("x", encoding="utf-8")
    calls = []

    def fake_ingest(ns):
        calls.append(ns)
        return {"accepted": True, "version": "v0001",
                "path": str(accepted), "before": {}, "after": {}}

    args = make_args(tmp_path, out / "watch_state.json")
    state = {"active_model": args.model, "active_version": None,
             "processed": {}}
    scan_once(args, state, ingest_fn=fake_ingest)
    (inbox / "b.txt").write_text("same", encoding="utf-8")
    entries = scan_once(args, state, ingest_fn=fake_ingest)
    assert len(calls) == 1
    assert entries[0]["status"] == "duplicate"
    assert list((out / "duplicates").iterdir())


def test_failed_file_is_archived(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "bad.txt").write_text("bad", encoding="utf-8")
    out = tmp_path / "out"
    args = make_args(tmp_path, out / "watch_state.json")

    def broken(ns):
        raise RuntimeError("boom")

    state = {"active_model": args.model, "active_version": None,
             "processed": {}}
    entry = process_file(inbox / "bad.txt", args, state, ingest_fn=broken)
    assert entry["status"] == "error"
    assert list((out / "failed").iterdir())
    assert state["processed"][entry["id"]]["status"] == "error"
