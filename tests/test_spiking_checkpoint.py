from streamft.spiking import SpikingConfig, SpikingLM, save_checkpoint


def test_save_checkpoint_from_same_directory_tokenizer(tmp_path):
    model = SpikingLM(SpikingConfig(
        vocab=32, dim=16, layers=1, memory=False))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    out = tmp_path / "checkpoint"
    save_checkpoint(model, out, tokenizer, {"step": 0})
    # Resuming into the same directory passes out/tokenizer.json as the source.
    save_checkpoint(model, out, out / "tokenizer.json", {"step": 1})
    assert (out / "model.safetensors").exists()
    assert (out / "tokenizer.json").exists()
