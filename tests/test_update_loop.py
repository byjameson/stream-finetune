from research.update_loop import add, export, rows


def test_learning_queue_add_list_export(tmp_path):
    queue = tmp_path / "queue.jsonl"
    add("hello", "Hello! How can I help?", queue)
    add("add task buy milk", "Added task 1: buy milk.", queue)
    data = rows(queue)
    assert len(data) == 2
    assert data[0]["user"] == "hello"
    out = export(queue, tmp_path / "export.txt")
    text = out.read_text(encoding="utf-8")
    assert "<|user|> hello" in text
    assert "<|assistant|> Added task 1" in text
    assert text.count("<|end|>") == 2
