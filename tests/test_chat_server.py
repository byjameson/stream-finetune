from atomzero.chat_server import chat_prompt


def test_chat_prompt_uses_training_format():
    prompt = chat_prompt([{"role": "user", "content": "hello"}])
    assert prompt == "<|user|> hello <|assistant|>"


def test_chat_prompt_includes_the_last_turns():
    prompt = chat_prompt([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "how are you?"},
    ])
    assert "<|user|> hi" in prompt
    assert "<|assistant|> Hello!" in prompt
    assert prompt.endswith("<|assistant|>")
