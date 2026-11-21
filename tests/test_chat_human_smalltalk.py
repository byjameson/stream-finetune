from research.chat_human import ChatHuman


def test_chat_human_fast_small_talk(tmp_path):
    human = ChatHuman(tmp_path / "memory.json")
    assert human.respond("hello") == "Hello! How can I help you today?"
    assert "well" in human.respond("how are you?").lower()
    assert human.respond("thank you") == "You are welcome."
