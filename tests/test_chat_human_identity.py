from research.chat_human import ChatHuman


def test_chat_human_answers_name_and_likes_from_memory(tmp_path):
    human = ChatHuman(tmp_path / "memory.json")
    human.respond("my name is Alex")
    human.respond("remember that I like green tea")
    assert human.respond("what is my name?") == "Your name is Alex."
    assert "green tea" in human.respond("what do I like?")
