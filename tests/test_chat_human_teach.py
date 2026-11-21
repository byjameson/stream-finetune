from research.chat_human import ChatHuman


def test_chat_human_teach_command_stores_a_fact(tmp_path):
    human = ChatHuman(tmp_path / "memory.json")
    assert human.respond("teach: the lab is on the second floor") == (
        "I learned that.")
    assert "second floor" in human.respond("what do you remember?")
    reloaded = ChatHuman(tmp_path / "memory.json")
    assert any("second floor" in fact["value"]
               for fact in reloaded.data["facts"])
