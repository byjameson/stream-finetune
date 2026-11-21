from research.chat_human import ChatHuman


def test_chat_human_format_skills(tmp_path):
    human = ChatHuman(tmp_path / "memory.json")
    assert human.respond("count from one to five") == (
        "One, two, three, four, five.")
    assert human.respond("be very brief: hello") == "Hi."
    assert human.respond("make it polite: open the window") == (
        "Could you open the window, please?")
    assert human.respond("give me two colors") == "Blue and green."
    assert human.respond("repeat after me: The book is on the desk.") == (
        "The book is on the desk.")
    assert human.respond(
        "summarize in one sentence: The cat sat on the mat.") == (
        "The cat sat on the mat.")
