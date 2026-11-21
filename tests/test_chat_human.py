from research.chat_human import ChatHuman, extract_facts


def test_extract_facts_from_english_sentences():
    facts = dict(extract_facts(
        "My name is Alex. I live in Berlin and I like green tea."))
    assert facts["name"].startswith("Alex")
    assert facts["location"].startswith("Berlin")
    assert "green tea" in facts["likes"]


def test_chat_human_memory_tasks_and_persistence(tmp_path):
    memory = tmp_path / "memory.json"
    human = ChatHuman(memory)

    assert human.respond("my name is Alex") == "I will remember that."
    assert human.respond("add task buy milk") == "Added task 1: buy milk"
    assert "buy milk" in human.respond("list tasks")
    assert human.respond("complete task 1") == "Completed task 1."
    assert human.respond("list tasks") == "There are no open tasks."
    assert "Alex" in human.respond("what do you remember")
    assert human.respond("what is the weather?") == (
        "I do not have a language model loaded yet.")

    reloaded = ChatHuman(memory)
    assert reloaded.data["facts"][0]["value"] == "Alex"
    assert reloaded.data["tasks"][0]["status"] == "done"
