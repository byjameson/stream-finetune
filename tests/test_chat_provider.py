from streamft.agent_runtime import AgentSession
from streamft.doc_memory import DocMemory
from research.chat_human import ChatHuman


def test_generator_handles_freeform_without_docs(tmp_path):
    human = ChatHuman(tmp_path / "memory.json",
                      generator=lambda text: f"provider: {text}")
    reply = human.respond("Tell me about photosynthesis")
    assert reply == "provider: Tell me about photosynthesis"


def test_general_chat_is_opt_in_when_docs_exist(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("law.txt", "The capital of France is Paris.")
    try:
        strict = ChatHuman(tmp_path / "strict.json", docs=docs,
                           generator=lambda text: "provider answer")
        assert "do not have a source" in strict.respond(
            "Tell me about photosynthesis")
        general = ChatHuman(tmp_path / "general.json", docs=docs,
                            generator=lambda text: "provider answer",
                            allow_general_chat=True)
        assert general.respond("Tell me about photosynthesis") == "provider answer"
    finally:
        docs.close()


def test_agent_session_forwards_generator(tmp_path):
    session = AgentSession(
        "provider-test", tmp_path / "memory.json",
        generator=lambda text: "provider answer")
    result = session.chat("Tell me about photosynthesis")
    assert result["reply"] == "provider answer"
    assert result["sources"] == []
