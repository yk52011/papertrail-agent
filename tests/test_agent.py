from langchain_core.messages import AIMessage

from papertrail.agent import ask
from papertrail.retrieval import RepositoryIndex


class FakeModel:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = 0

    def bind_tools(self, tools):
        assert {t.name for t in tools} == {"search_repository", "inspect_requirements"}
        return self

    def invoke(self, messages):
        self.calls += 1
        return next(self.messages)


def call():
    return AIMessage(content="", tool_calls=[{"name": "search_repository",
                     "args": {"query": "GPU"}, "id": "t1", "type": "tool_call"}])


def test_agent_executes_retrieval_and_collects_evidence(tmp_path):
    (tmp_path / "README.md").write_text("GPU required")
    model = FakeModel([call(), AIMessage(content="GPU required [README.md:1-1]; not run.")])
    result = ask(RepositoryIndex(tmp_path), "Requirements?", model=model)
    assert result["turns"] == 2
    assert result["evidence"][0]["citation"] == "README.md:1-1"
    assert result["execution_status"] == "not_run"
    assert not result["budget_exhausted"]


def test_model_budget_is_enforced(tmp_path):
    model = FakeModel([call(), call(), call()])
    result = ask(RepositoryIndex(tmp_path), "requirements?", model=model, max_turns=2)
    assert model.calls == 2
    assert result["budget_exhausted"]
    assert "incomplete" in result["answer"]
