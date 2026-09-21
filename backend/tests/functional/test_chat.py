"""
Tests fonctionnels de `/api/chat` via `TestClient`.

Le LLM (`app.agent.graph._llm`) est toujours remplacé par un double : aucun
test ici n'appelle l'API Anthropic. Le test qui exerce le tool RAG
(`search_payslip_knowledge_tool`) remplace en plus
`app.agent.tools.get_vector_store` par la doublure `fake_vector_store` de
conftest.py, pour ne jamais ouvrir la vraie collection ChromaDB — ce tool
appelle `get_vector_store()` directement (pas via `Depends`), donc
`app.dependency_overrides` seul ne suffit pas à l'intercepter.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

import app.agent.graph as graph_mod
import app.agent.tools as tools_mod


class _FakeLLM:
    """Même doublure que dans tests/unit/test_graph.py : `_llm` est un
    `RunnableBinding` Pydantic dont les attributs ne sont pas settables
    individuellement, donc on remplace l'objet module-level en entier."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = iter(responses)
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return next(self._responses)


class _CapturingLLM(_FakeLLM):
    """Comme `_FakeLLM`, mais garde aussi une copie des `messages` reçus à
    chaque appel — nécessaire pour vérifier que l'historique envoyé dans le
    corps JSON de la requête HTTP est bien transmis jusqu'au LLM."""

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return super().invoke(messages)


def test_chat_happy_path_direct_answer(client, monkeypatch):
    fake_llm = _FakeLLM([AIMessage(content="La CSG est une cotisation sociale.")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    response = client.post("/api/chat", json={"message": "C'est quoi la CSG ?"})

    assert response.status_code == 200
    # No tool call at all here: no RAG source to cite.
    assert response.json() == {"reply": "La CSG est une cotisation sociale.", "sources": None}


def test_chat_calls_the_rag_tool_end_to_end(client, monkeypatch, fake_vector_store):
    # search_payslip_knowledge_tool résout son VectorStore lui-même
    # (app.dependencies.get_vector_store), donc on patche le nom importé
    # dans tools.py plutôt que de compter sur app.dependency_overrides.
    monkeypatch.setattr(tools_mod, "get_vector_store", lambda: fake_vector_store)

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_payslip_knowledge_tool",
                        "args": {"query": "c'est quoi la CSG ?"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="La CSG déductible est assise sur le salaire brut."),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    response = client.post("/api/chat", json={"message": "C'est quoi la CSG ?"})

    assert response.status_code == 200
    # fake_vector_store.hits (conftest.py) has one entry: mois_annee
    # "03/2025", text "La CSG déductible est assise sur le salaire brut." —
    # remapped to the ChatSource shape (`extrait` instead of `text`).
    assert response.json() == {
        "reply": "La CSG déductible est assise sur le salaire brut.",
        "sources": [
            {"mois_annee": "03/2025", "extrait": "La CSG déductible est assise sur le salaire brut."}
        ],
    }
    assert fake_llm.call_count == 2


def test_chat_rag_tool_called_but_nothing_relevant_found_has_no_sources(
    client, monkeypatch, fake_vector_store
):
    fake_vector_store.hits = []
    monkeypatch.setattr(tools_mod, "get_vector_store", lambda: fake_vector_store)

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_payslip_knowledge_tool",
                        "args": {"query": "prime de partage de la valeur"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="Je n'ai trouvé aucune information à ce sujet dans vos bulletins importés."
            ),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    response = client.post(
        "/api/chat", json={"message": "Qu'est-ce que la prime de partage de la valeur ?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Je n'ai trouvé aucune information à ce sujet dans vos bulletins importés.",
        "sources": None,
    }


def test_chat_sends_the_provided_history_to_the_llm_in_order(client, monkeypatch):
    fake_llm = _CapturingLLM([AIMessage(content="En février, le total est de 100.0 euros.")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    response = client.post(
        "/api/chat",
        json={
            "message": "Et pour février ?",
            "history": [
                {"role": "user", "content": "Quel est le total en janvier ?"},
                {"role": "assistant", "content": "En janvier, le total est de 90.0 euros."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "En février, le total est de 100.0 euros."
    sent_messages = fake_llm.calls[0]
    # [0] = system prompt (reconstruit à chaque appel) ; le reste doit
    # porter l'historique fourni dans le corps JSON, puis le nouveau
    # message, dans cet ordre.
    assert [m.content for m in sent_messages[1:]] == [
        "Quel est le total en janvier ?",
        "En janvier, le total est de 90.0 euros.",
        "Et pour février ?",
    ]


def test_chat_without_a_history_field_in_the_request_body_still_works(client, monkeypatch):
    # Rétrocompatibilité : un ancien client qui n'envoie pas encore
    # `history` (le champ a une valeur par défaut `[]` dans ChatRequest)
    # continue de fonctionner exactement comme avant cette tâche.
    fake_llm = _CapturingLLM([AIMessage(content="Bonjour !")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    response = client.post("/api/chat", json={"message": "bonjour"})

    assert response.status_code == 200
    assert response.json()["reply"] == "Bonjour !"
    sent_messages = fake_llm.calls[0]
    assert [m.content for m in sent_messages[1:]] == ["bonjour"]


def test_chat_returns_500_when_the_llm_call_fails(client_no_raise, monkeypatch):
    # /api/chat n'a pas de try/except autour de run_chat() (contrairement à
    # /api/upload, qui catche explicitement l'échec d'extraction) : une
    # panne du LLM y remonte donc telle quelle en 500 FastAPI standard.
    # Documenté ici comme comportement actuel, pas comme un bug à corriger
    # dans le cadre de cette tâche (aucun changement de comportement).
    class _RaisingLLM:
        def invoke(self, messages):
            raise RuntimeError("Anthropic API indisponible")

    monkeypatch.setattr(graph_mod, "_llm", _RaisingLLM())

    response = client_no_raise.post("/api/chat", json={"message": "bonjour"})

    assert response.status_code == 500
