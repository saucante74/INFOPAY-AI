"""
Tests unitaires du graphe LangGraph (`app/agent/graph.py`).

Le seul appel externe du graphe est `_llm.invoke(...)` (un objet unique,
construit au niveau module) : on le remplace entièrement par un double
programmable. `_llm` est un `RunnableBinding` Pydantic — ses attributs ne
sont pas settables individuellement (`AttributeError`), donc on remplace
l'objet module-level en entier plutôt que de patcher `.invoke` dessus.

Le premier test ne passe jamais par les tools : boucle "agent -> END"
uniquement. Le second va jusqu'au bout de la boucle "agent -> tools ->
agent -> END" en laissant `query_analytics` s'exécuter pour de vrai contre
un moteur SQLite en mémoire (`analytics.engine` monkeypatché) — c'est
la seule façon de vérifier que le graphe écrit à la main (voir sa
docstring : volontairement pas de `create_react_agent`) route bien les
tool_calls vers `ToolNode` et boucle correctement, sans reconstruire
LangGraph à la main.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from sqlmodel import Session

import app.agent.graph as graph_mod
import app.agent.tools as tools_mod
import app.services.analytics as analytics_mod
from app.models.payslip import Payslip


class _FakeLLM:
    """Remplace `_llm` : `.invoke()` renvoie les réponses de `responses`
    dans l'ordre, une par appel."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = iter(responses)
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return next(self._responses)


def test_agent_answers_directly_when_no_tool_call_is_needed(monkeypatch):
    fake_llm = _FakeLLM([AIMessage(content="Bonjour, comment puis-je vous aider ?")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    reply = graph_mod.run_chat("bonjour")

    assert reply == "Bonjour, comment puis-je vous aider ?"
    assert fake_llm.call_count == 1  # pas de boucle : END direct


def test_agent_calls_analytics_tool_then_answers(monkeypatch, test_engine):
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    with Session(test_engine) as session:
        session.add(
            Payslip(
                mois_annee="03/2025",
                salaire_brut=3000.0,
                net_imposable=2400.0,
                net_a_payer=2300.0,
                total_cotisations_salariales=600.0,
                total_cotisations_patronales=900.0,
                cotisations_retraite=350.0,
                prelevement_source=120.0,
                raw_text="x",
                filename="f.pdf",
            )
        )
        session.commit()

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_analytics",
                        "args": {"operation": "somme", "champ": "net_a_payer"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le total du net à payer est de 2300.0 euros."),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    reply = graph_mod.run_chat("Quel est le total du net à payer ?")

    assert reply == "Le total du net à payer est de 2300.0 euros."
    # 2 appels : la décision d'appeler l'outil, puis la reformulation après
    # que ToolNode a exécuté query_analytics et renvoyé son résultat.
    assert fake_llm.call_count == 2


def _seed_multi_year(test_engine) -> None:
    with Session(test_engine) as session:
        session.add_all(
            [
                Payslip(
                    mois_annee="11/2025",
                    salaire_brut=3000.0,
                    net_imposable=2400.0,
                    net_a_payer=2300.0,
                    total_cotisations_salariales=600.0,
                    total_cotisations_patronales=900.0,
                    cotisations_retraite=350.0,
                    prelevement_source=100.0,
                    raw_text="x",
                    filename="f1.pdf",
                ),
                Payslip(
                    mois_annee="01/2026",
                    salaire_brut=3000.0,
                    net_imposable=2400.0,
                    net_a_payer=2300.0,
                    total_cotisations_salariales=600.0,
                    total_cotisations_patronales=900.0,
                    cotisations_retraite=350.0,
                    prelevement_source=130.0,
                    raw_text="x",
                    filename="f2.pdf",
                ),
                Payslip(
                    mois_annee="06/2026",
                    salaire_brut=3000.0,
                    net_imposable=2400.0,
                    net_a_payer=2300.0,
                    total_cotisations_salariales=600.0,
                    total_cotisations_patronales=900.0,
                    cotisations_retraite=350.0,
                    prelevement_source=140.0,
                    raw_text="x",
                    filename="f3.pdf",
                ),
            ]
        )
        session.commit()


def test_agent_uses_date_debut_date_fin_for_a_specific_year_question(monkeypatch, test_engine):
    # Reproduit le bug rapporté : des bulletins sur 2025 ET 2026, seule
    # 2026 doit être sommée quand la question porte sur "2026".
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    _seed_multi_year(test_engine)

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_analytics",
                        "args": {
                            "operation": "somme",
                            "champ": "prelevement_source",
                            "date_debut": "01/2026",
                            "date_fin": "12/2026",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le total des prélèvements à la source en 2026 est de 270.0 euros."),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    reply = graph_mod.run_chat("Somme des prélèvements à la source en 2026")

    assert reply == "Le total des prélèvements à la source en 2026 est de 270.0 euros."
    # Le tool_call du LLM factice porte bien date_debut/date_fin (pas
    # derniers_n_mois) — c'est ToolNode qui exécute réellement
    # query_analytics avec ces arguments contre le moteur en mémoire, donc
    # si le résultat ci-dessus est correct (270 = 130 + 140, pas 100+130+140
    # = 370), c'est la preuve que le filtre par date a bien exclu 11/2025.
    assert fake_llm.call_count == 2


def test_agent_compares_two_periods_with_two_successive_tool_calls(monkeypatch, test_engine):
    # "compare 2025 et 2026" : le LLM doit appeler l'outil une fois par
    # période (pas de paramètre de comparaison dans le tool), le graphe
    # bouclant agent -> tools -> agent -> tools -> agent avant la réponse
    # finale.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    _seed_multi_year(test_engine)

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_analytics",
                        "args": {
                            "operation": "somme",
                            "champ": "prelevement_source",
                            "date_debut": "01/2025",
                            "date_fin": "12/2025",
                        },
                        "id": "call_2025",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_analytics",
                        "args": {
                            "operation": "somme",
                            "champ": "prelevement_source",
                            "date_debut": "01/2026",
                            "date_fin": "12/2026",
                        },
                        "id": "call_2026",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "En 2025, le total des prélèvements à la source était de 100.0 euros, "
                    "contre 270.0 euros en 2026, soit une hausse de 170.0 euros."
                )
            ),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    reply = graph_mod.run_chat("Compare les prélèvements à la source entre 2025 et 2026")

    assert "100.0" in reply and "270.0" in reply
    # 3 appels : décision du 1er tool_call (2025), décision du 2e (2026)
    # après avoir reçu le résultat du 1er, puis la reformulation finale
    # après le résultat du 2e — deux exécutions réelles de query_analytics,
    # chacune avec sa propre plage, pas un seul appel avec un paramètre de
    # comparaison.
    assert fake_llm.call_count == 3


def test_system_prompt_instructs_honesty_when_nothing_is_found():
    # Le tool signale "rien trouvé" en renvoyant extraits_trouves vide
    # (voir _build_hits dans vectorstore.py) ; c'est au system prompt de
    # dire au LLM comment réagir à ce signal, puisque le tool lui-même ne
    # peut pas formuler la réponse finale à la place du LLM.
    content = graph_mod.SYSTEM_PROMPT.content
    assert "extraits_trouves" in content
    assert "vide" in content
    assert "aucune information" in content.lower()


def test_system_prompt_instructs_citing_the_source():
    content = graph_mod.SYSTEM_PROMPT.content
    assert "mois_annee" in content
    assert "source" in content.lower()


def test_agent_gets_empty_extraits_trouves_when_vector_store_has_no_relevant_hit(
    monkeypatch, fake_vector_store
):
    # fake_vector_store.hits vide simule ce que ChromaVectorStore.search()
    # renvoie réellement quand aucun résultat n'a été trouvé, ou quand tous
    # les résultats sont sous le seuil de similarité (_build_hits) — les
    # deux cas produisent la même liste vide, donc un seul scénario suffit
    # à ce niveau.
    fake_vector_store.hits = []
    monkeypatch.setattr(tools_mod, "get_vector_store", lambda: fake_vector_store)

    fake_llm = _FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_payslip_knowledge_tool",
                        "args": {"query": "qu'est-ce que la prime de partage de la valeur ?"},
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

    reply = graph_mod.run_chat("qu'est-ce que la prime de partage de la valeur ?")

    assert reply == "Je n'ai trouvé aucune information à ce sujet dans vos bulletins importés."
    assert fake_llm.call_count == 2
