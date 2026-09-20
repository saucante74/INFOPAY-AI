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

import json
from datetime import date

from langchain_core.messages import AIMessage, ToolMessage
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


def test_agent_answers_directly_when_no_tool_call_is_needed(monkeypatch, test_engine):
    # _build_system_prompt() interroge désormais analytics.get_available_period()
    # à chaque appel du nœud "agent" (voir graph.py) : même ce test, qui ne
    # passe par aucun tool, doit rediriger le moteur analytics vers la base
    # en mémoire pour ne jamais toucher backend/data/infopay.db.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    fake_llm = _FakeLLM([AIMessage(content="Bonjour, comment puis-je vous aider ?")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    result = graph_mod.run_chat("bonjour")

    assert result == {"reply": "Bonjour, comment puis-je vous aider ?", "sources": None}
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

    result = graph_mod.run_chat("Quel est le total du net à payer ?")

    assert result["reply"] == "Le total du net à payer est de 2300.0 euros."
    # query_analytics n'est pas le tool RAG : aucune source à citer.
    assert result["sources"] is None
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

    result = graph_mod.run_chat("Somme des prélèvements à la source en 2026")

    assert result["reply"] == "Le total des prélèvements à la source en 2026 est de 270.0 euros."
    assert result["sources"] is None
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

    result = graph_mod.run_chat("Compare les prélèvements à la source entre 2025 et 2026")

    assert "100.0" in result["reply"] and "270.0" in result["reply"]
    assert result["sources"] is None
    # 3 appels : décision du 1er tool_call (2025), décision du 2e (2026)
    # après avoir reçu le résultat du 1er, puis la reformulation finale
    # après le résultat du 2e — deux exécutions réelles de query_analytics,
    # chacune avec sa propre plage, pas un seul appel avec un paramètre de
    # comparaison.
    assert fake_llm.call_count == 3


def test_system_prompt_instructs_honesty_when_nothing_is_found(monkeypatch, test_engine):
    # Le tool signale "rien trouvé" en renvoyant extraits_trouves vide
    # (voir _build_hits dans vectorstore.py) ; c'est au system prompt de
    # dire au LLM comment réagir à ce signal, puisque le tool lui-même ne
    # peut pas formuler la réponse finale à la place du LLM.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "extraits_trouves" in content
    assert "vide" in content
    assert "aucune information" in content.lower()


def test_system_prompt_instructs_citing_the_source(monkeypatch, test_engine):
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "mois_annee" in content
    assert "source" in content.lower()


def test_system_prompt_discourages_the_rag_tool_for_general_knowledge_questions(
    monkeypatch, test_engine
):
    # Bug : "quelle différence entre salaire moyen et salaire médian ?"
    # déclenchait le tool RAG et remontait des chunks hors sujet, cités
    # comme "Sources" alors qu'ils n'avaient pas servi à la réponse. Le
    # seuil resserré (voir test_vectorstore.py) n'est qu'un filet de
    # sécurité après coup — la vraie prévention est de décourager l'appel
    # du tool sur ce type de question dès le prompt système.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "culture générale" in content.lower() or "sans lien avec le contenu réel" in content
    assert "salaire moyen" in content and "salaire médian" in content


def test_system_prompt_forbids_markdown_formatting(monkeypatch, test_engine):
    # ChatPanel.tsx affiche le texte de la réponse tel quel, sans rendu
    # Markdown : les symboles ** ou # apparaîtraient littéralement.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "markdown" in content.lower()
    assert "**" in content  # l'exemple de syntaxe à éviter, cité tel quel
    assert "texte brut" in content.lower()


def test_system_prompt_reflects_the_real_mocked_date(monkeypatch, test_engine):
    # La date du jour ne doit jamais venir de la mémoire d'entraînement du
    # LLM : on mocke la source (_today), pas seulement le formatage, pour
    # vérifier le mécanisme d'injection lui-même.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    monkeypatch.setattr(graph_mod, "_today", lambda: date(2026, 3, 15))

    content = graph_mod._build_system_prompt().content

    assert "15/03/2026" in content


def test_system_prompt_states_no_payslip_imported_when_the_database_is_empty(
    monkeypatch, test_engine
):
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "Aucun bulletin de paie n'a encore été importé." in content


def test_system_prompt_states_the_real_available_period_when_payslips_exist(
    monkeypatch, test_engine
):
    # Le cas précis du bug rapporté : bulletins tous en 2026, une question
    # sans année précisée ne doit pas pouvoir faire halluciner 2025 — le
    # prompt système doit porter la VRAIE plage disponible.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    _seed_multi_year(test_engine)

    content = graph_mod._build_system_prompt().content

    assert "11/2025" in content
    assert "06/2026" in content
    assert "3 bulletin" in content


def test_agent_uses_the_real_available_period_end_to_end_for_a_vague_period_question(
    monkeypatch, test_engine
):
    # Bout en bout : seuls des bulletins 2026 en base, une question sans
    # année ("février et mars") — le LLM factice ici simule la bonne
    # réaction (utiliser 2026, la seule année réellement présente) plutôt
    # que le comportement bogué (halluciner 2025). Ce test ne peut pas
    # vérifier que le VRAI Claude ferait ce choix ; il vérifie que le
    # mécanisme d'injection fonctionne bout en bout via run_chat().
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    with Session(test_engine) as session:
        session.add_all(
            [
                Payslip(
                    mois_annee="02/2026",
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
                    mois_annee="03/2026",
                    salaire_brut=3000.0,
                    net_imposable=2400.0,
                    net_a_payer=2300.0,
                    total_cotisations_salariales=600.0,
                    total_cotisations_patronales=900.0,
                    cotisations_retraite=350.0,
                    prelevement_source=110.0,
                    raw_text="x",
                    filename="f2.pdf",
                ),
            ]
        )
        session.commit()

    captured_messages: list = []

    class _CapturingLLM:
        """Comme _FakeLLM, mais garde aussi une copie des messages reçus à
        chaque appel — nécessaire ici pour inspecter le SystemMessage
        effectivement envoyé au LLM, pas seulement la réponse finale."""

        def __init__(self, responses: list[AIMessage]) -> None:
            self._responses = iter(responses)
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            captured_messages.append(messages)
            return next(self._responses)

    fake_llm = _CapturingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_analytics",
                        "args": {
                            "operation": "somme",
                            "champ": "cotisations_retraite",
                            "date_debut": "02/2026",
                            "date_fin": "03/2026",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le total des cotisations retraite est de 210.0 euros."),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    result = graph_mod.run_chat("Quelles sont mes cotisations retraite pour février et mars ?")

    assert result["reply"] == "Le total des cotisations retraite est de 210.0 euros."
    # Le prompt système envoyé au LLM (1er message de chaque appel) porte
    # bien la vraie plage disponible (2026), pas une année inventée.
    system_content = captured_messages[0][0].content
    assert "02/2026" in system_content
    assert "03/2026" in system_content


def test_agent_gets_empty_extraits_trouves_when_vector_store_has_no_relevant_hit(
    monkeypatch, test_engine, fake_vector_store
):
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
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

    result = graph_mod.run_chat("qu'est-ce que la prime de partage de la valeur ?")

    assert (
        result["reply"] == "Je n'ai trouvé aucune information à ce sujet dans vos bulletins importés."
    )
    # Le tool a été appelé mais n'a rien trouvé de pertinent (hits vide) :
    # pas de source à afficher, jamais une liste vide.
    assert result["sources"] is None
    assert fake_llm.call_count == 2


def _search_tool_message(*hits: dict, tool_call_id: str = "call_1") -> ToolMessage:
    """Construit le ToolMessage que ToolNode produit réellement pour
    search_payslip_knowledge_tool (voir SearchKnowledgeResult dans
    tools.py) : content = JSON sérialisé de {"extraits_trouves": [...]}."""
    return ToolMessage(
        content=json.dumps({"extraits_trouves": list(hits)}),
        name="search_payslip_knowledge_tool",
        tool_call_id=tool_call_id,
    )


def test_extract_sources_returns_none_when_the_search_tool_was_never_called():
    messages = [AIMessage(content="Bonjour !")]
    assert graph_mod._extract_sources(messages) is None


def test_extract_sources_returns_none_when_the_search_tool_found_nothing():
    messages = [_search_tool_message()]  # extraits_trouves: []
    assert graph_mod._extract_sources(messages) is None


def test_extract_sources_maps_hits_to_the_chatsource_shape():
    messages = [
        _search_tool_message(
            {"text": "La CSG déductible est assise sur le salaire brut.", "mois_annee": "03/2025"}
        )
    ]

    sources = graph_mod._extract_sources(messages)

    # Forme exacte du TypedDict ChatSource : mois_annee + extrait (pas
    # "text", la clé interne de vectorstore.py) — rien de plus.
    assert sources == [
        {"mois_annee": "03/2025", "extrait": "La CSG déductible est assise sur le salaire brut."}
    ]


def test_extract_sources_aggregates_across_several_search_tool_calls():
    # Le LLM peut appeler search_payslip_knowledge_tool plusieurs fois dans
    # la même conversation (ex: deux questions successives, ou une
    # reformulation) — chaque appel produit son propre ToolMessage.
    messages = [
        _search_tool_message(
            {"text": "extrait 1", "mois_annee": "01/2025"}, tool_call_id="call_1"
        ),
        AIMessage(content=""),  # tour intermédiaire, ignoré (pas un ToolMessage du tool RAG)
        _search_tool_message(
            {"text": "extrait 2", "mois_annee": "02/2025"}, tool_call_id="call_2"
        ),
    ]

    sources = graph_mod._extract_sources(messages)

    assert sources == [
        {"mois_annee": "01/2025", "extrait": "extrait 1"},
        {"mois_annee": "02/2025", "extrait": "extrait 2"},
    ]
