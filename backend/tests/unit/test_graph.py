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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


class _CapturingLLM:
    """Comme `_FakeLLM`, mais garde aussi une copie des `messages` reçus à
    chaque appel — nécessaire pour inspecter le SystemMessage ou
    l'historique effectivement envoyés au LLM, pas seulement la réponse
    finale. `self.calls` est une liste de listes de messages, une par
    appel à `.invoke()`, dans l'ordre."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = iter(responses)
        self.call_count = 0
        self.calls: list[list] = []

    def invoke(self, messages):
        self.call_count += 1
        self.calls.append(messages)
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


def test_agent_graph_actually_executes_query_analytics_via_the_real_tool_node(
    monkeypatch, test_engine
):
    # Descend un niveau sous test_agent_calls_analytics_tool_then_answers
    # ci-dessus : celui-ci ne vérifie que la réponse finale scriptée par le
    # faux LLM, qui ne dépend pas de ce que ToolNode a réellement produit —
    # un tool mal nommé (mismatch entre le nom du tool_call et le nom réel
    # de l'objet BaseTool) produit un ToolMessage d'erreur ("... is not a
    # valid tool ...") que le graphe absorbe silencieusement, puisqu'il
    # boucle "tools" -> "agent" quoi qu'il arrive. Ce cas concret s'est
    # produit pendant le développement de ce mécanisme (voir RAPPORT.md,
    # section diagnostic) : tools.py construit query_analytics via
    # `tool(_query_analytics)` pour pouvoir lui assigner une docstring
    # calculée à partir de FIELD_MAP (voir tools.py) — sans passer le nom
    # explicitement, l'outil se retrouvait exposé sous "_query_analytics"
    # au lieu de "query_analytics". Ce test regarde donc le ToolMessage
    # produit par le VRAI ToolNode, pas seulement la réponse finale.
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
            AIMessage(content="peu importe cette réponse, seul le ToolMessage compte ici"),
        ]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    result = graph_mod.agent_graph.invoke(
        {"messages": [HumanMessage(content="Quel est le total du net à payer ?")]}
    )

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert json.loads(tool_message.content) == {
        "operation": "somme",
        "champ": "net_a_payer",
        "periode": "1 mois (03/2025 à 03/2025)",
        "resultat": 2300.0,
    }


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


def test_system_prompt_states_the_field_ambiguity_principle(monkeypatch, test_engine):
    # Mécanisme général demandé par PROMPT.md : le prompt système doit
    # porter une RÈGLE DE COMPORTEMENT générique (jamais choisir un champ
    # en silence face à une correspondance non évidente), pas une liste de
    # termes déjà résolus — donc ce test vérifie la présence de la règle et
    # de ses mots-clés d'obligation, pas une liste fermée de synonymes.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "identité évidente" in content
    assert "clarification" in content.lower()
    # La règle doit explicitement se déclarer non limitée aux exemples
    # cités (cotisations sociales / charges / net / salaire) : c'est ce qui
    # la rend valable pour une formulation non anticipée aujourd'hui.
    assert "tout terme" in content.lower()


def test_system_prompt_distinguishes_calculation_periode_from_available_period(
    monkeypatch, test_engine
):
    # Bug de période rapporté : le LLM a une fois cité la période totale
    # des bulletins importés (get_available_period(), contexte général) à
    # la place de la période réellement couverte par UN calcul filtré (le
    # champ periode du résultat de CET appel à query_analytics). Cette
    # règle doit être générale (vaut pour toute réponse de calcul), donc ce
    # test vérifie la présence de la distinction elle-même, pas un exemple
    # chiffré particulier.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    content = graph_mod._build_system_prompt().content
    assert "champ periode" in content
    assert "période totale des bulletins importés" in content
    assert "JAMAIS" in content and "confonds" in content.lower()


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
    system_content = fake_llm.calls[0][0].content
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


def test_run_chat_sends_the_history_before_the_new_message_in_order(monkeypatch, test_engine):
    # Une question de suivi ("et pour février ?") ne peut être comprise
    # que si le LLM reçoit bien les échanges précédents, dans l'ordre,
    # avant le nouveau message.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    fake_llm = _CapturingLLM(
        [AIMessage(content="En février, le total est de 100.0 euros.")]
    )
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)
    history = [
        HumanMessage(content="Quel est le total de mes cotisations retraite en janvier ?"),
        AIMessage(content="En janvier, le total est de 90.0 euros."),
    ]

    result = graph_mod.run_chat("Et pour février ?", history=history)

    assert result["reply"] == "En février, le total est de 100.0 euros."
    sent_messages = fake_llm.calls[0]
    # [0] = le prompt système (reconstruit à chaque appel, voir
    # _build_system_prompt), [1:] = historique + nouveau message, dans
    # l'ordre exact où ils ont été fournis.
    assert [m.content for m in sent_messages[1:]] == [
        "Quel est le total de mes cotisations retraite en janvier ?",
        "En janvier, le total est de 90.0 euros.",
        "Et pour février ?",
    ]


def test_run_chat_truncates_history_to_the_last_n_messages(monkeypatch, test_engine):
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    fake_llm = _CapturingLLM([AIMessage(content="Réponse.")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)
    # Deux fois plus de messages que la limite : seuls les N derniers
    # doivent être envoyés au LLM, pas les plus anciens.
    history = [
        HumanMessage(content=f"message historique {i}")
        for i in range(graph_mod.MAX_HISTORY_MESSAGES * 2)
    ]

    graph_mod.run_chat("Nouvelle question", history=history)

    sent_messages = fake_llm.calls[0]
    # [0] = system prompt, le reste = historique tronqué + nouveau message.
    history_and_new_message = sent_messages[1:]
    assert len(history_and_new_message) == graph_mod.MAX_HISTORY_MESSAGES + 1
    # Les DERNIERS messages de l'historique sont gardés, pas les premiers.
    expected_oldest_kept_index = len(history) - graph_mod.MAX_HISTORY_MESSAGES
    assert history_and_new_message[0].content == f"message historique {expected_oldest_kept_index}"
    assert history_and_new_message[-2].content == f"message historique {len(history) - 1}"
    assert history_and_new_message[-1].content == "Nouvelle question"


def test_run_chat_without_history_behaves_like_before(monkeypatch, test_engine):
    # Rétrocompatibilité : `history` par défaut à None (voir la signature
    # de run_chat), donc un appelant qui ne le fournit pas — comme avant
    # cette tâche — continue de fonctionner exactement pareil.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    fake_llm = _CapturingLLM([AIMessage(content="Réponse sans historique.")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    result = graph_mod.run_chat("Une question isolée")

    assert result["reply"] == "Réponse sans historique."
    sent_messages = fake_llm.calls[0]
    assert [m.content for m in sent_messages[1:]] == ["Une question isolée"]


def test_run_chat_does_not_leak_a_previous_turns_rag_sources_into_the_current_one(
    monkeypatch, test_engine
):
    # `routers/chat.py`'s ChatHistoryMessage n'envoie jamais de ToolMessage
    # brut (rôle/contenu texte seulement), donc ce scénario précis ne peut
    # pas se produire via le routeur réel aujourd'hui — mais run_chat()
    # accepte `history: list[AnyMessage]` au sens large, et son contrat ne
    # doit pas dépendre de la discipline d'un appelant particulier. Ce test
    # construit directement un historique contenant le ToolMessage d'un
    # tour RAG précédent (comme le ferait un futur appelant, ou un
    # checkpointer LangGraph persistant), pour verrouiller que
    # run_chat() ne le confond jamais avec une source du tour actuel.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    history = [
        HumanMessage(content="C'est quoi la CSG ?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_payslip_knowledge_tool",
                    "args": {"query": "C'est quoi la CSG ?"},
                    "id": "call_previous",
                    "type": "tool_call",
                }
            ],
        ),
        _search_tool_message(
            {"text": "La CSG déductible est assise sur le salaire brut.", "mois_annee": "03/2025"},
            tool_call_id="call_previous",
        ),
        AIMessage(content="D'après votre bulletin de mars 2025 : la CSG est déductible."),
    ]

    fake_llm = _FakeLLM([AIMessage(content="Le total du net à payer est de 2300.0 euros.")])
    monkeypatch.setattr(graph_mod, "_llm", fake_llm)

    result = graph_mod.run_chat("Et le total du net à payer ?", history=history)

    # Cette question n'appelle pas le tool RAG (une seule réponse directe
    # dans fake_llm) : aucune source à citer pour CE tour, même si
    # l'historique rejoué en contient une d'un tour précédent.
    assert result["sources"] is None


def test_extract_sources_ignores_messages_from_before_the_cutoff_index():
    # Teste _extract_sources directement avec le même scénario, au niveau
    # le plus bas : une liste où seul le message APRÈS l'index de coupure
    # doit compter. run_chat() calcule ce sous-ensemble via
    # result["messages"][len(messages):] ; ici on le construit à la main
    # pour isoler la logique de filtrage de celle du graphe.
    previous_turn_tool_message = _search_tool_message(
        {"text": "extrait d'un tour précédent", "mois_annee": "01/2025"}
    )
    current_turn_messages = [AIMessage(content="Réponse du tour actuel, sans tool call.")]

    # Liste complète (comme result["messages"]) : contient l'ancien ToolMessage.
    assert graph_mod._extract_sources([previous_turn_tool_message, *current_turn_messages]) == [
        {"mois_annee": "01/2025", "extrait": "extrait d'un tour précédent"}
    ]
    # Sous-ensemble "ce tour seulement" (ce que run_chat() passe réellement) :
    # aucune source, cohérent avec l'absence de tool call ce tour-ci.
    assert graph_mod._extract_sources(current_turn_messages) is None


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
