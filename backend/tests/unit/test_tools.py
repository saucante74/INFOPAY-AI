"""
Tests unitaires des tools exposés au LLM (`app/agent/tools.py`).

Deux préoccupations spécifiques à ce fichier, au-delà du comportement
habituel des tests unitaires :

1. `query_analytics` est construit avec `tool("query_analytics")(_query_analytics)`
   plutôt que `@tool` (voir le commentaire dans tools.py) pour pouvoir lui
   assigner une docstring calculée à l'import à partir de FIELD_MAP. Ce
   détour a régressé une fois pendant le développement de ce mécanisme :
   `tool(_query_analytics)` sans nom explicite nomme l'outil d'après
   `_query_analytics.__name__` ("_query_analytics"), ce qui casse le
   routage de ToolNode (un tool_call "query_analytics" produit alors un
   ToolMessage d'erreur "not a valid tool", silencieusement absorbé par le
   graphe puisqu'il boucle vers "agent" quoi qu'il arrive) — voir
   RAPPORT.md, section diagnostic, pour la reproduction réelle de cette
   régression. `test_query_analytics_keeps_its_public_tool_name` verrouille
   spécifiquement ce point.
2. La docstring de `query_analytics` est le mécanisme par lequel le LLM
   reçoit la description de chaque champ (voir FIELD_MAP) : elle doit
   rester dérivée de FIELD_MAP, pas recopiée à la main, pour qu'un futur
   champ ajouté à FIELD_MAP y apparaisse automatiquement.
"""
from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.agent.tools as tools_mod
import app.services.analytics as analytics_mod
from app.models.payslip import Payslip
from app.services.analytics import FIELD_MAP


def test_query_analytics_keeps_its_public_tool_name():
    # Régression verrouillée : voir la docstring du module.
    assert tools_mod.query_analytics.name == "query_analytics"


def test_query_analytics_description_lists_every_field_map_entry_with_its_description():
    description = tools_mod.query_analytics.description
    for champ, info in FIELD_MAP.items():
        assert champ in description
        assert info["description"] in description


def test_champ_doc_reflects_a_newly_added_field_without_touching_tools_py(monkeypatch):
    # Preuve que la docstring ne recopie pas FIELD_MAP à la main : un champ
    # ajouté (même fictif, ici) apparaît dans la doc reconstruite sans
    # modifier tools.py.
    extended_field_map = {
        **FIELD_MAP,
        "champ_futur": {
            "colonne": "champ_futur",
            "description": "Description d'un champ ajouté après coup, à titre d'exemple.",
        },
    }
    monkeypatch.setattr(tools_mod, "FIELD_MAP", extended_field_map)

    doc = tools_mod._build_champ_doc()

    assert "champ_futur" in doc
    assert "Description d'un champ ajouté après coup" in doc


def test_query_analytics_description_states_the_ambiguity_principle():
    # Rappel court, dans le tool lui-même, de la consigne d'ambiguïté
    # portée par le prompt système (voir test_graph.py pour le texte
    # complet côté prompt système) — les deux mécanismes se renforcent.
    description = tools_mod.query_analytics.description
    assert "identité" in description.lower() or "évidente" in description.lower()


def test_query_analytics_description_distinguishes_periode_from_available_period():
    description = tools_mod.query_analytics.description
    assert "periode" in description
    assert "période totale des bulletins importés" in description


def test_query_analytics_tool_executes_end_to_end_against_a_real_engine(monkeypatch):
    # Exécute le VRAI tool (pas run_analytics_query directement) pour
    # vérifier que l'enveloppe tool("query_analytics")(_query_analytics)
    # route bien vers l'implémentation réelle et renvoie le bon résultat —
    # pas seulement que le nom du tool est correct (test dédié ci-dessus).
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(analytics_mod, "engine", engine)
    with Session(engine) as session:
        session.add(
            Payslip(
                mois_annee="01/2025",
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

    result = tools_mod.query_analytics.invoke(
        {"operation": "somme", "champ": "net_a_payer"}
    )

    assert result == {
        "operation": "somme",
        "champ": "net_a_payer",
        "periode": "1 mois (01/2025 à 01/2025)",
        "resultat": 2300.0,
    }
