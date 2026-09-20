"""
Tests unitaires du moteur de calcul exact (`app/services/analytics.py`).

`run_analytics_query` lit via le moteur SQL global `app.services.analytics.engine`
(pas via `get_session`, donc pas injectable par `app.dependency_overrides`) :
chaque test le remplace par un moteur SQLite en mémoire via `monkeypatch`,
pour ne jamais lire le vrai `backend/data/infopay.db`.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session

import app.services.analytics as analytics_mod
from app.models.payslip import Payslip
from app.services.analytics import run_analytics_query


def _make_payslip(mois_annee: str, **overrides: float) -> Payslip:
    base = dict(
        salaire_brut=3000.0,
        net_imposable=2400.0,
        net_a_payer=2300.0,
        total_cotisations_salariales=600.0,
        total_cotisations_patronales=900.0,
        cotisations_retraite=350.0,
        prelevement_source=120.0,
    )
    base.update(overrides)
    return Payslip(mois_annee=mois_annee, raw_text="x", filename="f.pdf", **base)


@pytest.fixture(autouse=True)
def _use_test_engine(test_engine, monkeypatch):
    """Redirige `analytics.py` vers le moteur en mémoire pour tous les tests
    de ce fichier — autouse pour ne pas avoir à le répéter à chaque test."""
    monkeypatch.setattr(analytics_mod, "engine", test_engine)
    return test_engine


def _seed(test_engine, *payslips: Payslip) -> None:
    with Session(test_engine) as session:
        session.add_all(payslips)
        session.commit()


def test_somme(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", net_a_payer=2000.0),
        _make_payslip("02/2025", net_a_payer=2200.0),
    )
    result = run_analytics_query(operation="somme", champ="net_a_payer")
    assert result == {
        "operation": "somme",
        "champ": "net_a_payer",
        "periode": "2 mois (01/2025 à 02/2025)",
        "resultat": 4200.0,
    }


def test_moyenne(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", salaire_brut=3000.0),
        _make_payslip("02/2025", salaire_brut=4000.0),
    )
    result = run_analytics_query(operation="moyenne", champ="salaire_brut")
    assert result["resultat"] == 3500.0


def test_min_et_max(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", net_imposable=1800.0),
        _make_payslip("02/2025", net_imposable=2600.0),
        _make_payslip("03/2025", net_imposable=2200.0),
    )
    assert run_analytics_query(operation="min", champ="net_imposable")["resultat"] == 1800.0
    assert run_analytics_query(operation="max", champ="net_imposable")["resultat"] == 2600.0


def test_derniers_n_mois_ne_considere_que_les_plus_recents(_use_test_engine):
    # Insérés dans le désordre chronologique : la fonction doit trier par
    # date avant d'appliquer `.tail()`, pas se fier à l'ordre d'insertion.
    _seed(
        _use_test_engine,
        _make_payslip("03/2025", net_a_payer=300.0),
        _make_payslip("01/2025", net_a_payer=100.0),
        _make_payslip("02/2025", net_a_payer=200.0),
    )
    result = run_analytics_query(operation="somme", champ="net_a_payer", derniers_n_mois=2)
    # Seuls février (200) et mars (300) sont les 2 plus récents.
    assert result["resultat"] == 500.0
    assert result["periode"] == "2 mois (02/2025 à 03/2025)"


def test_champ_inconnu_renvoie_une_erreur_sans_toucher_a_la_base(_use_test_engine):
    # Pas de seed : si la fonction lisait la base avant de valider `champ`,
    # elle échouerait différemment (base vide -> autre message d'erreur).
    result = run_analytics_query(operation="somme", champ="inexistant")
    assert result == {
        "error": "Champ inconnu: inexistant. Champs valides: "
        "['salaire_brut', 'net_imposable', 'net_a_payer', 'cotisations_salariales', "
        "'cotisations_patronales', 'cotisations_retraite', 'prelevement_source']"
    }


def test_aucun_bulletin_importe(_use_test_engine):
    result = run_analytics_query(operation="somme", champ="net_a_payer")
    assert result == {"error": "Aucun bulletin de paie n'a encore été importé."}


def test_operation_inconnue(_use_test_engine):
    _seed(_use_test_engine, _make_payslip("01/2025"))
    # `operation` est un Literal au niveau du type, mais rien n'empêche un
    # appelant externe (le tool LLM, via `cast`) de passer une valeur hors
    # de cette liste à l'exécution — d'où cette garde `else` dans le code.
    result = run_analytics_query(operation="mediane", champ="net_a_payer")  # type: ignore[arg-type]
    assert result == {"error": "Opération inconnue: mediane"}


def test_date_debut_date_fin_isolent_une_annee_complete(_use_test_engine):
    # Le bug qui a motivé cette fonctionnalité : des bulletins sur plusieurs
    # années, seule l'année demandée doit être prise en compte.
    _seed(
        _use_test_engine,
        _make_payslip("11/2025", prelevement_source=100.0),
        _make_payslip("12/2025", prelevement_source=110.0),
        _make_payslip("01/2026", prelevement_source=120.0),
        _make_payslip("06/2026", prelevement_source=130.0),
        _make_payslip("12/2026", prelevement_source=140.0),
        _make_payslip("01/2027", prelevement_source=150.0),
    )
    result = run_analytics_query(
        operation="somme",
        champ="prelevement_source",
        date_debut="01/2026",
        date_fin="12/2026",
    )
    assert result == {
        "operation": "somme",
        "champ": "prelevement_source",
        "periode": "3 mois (01/2026 à 12/2026)",
        "resultat": 390.0,
    }


def test_date_debut_date_fin_sur_quelques_mois_arbitraires(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", salaire_brut=1000.0),
        _make_payslip("02/2025", salaire_brut=2000.0),
        _make_payslip("03/2025", salaire_brut=3000.0),
        _make_payslip("04/2025", salaire_brut=4000.0),
    )
    result = run_analytics_query(
        operation="moyenne",
        champ="salaire_brut",
        date_debut="02/2025",
        date_fin="03/2025",
    )
    assert result["resultat"] == 2500.0
    assert result["periode"] == "2 mois (02/2025 à 03/2025)"


def test_date_debut_seul_va_jusquau_bulletin_le_plus_recent(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", salaire_brut=1000.0),
        _make_payslip("06/2025", salaire_brut=2000.0),
        _make_payslip("12/2025", salaire_brut=3000.0),
    )
    result = run_analytics_query(
        operation="somme", champ="salaire_brut", date_debut="06/2025"
    )
    assert result["resultat"] == 5000.0
    assert result["periode"] == "2 mois (06/2025 à 12/2025)"


def test_date_debut_ou_date_fin_prime_sur_derniers_n_mois(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", net_a_payer=100.0),
        _make_payslip("02/2025", net_a_payer=200.0),
        _make_payslip("03/2025", net_a_payer=300.0),
    )
    # derniers_n_mois=1 aurait pris mars seul (300) si respecté ; la plage
    # explicite doit primer et couvrir janvier + février (300 aussi, par
    # coïncidence de valeurs — le test vérifie la période retenue, pas
    # seulement le résultat).
    result = run_analytics_query(
        operation="somme",
        champ="net_a_payer",
        derniers_n_mois=1,
        date_debut="01/2025",
        date_fin="02/2025",
    )
    assert result["resultat"] == 300.0
    assert result["periode"] == "2 mois (01/2025 à 02/2025)"


def test_derniers_n_mois_toujours_valide_sans_dates(_use_test_engine):
    # Comportement inchangé si aucune date n'est fournie.
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", net_a_payer=100.0),
        _make_payslip("02/2025", net_a_payer=200.0),
        _make_payslip("03/2025", net_a_payer=300.0),
    )
    result = run_analytics_query(operation="somme", champ="net_a_payer", derniers_n_mois=2)
    assert result["resultat"] == 500.0
    assert result["periode"] == "2 mois (02/2025 à 03/2025)"


def test_aucun_filtre_prend_tout_lhistorique(_use_test_engine):
    _seed(
        _use_test_engine,
        _make_payslip("01/2025", net_a_payer=100.0),
        _make_payslip("02/2025", net_a_payer=200.0),
    )
    result = run_analytics_query(operation="somme", champ="net_a_payer")
    assert result["resultat"] == 300.0
    assert result["periode"] == "2 mois (01/2025 à 02/2025)"


def test_date_debut_posterieure_a_date_fin_renvoie_une_erreur_claire(_use_test_engine):
    _seed(_use_test_engine, _make_payslip("01/2025"))
    result = run_analytics_query(
        operation="somme",
        champ="net_a_payer",
        date_debut="12/2025",
        date_fin="01/2025",
    )
    assert result == {
        "error": "date_debut (12/2025) est postérieure à date_fin (01/2025)."
    }


def test_date_debut_posterieure_a_date_fin_nest_pas_une_comparaison_de_chaines(
    _use_test_engine,
):
    # '09/2025' > '01/2026' en comparaison de chaînes ("0" < "1" sur le
    # premier caractère du mois donnerait le mauvais résultat si comparé
    # naïvement) — la vraie date (septembre 2025) est bien antérieure à
    # janvier 2026, donc aucune erreur ne doit être levée ici.
    _seed(_use_test_engine, _make_payslip("10/2025"))
    result = run_analytics_query(
        operation="somme",
        champ="net_a_payer",
        date_debut="09/2025",
        date_fin="01/2026",
    )
    assert "error" not in result


def test_date_invalide_renvoie_une_erreur_claire(_use_test_engine):
    _seed(_use_test_engine, _make_payslip("01/2025"))
    result = run_analytics_query(
        operation="somme", champ="net_a_payer", date_debut="2025-01", date_fin="12/2025"
    )
    assert result == {"error": "date_debut invalide, format attendu MM/YYYY: '2025-01'"}
