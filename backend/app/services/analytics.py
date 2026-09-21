"""
Moteur de calcul exact. Aucune valeur numérique renvoyée par ces
fonctions ne passe par le LLM : le LLM choisit QUELS calculs faire
(operation, champ, période), mais l'exécution est du Pandas pur.
"""
from datetime import datetime
from typing import Literal, TypedDict

import pandas as pd
from sqlmodel import Session, select

from app.db import engine
from app.models.payslip import Payslip


class FieldInfo(TypedDict):
    """Une entrée de FIELD_MAP : la colonne DataFrame réelle derrière le
    nom exposé au LLM, et une description métier de ce que le champ
    couvre (et ne couvre pas).

    La description n'est PAS une table de correspondance "terme
    utilisateur -> champ" (ex: ne dit jamais littéralement "cotisations
    sociales -> cotisations_salariales") : une telle table serait fermée
    par construction, incapable de couvrir une formulation non anticipée
    aujourd'hui (voir RAPPORT.md, section "Mécanisme"). Elle donne plutôt
    au LLM la même matière qu'un lecteur humain du bulletin aurait pour
    juger LUI-MÊME si un terme y correspond clairement ou non — y compris
    un renvoi explicite vers les champs frères avec lesquels une confusion
    est plausible (ex: net_imposable renvoie vers net_a_payer et
    inversement), pour qu'un terme générique ("net", "cotisations
    sociales", "charges"...) déclenche la vigilance du LLM sans qu'aucune
    liste de termes n'ait eu besoin d'être écrite ici. Ajouter un nouveau
    champ à FIELD_MAP suffit à le faire apparaître dans le tool exposé au
    LLM (voir app/agent/tools.py, qui construit sa docstring à partir de
    ce dict) : rien d'autre à maintenir en parallèle."""

    colonne: str
    description: str


FIELD_MAP: dict[str, FieldInfo] = {
    "salaire_brut": {
        "colonne": "salaire_brut",
        "description": (
            "Salaire brut mensuel, avant toute cotisation ou retenue. "
            "Correspond au terme « brut » sur un bulletin de paie français, "
            "un terme standard et univoque en général. Si l'utilisateur dit "
            "simplement « salaire » ou « combien je gagne » SANS préciser "
            "« brut », il désigne le plus souvent ce qu'il perçoit "
            "réellement (voir net_a_payer), pas ce champ : ne suppose jamais "
            "lequel des deux il vise sans indice clair dans la question."
        ),
    },
    "net_imposable": {
        "colonne": "net_imposable",
        "description": (
            "Net imposable : la base retenue pour le calcul de l'impôt sur "
            "le revenu, distincte du montant réellement versé au salarié. "
            "Un bulletin de paie porte DEUX montants « nets » différents "
            "(celui-ci et net_a_payer) qui ne sont presque jamais égaux : le "
            "mot « net » seul, sans précision, ne permet pas de savoir "
            "lequel des deux l'utilisateur vise."
        ),
    },
    "net_a_payer": {
        "colonne": "net_a_payer",
        "description": (
            "Net à payer : le montant réellement versé au salarié, après "
            "toutes les cotisations et retenues (y compris le prélèvement à "
            "la source). C'est le montant qu'un utilisateur désigne le plus "
            "souvent par « salaire » ou par « net » seul dans le langage "
            "courant — mais voir net_imposable, qui porte aussi le mot "
            "« net » et représente une valeur différente : sans précision, "
            "ne choisis pas silencieusement entre les deux."
        ),
    },
    "cotisations_salariales": {
        "colonne": "total_cotisations_salariales",
        "description": (
            "Total des cotisations sociales retenues sur le salaire du "
            "salarié (part salariale uniquement) — n'inclut PAS la part "
            "payée par l'employeur (voir cotisations_patronales), ni le "
            "prélèvement à la source, qui est une retenue distincte des "
            "cotisations sociales (voir prelevement_source). Le terme "
            "générique « cotisations sociales », employé sans préciser "
            "« salariales » ou « patronales », peut désigner selon le "
            "contexte soit cette part salariale seule (ce que le salarié "
            "voit réellement déduit), soit l'ensemble salarié + employeur : "
            "ce choix n'est jamais évident, ne le tranche jamais en silence."
        ),
    },
    "cotisations_patronales": {
        "colonne": "total_cotisations_patronales",
        "description": (
            "Total des cotisations sociales payées par l'employeur (part "
            "patronale) : ce montant n'est jamais déduit du salaire du "
            "salarié, il figure sur le bulletin à titre indicatif. Un terme "
            "comme « charges », employé seul, est ambigu : il peut désigner "
            "cette part patronale, la part salariale (cotisations_"
            "salariales), les deux additionnées, ou même — hors du "
            "périmètre de cet outil — le coût total employeur (salaire brut "
            "+ cotisations patronales), qu'aucun champ ne calcule "
            "directement aujourd'hui."
        ),
    },
    "cotisations_retraite": {
        "colonne": "cotisations_retraite",
        "description": (
            "Total des cotisations retraite (base + complémentaire) telles "
            "qu'extraites du bulletin. Selon la présentation d'origine du "
            "bulletin, la part salariale et la part patronale de la "
            "retraite peuvent être confondues dans ce montant unique : ce "
            "champ ne garantit donc PAS de pouvoir isoler l'une des deux si "
            "l'utilisateur le demande explicitement (« cotisations retraite "
            "salariales » par exemple) — signale cette limite plutôt que de "
            "répondre comme si la distinction était garantie."
        ),
    },
    "prelevement_source": {
        "colonne": "prelevement_source",
        "description": (
            "Montant du prélèvement à la source (impôt sur le revenu "
            "prélevé directement sur le salaire). Généralement "
            "identifiable sans ambiguïté (« prélèvement à la source », "
            "« impôt », « PAS », « impôt sur le revenu ») car c'est la "
            "seule ligne d'imposition sur un bulletin de paie français — "
            "mais reste bien distinct des cotisations sociales "
            "(cotisations_salariales, cotisations_patronales, cotisations_"
            "retraite), qui ne sont PAS de l'impôt : un terme générique "
            "comme « prélèvements » ou « retenues » peut désigner "
            "n'importe laquelle de ces lignes, pas spécifiquement celle-ci."
        ),
    },
}

Operation = Literal["somme", "moyenne", "min", "max"]


class AnalyticsSuccess(TypedDict):
    """Forme renvoyée quand le calcul a pu être exécuté."""

    operation: Operation
    champ: str
    periode: str
    resultat: float


class AnalyticsError(TypedDict):
    """Forme renvoyée quand la requête est invalide ou ne peut aboutir
    (champ inconnu, opération inconnue, aucun bulletin importé)."""

    error: str


# Union à deux formes fixes plutôt que `dict[str, Any]` : le LLM et les
# appelants savent exactement à quoi s'attendre, succès ou erreur.
AnalyticsResult = AnalyticsSuccess | AnalyticsError


class AvailablePeriod(TypedDict):
    """Bornes réelles des bulletins actuellement en base. Utilisé pour
    injecter du contexte factuel dans le prompt système (voir
    app/agent/graph.py) — jamais pour qu'un LLM devine une période
    plausible à partir de sa mémoire d'entraînement."""

    premier_mois: str
    dernier_mois: str
    nombre_bulletins: int


def _load_dataframe() -> pd.DataFrame:
    with Session(engine) as session:
        payslips = session.exec(select(Payslip)).all()

    if not payslips:
        return pd.DataFrame()

    df = pd.DataFrame([p.model_dump() for p in payslips])
    # mois_annee au format 'MM/YYYY' -> colonne datetime triable
    df["_date"] = pd.to_datetime(df["mois_annee"], format="%m/%Y")
    return df.sort_values("_date")


def _parse_mois_annee(value: str) -> datetime | None:
    """Parse une chaîne 'MM/YYYY' en `datetime` comparable, ou `None` si le
    format est invalide. Utilisé pour comparer date_debut/date_fin sur leur
    valeur réelle (mois + année), pas sur l'ordre lexicographique des
    chaînes ('09/2026' > '12/2025' en comparaison de chaînes, faux
    chronologiquement)."""
    try:
        return datetime.strptime(value, "%m/%Y")
    except ValueError:
        return None


def get_available_period() -> AvailablePeriod | None:
    """Renvoie la plage de mois réellement couverte par les bulletins
    importés, ou None si aucun bulletin n'a encore été importé. Une simple
    lecture des bornes déjà triées par _load_dataframe() — pas un calcul
    agrégé (somme/moyenne), donc pas soumis à la même contrainte
    "jamais par le LLM" que run_analytics_query : c'est un fait descriptif
    sur les données, pas une valeur dérivée qu'il faudrait interpréter."""
    df = _load_dataframe()
    if df.empty:
        return None
    return {
        "premier_mois": df["mois_annee"].iloc[0],
        "dernier_mois": df["mois_annee"].iloc[-1],
        "nombre_bulletins": len(df),
    }


def run_analytics_query(
    operation: Operation,
    champ: str,
    derniers_n_mois: int | None = None,
    date_debut: str | None = None,
    date_fin: str | None = None,
) -> AnalyticsResult:
    """Exécute un calcul exact sur les bulletins stockés.

    Args:
        operation: 'somme' | 'moyenne' | 'min' | 'max'
        champ: une des clés de FIELD_MAP
        derniers_n_mois: si fourni et que ni date_debut ni date_fin ne le
            sont, ne considère que les N bulletins les plus récents.
            Ignoré si date_debut ou date_fin est fourni.
        date_debut: borne de début de période, format 'MM/YYYY'. Si fournie
            (seule ou avec date_fin), prime sur derniers_n_mois.
        date_fin: borne de fin de période, format 'MM/YYYY'. Voir
            date_debut.
    """
    if champ not in FIELD_MAP:
        return {"error": f"Champ inconnu: {champ}. Champs valides: {list(FIELD_MAP)}"}

    df = _load_dataframe()
    if df.empty:
        return {"error": "Aucun bulletin de paie n'a encore été importé."}

    if date_debut is not None or date_fin is not None:
        debut_dt = _parse_mois_annee(date_debut) if date_debut is not None else None
        if date_debut is not None and debut_dt is None:
            return {"error": f"date_debut invalide, format attendu MM/YYYY: {date_debut!r}"}

        fin_dt = _parse_mois_annee(date_fin) if date_fin is not None else None
        if date_fin is not None and fin_dt is None:
            return {"error": f"date_fin invalide, format attendu MM/YYYY: {date_fin!r}"}

        if debut_dt is not None and fin_dt is not None and debut_dt > fin_dt:
            return {
                "error": f"date_debut ({date_debut}) est postérieure à date_fin ({date_fin})."
            }

        if debut_dt is not None:
            df = df[df["_date"] >= debut_dt]
        if fin_dt is not None:
            df = df[df["_date"] <= fin_dt]

        if df.empty:
            return {
                "error": "Aucun bulletin de paie dans la période demandée "
                f"({date_debut or '…'} à {date_fin or '…'})."
            }
    elif derniers_n_mois:
        df = df.tail(derniers_n_mois)

    column = FIELD_MAP[champ]["colonne"]
    series = df[column]

    if operation == "somme":
        value = series.sum()
    elif operation == "moyenne":
        value = series.mean()
    elif operation == "min":
        value = series.min()
    elif operation == "max":
        value = series.max()
    else:
        return {"error": f"Opération inconnue: {operation}"}

    return {
        "operation": operation,
        "champ": champ,
        "periode": f"{len(df)} mois ({df['mois_annee'].iloc[0]} à {df['mois_annee'].iloc[-1]})",
        "resultat": round(float(value), 2),
    }
