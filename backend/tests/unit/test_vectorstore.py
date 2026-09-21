"""
Tests unitaires de `app/services/vectorstore.py` — limités à `_chunk_text`
et `_build_hits`, la logique métier pure du fichier (pas d'E/S).

`ChromaVectorStore` elle-même (index()/search()) n'est volontairement pas
testée ici : voir RAPPORT.md, "zones volontairement non testées", pour la
justification complète (E/S réelle vers ChromaDB, téléchargement réseau du
modèle d'embedding au premier appel). Son contrat (Protocol `VectorStore`)
est exercé de bout en bout ailleurs via `FakeVectorStore`
(tests/conftest.py) dans les tests fonctionnels de `/api/upload` et
`/api/chat`. `_build_hits` — le filtrage par seuil de distance qu'utilise
`search()` — est en revanche une fonction pure (assemblage + comparaison
de flottants, aucun accès à Chroma) et testable directement ici.
"""
from __future__ import annotations

from app.services.vectorstore import SIMILARITY_DISTANCE_THRESHOLD, _build_hits, _chunk_text


def test_chunk_text_empty_string_returns_one_empty_chunk():
    assert _chunk_text("") == [""]


def test_chunk_text_shorter_than_chunk_size_returns_a_single_chunk():
    assert _chunk_text("short text", chunk_size=800, overlap=100) == ["short text"]


def test_chunk_text_splits_with_overlap():
    text = "abcdefghijklmnopqrstuvwxy"  # 25 caractères
    chunks = _chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxy"]
    # Chaque chunk (sauf le dernier) chevauche le suivant sur `overlap`
    # caractères : c'est ce qui évite de couper une phrase pertinente pile
    # à la frontière entre deux chunks lors de la recherche par similarité.
    for current, following in zip(chunks, chunks[1:]):
        assert current[-3:] == following[:3]


def test_chunk_text_covers_the_full_text_without_gaps():
    # Un texte assez long pour produire plusieurs chunks (4, avec ces
    # paramètres) : chaque position du texte d'origine doit être couverte
    # par au moins un chunk, sans trou entre deux chunks consécutifs.
    text = "".join(f"{i:04d}" for i in range(50))  # 200 caractères, contenu vérifiable
    chunks = _chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) == 5
    step = 50 - 10
    for i, chunk in enumerate(chunks):
        start = i * step
        assert chunk == text[start : start + 50]
    assert chunks[0].startswith(text[:10])
    assert chunks[-1].endswith(text[-10:])


def test_build_hits_keeps_results_at_or_below_the_threshold():
    documents = ["texte pertinent", "texte juste à la limite"]
    metadatas = [{"mois_annee": "01/2025"}, {"mois_annee": "02/2025"}]
    distances = [0.5, SIMILARITY_DISTANCE_THRESHOLD]  # == le seuil, pas au-dessus

    hits = _build_hits(documents, metadatas, distances)

    assert hits == [
        {"text": "texte pertinent", "mois_annee": "01/2025"},
        {"text": "texte juste à la limite", "mois_annee": "02/2025"},
    ]


def test_build_hits_excludes_results_above_the_threshold():
    documents = ["texte pertinent", "texte hors sujet"]
    metadatas = [{"mois_annee": "01/2025"}, {"mois_annee": "02/2025"}]
    distances = [0.5, SIMILARITY_DISTANCE_THRESHOLD + 0.01]

    hits = _build_hits(documents, metadatas, distances)

    assert hits == [{"text": "texte pertinent", "mois_annee": "01/2025"}]


def test_build_hits_returns_empty_list_when_everything_is_above_the_threshold():
    documents = ["texte hors sujet 1", "texte hors sujet 2"]
    metadatas = [{"mois_annee": "01/2025"}, {"mois_annee": "02/2025"}]
    distances = [1.8, 2.4]

    assert _build_hits(documents, metadatas, distances) == []


def test_build_hits_returns_empty_list_for_empty_inputs():
    # Le cas "Collection.query() n'a renvoyé aucun résultat" (base vide ou
    # aucun chunk indexé) — même code de filtrage que "tout est hors seuil",
    # pas un cas spécial.
    assert _build_hits([], [], []) == []


def test_build_hits_accepts_a_custom_threshold():
    documents = ["texte"]
    metadatas = [{"mois_annee": "01/2025"}]
    distances = [0.9]

    assert _build_hits(documents, metadatas, distances, threshold=0.5) == []
    assert _build_hits(documents, metadatas, distances, threshold=1.0) == [
        {"text": "texte", "mois_annee": "01/2025"}
    ]


def test_similarity_threshold_was_tightened_from_the_previous_permissive_value():
    # Verrou explicite : 1.5 (rapport précédent) laissait passer
    # virtuellement tout ce qui contenait "salaire", pertinent ou non (voir
    # RAPPORT.md). Ce test échoue si un futur changement relâche le seuil
    # au-delà de la valeur mesurée dans cette tâche, sans passer par la
    # même méthode de mesure empirique.
    assert SIMILARITY_DISTANCE_THRESHOLD == 1.1


def test_build_hits_at_the_current_threshold_filters_the_bug_report_false_positive():
    """Verrouille le cas précis du bug rapporté, avec les distances
    mesurées empiriquement contre un corpus réaliste de bulletins (script
    de mesure et méthodologie complète dans RAPPORT.md) : la question
    hors-sujet "quelle différence entre salaire moyen et salaire médian ?"
    (distance mesurée : 1.133) ne doit plus passer le seuil actuel, alors
    qu'une question pertinente reformulée dans le même registre lexical
    (distance mesurée : 0.925, la moins bonne des 6 paraphrases testées)
    doit toujours passer."""
    documents = [
        "extrait pertinent (paraphrase de la CSG)",
        "extrait hors-sujet (salaire moyen vs salaire médian)",
    ]
    metadatas = [{"mois_annee": "03/2026"}, {"mois_annee": "04/2026"}]
    distances = [0.925, 1.133]

    hits = _build_hits(documents, metadatas, distances)

    assert hits == [{"text": documents[0], "mois_annee": "03/2026"}]
