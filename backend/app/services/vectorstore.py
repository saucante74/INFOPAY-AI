"""
Couche RAG : indexation du texte brut de chaque bulletin dans ChromaDB,
et recherche par similarité pour répondre aux questions explicatives
("à quoi correspond la ligne X ?").

ChromaDB tourne en mode embarqué (PersistentClient), pas besoin de
serveur séparé pour ce projet.

`ChromaVectorStore` est l'implémentation ChromaDB du Protocol
`app.interfaces.VectorStore`. Passer à FAISS/Pinecone revient à
écrire une classe exposant les mêmes `index()` / `search()` et à la câbler
dans `app/dependencies.py`.
"""
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions

CHROMA_PATH = "./data/chroma_data"
COLLECTION_NAME = "payslips"

# `Collection.query()` renvoie par défaut une distance L2 au carré sur des
# embeddings normalisés (all-MiniLM-L6-v2, la fonction par défaut de
# Chroma) : 0.0 = chunk identique à la requête, jusqu'à 4.0 = opposé.
# Vérifié empiriquement (voir RAPPORT.md) : un texte identique donne 0.0,
# une requête sans aucun rapport lexical donne ~1.8. Ce seuil est
# volontairement conservateur — il n'écarte que les cas franchement
# dégénérés. Les tests empiriques montrent que l'embedding par défaut (un
# modèle anglophone généraliste, sur du français, sur des chunks courts) ne
# sépare PAS de façon fiable une question française hors-sujet d'une
# question française pertinente dans la zone 1.0-1.3 : un seuil plus
# agressif y rejetterait autant de bonnes réponses que de mauvaises. Voir
# RAPPORT.md pour la méthodologie et ses limites.
SIMILARITY_DISTANCE_THRESHOLD = 1.5


def _build_hits(
    documents: list[str],
    metadatas: list[Mapping[str, Any]],
    distances: list[float],
    threshold: float = SIMILARITY_DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Assemble les hits (texte + source `mois_annee`) à partir des trois
    listes parallèles renvoyées par `Collection.query()`, en excluant les
    chunks dont la distance dépasse `threshold` — trop éloignés de la
    requête pour être présentés comme une réponse pertinente. Fonction pure
    (pas d'E/S), donc testable sans ouvrir de collection ChromaDB réelle,
    contrairement à `ChromaVectorStore.search()` qui l'appelle."""
    hits = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        if distance > threshold:
            continue
        hits.append({"text": doc, "mois_annee": meta.get("mois_annee")})
    return hits


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Découpage simple par fenêtre glissante. Les bulletins sont courts
    (1-2 pages) donc un chunking basique suffit largement."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks or [text]


class ChromaVectorStore:
    """Implémentation de `VectorStore` adossée à une collection ChromaDB
    persistante locale."""

    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME) -> None:
        client = chromadb.PersistentClient(path=path)
        # Embedding function par défaut de Chroma (all-MiniLM-L6-v2, local, gratuit).
        # Suffisant pour ce cas d'usage et évite une dépendance à une API d'embedding.
        # cast : la fonction par défaut de Chroma est déclarée
        # EmbeddingFunction[Documents], alors que get_or_create_collection
        # attend l'EmbeddingFunction[Embeddable] plus large. Invariance des
        # génériques côté chromadb, pas une vraie incompatibilité.
        embedding_fn = cast(EmbeddingFunction[Embeddable],
                            embedding_functions.DefaultEmbeddingFunction())
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    def index(self, payslip_id: int, mois_annee: str, raw_text: str) -> None:
        chunks = _chunk_text(raw_text)
        ids = [f"payslip-{payslip_id}-chunk-{i}" for i in range(len(chunks))]
        metadatas: list[Mapping[str, str | int | float | bool]] = [
            {"payslip_id": payslip_id, "mois_annee": mois_annee} for _ in chunks
        ]

        self._collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    def search(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        results = self._collection.query(query_texts=[query], n_results=n_results)

        # QueryResult est un TypedDict total=False : les clés existent mais
        # peuvent valoir None. `or` couvre les deux cas (absente ou None).
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return _build_hits(documents, metadatas, distances)

    def delete(self, payslip_id: int) -> None:
        self._collection.delete(where={"payslip_id": payslip_id})


@lru_cache(maxsize=1)
def get_default_vector_store() -> ChromaVectorStore:
    """Instance partagée par défaut. Câblée dans `app/dependencies.py`.

    Construite à la première utilisation plutôt qu'à l'import du module :
    le client Chroma n'est ouvert que si l'application s'en sert réellement.
    """
    return ChromaVectorStore()
