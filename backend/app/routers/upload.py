import logging
from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import ValidationError
from sqlmodel import Session

from app.db import get_session
from app.dependencies import get_extractor, get_vector_store
from app.interfaces import Extractor, VectorStore
from app.models.payslip import Payslip
from app.rate_limit import upload_rate_limit
from app.services.extraction import PayslipExtractionError, extract_text_from_pdf

router = APIRouter(prefix="/api", tags=["upload"])
logger = logging.getLogger(__name__)

# Message générique volontairement : le détail Pydantic brut (noms de champs
# internes, type attendu, lien vers errors.pydantic.dev...) n'a aucun sens
# pour l'utilisateur final et ne doit jamais atteindre le frontend — voir
# PayslipExtractionError et le `except` ci-dessous, qui logue le détail
# technique côté serveur avant de renvoyer ce texte.
EXTRACTION_ERROR_MESSAGE = (
    "Certaines informations n'ont pas pu être lues correctement sur ce bulletin "
    "(mise en page inhabituelle ou texte peu lisible). Essayez avec un bulletin "
    "au format plus standard, ou consultez les exemples fournis dans la page Aide."
)


@router.post("/upload", dependencies=[Depends(upload_rate_limit)])
async def upload_payslip(
    file: UploadFile,
    session: Session = Depends(get_session),
    extractor: Extractor = Depends(get_extractor),
    vector_store: VectorStore = Depends(get_vector_store),
) -> Payslip:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    file_bytes = await file.read()

    raw_text = extract_text_from_pdf(file_bytes)
    try:
        extracted = extractor.extract(raw_text)
    except (PayslipExtractionError, ValidationError) as exc:
        logger.warning("Échec d'extraction pour %r : %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=EXTRACTION_ERROR_MESSAGE) from exc

    payslip = Payslip(
        **extracted.model_dump(),
        raw_text=raw_text,
        filename=file.filename,
    )
    session.add(payslip)
    session.commit()
    session.refresh(payslip)

    # Indexation vectorielle pour le RAG, après avoir obtenu l'ID en base.
    # session.refresh() a peuplé la clé primaire auto-incrémentée : l'Optional
    # du modèle SQLModel n'est plus possible ici.
    assert payslip.id is not None
    vector_store.index(payslip.id, payslip.mois_annee, raw_text)

    return payslip


@router.get("/payslips")
def list_payslips(session: Session = Depends(get_session)) -> Sequence[Payslip]:
    from sqlmodel import select

    payslips = session.exec(select(Payslip).order_by(Payslip.mois_annee)).all()
    return payslips


@router.delete("/payslips/{payslip_id}", status_code=204)
def delete_payslip(
    payslip_id: int,
    session: Session = Depends(get_session),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    payslip = session.get(Payslip, payslip_id)
    if payslip is None:
        raise HTTPException(status_code=404, detail=f"Bulletin {payslip_id} introuvable.")

    vector_store.delete(payslip_id)

    session.delete(payslip)
    session.commit()
