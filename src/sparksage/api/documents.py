"""FastAPI routes for the knowledge-document management API.

This module registers the CRUD + tag-management routes onto an existing
FastAPI app. Like :mod:`sparksage.api.app`, the web layer is a thin shell: it
only does HTTP-shaped I/O (file upload, form / JSON parsing, serialization) and
delegates every piece of real work to the framework-agnostic
:class:`~sparksage.documents.DocumentService`. FastAPI is an *optional*
dependency, imported lazily inside :func:`register_document_routes`.

Note: this module deliberately omits ``from __future__ import annotations`` so
FastAPI can resolve the lazily-imported route-parameter types (``UploadFile`` /
``File`` / ``Form`` / ``Body`` / ``Path``) via eager annotation evaluation --
see the matching note in :mod:`sparksage.api.app`.

RESTful surface (all under ``/api/v1``):

* ``POST   /documents``                 -- upload a file -> create a document
* ``GET    /documents``                 -- list (optional ``?tag=`` filter / paging)
* ``GET    /documents/{id}``            -- retrieve one
* ``PATCH  /documents/{id}``            -- update title / summary / tags
* ``DELETE /documents/{id}``            -- delete
* ``POST   /documents/{id}/tags``       -- merge tags onto a document
* ``POST   /documents/{id}/tags:auto``  -- auto-generate tags from content
* ``DELETE /documents/{id}/tags/{tag}`` -- remove a single tag
* ``GET    /tags``                      -- every distinct tag with counts
"""

import logging
from typing import Annotated, Any

from sparksage.api.schemas import (
    AddTagsRequest,
    AutoTagRequest,
    DocumentCreateResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentUpdateRequest,
    TagsResponse,
    to_document_list_response,
    to_document_out,
    to_tags_response,
)
from sparksage.documents.service import (
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceError,
)

_logger = logging.getLogger(__name__)

#: Default page size for ``GET /documents``.
DEFAULT_LIST_LIMIT = 50

#: Maximum page size for ``GET /documents``.
MAX_LIST_LIMIT = 200


def _parse_tags_field(raw: str | None) -> list[str] | None:
    """Parse a comma-separated ``tags`` form value into a clean list.

    Returns ``None`` when ``raw`` is empty so the service can fall back to
    auto-extraction instead of being told "no tags".
    """
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    tags = [p for p in parts if p]
    return tags or None


#: Default body for the auto-tag route (module-level so it is not rebuilt per
#: request and stays a valid FastAPI default).
_DEFAULT_AUTO_TAG_BODY = AutoTagRequest()


def register_document_routes(app: Any, service: DocumentService) -> Any:
    """Register the document-management routes onto ``app``.

    FastAPI / python-multipart must be installed (the ``sparksage[api]`` extra).
    """
    try:
        from fastapi import (
            Body,
            File,
            Form,
            HTTPException,
            Path,
            Query,
            UploadFile,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The SparkSage document API requires 'fastapi' and 'python-multipart'. "
            "Install them with: pip install 'sparksage[api]'"
        ) from exc

    def _not_found(exc: DocumentNotFoundError) -> None:
        raise HTTPException(status_code=404, detail=str(exc) or "document not found") from exc

    def _bad_request(exc: Exception, code: int = 422) -> None:
        raise HTTPException(status_code=code, detail=str(exc) or "bad request") from exc

    @app.post(
        "/api/v1/documents",
        response_model=DocumentCreateResponse,
        summary="Upload a Markdown document, parse it and store it",
        tags=["documents"],
    )
    async def create_document(
        file: Annotated[
            UploadFile,
            File(description="The document to upload (any convertible format)."),
        ],
        tags: Annotated[
            str | None,
            Form(description="Comma-separated tags. Omit to auto-generate from content."),
        ] = None,
        clean: Annotated[
            bool, Form(description="Apply text cleaning before parsing.")
        ] = True,
        language: Annotated[
            str, Form(description="BCP-47 language code recorded on the document.")
        ] = "en",
        auto_tag: Annotated[
            bool,
            Form(description="Auto-generate tags when none are supplied."),
        ] = True,
        top_n: Annotated[
            int, Form(ge=1, le=32, description="Max auto-generated tags.")
        ] = 5,
    ) -> DocumentCreateResponse:
        data = await file.read()
        try:
            document = service.upload(
                data,
                file.filename,
                tags=_parse_tags_field(tags),
                clean=clean,
                language=language,
                auto_tag=auto_tag,
                top_n=top_n,
            )
        except DocumentServiceError as exc:
            _bad_request(exc, code=400)
        except ValueError as exc:
            _bad_request(exc)
        assert document is not None
        return DocumentCreateResponse(document=to_document_out(document))

    @app.get(
        "/api/v1/documents",
        response_model=DocumentListResponse,
        summary="List stored documents (optionally filtered by tag)",
        tags=["documents"],
    )
    async def list_documents(
        tag: Annotated[str | None, Query(description="Only documents carrying this tag.")] = None,
        limit: Annotated[
            int, Query(ge=1, le=MAX_LIST_LIMIT, description="Page size.")
        ] = DEFAULT_LIST_LIMIT,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> DocumentListResponse:
        items = service.list(tag=tag, limit=limit, offset=offset)
        total = service.count(tag=tag)
        return to_document_list_response(items, total=total, tag=tag)

    @app.get(
        "/api/v1/documents/{document_id}",
        response_model=DocumentOut,
        summary="Retrieve a single document",
        tags=["documents"],
    )
    async def get_document(
        document_id: Annotated[str, Path(description="Document UUID.")],
    ) -> DocumentOut:
        try:
            document = service.get(document_id)
        except DocumentNotFoundError as exc:
            _not_found(exc)
        assert document is not None
        return to_document_out(document)

    @app.patch(
        "/api/v1/documents/{document_id}",
        response_model=DocumentOut,
        summary="Update a document's title, summary or tags",
        tags=["documents"],
    )
    async def update_document(
        document_id: Annotated[str, Path(description="Document UUID.")],
        payload: DocumentUpdateRequest,
    ) -> DocumentOut:
        try:
            document = service.update(
                document_id,
                title=payload.title,
                summary=payload.summary,
                tags=payload.tags,
            )
        except DocumentNotFoundError as exc:
            _not_found(exc)
        except ValueError as exc:
            _bad_request(exc)
        assert document is not None
        return to_document_out(document)

    @app.delete(
        "/api/v1/documents/{document_id}",
        status_code=204,
        summary="Delete a document",
        tags=["documents"],
    )
    async def delete_document(
        document_id: Annotated[str, Path(description="Document UUID.")],
    ) -> None:
        deleted = service.delete(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="document not found")

    @app.post(
        "/api/v1/documents/{document_id}/tags",
        response_model=DocumentOut,
        summary="Add tags to a document",
        tags=["documents", "tags"],
    )
    async def add_tags(
        document_id: Annotated[str, Path(description="Document UUID.")],
        payload: AddTagsRequest,
    ) -> DocumentOut:
        try:
            document = service.add_tags(document_id, payload.tags)
        except DocumentNotFoundError as exc:
            _not_found(exc)
        except ValueError as exc:
            _bad_request(exc)
        assert document is not None
        return to_document_out(document)

    @app.post(
        "/api/v1/documents/{document_id}/tags:auto",
        response_model=DocumentOut,
        summary="Auto-generate tags from document content",
        tags=["documents", "tags"],
    )
    async def auto_tag_document(
        document_id: Annotated[str, Path(description="Document UUID.")],
        payload: Annotated[AutoTagRequest, Body()] = _DEFAULT_AUTO_TAG_BODY,
    ) -> DocumentOut:
        try:
            document = service.extract_tags(
                document_id,
                top_n=payload.top_n,
                replace_existing=payload.replace_existing,
            )
        except DocumentNotFoundError as exc:
            _not_found(exc)
        assert document is not None
        return to_document_out(document)

    @app.delete(
        "/api/v1/documents/{document_id}/tags/{tag}",
        response_model=DocumentOut,
        summary="Remove a single tag from a document",
        tags=["documents", "tags"],
    )
    async def remove_tag(
        document_id: Annotated[str, Path(description="Document UUID.")],
        tag: Annotated[str, Path(description="Tag to remove (case-insensitive).")],
    ) -> DocumentOut:
        try:
            document = service.remove_tag(document_id, tag)
        except DocumentNotFoundError as exc:
            _not_found(exc)
        assert document is not None
        return to_document_out(document)

    @app.get(
        "/api/v1/tags",
        response_model=TagsResponse,
        summary="List every distinct tag with its document count",
        tags=["tags"],
    )
    async def list_tags() -> TagsResponse:
        return to_tags_response(service.tags())

    return app
