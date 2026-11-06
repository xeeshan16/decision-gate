"""HTTP endpoints for drift monitoring: run checks, read the check history."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from decision_gate.db.models import DriftCheckRecord
from decision_gate.db.repository import DriftCheckRepository
from decision_gate.drift import CheckStatus
from decision_gate.drift_monitor import DriftMonitor
from decision_gate.questions import QuestionCatalog
from decision_gate.reviews import as_utc
from decision_gate.routing import Router
from decision_gate.selection import SelectionError

AlertState = Literal["sent", "failed", "unconfigured"]


class CheckOut(BaseModel):
    id: uuid.UUID
    question: str
    question_version: int
    status: CheckStatus = Field(
        description="'drift' if any signal drifted, 'ok' if at least one could be measured, "
        "otherwise 'insufficient'."
    )
    checked_at: datetime
    window_start: datetime
    window_end: datetime
    reference_start: datetime
    reference_end: datetime
    signals: list[dict[str, Any]]
    summary: dict[str, Any]
    action: str = Field(
        description="none, tightened, suspended, already_tightened, not_tightened or skipped."
    )
    action_reason: str
    previous_rule_id: uuid.UUID | None
    rule_id: uuid.UUID | None = Field(description="The tightened rule, if one was installed.")
    alert: AlertState | None
    alert_event: str | None
    alert_error: str | None
    policy: dict[str, Any]

    @classmethod
    def from_record(cls, record: DriftCheckRecord) -> CheckOut:
        return cls(
            id=record.id,
            question=record.question_id,
            question_version=record.question_version,
            status=record.status,
            checked_at=as_utc(record.checked_at),
            window_start=as_utc(record.window_start),
            window_end=as_utc(record.window_end),
            reference_start=as_utc(record.reference_start),
            reference_end=as_utc(record.reference_end),
            signals=record.signals,
            summary=record.summary,
            action=record.action,
            action_reason=record.action_reason,
            previous_rule_id=record.previous_rule_id,
            rule_id=record.rule_id,
            alert=record.alert,
            alert_event=record.alert_event,
            alert_error=record.alert_error,
            policy=record.policy,
        )


class CheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(
        default=None, description="Check one question; omit to check every monitored one."
    )
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")


def drift_router(
    catalog: QuestionCatalog,
    monitor: DriftMonitor,
    checks: DriftCheckRepository,
    router: Router,
) -> APIRouter:
    api = APIRouter(prefix="/v1/drift", tags=["drift"])

    @api.post(
        "/check",
        responses={
            404: {"description": "Unknown question, or it is not monitored."},
            422: {"description": "The stored decisions cannot be analysed."},
        },
    )
    async def run_check(request: CheckIn) -> dict[str, list[CheckOut]]:
        """Measure drift now. On drift, tighten routing (if the policy says so) and alert.

        Run it on a schedule, e.g. hourly from a cron job.
        """
        if request.question is not None:
            try:
                targets = [catalog.get(request.question, request.version)]
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
            if not monitor.monitored(targets[0]):
                raise HTTPException(
                    status_code=404,
                    detail=f"{request.question} has no routing policy or drift is disabled",
                )
        else:
            targets = list(catalog.questions)
        try:
            records = await monitor.check_all(targets)
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            router.invalidate()
        return {"checks": [CheckOut.from_record(r) for r in records]}

    @api.get("/status")
    async def drift_status() -> dict[str, list[dict[str, Any]]]:
        """The latest check of every monitored question version (``null`` if never checked)."""
        out: list[dict[str, Any]] = []
        for question in catalog.questions:
            if not monitor.monitored(question):
                continue
            latest = await checks.latest(question.id, question.version)
            out.append(
                {
                    "question": question.id,
                    "question_version": question.version,
                    "latest": CheckOut.from_record(latest) if latest else None,
                }
            )
        return {"questions": out}

    @api.get("/checks")
    async def list_checks(
        question: str | None = None,
        version: Annotated[int | None, Query(ge=1)] = None,
        status: CheckStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, list[CheckOut]]:
        """Check history, newest first."""
        records = await checks.find(
            question_id=question, question_version=version, status=status, limit=limit
        )
        return {"checks": [CheckOut.from_record(r) for r in records]}

    @api.get("/checks/{check_id}")
    async def get_check(check_id: uuid.UUID) -> CheckOut:
        record = await checks.get(check_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown drift check")
        return CheckOut.from_record(record)

    return api


__all__ = ["CheckIn", "CheckOut", "drift_router"]
