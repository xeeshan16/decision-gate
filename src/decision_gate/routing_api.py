"""HTTP endpoints for routing: policies, rule refreshes and the rule history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from decision_gate.db.models import RoutingRuleRecord
from decision_gate.db.repository import RoutingRuleRepository
from decision_gate.questions import QuestionCatalog
from decision_gate.routing import Router, RoutingRefresher
from decision_gate.selection import AnyQuestion, SelectionError

RuleStatus = Literal["active", "retired"]
RoutingState = Literal["no_policy", "no_rule", "policy_changed", "calibrator_changed", "current"]


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


class RuleOut(BaseModel):
    id: uuid.UUID
    question: str
    question_version: int
    status: RuleStatus
    policy_fingerprint: str
    policy: dict[str, Any]
    calibrator_id: uuid.UUID | None
    auto_threshold: float | None = Field(
        description="Calibrated probability at or above which decisions are automated."
    )
    verify_threshold: float | None = Field(
        description="At or above this (and below auto), a person verifies the answer."
    )
    conformal: dict[str, Any] | None = Field(description="Answer-set predictor parameters.")
    labeled: int
    eligible: int = Field(description="Labeled decisions the thresholds were selected on.")
    reason: str
    evaluation: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_record(cls, record: RoutingRuleRecord) -> RuleOut:
        return cls(
            id=record.id,
            question=record.question_id,
            question_version=record.question_version,
            status=record.status,
            policy_fingerprint=record.policy_fingerprint,
            policy=record.policy,
            calibrator_id=record.calibrator_id,
            auto_threshold=record.auto_threshold,
            verify_threshold=record.verify_threshold,
            conformal=record.conformal,
            labeled=record.labeled,
            eligible=record.eligible,
            reason=record.reason,
            evaluation=record.evaluation,
            created_at=_utc(record.created_at),
        )


class RefreshIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(
        default=None, description="Refresh one question; omit to refresh every routed one."
    )
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")


class StatusOut(BaseModel):
    question: str
    question_version: int
    state: RoutingState
    policy: dict[str, Any] | None
    policy_fingerprint: str | None
    rule: RuleOut | None


def routing_router(
    catalog: QuestionCatalog,
    router: Router,
    refresher: RoutingRefresher,
    rules: RoutingRuleRepository,
) -> APIRouter:
    api = APIRouter(prefix="/v1/routing", tags=["routing"])
    config = router.config

    def routed(question: AnyQuestion) -> bool:
        return config.policy_for(question.id) is not None

    @api.get("/status")
    async def routing_status() -> dict[str, list[StatusOut]]:
        """For every question version: its policy, its active rule and whether that rule
        still applies (``current``) or every decision is going to a human and why."""
        out: list[StatusOut] = []
        for question in catalog.questions:
            policy = config.policy_for(question.id)
            rule = await rules.active(question.id, question.version)
            state: RoutingState
            if policy is None:
                state = "no_policy"
            elif rule is None:
                state = "no_rule"
            elif rule.policy_fingerprint != policy.fingerprint():
                state = "policy_changed"
            else:
                active = await refresher.active_calibrator_id(question)
                state = "calibrator_changed" if rule.calibrator_id != active else "current"
            out.append(
                StatusOut(
                    question=question.id,
                    question_version=question.version,
                    state=state,
                    policy=policy.model_dump(mode="json") if policy else None,
                    policy_fingerprint=policy.fingerprint() if policy else None,
                    rule=RuleOut.from_record(rule) if rule else None,
                )
            )
        return {"questions": out}

    @api.post(
        "/refresh",
        responses={
            404: {"description": "Unknown question, or no policy covers it."},
            422: {"description": "The stored decisions cannot be analysed."},
        },
    )
    async def refresh(request: RefreshIn) -> dict[str, list[RuleOut]]:
        """Select thresholds from labeled decisions and store them as the active rules.

        Run it after a refit, after changing a policy, or on a schedule as labels arrive.
        """
        if request.question is not None:
            try:
                targets = [catalog.get(request.question, request.version)]
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
            if not routed(targets[0]):
                raise HTTPException(
                    status_code=404, detail=f"no routing policy covers {request.question}"
                )
        else:
            targets = [q for q in catalog.questions if routed(q)]
        stored: list[RuleOut] = []
        try:
            for question in targets:
                result = await refresher.refresh(question)
                stored.append(RuleOut.from_record(result.rule))
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            router.invalidate()
        return {"rules": stored}

    @api.get("/rules")
    async def list_rules(
        question: str | None = None,
        version: Annotated[int | None, Query(ge=1)] = None,
        status: RuleStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, list[RuleOut]]:
        """Rule history, newest first within each question version."""
        records = await rules.find(
            question_id=question, question_version=version, status=status, limit=limit
        )
        return {"rules": [RuleOut.from_record(r) for r in records]}

    @api.get("/rules/{rule_id}")
    async def get_rule(rule_id: uuid.UUID) -> RuleOut:
        record = await rules.get(rule_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown routing rule")
        return RuleOut.from_record(record)

    return api


__all__ = ["RefreshIn", "RuleOut", "StatusOut", "routing_router"]
