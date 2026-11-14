from fastapi import APIRouter, Request

from orchestrator.agents.catalog import AgentCatalog
from orchestrator.api.schemas import AgentOut

router = APIRouter(tags=["agents"])


@router.get("/agents")
async def list_agents(request: Request) -> list[AgentOut]:
    """Every agent the planner may use, what each action does, and whether it searches the
    web. The same catalog the planner's prompt and plan validation are built from."""
    catalog: AgentCatalog = request.app.state.catalog
    return [AgentOut.of(spec) for spec in catalog.specs]
