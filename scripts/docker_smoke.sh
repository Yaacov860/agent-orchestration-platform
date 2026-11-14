#!/usr/bin/env bash
# Docker smoke test: build the image, start it next to a throwaway Postgres, and require the
# API container to come up and answer GET /health with 200, once with LLM_PROVIDER=none and
# once with LLM_PROVIDER=gemini. Neither run has a Gemini key, so no model is ever called.
#
#   scripts/docker_smoke.sh
#
# SMOKE_PORT (default 18000) is the loopback port used for the check; SMOKE_TIMEOUT (default 60)
# is how many seconds each step may take; SMOKE_NO_BUILD=1 reuses an image already built.
# Everything it creates (network, containers) is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="orchestrator-smoke-$$"
IMAGE="${SMOKE_IMAGE:-orchestrator-smoke:local}"
PORT="${SMOKE_PORT:-18000}"
TIMEOUT="${SMOKE_TIMEOUT:-60}"

cleanup() {
  docker rm -f "$NAME-api" "$NAME-db" >/dev/null 2>&1 || true
  docker network rm "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

[ "${SMOKE_NO_BUILD:-0}" = 1 ] || docker build -t "$IMAGE" .

docker network create "$NAME" >/dev/null
docker run -d --name "$NAME-db" --network "$NAME" \
  -e POSTGRES_USER=orchestrator -e POSTGRES_PASSWORD=orchestrator -e POSTGRES_DB=orchestrator \
  postgres:17-alpine >/dev/null
# Over TCP: during initdb the image's temporary server listens on the socket only.
for _ in $(seq "$TIMEOUT"); do
  docker exec "$NAME-db" pg_isready -h 127.0.0.1 -U orchestrator -d orchestrator >/dev/null 2>&1 && break
  sleep 1
done

check() {
  local provider=$1 code=""
  docker rm -f "$NAME-api" >/dev/null 2>&1 || true
  docker run -d --name "$NAME-api" --network "$NAME" -p "127.0.0.1:$PORT:8000" \
    -e DATABASE_URL="postgresql+asyncpg://orchestrator:orchestrator@$NAME-db:5432/orchestrator" \
    -e LLM_PROVIDER="$provider" -e GEMINI_API_KEY= \
    "$IMAGE" >/dev/null
  for _ in $(seq "$TIMEOUT"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
    if [ "$code" = 200 ]; then
      echo "ok: LLM_PROVIDER=$provider, no key: GET /health -> 200"
      return 0
    fi
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME-api")" = true ] || break
    sleep 1
  done
  echo "FAIL: LLM_PROVIDER=$provider: GET /health -> ${code:-no answer}" >&2
  docker logs --tail 40 "$NAME-api" >&2
  return 1
}

check none
check gemini
