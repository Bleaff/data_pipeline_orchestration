# neudc frontend (MVP, #23)

Read-only dashboard over the [control-plane API](../README.md#-control-plane-api) (#22):
pipeline list, a per-pipeline node graph (health/throughput/queue depth), live metrics
over `/ws/metrics`, and a feed of recent frames from `/pipelines/{name}/preview`.

## Getting started

```bash
npm install
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_BASE_URL at your API server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API server must be running
separately (`python -m neudc.entrypoints.api_server`) and reachable from the browser —
its CORS defaults allow any origin, so no extra config is needed for local dev.

## Scripts

- `npm run dev` / `npm run build` / `npm run start` — Next.js dev/build/serve.
- `npm run lint` — ESLint.
- `npm test` — Vitest unit tests for the pure logic in `src/lib/` (graph layout, frame
  feed dedup, metrics-snapshot reads). No React component or E2E tests yet.

## Layout

- `src/lib/api.ts` — typed control-plane client.
- `src/lib/graphLayout.ts`, `frameFeed.ts`, `metrics.ts` — pure, unit-tested logic.
- `src/hooks/` — polling/WebSocket hooks that call into `src/lib/`.
- `src/components/PipelineGraph.tsx` (React Flow), `FrameFeed.tsx`, `StatusBadge.tsx`.
- `src/app/` — `/` (pipeline list) and `/pipelines/[name]` (graph + metrics + frames).
