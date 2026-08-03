# neudc frontend (#23, #24)

Next.js app over the [control-plane API](../README.md#-control-plane-api) (#22):

- **Dashboard** (#23, read-only): pipeline list, a per-pipeline node graph
  (health/throughput/queue depth), live metrics over `/ws/metrics`, and a feed of
  recent frames from `/pipelines/{name}/preview`.
- **Config builder** (#24, `/builder`): add/connect nodes from the `/node-types`
  catalog, edit their fields, live YAML preview, validate and start the pipeline.
- **Pre-label review** (#24, `/pipelines/{name}/review`): view/correct a pipeline's
  `CreateDataset` output — drag to add/move/resize boxes, edit class ids, save.

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
  feed dedup, metrics-snapshot reads, builder graph state, box geometry). No React
  component or E2E tests yet.

## Layout

- `src/lib/api.ts` — typed control-plane client.
- `src/lib/graphLayout.ts`, `frameFeed.ts`, `metrics.ts` — dashboard's pure logic.
- `src/lib/builderGraph.ts`, `yamlExport.ts` — config builder's pure logic (graph
  state, YAML rendering).
- `src/lib/boxGeometry.ts` — pre-label review's pure logic (normalized-box math for
  drag/move/resize).
- `src/hooks/` — polling/WebSocket hooks that call into `src/lib/`.
- `src/components/` — `PipelineGraph.tsx`/`FrameFeed.tsx`/`StatusBadge.tsx` (dashboard),
  `NodePalette.tsx`/`BuilderCanvas.tsx`/`NodeConfigForm.tsx` (builder), `BoxOverlay.tsx`
  (review).
- `src/app/` — `/` + `/pipelines/[name]` (dashboard), `/builder` (config builder),
  `/pipelines/[name]/review` (pre-label review).
