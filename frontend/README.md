# KernelSight web UI

A Next.js (App Router) single-page UI over the KernelSight API. It is built
as a static export, so `kernelsight serve` can host it without Node.

```powershell
# development: UI on :3000 with hot reload, API on :8000
kernelsight serve --reload        # terminal 1 (repo root, venv active)
npm run dev                       # terminal 2 (this folder)

# production-style: build once, then one server for API + UI on :8000
npm run build                     # writes out/
kernelsight serve
```

| File | Purpose |
|------|---------|
| `src/lib/api.ts` | typed API client; types mirror the JSON report schema |
| `src/app/page.tsx` | page state: mode, inputs, results, export |
| `src/components/SourceEditor.tsx` | source input, examples, compiler options |
| `src/components/AnalysisView.tsx` | metrics, instruction mix, findings |
| `src/components/CfgGraph.tsx` | SVG arc diagram of the control-flow graph |
| `src/components/CompareView.tsx` | metric deltas, changed mnemonics, normalised diff |

Deep links such as `/?example=matmul.cu&run=analyze` or
`/?example=math_heavy.cu&run=compare` preload an example and run it.
