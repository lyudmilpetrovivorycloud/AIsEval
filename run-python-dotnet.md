# Running the AiDotNet & PyTorch Benchmark Projects

This guide explains how to start both benchmark hosts, lists the API endpoints
used to compare performance, and gives example Postman / HTTP calls for each
endpoint. Where an endpoint expects a CSV upload, the body section is annotated
with a comment showing which file to load.

All commands are written for **Windows PowerShell** (the repo's primary shell);
Git Bash / macOS / Linux equivalents are noted where they differ.

---

## 1. Run the projects

### AiDotNet host (C# / .NET)

From `aidotnet-benchmarks/`:

```powershell
dotnet restore
dotnet build -c Release
dotnet run -c Release
```

Binds (from `Properties/launchSettings.json`):

- `https://localhost:7001`
- `http://localhost:7000`

> The dev cert is self-signed — pass `-k` to curl, or disable SSL verification
> in Postman (**Settings → General → SSL certificate verification: OFF**) when
> calling the `https` URL.

### PyTorch host (Python / FastAPI)

One-time setup from `pytorch-benchmarks/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked by execution policy, call .\.venv\Scripts\python.exe directly
python -m pip install --upgrade pip
python -m pip install -e .
```

Start the API:

```powershell
python -m uvicorn pytorch_benchmarks.api:app --host 127.0.0.1 --port 8000
# or the installed console script:
pytorch-bench-api
```

Binds: `http://localhost:8000`

### Recommended layout for a head-to-head run

| Terminal | Directory | Command |
| --- | --- | --- |
| 1 | `aidotnet-benchmarks` | `dotnet run -c Release` |
| 2 | `pytorch-benchmarks` | `python -m uvicorn pytorch_benchmarks.api:app --host 127.0.0.1 --port 8000` |
| 3 | (any) | the comparison POST (see `/api/Both/Benchmark` below) |

> A full four-model benchmark saturates the CPU for several minutes per side.
> Each host rejects a second concurrent benchmark run with **HTTP 409** — never
> run two at once or the measurements corrupt each other.

---

## 2. API endpoints for performance comparison

The two hosts expose mirrored routes so the same workload can be measured on
each framework. `{ai}` = `https://localhost:7001`, `{py}` = `http://localhost:8000`.

### Neural-network benchmark (MLP / CNN / LSTM / Transformer)

| Method | Path | AiDotNet | PyTorch | Purpose |
| --- | --- | --- | --- | --- |
| GET | `/api/Benchmark/Test` | ✅ | ✅ | Readiness probe |
| POST | `/api/Benchmark/Models` | ✅ | ✅ | Run the four-model benchmark, return JSON report |

Query parameters (identical on both sides):
`models` (default `mlp,cnn,lstm,transformer`), `epochs` (3), `trainBatches` (20),
`batchSize` (64), `inferenceIterations` (100), `warmupIterations` (10), `seed` (1234).
The PyTorch side also accepts `device` (default `cpu`) and `threads` (default 8).

### CSV regression

| Method | Path | AiDotNet | PyTorch | Notes |
| --- | --- | --- | --- | --- |
| GET | `/api/Regression/Test` | ✅ | ✅ | Returns `ping` |
| POST | `/api/Regression/SimpleRegression` | ✅ | ✅ | CSV upload → predictions (see fairness note) |
| POST | `/api/Regression/MultipleRegression` | ✅ (via SimpleRegression*) | ✅ | nn.Linear + Adam loop on PyTorch side |
| POST | `/api/Regression/Predict` | ❌ | ✅ | PyTorch only — raw `torch.linalg.lstsq` (LAPACK) |

\* On the AiDotNet side a single `SimpleRegression` route handles both shapes:
one feature column runs `SimpleRegression<double>`, multiple feature columns run
`MultipleRegression<double>` through the full `AiModelBuilder` lifecycle.

Query parameter: `UseGPU` (default `false`; currently CPU-only on both sides —
`UseGPU=true` is echoed as `gpuRequested: true` but `gpuUsed` stays `false`).

> **Fairness note:** PyTorch `/Predict` (raw LAPACK) is *not* an apples-to-apples
> comparison against AiDotNet's builder pipeline. For a like-for-like regression
> comparison use `/MultipleRegression` on both hosts.

### Fan-out (drive both frameworks from one call — AiDotNet host only)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/Both/Benchmark` | Runs `/api/Benchmark/Models` on **both** hosts sequentially, returns both reports side by side |
| POST | `/api/Both/SimpleRegression` | Forwards a CSV upload to both regression endpoints, returns both responses |

> `/api/Both/Benchmark` addresses the AiDotNet side via the incoming request's
> own scheme/host, then calls the PyTorch host at `http://localhost:8000`.
> `/api/Both/SimpleRegression` forwards to a **hardcoded** AiDotNet endpoint at
> `https://localhost:50724` and PyTorch at `http://localhost:8000` — to use it,
> start the AiDotNet host on port 50724 (e.g. `dotnet run -c Release --urls https://localhost:50724`).

---

## 3. Example HTTP / Postman calls

Body comments use `// ...` to indicate which CSV file to load. Sample CSVs live
under `TestData/RegressionTestData/` (`Simple/SmallSet`, `Complex/SmallSet`,
`Complex/LargeSet`), each containing a `features.csv` (feature columns + target
in the last column) and a `tests.csv` (feature columns only).

### Readiness probes

```http
GET https://localhost:7001/api/Benchmark/Test
GET http://localhost:8000/api/Benchmark/Test
GET https://localhost:7001/api/Regression/Test
GET http://localhost:8000/api/Regression/Test
```

curl:

```bash
curl -k https://localhost:7001/api/Benchmark/Test
curl    http://localhost:8000/api/Benchmark/Test
```

### Run the four-model benchmark — AiDotNet

```http
POST https://localhost:7001/api/Benchmark/Models?models=mlp,cnn,lstm,transformer&epochs=3&trainBatches=20&batchSize=64&inferenceIterations=100&warmupIterations=10&seed=1234

# No body. Returns the JSON benchmark report.
```

### Run the four-model benchmark — PyTorch

```http
POST http://localhost:8000/api/Benchmark/Models?models=mlp,cnn,lstm,transformer&device=cpu&threads=8

# No body. Returns the JSON benchmark report.
```

curl quick-checks (single model):

```bash
curl -k -X POST "https://localhost:7001/api/Benchmark/Models?models=mlp"   # AiDotNet only
curl    -X POST "http://localhost:8000/api/Benchmark/Models?models=mlp"    # PyTorch only
```

### Run BOTH benchmarks from one call (fan-out)

```http
POST https://localhost:7001/api/Both/Benchmark?models=mlp,cnn,lstm,transformer

# No body. Runs AiDotNet then PyTorch sequentially; returns both reports.
# Use a long client timeout (a full run is minutes per side).
```

curl:

```bash
curl -k -X POST "https://localhost:7001/api/Both/Benchmark?models=mlp,cnn,lstm,transformer" -o both.json
```

### CSV regression — AiDotNet

```http
POST https://localhost:7001/api/Regression/SimpleRegression?UseGPU=false
Content-Type: multipart/form-data        # let Postman generate the boundary — do NOT set this header manually

# Body → form-data:
#   features = [File]  // load TestData/RegressionTestData/Simple/SmallSet/features.csv
#   tests    = [File]  // load TestData/RegressionTestData/Simple/SmallSet/tests.csv
```

### CSV regression — PyTorch (framework-symmetric route)

```http
POST http://localhost:8000/api/Regression/MultipleRegression?UseGPU=false
Content-Type: multipart/form-data        # auto-generated boundary; do NOT set manually

# Body → form-data:
#   features = [File]  // load TestData/RegressionTestData/Complex/SmallSet/features.csv
#   tests    = [File]  // load TestData/RegressionTestData/Complex/SmallSet/tests.csv
```

### CSV regression — fan-out to both

```http
POST https://localhost:50724/api/Both/SimpleRegression?UseGPU=false
Content-Type: multipart/form-data        # auto-generated boundary; do NOT set manually

# Start the AiDotNet host on 50724 first (dotnet run -c Release --urls https://localhost:50724).
# Body → form-data:
#   features = [File]  // load TestData/RegressionTestData/Simple/SmallSet/features.csv
#   tests    = [File]  // load TestData/RegressionTestData/Simple/SmallSet/tests.csv
```

curl equivalent (file upload):

```bash
curl -k -X POST "https://localhost:7001/api/Regression/SimpleRegression?UseGPU=false" \
  -F "features=@TestData/RegressionTestData/Simple/SmallSet/features.csv" \
  -F "tests=@TestData/RegressionTestData/Simple/SmallSet/tests.csv"

curl -X POST "http://localhost:8000/api/Regression/MultipleRegression?UseGPU=false" \
  -F "features=@TestData/RegressionTestData/Complex/SmallSet/features.csv" \
  -F "tests=@TestData/RegressionTestData/Complex/SmallSet/tests.csv"
```

### Pasting CSV instead of uploading a file

Both regression endpoints also accept the CSV as **Text** form-data values
(handy in Postman when file selection misbehaves): set the `features` and
`tests` rows to **Text** and paste the CSV content directly, e.g.

```
features = x,target\n1,2.1\n2,4\n...   // pasted instead of loading features.csv
tests    = x\n6\n7\n...                // pasted instead of loading tests.csv
```

---

## 4. Postman tips for the CSV uploads

If a regression call returns `No multipart/form-data body was received`:

1. **Headers** → delete or disable any manual `Content-Type` header. Postman
   must generate `multipart/form-data; boundary=...` from the Body tab itself.
2. **Body** → select **form-data**.
3. Add rows with exact keys `features` and `tests`; switch each row type from
   **Text** to **File** and choose the CSV (or keep **Text** and paste CSV).
4. A yellow warning icon next to a file means Postman can't read it — reselect
   it, or add its folder under **Settings → Working Directory**.
5. For the `https://localhost:7001` host, turn **SSL certificate verification
   OFF** (self-signed dev cert).

An existing Postman collection with these requests pre-built lives at
`postman/Testing.postman_collection.json` (folders: **AIDotNet**, **PyTourch**,
**Both**).

---

## 5. Reading the response

Each regression response reports `timings` in milliseconds (`timing_unit:
"milliseconds"`) and also sets a `Server-Timing: app;dur=...` header so you can
compare server-side duration against Postman's client-side round-trip time. The
benchmark report carries per-model training/inference timing, CPU/GPU
utilization, peak memory, throughput at batch sizes 1/8/32/128, and
warm-up-versus-steady-state numbers. Published methodology and numbers are in
`Reporting/findings.md`.
