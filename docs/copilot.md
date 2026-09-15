# HDB Resale Decision Copilot

The copilot extends the existing analytics project with an optional,
evidence-grounded language layer. It is designed to explain verified market
summaries, comparable transactions and model diagnostics in plain language.
It does not replace the analytical pipeline and does not value or forecast a
specific flat.

## Design principle

Every answer follows one controlled path:

~~~text
Explicit user controls
        |
        v
Deterministic pandas calculations
        |
        v
Small evidence packet with stable fact IDs
        |
        v
Structured OpenAI evidence-ID selection
        |
        v
Citation and draft validation
        |
        v
Deterministic visible-claim rendering
~~~

The full transaction snapshot is never sent to the language model. The model
receives only the user's question and a compact JSON evidence packet calculated
by the application. Its prose fields are treated as untrusted scratch output and
never displayed. Only accepted evidence-ID selections cross the boundary; the
application reconstructs every visible point from the cited record.

## Supported analyses

- **Market brief:** transaction count, median resale price, median price per
  square metre, observed coverage and endpoint movement for explicit filters.
- **Comparable sales:** the existing staged comparable-sales matcher, observed
  distribution statistics and five deterministically ranked transactions.
- **Town comparison:** like-for-like summary for selected towns and flat type.
- **Model reliability:** chronological holdout and rolling-backtest metrics,
  interval coverage and the documented intended-use limits.

Market briefs and town comparisons require at least 20 matching transactions.
Comparable sales use the selected minimum and expose each deterministic widening
step when the preferred match is too thin.

The first release intentionally excludes:

- unit valuations or appraisals;
- future resale-price forecasts;
- buy, sell, offer or financial recommendations;
- claims about causal price effects;
- live property listings or asking prices; and
- MRT-adjusted transaction analysis while verified block coordinates are
  unavailable.

## Run locally

Install the pinned dependencies:

~~~powershell
python -m pip install -r requirements.txt
~~~

The app works without an API key by returning a deterministic evidence brief.
To enable the language layer for the current PowerShell session:

~~~powershell
$env:OPENAI_API_KEY = "your-key"
$env:HDB_COPILOT_MODEL = "gpt-5.6-terra"
$env:HDB_COPILOT_MAX_REQUESTS = "8"
streamlit run copilot_app.py
~~~

Never place a real key in source code, .env.example or
.streamlit/secrets.toml committed to Git. For Streamlit deployment, put the
key in the platform secret manager.

HDB_COPILOT_MAX_REQUESTS is a basic per-session portfolio-demo guard, not a
complete global rate limiter. A public production deployment should also have
provider-side spend limits, authentication or a gateway-level quota.

## Reliability controls

The copilot boundary uses the OpenAI Responses API with Pydantic Structured
Outputs, `store=False`, a bounded output-token budget, a 30-second client
timeout and one retry. A model evidence selection is accepted only when:

1. every cited evidence ID exists;
2. every evidence point has at least one citation;
3. displayed numbers and recognized explicit units are supported by the cited
   deterministic evidence;
4. uncited headline, summary, limitation and follow-up fields contain no numbers;
   and
5. the draft does not present itself as a valuation, forecast or financial
   recommendation.

Even after those checks, model-written wording is discarded. The visible answer
uses deterministic labels and display values from the accepted evidence IDs,
which removes unrestricted model paraphrases from the product boundary.

If the SDK, API key or network is unavailable, or if validation fails, the app
returns a deterministic brief built directly from the same evidence packet.
The analytical result therefore remains usable without a model call.

OpenAI implementation references:

- [Developer quickstart](https://developers.openai.com/api/docs/quickstart)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Model catalog](https://developers.openai.com/api/docs/models)

## Evaluation

The frozen cases in evals/copilot_cases.json cover ordinary questions,
sparse segments, invalid inputs, prompt injection, unsupported valuation and
forecast requests, and numerical-grounding checks.

`scripts/evaluate_copilot.py` loads the tracked HDB Parquet snapshot and model
report, routes every case's actual inputs through the corresponding deterministic
evidence builder, and checks declared outcomes, filters, evidence surfaces,
sparse-data disclosures and scope guardrails. For cases that reach the answer
boundary, it forces the no-key path and checks the fallback twice without network access for
determinism and question-controlled instruction or number leakage. Mocked
boundary tests separately exercise accepted and rejected structured AI outputs,
including unit swaps. The 40/40 offline result is therefore a real-data analytics
and fallback safety check, not a live-model quality score; a public AI deployment
should run the same catalogue with controlled credentials and human review before
release.

The lexical scope and unit checks remain defense-in-depth, not a semantic proof
over every possible English paraphrase. Product safety does not rely on that
blacklist: unrestricted model prose is discarded, underlying facts stay visible,
and the app takes no purchase action. A public AI release should still run
controlled live-model red-team evaluation.

Minimum release checks:

- deterministic calculations match direct pandas results;
- every externally visible number is traceable to evidence;
- all 40 offline cases pass their declared calculation, outcome and leakage
  checks;
- mutation tests confirm that unsupported numbers, echoed instructions, omitted
  answer values, missing snapshot disclosure and incorrect custom-report
  provenance cause evaluation failures;
- every offline result must exactly match the canonical deterministic renderer,
  so appended model or monkeypatched prose also fails evaluation;
- the dataset cutoff and sample size remain visible;
- malicious instructions inside questions or evidence do not change system
  behavior;
- missing credentials and API failures produce a useful fallback; and
- the original analytics dashboard and its tests remain unchanged.

Run the complete checks with:

~~~powershell
python -m pip check
python -m compileall -q src app.py copilot_app.py
python -m ruff check .
python -m unittest discover -s tests -v
python scripts/evaluate_copilot.py
~~~

## Future price-range work

The existing Ridge model is an evaluated market-monitoring model, not a
persisted production valuation model. Its empirical ranges currently
under-cover their nominal target. A future scenario-estimation feature requires
a separately versioned inference artifact trained on all complete months,
checksum and schema validation, out-of-distribution checks, improved interval
calibration and updated model-card language before it can be exposed.
