# ReviewSense AI

**Cloud-powered product review sentiment analysis and customer insights**

ReviewSense AI is a responsive Streamlit website for classifying product reviews as **Positive**, **Negative**, or **Neutral**. Its primary inference engine is Cloudflare Workers AI, so visitors' phones and computers do not download or run a large model. VADER is an optional, explicitly labeled fallback. The project includes CSV upload, aspect-level predictions, keyword insights, interactive charts, review filters, evaluation against human labels, and CSV export.

> **Live demo:** [https://reviewsense-ai-nves.onrender.com](https://reviewsense-ai-nves.onrender.com) · Hosted on Render's free instance. The instant sample demo and VADER work now; Cloudflare Workers AI is not yet configured, so live AI analysis requires adding your own Cloudflare credentials to Render's environment settings.

## What runs where?

| Component | Runs at | Data handling |
|---|---|---|
| Streamlit website | [Render live demo](https://reviewsense-ai-nves.onrender.com) | Receives uploads and holds data in application/session memory; no project database is used |
| Sentiment and aspects | Cloudflare Workers AI | Review text is sent server-to-server from Streamlit to Cloudflare |
| VADER fallback | Streamlit app process | Rule/lexicon inference; it is not represented as an AI-model prediction |
| Instant sample demo | Streamlit app process | Reads bundled, pre-analyzed sample predictions; makes no AI API request |

Uploaded data is not intentionally written to a database or file by the application. Render and Cloudflare are separate providers and their current policies apply. Obtain permission before processing reviews, avoid personal or sensitive information, and do not upload customer names, email addresses, IDs, confidential text, or regulated data. A public app has no user login; its in-process quota is only a best-effort guard, not a global abuse-prevention system.

## Features

- Cloudflare Workers AI classifies complete reviews with a constrained three-class JSON prompt.
- The model can return short evidence-based reasons and per-aspect sentiment labels.
- Strict schema checks reject malformed responses, unsupported labels, and missing results.
- Cloud AI confidence is shown as **unavailable** because the selected text-generation model does not return class probabilities.
- VADER can be selected manually. Cloud failures never silently switch the prediction source.
- CSV upload with detected review columns, preview through Streamlit, missing review handling, duplicate flags, UTF-8/CSV validation, a **2 MB** file cap, and a **100-review** cap.
- Cloud inference is batched in groups of four, with a 4,000-character per-review limit, bounded retries, a 30-request/minute process limit, and a 100-review/day process safety limit.
- Session-level prediction caching avoids repeat API calls when navigating pages or applying filters.
- Sentiment, rating, aspect, product-comparison, and AI/VADER charts use the active dataset.
- Analytics also show rating distributions by predicted sentiment and phrase-frequency charts for positive and negative feedback.
- Searchable review explorer, product/aspect/rating filters, extracted phrase counts, and enriched CSV download.
- Overview cards include a data-derived average rating and the most frequent positive/negative multi-word phrase when the uploaded data supports them.
- Accuracy, macro precision/recall/F1, and confusion matrix only when the user supplies ground-truth labels.
- Provider and model identifiers are kept out of the regular UI and exported analysis source labels; the interface describes remote inference generically.

## Website and app downloads

The public website is usable in a modern desktop or mobile browser. The **Apps & Downloads** page reports platform availability honestly. At present, there is no installable PWA or verified Windows, macOS, Linux, Android, or iOS package in this repository.

A real PWA requires a web app manifest and a service worker served with a scope that can control the app page. The current Streamlit deployment does not provide that product shell, so it is not labeled as installable. Native packages likewise require a separate packaging/build pipeline, platform testing, and—in some cases—signing credentials or platform-specific build machines. No download buttons are shown for packages that do not exist.

The app remains a Streamlit website rather than the separate React/TypeScript frontend and FastAPI backend described as a possible future architecture. That choice preserves the existing working dashboard and keeps the deployment footprint small. Moving to a standalone frontend/backend split should be a separately scoped project before adding PWA and native builds.

The local usage counter is **not Cloudflare's actual Neurons meter** and resets if the app process restarts. Cloudflare free allocations are account-wide and subject to provider policy. Review actual usage in the Cloudflare dashboard. On Workers Free, the published daily allocation is 10,000 Neurons and requests beyond the free allocation fail; paid plans may bill for usage beyond it. Do not enable billing or upgrade if you require a hard no-charge ceiling.

## Technology

- Python 3.12
- Render Web Service
- Cloudflare Workers AI REST API
- IBM Granite 4.0 H Micro (`@cf/ibm-granite/granite-4.0-h-micro`) by default
- Pandas, Plotly, scikit-learn, Requests, vaderSentiment
- CSV; no database required

The prompt-suggested `@cf/meta/llama-3.1-8b-instruct-fast` is not present in the current Workers AI model catalog checked for this version. The default is the documented, currently listed Granite 4.0 H Micro model. Change `CLOUDFLARE_MODEL` only to an ID available to your Cloudflare account. The model is configurable to accommodate catalog changes.

Model and pricing references (checked 2026-11-11):

- [Granite 4.0 H Micro model card](https://developers.cloudflare.com/workers-ai/models/granite-4.0-h-micro/)
- [Workers AI REST API setup](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
- [Workers AI model catalog](https://developers.cloudflare.com/workers-ai/models/)
- [Workers AI pricing and daily free allocation](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Render Streamlit deployment guide](https://render.com/docs/deploy-streamlit)
- [Render Blueprint specification](https://render.com/docs/blueprint-spec)

## Project structure

```text
ReviewSenseAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .python-version
├── render.yaml
├── sample_reviews.csv
├── sample_reviews_preanalyzed.csv
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── assets/
│   ├── logo.png
│   └── logo.svg
├── tests/
│   └── test_sentiment.py
└── utils/
    ├── __init__.py
    ├── cloud_ai.py
    ├── data_processor.py
    ├── aspect_miner.py
    ├── vader_analyzer.py
    └── evaluator.py
```

### File guide

- `app.py` builds the Streamlit pages, user controls, charts, session cache, privacy confirmation, and download controls.
- `utils/cloud_ai.py` calls the Cloudflare API server-side, constructs the untrusted-review prompt, batches requests, retries transient failures, validates JSON, and tracks best-effort process quotas.
- `utils/data_processor.py` validates CSV uploads, detects review columns, cleans missing text, and adds safe analysis output columns.
- `utils/vader_analyzer.py` computes independent VADER labels and compound scores.
- `utils/aspect_miner.py` counts common bigrams/trigrams. These are frequency-based hints, not generated explanations.
- `utils/evaluator.py` calculates metrics only from independent human labels.
- `sample_reviews.csv` contains 60 illustrative reviews.
- `sample_reviews_preanalyzed.csv` is a no-network demo with older, precomputed Transformer/VADER labels. It is explicitly identified as a **pre-analyzed Transformer demo**, not a current Cloudflare prediction.
- `tests/test_sentiment.py` tests the CSV, VADER, output validation, retry handling, and metrics without calling paid or live cloud APIs.
- `.streamlit/secrets.toml.example` shows the format only; it contains no credentials.
- `render.yaml` defines the lightweight Render web service and prompts for Cloudflare credentials at deploy time.

## Run locally

The browser UI runs locally, but the **Cloudflare inference is still remote**. Without Cloudflare credentials, choose the instant demo or VADER fallback.

### Linux or macOS

```bash
cd ReviewSenseAI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd ReviewSenseAI
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the address printed by Streamlit, usually `http://localhost:8501`. Since the model runs on Cloudflare, the local app does not load PyTorch or download model weights.

For local cloud inference, create `.streamlit/secrets.toml` from the example file and add your own credentials. `.gitignore` excludes the real secrets file. Never paste secrets into source code, issues, screenshots, or chat messages.

## Public deployment on Render

### 1. Get the source from GitHub

The public repository is [https://github.com/silentboy765/ReviewSenseAI](https://github.com/silentboy765/ReviewSenseAI). To work on another computer, clone it:

```bash
git clone https://github.com/silentboy765/ReviewSenseAI.git
cd ReviewSenseAI
```

For future changes, use `git status`, then `git add`, `git commit`, and `git push`. Confirm `.streamlit/secrets.toml`, `.env`, `.venv`, and `__pycache__` are not staged. Never put a GitHub token in the remote URL.

### 2. Configure Cloudflare Workers AI

1. Create/sign into a Cloudflare account at [dash.cloudflare.com/sign-up](https://dash.cloudflare.com/sign-up) and open **Workers AI**. Account creation can require email verification.
2. Stay on a plan with the free allocation if you require a no-charge limit. Do not enable paid billing. Provider limits, quotas, model access, and free allocations can change.
3. From Workers AI, create a Workers AI API token using Cloudflare's recommended token flow. For a manually created token, grant only the needed Workers AI permissions (Cloudflare documents Workers AI Read and Edit for REST inference).
4. Copy the account ID and token into a secure password manager. Do not commit them to GitHub.
5. Check the Cloudflare dashboard after tests to monitor actual Neurons usage. The app's local review counter is not provider billing telemetry.

### 3. Deploy the website to Render

The live service is already connected to `silentboy765/ReviewSenseAI` on branch `main`. To reproduce it, sign in to [Render](https://dashboard.render.com/), create a Web Service from the repository, and set:

- Runtime: Python 3
- Build command: `python -m pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
- Instance: **Free** ($0/month; it can spin down when idle)

These settings are also recorded in `render.yaml`. The `.python-version` file selects Python 3.12.10.

To enable live Cloudflare AI on the existing service, open **Environment** in the Render dashboard and add `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. `CLOUDFLARE_MODEL` defaults to the model configured in the app. Save the values and redeploy. Do not send credentials in chat or commit them to GitHub.

The public website is [https://reviewsense-ai-nves.onrender.com](https://reviewsense-ai-nves.onrender.com). The sample demo works without Cloudflare credentials. After setting them, test a few non-sensitive reviews using Cloudflare AI, then verify the VADER comparison, export, and actual Cloudflare usage before sharing.

The API token is stored in Render's environment settings, not in GitHub or the browser. The app does not need a database or a separate Cloudflare Worker. Do not enable a paid Render instance or Cloudflare billing unless you have reviewed the costs and limits.

### Free-tier and public-app limits

- Workers AI's free allowance is shared across the Cloudflare account, not reserved per user or project. The app does not promise unlimited free inference.
- This app enforces a 100-review/day and 30-request/minute in-memory safety cap per running app process. It is not durable, cross-replica rate limiting and resets when a process restarts.
- The selected model's response is validated, but an invalid response or provider outage stops that analysis and offers the user VADER explicitly. There is no automatic paid-provider fallback.
- Requests are retried up to three times for timeouts and transient HTTP errors with exponential backoff. Repeated 429/quota failures become a clear error.
- Cloudflare AI produces text labels, not class probabilities. The dashboard does not invent confidence scores or claim a fixed accuracy.
- Cloud-hosted user uploads go to the Render server and review text goes to Cloudflare for AI analysis. Do not upload sensitive data.
- Render's free web service can sleep after inactivity and has resource limits; the first request after sleeping may take longer. This demonstration has no account system or globally enforced abuse protection.

## How sentiment analysis works

1. Pandas loads and validates the CSV; the user selects the review-text column.
2. Missing review cells become empty strings. Empty reviews receive Neutral without a Cloudflare request. Duplicate reviews are flagged.
3. Non-empty reviews are encoded as JSON data and sent in batches of four to the configured Cloudflare Workers AI model. The system prompt says reviews are untrusted data and cannot change classification instructions.
4. The model is asked for strict JSON with `sentiment`, short `reason`, and zero to four aspect labels. The app validates result count, fields, and each label against Positive/Negative/Neutral before using any result.
5. The model's output is a prediction, not a verified fact. Since no class probabilities are supplied, confidence remains unavailable. VADER independently supplies a compound score and separate result.
6. Plotly charts, customer phrase counts, filters, metrics, and CSV export are calculated from the current dataset.
7. Accuracy is calculated only if the user selects an actual human-labeled sentiment column. AI output is never used as its own ground truth.

### VADER thresholds

VADER's normalized compound score is between -1 and +1:

- `>= 0.05`: Positive
- `<= -0.05`: Negative
- Between those thresholds: Neutral

VADER is a lexicon-and-rule method; Cloudflare AI is generative contextual language inference. Neither is guaranteed correct, and their agreement is not an accuracy measure.

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py utils tests
```

The unit tests mock HTTP calls and do not send review text to Cloudflare. A real Cloudflare test requires your own configured account/token and may consume free Neurons. Test positive, negative, neutral, mixed, empty, long, malformed-CSV, invalid-model-response, rate-limit, and missing-ground-truth cases before a public demonstration. Sarcasm, Hinglish, slang, domain-specific terms, and hallucinated aspects remain limitations.

## College project report

1. **Abstract:** ReviewSense AI is a responsive web application that uses a hosted open model to classify customer reviews and presents sentiment, aspects, keyword counts, and interactive analytics.
2. **Introduction:** Customer reviews are useful but costly to inspect manually. A sentiment dashboard helps organize feedback for human review.
3. **Problem statement:** Develop a cross-device, beginner-friendly review dashboard without requiring visitors to run a large local model.
4. **Objectives:** Accept CSV files, analyze English reviews in the cloud, compare with VADER, show aspects and analytics, export results, and evaluate only against independent human labels.
5. **Existing system:** Manual review and spreadsheets are labor-intensive; local large models may consume device resources; paid APIs may incur charges.
6. **Proposed system:** Render hosts the Streamlit UI and Python server; Cloudflare Workers AI performs remote inference; VADER provides an optional separate fallback.
7. **System architecture:** Browser → Streamlit server → CSV validation/Pandas → Cloudflare Workers AI → output validation → analytics and export. VADER is a separate local-in-process comparison path.
8. **Technologies:** Python, Streamlit, Cloudflare Workers AI REST API, Pandas, Plotly, VADER, scikit-learn, Requests, and CSV.
9. **AI model:** IBM Granite 4.0 H Micro is configured as an instruction-following model for JSON classification. Model availability and pricing should be rechecked before a new deployment.
10. **Methodology:** Reviews are JSON-encoded, sent in bounded batches, classified using a fixed system prompt, and validated before the dashboard accepts the result.
11. **Implementation:** The application separates interface, API client, CSV processing, VADER, phrase counts, and evaluation into readable modules.
12. **Testing:** Unit tests mock the API and cover sentiment data handling, output schema, retries, CSV errors, and ground-truth evaluation.
13. **Results/screenshots:** Capture the deployed Overview, Analytics, review explorer, aspect view, and evaluation page using the sample dataset. Do not present a screenshot as model accuracy evidence.
14. **Limitations:** Model mistakes, sarcasm, Hinglish, domain vocabulary, hallucinated aspect tags, quota exhaustion, provider outages, public-app abuse, and free-host sleep.
15. **Future scope:** Use durable shared rate limiting, collect consented human labels, evaluate model quality, monitor actual provider usage, and add auth for a larger deployment.
16. **Conclusion:** The project demonstrates a hosted end-to-end review analysis workflow with transparent uncertainty, source labeling, and no local large-model requirement.

### Viva questions and answers

1. **Why do model outputs not have confidence percentages?** This text-generation endpoint does not return calibrated class probabilities; showing a made-up percentage would be misleading.
2. **How does the app reduce prompt injection risk?** Reviews are passed as JSON user data while the system instructions say not to follow instructions inside reviews; outputs are schema-validated.
3. **Why use a cloud model?** Inference runs on Cloudflare infrastructure rather than consuming the visitor's CPU/GPU or downloading model weights.
4. **Is cloud inference unlimited and free?** No. Cloudflare publishes a limited free Neurons allocation and quotas. The app also applies a process-local cap.
5. **What does VADER do?** It uses a lexicon and rules, producing a compound score mapped with fixed thresholds. It is separate from the generative model.
6. **How are aspects found?** The model returns feature names and a sentiment for each within a constrained schema. These are predictions, not verified or exhaustive facts.
7. **How do you calculate model accuracy?** Compare predictions to independent human labels, then calculate metrics such as accuracy and macro F1.
8. **What is a confusion matrix?** A table comparing each human label against the model's predicted class.
9. **How are missing reviews handled?** They become empty strings, are assigned Neutral without an API call, and remain in the output.
10. **What are the privacy concerns?** The Render host receives uploaded CSV content, and Cloudflare receives review text for inference; users must avoid sensitive data.
11. **What is a main limitation?** Sarcasm, slang, Hinglish, and domain-specific meaning can be misclassified; model-generated aspects can also be wrong.
12. **Does this app store a review database?** No database is included; results are held in app/session memory and can be downloaded by the user.

### Presentation outline

1. Problem and motivation.
2. Architecture and data flow.
3. Cloudflare Workers AI and secure secret configuration.
4. Upload a sample CSV and select the review column.
5. Demonstrate Overview, Analytics, aspects, and review filters.
6. Compare Cloud AI and VADER; discuss unavailable confidence honestly.
7. Show evaluation with a human-labeled column.
8. Discuss privacy, free quota, test evidence, limitations, and future work.
