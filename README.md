# ComPlaintAI-Bank

ComPlaintAI-Bank is a university AI project for public banking-complaint intake and policy assistance. A local NLP pipeline predicts complaint intent, extracts selected entities, and proposes internal priority and routing. A PostgreSQL boundary records guest complaints and returns a case reference only after commit confirmation. A separate policy FAQ retrieves passages from the approved policy PDFs and can use Groq to draft source-validated answers.

The customer chatbot and restricted administrator dashboard are separate Streamlit applications: `customer_app.py` and `admin_app.py`. The customer app does not require an account. Contact details are follow-up channels, not proof of identity. Predicted intent is not a verified allegation, and internal priority/routing metadata is not shown to customers.

## Architecture

- `src/bert_intent_classifier.py` uses the fine-tuned `bert-base-uncased` classifier trained on BANKING77. Model files are kept outside Git.
- `src/entity_extractor.py` uses a blank English spaCy pipeline with explicit extraction patterns and credential-disclosure protection.
- `src/priority_engine.py` applies project-configured rules for internal priority and department suggestions. These mappings are not policy-mandated outcomes for every intent.
- `src/complaint_workflow.py` combines local classification, entity extraction, and internal triage.
- `src/complaint_repository.py` validates, redacts labelled credentials, and persists complaints transactionally in PostgreSQL.
- `src/policy_rag.py` embeds policy PDFs with `sentence-transformers/all-MiniLM-L6-v2` and retrieves them from ChromaDB.
- `src/policy_assistant.py` sends a standalone policy question and retrieved policy excerpts to Groq when answer generation is enabled. It does not receive complaint records or complaint-intake contact details.
- `src/admin_service.py` authorizes administrator operations on the server before database access. `admin_app.py` is separate from the public customer app.
- `migrations/001_create_complaints.sql`, `002_create_customers.sql`, and `003_guest_complaint_contacts.sql` define the current schema sequence. No deployment should assume these have been applied to a new database.

The eight authoritative PDFs in `knowledge_base/policies/` are included in the repository. The generated ChromaDB index and local BERT weights are ignored and are not deployment inputs.

## Local setup

Use Python 3.12, which is also the current Streamlit Community Cloud default. Create and activate a virtual environment, then install the declared dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep local settings in the ignored project-root `.env`. The application reads the process environment first, then Streamlit secrets, while `python-dotenv` loads `.env` without overriding existing environment variables. Never put credentials in source files or commit `.env` or `.streamlit/secrets.toml`.

Local configuration names are `DATABASE_URL`, `GROQ_API_KEY`, optional `GROQ_MODEL`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `BERT_MODEL_REPO`, `BERT_MODEL_REVISION`, optional `HF_TOKEN`, and optional `CHROMA_PERSIST_DIRECTORY`. No values are included here. Administrator passwords shorter than 16 characters are rejected. The existing local demo credential should be replaced with a strong value before local admin use or deployment; do not reuse a demonstration password.

Run the applications from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run customer_app.py
.\.venv\Scripts\python.exe -m streamlit run admin_app.py
```

Apply migrations to a newly configured PostgreSQL database in numeric order (`001`, `002`, `003`) using a PostgreSQL migration tool or an authenticated `psql` session. Do not run migrations from the app process. Use separate, least-privilege database roles for complaint insertion and administrator reading/status updates where the schema and hosting setup permit it. No live database connection or migration is performed by the project setup commands above.

## Model and policy data in deployment

### Fine-tuned BERT

`models/bert_banking77_model/` is ignored by Git and contains about 438 MB of weights locally. A fresh Community Cloud build will not have that directory. The classifier now supports downloading the **existing trained artifact** from a configured Hugging Face-compatible model repository; it does not retrain or replace the model. Before deployment:

1. The existing fine-tuned model is published at `Kavinvelavan/complaintai-bank-bert-banking77`.
2. Pin `BERT_MODEL_REVISION` to the immutable commit `913bc6bdfc2bbbe2424559a511c8bfd343a6b19a` so deployments use the reviewed artifact.
3. Set `BERT_MODEL_REPO` to `Kavinvelavan/complaintai-bank-bert-banking77` and `BERT_MODEL_REVISION` to that commit in the customer app's secrets. Add `HF_TOKEN` only if repository access requires it; keep the token private.

The first complaint on a fresh app instance downloads those files to the Hugging Face cache. The app verifies the required files and loads them locally only. The repository and revision above are deployment configuration values; verify repository access and a fresh cloud download before release. The model download is large and may exceed Community Cloud build/runtime resource limits; verify the target plan and cold-start behavior before release.

### Policy FAQ and ChromaDB

The local `data/chroma_db/` directory is ignored and is not copied to a cloud app. On first FAQ use, the customer app opens the configured ChromaDB path and initializes an empty or incomplete index from the eight committed PDFs. Initialization uses the existing deterministic chunk IDs and non-destructive `upsert` path (`rebuild=False`), under a process lock; a valid existing local ingestion manifest is detected and left unchanged. Repeated ingestion uses the same IDs and does not duplicate chunks. The embedding model downloads from its configured public model source if not cached. No app path resets or deletes the collection.

Community Cloud does not guarantee persistence of generated local files across app sessions or restarts. The index can therefore be recreated from the committed PDFs when necessary. PostgreSQL, rather than local files, is the persistent complaint store.

## Secrets per Streamlit app

Configure secrets separately in each app's Community Cloud settings. The names below are required or optional; do not commit their values.

| App | Required secrets | Optional secrets |
| --- | --- | --- |
| Customer | `DATABASE_URL`, `GROQ_API_KEY`, `BERT_MODEL_REPO`, `BERT_MODEL_REVISION` | `GROQ_MODEL`, `HF_TOKEN`, `CHROMA_PERSIST_DIRECTORY` |
| Admin | `DATABASE_URL`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` | — |

The FAQ and complaint registration fail independently when their own service configuration is missing. The customer app sends only standalone policy FAQ questions and retrieved policy excerpts to Groq. It must never send complaint descriptions or contact information to Groq.

The admin password must be a strong, unique secret of at least 16 characters. The app fails closed if administrator credentials are missing or the password is shorter. Do not place the current weak local demonstration password in deployment secrets.

## Testing

Run the offline unit suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use mocks or temporary test data and should not require a live database or Groq call. Keep BERT-model and policy-index tests isolated from production data. Do not count a migration, cloud download, database operation, or provider request as verified unless it was actually performed and reported.

## Streamlit Community Cloud deployment

The two root-level entry points can be deployed as two apps from the same GitHub repository and branch. Community Cloud copies the files tracked in the selected repository; ignored local weights, `.env`, and `data/chroma_db/` are not available there. The eight policy PDFs are tracked and will be available. Python 3.12 is the selected target because it is the current Community Cloud default; confirm dependency compatibility and memory needs in a clean Linux deployment before publishing.

1. Review `git status` and explicitly select only intended source, migration, dependency, test, README, and policy files for the first push. Never include `.env`, secrets, complaint data, model weights, or local ChromaDB data.
2. Create the GitHub repository and push the reviewed branch only after local review/approval.
3. In Community Cloud, create one app with entry point `customer_app.py`; configure the customer secrets listed above.
4. Create another app with entry point `admin_app.py`; configure its own database and admin secrets. In the app's **Sharing** settings, restrict viewing to specific invited people. Do not publish the admin dashboard as a public app merely because it also has a login form.
5. Community Cloud currently allows one private app per account/workspace. Plan accordingly if this account already has another private app. A private-app gate restricts viewers, but does not replace application authentication, server-side authorization, least-privilege database credentials, or operational security.
6. Apply migrations to the configured PostgreSQL database through a controlled migration process, then verify each app independently. Do not claim registration, policy generation, or admin operations work until their cloud configuration and behavior have been tested.

Community Cloud resource limits, cold starts, generated-file persistence, private-app availability, and package installation behavior can change. Review current platform documentation and test the chosen deployment tier before relying on it for complaint records. The current admin authentication is a demonstration foundation, not production-ready authentication; session hardening, rate limiting, recovery controls, transport/deployment review, and database-role separation remain necessary.

## Project safeguards

- Never use real financial-institution names in policy content or customer-facing examples.
- Do not modify policy PDFs unless explicitly asked; do not add effective dates or printed page numbers.
- Keep original complaint text separate from model predictions and extracted entities.
- Do not expose internal priority, confidence, routing, notes, or complaint-register data to customers.
- Do not claim persistence or external actions without confirmation from the relevant system.
- Pattern-based credential redaction is not exhaustive; ambiguous labelled values must fail closed.
- Build incrementally, inspect existing files before changes, run appropriate tests, and report failures honestly.
