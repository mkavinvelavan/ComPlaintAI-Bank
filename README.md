# ComPlaintAI-Bank

**An AI-assisted banking complaint registration, intent classification, prioritisation, and policy-assistance system.**

ComPlaintAI-Bank is an academic project that brings together Natural Language Processing (NLP), a fine-tuned BERT classifier, rule-based complaint triage, a Retrieval-Augmented Generation (RAG) policy assistant, and a PostgreSQL complaint register. It provides two separate Streamlit applications: a public-facing customer application and a restricted administrator dashboard.

> **Project notice:** This repository is an academic demonstration, not a banking service. It does not connect to or represent a real financial institution. A predicted intent is a model output, not verification that a customer's allegation is true. Priority and department values are internal suggestions, not proof that a complaint has been escalated or actioned.

---

## Table of contents

- [Project objectives](#project-objectives)
- [Main features](#main-features)
- [System architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Complaint classification model](#complaint-classification-model)
- [Complaint processing and prioritisation](#complaint-processing-and-prioritisation)
- [Policy assistant: RAG workflow](#policy-assistant-rag-workflow)
- [Database](#database)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Run the applications](#run-the-applications)
- [Build or refresh the policy index](#build-or-refresh-the-policy-index)
- [Run the tests](#run-the-tests)
- [Deployment notes](#deployment-notes)
- [Security, limitations, and responsible use](#security-limitations-and-responsible-use)
- [References](#references)

---

## Project objectives

The project aims to:

1. Allow a customer to describe a banking-related issue in natural language and submit it as a complaint.
2. Identify the likely complaint category using a fine-tuned 77-class BERT intent classifier.
3. Extract selected details, such as transaction references, dates, monetary amounts, and card or account suffixes, when they match supported patterns.
4. Apply configurable rules to suggest an internal priority and department queue.
5. Redact supported, explicitly labelled credential values before complaint text is stored.
6. Register complaints in PostgreSQL and return a case reference only after the database transaction is committed.
7. Answer general policy questions using passages retrieved from the policy PDFs included with the project.
8. Provide a separate administrator interface to review complaint records and update their status.

## Main features

### Customer application

- Conversational complaint-intake interface.
- Guest complaint submission; a customer account is not required by the current complaint form.
- Requires at least one valid follow-up contact method: an email address, a phone number, or both.
- Contact details are used for follow-up and are **not** treated as identity verification.
- Local complaint analysis using the fine-tuned BERT model and the spaCy-based entity extractor.
- Policy FAQ interface with source information shown alongside answers.

### Complaint analysis

- 77-class banking intent classification.
- Up to three candidate intent predictions returned by the classifier.
- Pattern-based extraction of supported transaction references, money amounts, dates, and card/account last-four digits.
- Narrow, deterministic correction rules for clear card-delivery wording.
- Rule-based priority and department suggestions.
- Credential-disclosure detection and redaction of supported labelled values before persistence.

### Administrator dashboard

- Separate Streamlit entry point from the public customer application.
- Server-side username/password verification using configured secrets.
- Overview of registered complaints, status counts, and priority counts.
- Complaint search by case reference, with status and priority filters.
- Complaint detail view for authorised administrators.
- Explicit complaint-status updates.

---

## System architecture

The system has two main workflows. Complaint registration and policy question answering are separate paths.

### 1. Complaint registration workflow

```mermaid
flowchart TD
    A[Customer] --> B[Customer Streamlit App]
    B --> C[Complaint description and contact details]
    C --> D[Complaint Registration Service]
    D --> E[Complaint Workflow]
    E --> F[BERT Intent Classifier]
    E --> G[spaCy Matcher and Regex Entity Extractor]
    F --> H[Combined Complaint Assessment]
    G --> H
    H --> I[Rule-based Priority Engine]
    I --> J[Intent, priority and department suggestion]
    J --> K[Complaint Repository]
    K --> L[Validate contact and assessment]
    L --> M[Redact supported labelled credentials]
    M --> N[(PostgreSQL)]
    N --> O{Transaction committed?}
    O -->|Yes| P[Return case reference]
    O -->|No| Q[Return safe registration error]
```

**Important distinction:** BERT predicts the intent. The separate Python priority engine applies configured rules to produce the priority and department suggestion. Priority is not predicted by a second machine-learning model.

### 2. Policy FAQ workflow

```mermaid
flowchart TD
    A[Customer policy question] --> B[Sensitive-query guard]
    B --> C[Policy Assistant]
    C --> D[Sentence Transformer query embedding]
    D --> E[(ChromaDB policy collection)]
    E --> F[Top matching policy passages]
    F --> G[Question and retrieved excerpts]
    G --> H[Groq chat completion]
    H --> I[Answer with policy citations]
    I --> J[Display answer and source filename/page]
```

Policy documents are processed locally for retrieval. When answer generation is enabled, the current policy question, any prior cited policy-answer context used for a follow-up, and retrieved policy excerpts are sent to the configured Groq API. The complaint-intake path does not send complaint descriptions or contact details to Groq.

### 3. Administrator workflow

```mermaid
flowchart TD
    A[Administrator] --> B[Admin Streamlit App]
    B --> C[Server-side credential check]
    C --> D[Admin Complaint Service]
    D --> E[Authorisation check]
    E --> F[Admin Repository]
    F --> G[(PostgreSQL)]
    G --> H[Overview, filtered records and complaint details]
    H --> I[Administrator updates complaint status]
    I --> F
```

---

## Technology stack

| Area | Technology | Role in the project |
|---|---|---|
| Customer interface | Streamlit | Conversational complaint form and policy FAQ |
| Administrator interface | Streamlit | Restricted complaint dashboard |
| Programming language | Python | Application logic and processing |
| Intent classification | Hugging Face Transformers, BERT-base uncased | Predicts one of 77 complaint intents |
| Deep learning framework | PyTorch | Runs and fine-tunes the BERT model |
| NLP and entity patterns | spaCy, regular expressions | Extracts selected entities and detects supported credential patterns |
| Dataset | PolyAI Banking77 | Labelled examples for 77 banking intents |
| Embedding model | Sentence Transformers, `all-MiniLM-L6-v2` | Converts policy passages and questions into vectors |
| Vector database | ChromaDB | Stores and retrieves policy passage embeddings |
| LLM API | Groq | Generates policy FAQ answers from retrieved evidence |
| Relational database | PostgreSQL | Stores registered complaint records and related fields |
| Database driver | Psycopg | Connects Python services to PostgreSQL |
| PDF processing | pypdf | Extracts text from policy PDFs |
| Configuration | python-dotenv and Streamlit secrets | Loads local and hosted application settings |

---

## Complaint classification model

### Dataset

The intent classifier was fine-tuned using the [PolyAI Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77). It contains short banking-related customer queries labelled with one of 77 intent categories.

| Split | Number of examples | Use |
|---|---:|---|
| Original training split | 10,003 | Source for training and validation |
| Training subset | 9,002 | Model fine-tuning |
| Validation subset | 1,001 | Model selection and early stopping checks |
| Official test split | 3,080 | Final held-out evaluation |
| **Total** | **13,083** | **Complete dataset** |

A stratified 10% validation subset was created from the official training split using random seed 42. The official test split was kept separate from training and validation.

### Model configuration

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Task | 77-class sequence classification |
| Transformer layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Maximum sequence length | 128 tokens |
| Learning rate | `2e-5` |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Maximum training epochs | 4 |
| Weight decay | `0.01` |
| Warmup ratio | `0.10` |
| Optimizer | AdamW (Transformers Trainer default) |
| Validation frequency | After each epoch |
| Best-model selection | Validation macro-F1 |
| Early-stopping patience | 2 evaluation rounds |
| Random seed | 42 |

The training notebook uses dynamic padding through `DataCollatorWithPadding`. Mixed-precision FP16 is enabled when CUDA is available. The best checkpoint was selected using validation macro-F1.

### Evaluation results

The following results are from evaluation on the held-out official test split of 3,080 examples.

| Metric | Result |
|---|---:|
| Accuracy | **90.62%** |
| Macro precision | 91.26% |
| Macro recall | 90.62% |
| Macro F1-score | **90.59%** |
| Weighted F1-score | 90.59% |
| Balanced accuracy | 90.62% |
| Top-3 accuracy | 97.11% |

**Metric note:** Top-3 accuracy means that the correct intent appears among the model's three highest-ranked candidates. It is not the same as top-1 accuracy. The model's output is a classification aid and does not establish whether a complaint allegation is true.

The training workflow and evaluation code are available in [`ComPlaintAI_BERT_FineTuning_BANKING77.ipynb`](ComPlaintAI_BERT_FineTuning_BANKING77.ipynb).

### Model artifact

The trained model weights are not included in this Git repository because of their size. The application can load a complete local model from `models/bert_banking77_model/` or download the existing trained artifact from the configured Hugging Face repository at a pinned revision.

- Model repository: [`Kavinvelavan/complaintai-bank-bert-banking77`](https://huggingface.co/Kavinvelavan/complaintai-bank-bert-banking77)
- Reviewed model revision: `913bc6bdfc2bbbe2424559a511c8bfd343a6b19a`

For a fresh deployment, configure `BERT_MODEL_REPO` and `BERT_MODEL_REVISION` as described in [Configuration](#configuration). The model loader requires an immutable 40-character commit SHA for remote downloads. A Hugging Face token is only needed if the repository requires authentication.

---

## Complaint processing and prioritisation

The complaint workflow combines the classifier, entity extractor, and priority engine.

1. **Input validation:** The application checks the complaint description and validates at least one contact method for guest submission.
2. **Intent classification:** BERT predicts the most likely Banking77 intent and returns up to three scored candidates.
3. **Entity extraction:** A blank English spaCy pipeline, explicit matcher patterns, and regular expressions identify supported details. The extractor does not load a pretrained spaCy language model.
4. **Intent correction:** A narrow rule handles clear card-delivery statements and delivery-time questions. Ambiguous wording is left to the model prediction.
5. **Priority assessment:** A deterministic rule table maps the final intent to an internal priority and suggested queue. The baseline mapping covers 77 labels: 6 HIGH, 34 MEDIUM, and 37 LOW defaults. The assessment can also flag conditions for manual review, including ambiguous high-risk predictions, missing or invalid confidence, and possible credential exposure. The current database schema does not persist a separate review-queue state.
6. **Credential handling:** Supported, explicitly labelled password, passcode, PIN, CVV/CVC, OTP, and verification-code values are redacted before persistence. Ambiguous labelled values are rejected rather than stored unchanged.
7. **Database registration:** The repository validates the data and writes the complaint transactionally. A case reference is returned only after a confirmed commit.

The department value is a **suggestion** generated by project rules. It does not represent an actual transfer to a bank department or an external workflow.

---

## Policy assistant: RAG workflow

The policy FAQ uses Retrieval-Augmented Generation to answer questions using the PDF documents included in `knowledge_base/policies/`.

### Policy knowledge base

- **Source documents:** 8 project policy PDFs.
- **PDF processing:** `pypdf` extracts text page by page.
- **Chunk size:** 800 characters.
- **Chunk overlap:** 120 characters.
- **Chunk boundaries:** The splitter prefers paragraph, line, sentence, and word boundaries where possible.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Embedding dimensions:** 384.
- **Vector database:** ChromaDB.
- **Collection name:** `complaintai_bank_policies`.
- **Distance metric:** Cosine distance.
- **Default retrieval:** Top 3 matching passages.
- **Default answer model:** `openai/gpt-oss-120b` through Groq; can be changed with `GROQ_MODEL`.

Each chunk includes source metadata such as the PDF filename, policy identifier, page number, chunk index, and embedding model. Chunk IDs are deterministic, so repeated ingestion can use upserts without duplicating unchanged chunks.

### Grounding and citations

The policy assistant is instructed to answer from the retrieved excerpts and state when the available documents do not contain enough information. It displays policy identifiers, source filenames, and page numbers for returned sources. The application checks that cited policy identifiers and pages correspond to retrieved evidence; this check does not independently prove that every generated sentence is semantically entailed by the cited text.

The cosine similarity value is a retrieval measure, not a calibrated confidence score or a policy decision. The project does not currently report a formal quantitative RAG accuracy or hallucination evaluation.

### Included policy PDFs

The repository contains eight policy documents covering:

1. Account management and banking services.
2. Debit and credit card services.
3. Fraud, unauthorised transactions, and disputes.
4. Fund transfers, remittances, and refunds.
5. Banking fees, charges, and interest.
6. Digital banking and authentication.
7. Customer complaint registration and resolution.
8. Complaint priority, department routing, and data privacy.

These are the policy reference documents supplied with this academic project. They should not be represented as the policies of a real bank.

---

## Database

The application uses PostgreSQL for persistent complaint records. The SQL migrations are in the `migrations/` directory and must be applied in numeric order:

1. `001_create_complaints.sql` — creates the complaint table, indexes, and update timestamp trigger.
2. `002_create_customers.sql` — creates the customer account table and related constraints. The current public complaint form uses guest submission.
3. `003_guest_complaint_contacts.sql` — allows guest complaint records and adds email/phone follow-up fields and validation constraints.

The complaint record includes a case reference, complaint text after supported credential redaction, predicted intent, priority, department suggestion, credential-disclosure indicator, status, timestamps, and optional contact details. The initial status is `RECEIVED`; the administrator interface supports `RECEIVED`, `IN_REVIEW`, `RESOLVED`, and `CLOSED`.

**Migrations are not run automatically.** Apply them to the intended PostgreSQL database through a controlled migration process before using the application with that database. The application setup does not create or modify a live database by itself.

---

## Repository structure

```text
ComPlaintAI-Bank/
├── customer_app.py                    # Public customer application
├── admin_app.py                       # Separate administrator dashboard
├── ComPlaintAI_BERT_FineTuning_BANKING77.ipynb
├── requirements.txt
├── migrations/
│   ├── 001_create_complaints.sql
│   ├── 002_create_customers.sql
│   └── 003_guest_complaint_contacts.sql
├── knowledge_base/
│   └── policies/                      # Eight policy reference PDFs
├── scripts/
│   ├── ingest_policies.py             # Build/refresh local ChromaDB index
│   └── smoke_test_policy_assistant.py
├── src/
│   ├── bert_intent_classifier.py
│   ├── model_artifacts.py
│   ├── complaint_processor.py
│   ├── complaint_workflow.py
│   ├── intent_corrections.py
│   ├── entity_extractor.py
│   ├── credential_redaction.py
│   ├── priority_engine.py
│   ├── complaint_registration_service.py
│   ├── complaint_repository.py
│   ├── policy_rag.py
│   ├── policy_assistant.py
│   ├── admin_auth.py
│   ├── admin_service.py
│   ├── admin_repository.py
│   └── ...                             # Supporting controllers and utilities
└── tests/                              # Offline unit and component tests
```

Generated model weights, `.env`, and the local ChromaDB index are excluded from Git by `.gitignore`.

---

## Getting started

### Prerequisites

- Python 3.12 recommended for the local application environment.
- Git (optional, if cloning the repository).
- A PostgreSQL database for complaint registration and administrator access.
- A Groq API key for policy answer generation.
- Access to the fine-tuned BERT model artifact, either locally or through the configured Hugging Face repository.
- Internet access on first use if model or embedding files are not already cached.

### 1. Clone the repository

```bash
git clone https://github.com/mkavinvelavan/ComPlaintAI-Bank.git
cd ComPlaintAI-Bank
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the environment

Create a `.env` file in the repository root. See [Configuration](#configuration) for the supported variable names. The `.env` file is ignored by Git; never commit credentials or customer data.

### 4. Prepare the BERT model

The repository ZIP does not include the model weights. Use one of these options:

- Place the complete trained model artifact in `models/bert_banking77_model/`; or
- Configure `BERT_MODEL_REPO` and `BERT_MODEL_REVISION` so the application can download the existing artifact.

The downloaded artifact must contain the required model, tokenizer, and label-mapping files. The application loads the existing fine-tuned model; it does not train or fine-tune BERT when a customer submits a complaint.

### 5. Prepare PostgreSQL

Create or select a PostgreSQL database, then apply migrations `001`, `002`, and `003` in order. For example, from the repository root in **macOS / Linux / Git Bash**, with `DATABASE_URL` exported:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_create_complaints.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/002_create_customers.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/003_guest_complaint_contacts.sql
```

In **Windows PowerShell**, use `$env:DATABASE_URL` instead:

```powershell
psql "$env:DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_create_complaints.sql
psql "$env:DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/002_create_customers.sql
psql "$env:DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/003_guest_complaint_contacts.sql
```

These commands require a valid database connection and appropriate migration permissions. They are examples for an operator-controlled setup; they are not executed by the application automatically.

---

## Configuration

The application reads settings from process environment variables first, then Streamlit secrets. For local development, `python-dotenv` loads the project-root `.env` file without overriding values already present in the process environment.

| Variable | Used by | Required when | Description |
|---|---|---|---|
| `DATABASE_URL` | Customer and admin apps | Registering or reading complaints | PostgreSQL connection string |
| `GROQ_API_KEY` | Customer policy FAQ | Generating policy answers | API key for Groq |
| `GROQ_MODEL` | Customer policy FAQ | Optional | Groq model name; default is `openai/gpt-oss-120b` |
| `BERT_MODEL_REPO` | Customer complaint processing | When local BERT files are absent | Hugging Face model repository ID |
| `BERT_MODEL_REVISION` | Customer complaint processing | When local BERT files are absent | Immutable 40-character model commit SHA |
| `HF_TOKEN` | Model download | Only if repository access requires it | Hugging Face access token |
| `CHROMA_PERSIST_DIRECTORY` | Customer policy FAQ | Optional | Local ChromaDB persistence directory; default is `data/chroma_db` |
| `ADMIN_USERNAME` | Admin app | Administrator sign-in | Administrator username |
| `ADMIN_PASSWORD` | Admin app | Administrator sign-in | Strong administrator password; at least 16 characters |

Example `.env` layout (replace every placeholder with your own values):

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DATABASE
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
BERT_MODEL_REPO=Kavinvelavan/complaintai-bank-bert-banking77
BERT_MODEL_REVISION=913bc6bdfc2bbbe2424559a511c8bfd343a6b19a
# HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
# CHROMA_PERSIST_DIRECTORY=data/chroma_db
ADMIN_USERNAME=YOUR_ADMIN_USERNAME
ADMIN_PASSWORD=REPLACE_WITH_A_STRONG_PASSWORD_OF_AT_LEAST_16_CHARACTERS
```

Do not paste real credentials into source code, notebooks, screenshots, public issues, or this README. In hosted environments, configure secrets separately for each Streamlit app.

---

## Run the applications

Run each application in a separate terminal from the repository root, with the virtual environment activated.

**Customer application**

```bash
python -m streamlit run customer_app.py
```

**Administrator dashboard**

```bash
python -m streamlit run admin_app.py
```

The customer app provides two tabs: **File a Complaint** and **Policy FAQ**. Sending a chat message only adds information to the current conversation; the customer must use the complaint submission form and provide at least one valid contact method to register the complaint.

The admin app requires the configured administrator credentials. There is no public administrator sign-up flow.

---

## Build or refresh the policy index

The customer app can initialise an incomplete local policy index from the eight PDFs when the Policy FAQ is first used. You can also run the ingestion script manually:

```bash
python scripts/ingest_policies.py
```

By default, the script reads PDFs from `knowledge_base/policies/` and stores the ChromaDB index under `data/chroma_db/`. It uses deterministic chunk identifiers and non-destructive upserts. The embedding model may be downloaded on first ingestion if it is not already cached.

The local ChromaDB directory is generated data and is not included in the repository ZIP. PostgreSQL is the persistent store for registered complaint records; do not treat the local vector-index directory as the complaint database.

---

## Run the tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The test suite is designed to exercise application components using isolated test data and mocks. The unit tests do not constitute a live integration test of a production database, the Groq API, a hosted Streamlit deployment, or a remote model download.

---

## Deployment notes

The customer and administrator entry points can be deployed as separate Streamlit Community Cloud apps from the same repository and branch:

| App | Entry point | Main configuration |
|---|---|---|
| Customer | `customer_app.py` | `DATABASE_URL`, `GROQ_API_KEY`, `BERT_MODEL_REPO`, `BERT_MODEL_REVISION`; optional `GROQ_MODEL`, `HF_TOKEN`, and `CHROMA_PERSIST_DIRECTORY` |
| Administrator | `admin_app.py` | `DATABASE_URL`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` |

Before deployment:

1. Apply the PostgreSQL migrations to the intended database through a controlled process.
2. Configure secrets independently for each app; never commit `.env` or Streamlit secrets.
3. Confirm the pinned BERT model artifact can be downloaded by a fresh app instance.
4. Confirm the policy PDFs are present and the policy index can be generated in the target environment.
5. Restrict access to the administrator app using the hosting platform's access controls, in addition to the application's sign-in check.
6. Test complaint registration, policy answers, administrator access, and status updates in the target environment before relying on the deployment.

Streamlit-hosted filesystems may be ephemeral. The local ChromaDB index can be rebuilt from the policy PDFs if it is missing. PostgreSQL is used for complaint persistence. Model downloads, cold starts, package compatibility, memory use, and platform limits must be checked on the selected hosting plan.

The current administrator authentication is a project-level demonstration mechanism. It does not provide multi-factor authentication, rate limiting, account recovery, or full production session hardening. Do not use this implementation to handle real banking complaints or sensitive financial data without a separate security review and appropriate operational controls.

---

## Security, limitations, and responsible use

- **Do not use real customer information** in the project demonstration, test suite, screenshots, or public repository.
- The public complaint form collects an email address, phone number, or both for follow-up. These details do not prove the submitter's identity.
- Complaint intent is predicted by BERT. The result can be wrong and must not be treated as a verified finding.
- Priority and department suggestions are determined by configured rules. They are not generated by BERT and do not mean a real department has received or acted on a case.
- Credential redaction recognises supported, explicitly labelled patterns. It is best-effort and is not a complete secret-detection system. Users should not enter passwords, PINs, CVVs, or one-time codes.
- Ambiguous labelled credential values are rejected by the persistence layer rather than saved unchanged.
- The complaint path uses local model processing and does not send complaint descriptions or contact details to Groq. The policy FAQ sends the current policy question, any prior cited answer used as follow-up context, and retrieved policy excerpts to Groq for answer generation.
- Policy answers depend on the coverage and quality of the included PDF documents and the passages retrieved. A citation identifier check does not guarantee that every generated statement is correct or fully supported.
- Entity extraction is pattern-based and supports only defined formats. For example, numeric dates can be ambiguous, and only supported currency patterns and explicitly cued transaction/card/account details are recognised.
- Top-3 model accuracy is not top-1 accuracy. Model scores are not a measure of the truthfulness of a complaint.
- The project has a measured held-out test result for BERT intent classification. It does not currently report a formal quantitative evaluation of RAG answer quality, hallucination rate, or end-to-end production performance.

---

## References

1. Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. [BANKING77 dataset on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) · [Paper](https://arxiv.org/abs/2003.04807).
2. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [ACL Anthology](https://aclanthology.org/N19-1423/).
3. Hugging Face. [Transformers documentation](https://huggingface.co/docs/transformers/index).
4. Hugging Face. [`bert-base-uncased` model card](https://huggingface.co/google-bert/bert-base-uncased).
5. Sentence Transformers. [`all-MiniLM-L6-v2` model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
6. Chroma. [Documentation](https://docs.trychroma.com/).
7. Groq. [API documentation](https://console.groq.com/docs/api-reference).
8. Streamlit. [Documentation](https://docs.streamlit.io/).
9. spaCy. [Usage documentation](https://spacy.io/usage).
10. pypdf. [Documentation](https://pypdf.readthedocs.io/).

---

**ComPlaintAI-Bank — Academic project demonstration.**

**KAVINVELAVAN MANIVASAKAN**

**STUDENT ID: 1704564**

