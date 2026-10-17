# ComPlaintAI-Bank

ComPlaintAI-Bank is a planned NLP, LLM, and retrieval-augmented generation (RAG) system for banking complaint registration and policy assistance. The customer application will provide a policy assistant grounded in approved policy text and a complaint registration workflow. A separate administrator application will support authorized complaint operations.

This repository is at the project setup stage. Application source files and model artifacts have not yet been created. The eight existing PDFs in `knowledge_base/policies/` are the planned policy knowledge base. Preserve these source documents; do not add effective dates or page numbers to them.

## Planned architecture

- Fine-tune `bert-base-uncased` on BANKING77 in Google Colab for complaint intent classification. Keep a backup of the trained model in Google Drive and arrange deployment access separately.
- Use spaCy with EntityRuler for relevant entity extraction.
- Use explicit rule-based logic for priority assignment and internal routing.
- Embed policy text with `all-MiniLM-L6-v2` and retrieve it with ChromaDB.
- Generate policy answers from retrieved text using Llama 3.1 through Groq, subject to model availability. Answers must remain grounded in retrieved policy text.
- Build two separate Streamlit applications: `customer_app.py` and `admin_app.py`.
- Store complaints in persistent PostgreSQL, currently planned for Neon.
- Plan separate deployments for the two applications using GitHub and Streamlit Community Cloud, subject to their current free-tier limits.

## Environment setup

No application dependencies or source files are defined yet. When implementation begins, use a Python virtual environment and install only the dependencies needed for the component being developed. Keep credentials in environment variables or a local `.env` file that is excluded from version control. Never commit API keys, passwords, database credentials, complaint data, or local model weights.

Deployment configuration will need, at minimum, appropriately scoped database credentials and a Groq API key. Configure secrets in the deployment platform rather than storing them in source control. Verify model and service availability, database access controls, and platform limits before deployment.

## Training workflow

1. Prepare and fine-tune `bert-base-uncased` on BANKING77 in Google Colab.
2. Evaluate the classifier on held-out data and record the evaluation method and results; do not present unverified performance claims.
3. Back up the trained model in Google Drive. Set up and verify deployment access to the model separately from the backup.
4. Keep complaint text, classifier predictions, and extracted entities as distinct data. Establish privacy and retention controls before using real complaint data.

## Development and testing approach

Develop and test components incrementally before integrating them. Start by inspecting the repository and existing policy documents. Define component boundaries and tests before implementation; test classification, extraction, priority/routing rules, policy retrieval and answer grounding, complaint persistence, authorization, and each user interface independently as appropriate. Then test their integration. Report test commands, results, and failures honestly; do not claim a check passed unless it was run and verified.

The customer interface must not expose the complaint register, internal priority, confidence scores, administrator notes, or routing metadata. Generate a complaint reference only after the database confirms successful registration. Do not state that a complaint was saved, a refund was processed, a department was contacted, or another external action occurred without confirmation.

## Deployment plan

Deploy `customer_app.py` and `admin_app.py` as separate Streamlit Community Cloud applications connected to their intended GitHub repository and branch. Confirm current platform and free-tier limits, persistent database configuration, secrets handling, model access, and health of each application before release. Administrator authentication and authorization must be enforced server-side, with database permissions scoped to the required operations. Never rely on hiding interface elements as the authorization control.

## Project safeguards

- Never mention real bank names in policy content or customer-facing project examples.
- Do not modify the policy PDFs unless explicitly asked. Do not add effective dates or page numbers to them.
- Do not commit secrets, credentials, `.env` files, complaint data, or local model weights.
- Preserve original complaint text separately from predictions and extracted entities.
- Do not expose internal administrative fields or the complaint register to customers.
- Do not claim an operation succeeded without confirmation from the relevant system.
- Do not install large ML packages or create application source files during this setup stage.
