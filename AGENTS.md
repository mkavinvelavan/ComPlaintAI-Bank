# Development instructions

## Project rules

- This project is a planned NLP, LLM, and RAG-based banking complaint registration and policy assistance system. Follow the architecture and staged development plan in `README.md`.
- Never mention real bank names in policy content or customer-facing project examples.
- Treat the eight existing PDFs in `knowledge_base/policies/` as source policy documents. Do not modify them unless explicitly asked; never add effective dates or page numbers to them.
- Do not commit secrets, API keys, passwords, database credentials, `.env` files, complaint data, or local model weights.
- Do not claim that a complaint was saved, a refund processed, a department contacted, or another external action completed without confirmation from the relevant system.
- Keep original complaint text separate from model predictions and extracted entities.
- Do not expose internal priority, confidence scores, administrator notes, routing metadata, or the complaint register to customers.
- Protect administrator functionality with authentication and server-side authorization. Use authorized, appropriately scoped database permissions.
- Generate a complaint reference only after the database confirms successful registration.
- Policy answers must be grounded in retrieved policy text.
- Test each component before integrating it. Build incrementally; do not generate the entire application in one attempt.
- Do not install large ML packages or create application source files during the initial setup stage unless a later task explicitly calls for that work.

## Instructions for future coding tasks

1. Inspect the existing files and repository state before making changes. Identify relevant instructions and preserve the existing policy PDFs.
2. Implement the requested work incrementally, changing only files needed for the task. Avoid unrelated edits and generated artifacts.
3. Explain what changed and why in the completion summary.
4. Run appropriate checks or tests for the changes when requested or needed to verify the work. Report the exact checks performed and their results, including failures; never claim unrun checks passed.
5. Do not commit, publish, deploy, or perform external actions unless explicitly requested and authorized.
