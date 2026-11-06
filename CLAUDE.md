# Standing Rules for Every Project

These rules apply to every project created in this folder. Follow them unless I explicitly say otherwise for a specific project.

## Workflow
- **Always show me the plan first** and wait for approval before building.
- **Ask before every `git push`.** Never push without explicit confirmation.

## GitHub
- GitHub account: `arbablatif`.
- Each project is its own **public** repository, created with the `gh` CLI.
- In commit messages and PRs, do not add Co-Authored-By lines or any Claude attribution.
## Code
- Python 3.12, with a `requirements.txt`.
- Clear, commented, modular code.

## Secrets
- API keys go only in a `.env` file.
- Add `.env` to `.gitignore` **before the first commit**.
- Include a `.env.example` with placeholder values.

## LLMs and Embeddings
- Default LLM: Google Gemini (cheapest Flash model).
- Provider must be configurable to Anthropic or OpenAI.
- Embeddings: local `sentence-transformers` (no API calls).

## Keep API Usage Minimal
- Cache every LLM response to disk.
- Keep evaluation sets small.
- Provide a `--mock` mode that runs end to end with no API calls.
- **Never call the API in a loop without asking me first.**

## Documentation
Every repo gets:
- A professional `README.md` with: problem, approach, architecture diagram (Mermaid), results, how to run, limitations.
- A `LEARNING.md` explaining the key concepts and design decisions in plain language, so the owner can explain them to clients.

## Testing and Results
- Include a small test or evaluation script.
- Put real results (from actually running it) in the README.

## Confidentiality
- No client names, client data, or credentials anywhere in the code, docs, or commit history.
