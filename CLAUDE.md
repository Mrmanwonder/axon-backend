# CLAUDE.md

Project context for agents working in axon-backend (Cloudflare Workers + shared library for
the mastery-* pipeline). The product rules (the model never assigns or disputes marks; never
fabricate a marking scheme; unsure data never reaches analytics; fail visibly; no silent
fallback to Cambridge) are defined in the Axon-Site repo's `CLAUDE.md` and apply here too.

Before merge: `npm run typecheck`, `npm test`, `npm run dry-run` (all Workers), and
`npm run check:contract-parity`. Gemini and other keys exist in the Workers' environment;
never print or commit a secret. A merged Worker is not live until it is deployed and the
runtime is checked.

## Linear-first execution protocol

Linear (team Axon-26, project Axon) is the system of record. Repos, CI, Supabase and
Cloudflare hold evidence; **nothing is planned, active, blocked, verified or complete
unless Linear says so** (AXO-28). Agents follow this before touching code.

1. **Read before you act.** Read the issue in full (`get_issue` with relations), plus its
   parent and children. The list view truncates descriptions. Each `### [ ]` / `- [ ]`
   box is an acceptance item with its own How / Where / Verify.
2. **Tick only with evidence.** Put the evidence (commit SHA, PR, CI run id, query output,
   live URL response, screenshot, test name) in a comment on that issue. "Merged" is not
   "done"; a migration file is not "applied"; merged backend code is not "live".
3. **Move status only on evidence.** All boxes ticked and verified → Done. Needs
   something only a human can do (real device, legal sign-off, third-party account,
   product decision) → stays open with a comment starting `BLOCKED-ON-HUMAN:` that states
   the exact ask. Status changes happen as work changes state, not afterwards.
4. **Never invent results.** If you could not run something, say so, do the preparatory
   work and leave it open. Never write "tested on mobile" without a real device.
5. **Parents close only when every child is done.** Otherwise comment what remains.
6. **New work is recorded first.** A bug, risk, dependency or scope change found mid-task
   goes into Linear before it is acted on. If a task is unnecessary, record why before
   cancelling it. Blockers are recorded with a blocked relation immediately.
7. **Open product decisions are not yours.** Write up options and a recommendation and ask
   the owner; do not resolve them in code.

Verification before requesting merge: run the repo's typecheck, lint, unit/integration
tests and build; confirm CI is green on the exact head; re-read your own diff. For database
changes: a versioned migration file in `supabase/migrations/`, applied, then Supabase
advisors re-run with no new warnings; production state must never differ from the repo.
Real-device requirements are not satisfied by automated tests.

Status policy: Backlog (valid, not committed) · Todo (fully specified) · In Progress
(work is happening) · In Review (implementation exists; verification, rollout or acceptance
remains) · Done (implementation + tests + deployment + acceptance evidence) · Duplicate
(points at one canonical issue).

Reports and specs are standalone files named `claude_[descriptor]-[YYYY-MM-DD].md`, linked
from the relevant Linear issue. Secrets are never printed or committed.
