# Dynamic Skill Selection in Simple Words — Real-World QA Scenarios

A plain-English guide to why AI agents should only pick up the tools they actually need for each job — with four domain examples showing the difference it makes.

---

## Core Idea

Imagine hiring a contractor who shows up to fix your kitchen faucet carrying every tool they own — circular saw, jackhammer, pipe threader, electrical panel tester — and dumps them all on your kitchen floor before deciding which one to use. That's what most AI agents do with their tools today. Every turn, every question, the full toolkit lands in front of the model and it has to figure out what's relevant.

Dynamic Skill Selection is the practice of handing the contractor only the plumbing tools before they walk in the door. A fast, cheap routing step reads the incoming request, figures out what domain it belongs to, and loads only the relevant subset of capabilities into the AI's working memory. The other tools stay in the van.

The routing step is not the smart part of the system — it doesn't need to be. It just needs to be fast and accurate enough to get the right toolbox into the room. The LLM is the smart part, and it does better work when it isn't distracted by 40 tools it wasn't going to use anyway.

| Concept | Analogy |
|---|---|
| Skill registry | A complete tool inventory stored in a warehouse |
| Query embedding | Reading the work order to understand the job |
| Cosine similarity | Matching the job description to the right tool category |
| Top-k selection | Pulling only the relevant tools from the warehouse shelf |
| Permission filtering | Checking which tools the contractor is licensed to use |
| Eviction policy | Returning tools to the shelf that haven't been used in weeks |

---

## Scenario 1 — Customer Support: Telecom Company

### Problem Statement

A telecom company builds a support chatbot that handles billing questions, technical troubleshooting, account changes, and fraud reports. All four departments contributed tools: 8 billing tools, 9 network diagnostic tools, 7 account management tools, and 6 fraud investigation tools — 30 tools total. The bot loads all 30 for every customer message. When a customer asks "Why is my internet slow?", the bot occasionally responds by initiating a billing dispute process, the only tool it could find that mentioned "charges" in its description — the word appeared in a network speed test tool's error message handling.

### Solution

At registration, each tool's description is embedded and tagged with a department label. When a customer message arrives, the router embeds the query and computes similarity against all 30 tool descriptions. It returns the top 5. "Why is my internet slow?" scores highest against network diagnostic tools. Billing tools don't appear in the top 5 at all.

**Layman version:** Think of it like a call centre where the routing menu — "Press 1 for billing, press 2 for technical support" — happens automatically and invisibly before the conversation even reaches an agent. The customer never sees the menu, but the agent who answers already knows this is a network call and has the right scripts open.

### Outcome
- Billing-related tool misfires on technical queries drop by 91%
- Average resolution time falls from 4.2 minutes to 2.7 minutes because the model isn't sifting through irrelevant tools
- Customer satisfaction score for first-contact resolution improves by 18 percentage points

### Benefits
- **Accuracy:** The model focuses on the three to five tools that are actually useful, instead of picking from a crowd
- **Speed:** Fewer tool definitions in the prompt means faster LLM response — 3,200 fewer input tokens per turn
- **Reliability:** Billing actions can no longer be accidentally triggered by a technical support query

### Best Practices
- Write tool descriptions using the same words customers actually use in support tickets, not internal engineering names
- Test routing accuracy monthly using a sample of real customer queries as ground truth
- Set the top-k cap to 5 for a single-turn support bot — increase to 7 only if multi-step workflows require it

---

## Scenario 2 — Healthcare: Clinical Documentation Assistant

### Problem Statement

A hospital deploys an AI assistant for clinical staff. It has tools for retrieving patient records, ordering lab tests, looking up drug interaction data, submitting billing codes, scheduling appointments, and escalating to on-call physicians — 28 tools across 5 clinical domains. A nurse asks: "What are the contraindications for metformin in a patient with renal impairment?" The model, seeing the full tool list including order-lab-test and submit-billing-code, spends two LLM turns exploring whether it should order a creatinine test and generate a billing code before answering the drug interaction question — a task that required only the drug reference tool.

### Solution

The router classifies the query as a clinical reference question and selects the top 4 tools: drug interaction lookup, clinical guideline search, medication reference, and patient allergy checker. Lab ordering and billing tools are not in scope. The model answers in one turn with the correct contraindication information.

**Layman version:** It's the difference between a nurse asking a colleague a question and that colleague immediately pulling out the relevant reference manual versus first spreading every manual, order form, and billing guide across the desk before responding. The right reference goes on the desk; everything else stays in the cabinet.

### Outcome
- Clinical reference queries resolve in 1 LLM call instead of 2.4 average calls
- Zero billing-code submissions triggered by clinical reference queries
- Nurses report 35% higher confidence in AI responses because answers are more focused and less cluttered with irrelevant tool-call output

### Benefits
- **Safety:** High-risk tools (order-lab-test, prescribe-medication) are structurally invisible when the query doesn't warrant them
- **Compliance:** Audit logs show clean separation between reference queries and action-taking queries, simplifying HIPAA audit trails
- **Latency:** Average response time drops from 3.1 seconds to 1.4 seconds for reference queries

### Best Practices
- Apply strict role-based permission filtering: nurses, physicians, and billing staff see different tool subsets regardless of query similarity
- Never cache user roles — re-load from the hospital's IAM system on every turn; role changes (e.g., a rotating resident leaving) must take effect immediately
- Log every tool selection event, not just tool calls — selection logs reveal when the routing is working and when it is drifting

---

## Scenario 3 — Finance: Wealth Management Advisor Bot

### Problem Statement

A wealth management firm builds an internal AI assistant for advisors. Tools include portfolio analysis, trade execution, regulatory lookup, client communication templates, market data retrieval, and risk assessment — 22 tools. Advisors use it throughout the day for both client preparation (research, analysis) and active trading sessions (execution, risk). During a preparation session, an advisor asks "What is the dividend yield trend for MSFT over the last 5 years?" The model, with trade execution tools in context, responds with both the trend analysis and a draft trade order for MSFT, which the advisor has to manually dismiss before sharing the analysis with the client.

### Solution

The router detects "dividend yield trend" as a research and analysis intent. It selects the market data retrieval, portfolio analysis, and financial metrics tools. Trade execution tools require both semantic relevance AND an explicit trading-session flag in the user context. During a research session, the flag is not set, so execution tools are filtered out by the permission layer regardless of their similarity score.

**Layman version:** Imagine a car with separate keys for different modes: a standard key for driving and a special key that must be turned first to enable the launch control system. The engine always works. Launch control only activates when you deliberately engage it. The AI's execution tools are the launch control — always available when the right mode is active, structurally unavailable when it isn't.

### Outcome
- Draft trade orders triggered during research sessions drop to zero
- Advisor preparation time for client meetings decreases by 22% due to cleaner, more focused AI responses
- Compliance team signs off on dynamic selection as a control mechanism, reducing manual review burden by 30%

### Benefits
- **Risk control:** Execution tools cannot fire during sessions where they weren't intended — no accidental trades
- **Focus:** Research queries produce research answers; the model isn't tempted to take action when the task is analysis
- **Auditability:** Every tool selection is logged with session context, creating a clear audit trail for regulatory review

### Best Practices
- Model session state as a first-class parameter in the permission filter — "research mode" vs "trading mode" is not inferred from the query, it is explicitly set by the user
- Include domain prefixes in tool names: `trade.execute_order` vs `research.get_dividend_data` — this makes audit logs self-explanatory
- Run weekly accuracy checks: pull 50 advisor queries from logs, manually label the correct tool, and measure routing precision

---

## Scenario 4 — IT Helpdesk: Internal Employee Support Bot

### Problem Statement

A large enterprise deploys an IT helpdesk bot for 5,000 employees. Tools cover password reset, VPN troubleshooting, software installation requests, hardware ticket creation, access provisioning, and system health dashboards — 24 tools. Standard employees, IT administrators, and security engineers all use the same bot, but with very different access needs. When a standard employee asks "I can't access the shared drive," the model selects access provisioning tools (which should be IT-admin-only) because the description "manage access to shared resources" scores highly. The employee receives an error — the tool call fails because they don't have execution rights — but only after the model already attempted the call and consumed tokens.

### Solution

Permission filtering is applied before tool execution but, critically, also at the selection stage. Standard employee role is not permitted to see provisioning tools in the selection output at all — they don't appear in the top-k list regardless of similarity score. The model selects from access-request tools instead (which let the employee submit a ticket for an admin to action) and a shared-drive troubleshooting guide. The model answers with the right workflow for a standard employee: submit an access request, and here is the expected SLA.

**Layman version:** Think of a hotel keycard. The keycard for room 402 opens room 402, the gym, and the pool. It does not open the kitchen, the server room, or other guest rooms — not because the keycard holder couldn't physically walk to those doors, but because the card simply doesn't register as valid. The guest never knows those doors exist as options for them. The AI works the same way: if you don't have the role, the tool is invisible to you.

### Outcome
- Failed tool call attempts by non-privileged users drop from ~40/day to zero
- Token cost per IT query drops by 58% — provisioning tool schemas (large, complex JSON) no longer appear in employee prompts
- IT admins report better bot performance because their tool list (admin-only tools included) is now also smaller and more focused

### Benefits
- **Security:** Tool invisibility is stronger than tool-call failure. If the tool isn't in the prompt, the model cannot attempt to call it, even under adversarial prompting
- **Cost:** Fewer tool schemas in the prompt = fewer input tokens = lower cost per query
- **User experience:** Employees receive the correct workflow for their role immediately, without encountering confusing error messages from tools they can't use

### Best Practices
- Implement tool visibility as a hard filter, not just a runtime permission check — if a user's role excludes a tool, it must not appear in their prompt
- Review role-to-tool mappings quarterly with the security team — access creep is real and the mapping can drift from the intended policy
- Test permission filtering with a synthetic user for each role type before deploying changes to the registry

---

## Summary

| | Without Dynamic Skill Selection | With Dynamic Skill Selection |
|---|---|---|
| Tool definitions per prompt | All registered (20–40+) | Top-k relevant (3–7) |
| Input tokens per turn | 3,000–6,000 for tool schemas alone | 300–900 for selected tool schemas |
| Tool selection accuracy | Degrades past ~15 tools | Maintained at >90% top-5 |
| Cross-domain tool misfires | Common — model sees all tools | Rare — wrong-domain tools not visible |
| Permission enforcement | At execution time (fail loud) | At selection time (invisible to model) |
| Routing overhead | None | ~5–15ms, ~100 tokens per turn |
| Registry scalability | Breaks above ~20 tools in practice | Scales to 500+ tools with FAISS index |
