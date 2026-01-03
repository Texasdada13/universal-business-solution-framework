# POC #5: Compliance and JJDPA Guardrail Agent

**Project Type:** LangGraph Background Agent with LangChain RAG over Regulatory Documents
**Complexity:** High
**Timeline Estimate:** 6-8 weeks for POC
**Primary Value:** Prevent compliance violations, reduce legal risk, ensure youth rights protected

---

## Problem Statement

Juvenile justice systems must comply with complex federal (JJDPA), state, and local regulations governing detention, placement, and treatment of youth. Violations can result in:
- Legal liability and lawsuits
- Loss of federal funding
- Harm to youth (inappropriate detention, rights violations)
- Reputational damage

Currently, compliance checking is:
- Manual and time-consuming
- Inconsistent across staff
- Reactive (violations discovered after the fact)
- Difficult (regulations complex and frequently updated)
- Incomplete (staff may not know all applicable rules)

**Goal:** A background agent that proactively reviews planned placements, detention decisions, and sanctions against JJDPA and state/local rules, flagging potential violations such as inappropriate detention of low-risk youth, with complete audit trail of rationale and citations.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│         COMPLIANCE & JJDPA GUARDRAIL AGENT                │
└──────────────────────────────────────────────────────────┘

LAYER 1: Regulatory Knowledge Base (LangChain RAG)
┌───────────────────────────────────────────────────────┐
│ REGULATORY DOCUMENTS:                                 │
│  • JJDPA federal statute and regulations              │
│  • State juvenile justice statutes                    │
│  • County/local policies and procedures               │
│  • Court rules and standing orders                    │
│  • Settlement agreements (if applicable)              │
│  • DOJ guidance documents                             │
│  • Case law (relevant precedents)                     │
│                                                       │
│ COMPLIANCE AREAS:                                     │
│  • Detention criteria (when detention allowed)        │
│  • Placement restrictions (e.g., no adult jails)      │
│  • Disproportionate Minority Contact (DMC) tracking   │
│  • Status offense handling (e.g., no detention)       │
│  • Sight and sound separation (from adult offenders)  │
│  • Youth rights (attorney access, notice, etc.)       │
│  • Sanctions and conditions (proportionality)         │
│                                                       │
│ [Document Loaders] → [Legal Text Splitting]          │
│        ↓                                              │
│ [Embeddings] → [Vector Store: Regulations]           │
│ [Metadata: jurisdiction, topic, effective_date]      │
└───────────────────────────────────────────────────────┘

LAYER 2: LangGraph Compliance Checking Pipeline
┌───────────────────────────────────────────────────────┐
│  TRIGGER: Proposed Action (placement, detention, etc.)│
│    ↓                                                  │
│  [Action Intake Node]                                 │
│    ├─ Parse proposed action details                  │
│    ├─ Identify action type (detention, placement,    │
│    │   sanction, condition, release, etc.)           │
│    ├─ Extract key facts (youth age, offense, risk,   │
│    │   proposed placement/sanction)                  │
│    └─ Validate required data present                 │
│    ↓                                                  │
│  [Retrieve Applicable Law/Policy Node]                │
│    ├─ Identify relevant regulations based on action  │
│    ├─ Query vector store by action type & context    │
│    ├─ Retrieve: JJDPA, state law, local policy       │
│    ├─ Filter by jurisdiction and effective date      │
│    └─ Retrieve precedents (prior violations/issues)  │
│    ↓                                                  │
│  [Evaluate Compliance Node]                           │
│    ├─ Match action facts to regulatory requirements  │
│    ├─ Check each applicable rule:                    │
│    │   • Detention criteria met?                     │
│    │   • Placement appropriate?                      │
│    │   • Youth rights protected?                     │
│    │   • Sanction proportional?                      │
│    │   • DMC considerations?                         │
│    ├─ Identify violations or concerns                │
│    └─ Cite specific statute/regulation for each      │
│    ↓                                                  │
│  [Emit Flags and Rationales Node]                    │
│    ├─ For each issue:                                │
│    │   • Severity (critical, high, medium, low)      │
│    │   • Violation type (detention, rights, DMC)     │
│    │   • Rationale (why it's a violation)            │
│    │   • Citation (JJDPA § X, State Code § Y)        │
│    │   • Recommended action (alternative)            │
│    ├─ Generate compliance report                     │
│    └─ Alert staff if critical violations             │
│    ↓                                                  │
│  [Log for Audit Node]                                 │
│    ├─ Store in audit database:                       │
│    │   • Proposed action                             │
│    │   • Compliance check results                    │
│    │   • Flags raised                                │
│    │   • Staff response (proceeded, modified, etc.)  │
│    │   • Timestamp and reviewer                      │
│    └─ Enable compliance reporting and monitoring     │
│    ↓                                                  │
│  [Human Review] ◄─── INTERRUPT() if violations       │
│    ├─ Staff reviews flags                            │
│    ├─ Options:                                       │
│    │   • Modify action to comply                     │
│    │   • Request exception (rare, supervisor)        │
│    │   • Abandon proposed action                     │
│    └─ Document decision                              │
│    ↓                                                  │
│  END → Action Approved or Modified                   │
└───────────────────────────────────────────────────────┘

LAYER 3: State Management
┌───────────────────────────────────────────────────────┐
│ ComplianceState:                                      │
│  - action_id: str                                     │
│  - action_type: str (detention, placement, sanction)  │
│  - action_details: Dict (youth, proposed action)      │
│  - retrieved_regulations: List[Document]              │
│  - compliance_checks: List[Dict] (rule → result)      │
│  - flags: List[Dict] (violations and concerns)        │
│  - staff_decision: str (approved, modified, abandoned)│
│  - audit_log: Dict (for compliance reporting)         │
└───────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Regulatory Knowledge Base (LangChain RAG)

**Document Sources:**

1. **Federal (JJDPA):**
   - Juvenile Justice and Delinquency Prevention Act (34 U.S.C. § 11101 et seq.)
   - Core protections:
     - **Deinstitutionalization of Status Offenders (DSO):** No detention for status offenses (truancy, runaway, etc.)
     - **Sight and Sound Separation:** Juveniles not detained/confined in facilities where they have contact with adult inmates
     - **Jail Removal:** Juveniles not detained in adult jails (with limited exceptions and 6-hour rule)
     - **Disproportionate Minority Contact (DMC):** Jurisdictions must address overrepresentation

2. **State Statutes:**
   - State juvenile justice code
   - Detention criteria (e.g., "clear and convincing evidence youth is flight risk or danger")
   - Placement standards
   - Youth rights (attorney access, notification, hearings)

3. **Local Policies:**
   - County juvenile justice policies
   - Detention screening tools and thresholds
   - Placement protocols
   - Sanction guidelines

4. **Court Orders:**
   - Standing orders from judges
   - Consent decrees or settlement agreements (if system under oversight)

5. **Guidance Documents:**
   - DOJ guidance on JJDPA compliance
   - State oversight agency guidance
   - Best practice manuals

**Document Processing:**
```python
Document Loaders:
  - PDFLoader for statutes and regulations (official versions)
  - WebBaseLoader for federal register, agency websites
  - TextLoader for court orders
  - JSONLoader for structured compliance rules

Text Splitting:
  - Legal text splitting (respect statutory sections, subsections)
  - chunk_size: 1500 tokens (legal provisions often long)
  - chunk_overlap: 200 tokens (maintain context across sections)
  - Preserve structure: § 11133(a)(11)(A) as unit

Metadata Tagging:
  - jurisdiction: "federal", "state", "county"
  - regulation_type: "detention_criteria", "placement_standards", "youth_rights", "DMC", "sanctions"
  - effective_date: Date (for filtering superseded regulations)
  - authority_level: "statute", "regulation", "policy", "guidance"
  - citation: "JJDPA § 11133(a)(11)", "State Code § 123.45"
```

**Embeddings:**
- Legal-domain embeddings if available (or general OpenAI embeddings)
- Semantic understanding of legal language

**Vector Store:**
- Development: ChromaDB
- Production: Pinecone or Qdrant
- Separate collections by jurisdiction (federal, state, local) for efficient filtering

### 2. LangGraph Compliance Pipeline Nodes

#### A. Action Intake Node
**Purpose:** Parse and validate proposed action details

**Logic:**
1. Accept input: Proposed action (detention decision, placement plan, sanction, etc.)
2. Identify action type:
   - Detention (pre-trial or post-disposition)
   - Placement (secure facility, group home, foster care, etc.)
   - Sanction (community service, restitution, probation conditions, etc.)
   - Release decision
   - Transfer to adult court

3. Extract key facts:
   - **Youth info:** Age, gender, race/ethnicity, offense type, risk level
   - **Proposed action:** Specific facility/program, duration, conditions
   - **Rationale:** Why this action? (flight risk, public safety, treatment needs)

4. Validate completeness: Required data present? (If missing, alert staff)

**Input Format (Example):**
```json
{
  "action_type": "detention",
  "youth": {
    "age": 15,
    "gender": "male",
    "race": "Black",
    "offense": "shoplifting (petit theft)",
    "offense_type": "misdemeanor",
    "status_offense": false,
    "prior_record": "none",
    "risk_assessment_score": "low"
  },
  "proposed_action": {
    "facility": "County Juvenile Detention Center",
    "duration": "pending adjudication (est. 30 days)",
    "rationale": "Parent unable to supervise; youth missed prior court date"
  },
  "staff": "Officer Jane Doe",
  "timestamp": "2024-03-15T10:30:00Z"
}
```

**No LLM needed** - structured data parsing

#### B. Retrieve Applicable Law/Policy Node
**Purpose:** Find all relevant regulations for this action type

**Logic:**
1. Based on action type, identify relevant compliance areas:
   - **Detention:** Detention criteria, JJDPA Jail Removal, Sight & Sound Separation, DMC
   - **Placement:** Placement standards, least restrictive setting, JJDPA core protections
   - **Sanction:** Proportionality, youth rights, evidence-based practices

2. Query vector store:
   - Semantic search: "detention criteria for low-risk youth"
   - Metadata filter:
     - jurisdiction: "federal" OR "state" OR "county"
     - regulation_type: "detention_criteria"
     - effective_date: <= current_date (exclude superseded regs)

3. Retrieve top 10-15 regulatory passages

4. Additional retrievals:
   - Prior violations or issues (historical compliance data)
   - Related case law or guidance

**Retrieval Queries (Examples):**

For detention decision:
- "When is pre-trial detention of juveniles permitted under JJDPA?"
- "Detention criteria for low-risk youth in [State]"
- "JJDPA status offense deinstitutionalization requirements"
- "Sight and sound separation requirements for juveniles"

For placement:
- "Placement standards for juvenile offenders"
- "Least restrictive setting requirements"
- "JJDPA jail removal exceptions"

**Tools:**
- MultiQueryRetriever (multiple perspectives on legal question)
- Metadata filtering for jurisdiction and date
- Parent-Child chunking (child for precise match, parent for full legal context)

#### C. Evaluate Compliance Node
**Purpose:** Assess proposed action against each applicable regulation

**Logic:**

1. **For each retrieved regulation:**
   - Extract requirements (e.g., "Detention only if clear and convincing evidence of flight risk or danger")
   - Match youth/action facts to requirements
   - Determine: COMPLIANT, VIOLATION, or CONCERN (borderline)

2. **Specific Checks:**

   **Detention Criteria:**
   - Does youth meet statutory criteria for detention?
   - Evidence standard met? (preponderance, clear & convincing, etc.)
   - Least restrictive alternative considered?
   - Risk assessment supports detention?

   **JJDPA Core Protections:**
   - **DSO:** Is offense a status offense? (If yes, detention prohibited)
   - **Sight & Sound:** Will youth have contact with adults? (If yes, violation)
   - **Jail Removal:** Is facility an adult jail? (If yes, check exceptions)
   - **DMC:** Is youth a minority? (Track for DMC compliance)

   **Youth Rights:**
   - Attorney access provided?
   - Parent/guardian notified?
   - Hearing within required timeframe?

   **Proportionality (Sanctions):**
   - Sanction proportional to offense severity?
   - Excessive conditions? (e.g., 200 hours community service for shoplifting)

3. **Generate Compliance Assessment:**
   - For each rule: Compliant, Violation, or Concern
   - Cite specific statutory/regulatory provision
   - Explain reasoning

**LLM Prompt (Evaluate Compliance):**
```
Evaluate whether the proposed action complies with the following regulation.

Proposed Action:
- Youth: {youth_info}
- Action: {proposed_action}

Regulation:
{regulation_text}

Determine:
1. Does the proposed action comply with this regulation? (COMPLIANT / VIOLATION / CONCERN)
2. Reasoning: Explain why, matching facts to legal requirements
3. Citation: Cite the specific section of the regulation
4. If VIOLATION or CONCERN: What would make it compliant? (Alternative action)

Output structured JSON.
```

**Output Format:**
```json
{
  "regulation": "JJDPA § 11133(a)(11) - Deinstitutionalization of Status Offenders",
  "citation": "34 U.S.C. § 11133(a)(11)(A)",
  "compliance_status": "COMPLIANT",
  "reasoning": "Youth's offense (shoplifting) is a delinquent act, not a status offense. DSO provision does not apply.",
  "alternative_action": null
}
```

**Example of Violation:**
```json
{
  "regulation": "State Juvenile Code § 45.67 - Detention Criteria",
  "citation": "State Code § 45.67(b)",
  "compliance_status": "VIOLATION",
  "reasoning": "State law requires 'clear and convincing evidence' that youth is a flight risk or danger to public safety for pre-trial detention. Youth's risk assessment is LOW, and the rationale ('parent unable to supervise') does not meet the legal standard. Missed court date alone may not constitute clear and convincing evidence of flight risk without additional factors.",
  "alternative_action": "Consider electronic monitoring, intensive supervision, or court reminder services instead of detention. If detention deemed necessary, document additional flight risk factors beyond single missed court date."
}
```

#### D. Emit Flags and Rationales Node
**Purpose:** Compile and prioritize compliance issues for staff review

**Logic:**

1. **Categorize Flags by Severity:**
   - **CRITICAL:** Clear statutory violation (e.g., detaining status offender, adult jail placement without exception)
   - **HIGH:** Likely violation or significant concern (e.g., detention criteria not met, proportionality issue)
   - **MEDIUM:** Borderline, needs careful review (e.g., DMC consideration, best practice vs. requirement)
   - **LOW:** Informational (e.g., reminder of reporting requirement)

2. **For each flag:**
   - Violation type (detention_criteria, JJDPA_core, youth_rights, proportionality, DMC)
   - Regulation violated
   - Rationale (why it's an issue)
   - Citation (specific statute/regulation)
   - Recommended alternative action

3. **Generate Compliance Report:**
   - Summary: Number of flags by severity
   - Detailed findings: Each flag with full rationale
   - Overall recommendation: APPROVE, MODIFY, or REJECT proposed action

4. **Alert Staff:**
   - If CRITICAL flags: Immediate alert (email, system notification)
   - If HIGH flags: Notification requiring review before proceeding
   - If MEDIUM/LOW: Information in report, no blocking

**Compliance Report Template:**
```
COMPLIANCE REVIEW REPORT

Proposed Action: Detention of [Youth Name]
Staff: Officer Jane Doe
Timestamp: 2024-03-15 10:30 AM

=== SUMMARY ===
Critical Flags: 0
High Flags: 1
Medium Flags: 1
Low Flags: 0

Overall Recommendation: MODIFY ACTION

=== DETAILED FINDINGS ===

[HIGH] Violation: Detention Criteria Not Met
Regulation: State Juvenile Code § 45.67(b)
Citation: State Code § 45.67(b)
Reasoning: State law requires "clear and convincing evidence" of flight risk or danger
for pre-trial detention. Youth's risk assessment is LOW, and rationale provided
("parent unable to supervise; missed one court date") does not meet the legal standard.

Recommended Action: Consider alternatives to detention such as electronic monitoring,
intensive supervision, or court reminder services. If detention is deemed necessary,
document additional flight risk factors beyond the single missed court date.

[MEDIUM] Consideration: Disproportionate Minority Contact (DMC)
Regulation: JJDPA § 11133(a)(15) - DMC Reduction
Citation: 34 U.S.C. § 11133(a)(15)
Reasoning: Youth is African American. Jurisdiction must monitor and address DMC.
This detention decision should be reviewed to ensure it is not contributing to
disproportionate detention rates.

Recommended Action: Document rationale for detention carefully. Consider whether
similarly situated white youth would receive the same decision. Track this case
for DMC reporting.

=== CITATIONS ===
1. 34 U.S.C. § 11133(a)(15) (JJDPA - DMC Reduction)
2. State Juvenile Code § 45.67(b) (Detention Criteria)

=== NEXT STEPS ===
Review findings and modify proposed action to address HIGH flag before proceeding.
Document decision-making process for audit trail.
```

#### E. Log for Audit Node
**Purpose:** Create complete audit trail for compliance monitoring and reporting

**Logic:**

1. Store in compliance audit database:
   - Proposed action (all details)
   - Compliance check results (all rules evaluated)
   - Flags raised (violations and concerns)
   - Staff decision (approved as-is, modified, abandoned)
   - Timestamp and reviewer

2. Enable queries for:
   - **Compliance Reporting:** How many detention decisions reviewed? How many violations detected?
   - **DMC Monitoring:** Track detention/placement decisions by race/ethnicity
   - **Trend Analysis:** Are certain staff/facilities generating more violations?
   - **Federal Reporting:** JJDPA compliance reports to OJJDP

3. Retention: Minimum 5 years (or per state law)

**Database Schema (Example):**
```sql
CREATE TABLE compliance_audits (
  audit_id SERIAL PRIMARY KEY,
  action_type VARCHAR(50),
  youth_demographics JSONB,  -- age, race, gender, offense
  proposed_action JSONB,
  compliance_results JSONB,  -- all checks performed
  flags JSONB,  -- violations and concerns
  staff_decision VARCHAR(50),  -- approved, modified, abandoned
  staff_user_id INT,
  timestamp TIMESTAMP,
  jurisdiction VARCHAR(50)
);

-- Indexes for reporting queries
CREATE INDEX idx_action_type ON compliance_audits(action_type);
CREATE INDEX idx_timestamp ON compliance_audits(timestamp);
CREATE INDEX idx_flags ON compliance_audits USING GIN (flags);
```

**No LLM needed** - straightforward database logging

#### F. Human Review Node (HITL)
**Purpose:** Staff review and decision on flagged violations

**Logic:**

1. If CRITICAL or HIGH flags: `interrupt()` - block action until review

2. Display compliance report to staff

3. Staff options:
   - **Modify Action:** Change proposed action to address flags (e.g., choose alternative to detention)
   - **Request Exception:** Rare; supervisor approval required; must document extraordinary circumstances
   - **Abandon Action:** Don't proceed with proposed action

4. Staff documents decision:
   - What did they decide?
   - Why? (If proceeding despite flag, extraordinary justification)
   - Supervisor approval (if exception)

5. Update audit log with staff decision

**UI/UX:**
- Clear presentation of compliance report
- Highlight CRITICAL/HIGH flags
- Show recommended alternatives
- Inline editing to modify proposed action
- Approval workflow for exceptions

**Implementation:**
```python
def human_review_node(state):
    if has_critical_or_high_flags(state["flags"]):
        # Block and require review
        interrupt("Compliance violations detected. Review required before proceeding.")

        # After human resumes
        if state["staff_decision"] == "exception_requested":
            # Route to supervisor approval
            interrupt("Exception requires supervisor approval.")

        # Log decision
        log_staff_decision(state)

    return state
```

### 3. State Management

**StateGraph Schema:**
```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class ComplianceState(TypedDict):
    # Action details
    action_id: str
    action_type: str  # detention, placement, sanction, release
    youth: Dict[str, Any]  # age, race, gender, offense, risk_level
    proposed_action: Dict[str, Any]  # facility, duration, rationale

    # Regulatory retrieval
    retrieved_regulations: List[Dict]  # full regulatory passages
    applicable_rules: List[str]  # rule names

    # Compliance evaluation
    compliance_checks: List[Dict]  # [{regulation, status, reasoning, citation}]

    # Flags
    flags: List[Dict]  # violations and concerns with severity
    critical_flags: int
    high_flags: int

    # Staff review
    staff_decision: Optional[str]  # approved, modified, abandoned, exception_requested
    staff_notes: Optional[str]
    supervisor_approved: Optional[bool]

    # Audit
    audit_log: Dict
    timestamp: datetime
```

**Checkpointing:**
- PostgresSaver for production
- Checkpoint after each node
- Enables supervisory review of compliance logic
- Audit trail reconstruction (time-travel)

### 4. Conditional Routing

**Routing Logic:**

1. **After Emit Flags Node:**
   - If CRITICAL or HIGH flags → Human Review Node (HITL)
   - Otherwise → Log for Audit, then END

2. **After Human Review:**
   - If modified → Re-run compliance check on modified action (loop back)
   - If exception requested → Supervisor approval (additional HITL)
   - If approved or abandoned → Log for Audit, then END

**Implementation:**
```python
def route_after_flags(state):
    if state["critical_flags"] > 0 or state["high_flags"] > 0:
        return "human_review"
    else:
        return "log_for_audit"

graph.add_conditional_edges(
    "emit_flags",
    route_after_flags,
    {
        "human_review": "human_review_node",
        "log_for_audit": "log_for_audit_node"
    }
)
```

---

## Data Flow

### Input
**Proposed Action:**
- Action type (detention, placement, sanction, release, etc.)
- Youth demographics and case details
- Specific proposed action (facility, duration, conditions)
- Rationale

### Processing
1. **Intake:** Parse and validate action details
2. **Retrieve:** Find applicable regulations (JJDPA, state, local)
3. **Evaluate:** Check compliance with each regulation
4. **Flag:** Identify violations and concerns, prioritize by severity
5. **Review (if violations):** Staff reviews and decides
6. **Log:** Audit trail for reporting and monitoring

### Output
1. **Compliance Report:**
   - Summary of flags
   - Detailed findings with citations
   - Recommended alternatives
   - Overall recommendation (approve, modify, reject)

2. **Alerts (if critical violations):**
   - Immediate notification to staff and supervisor
   - Block action until addressed

3. **Audit Log:**
   - Complete record for compliance monitoring
   - Enables DMC tracking, federal reporting, trend analysis

---

## Technical Implementation Considerations

### 1. LLM Selection
- **Primary:** Claude Sonnet 4.5
  - Excellent reasoning for legal compliance logic
  - Strong citation capabilities (critical for regulatory compliance)
  - Handles long context (legal documents)
- **Fallback:** GPT-4

### 2. Prompt Engineering

**Compliance Evaluation:**
- Temperature: 0.1 (consistency critical for legal compliance)
- Chain-of-thought: "First, identify the legal requirements. Then, match facts to requirements. Then, determine compliance."
- Few-shot examples: Show examples of COMPLIANT, VIOLATION, CONCERN determinations
- Explicit instruction: "Cite specific statutory section"

**Legal Text Interpretation:**
- Careful phrasing: Legal language precise, don't paraphrase loosely
- Conservatism: When uncertain, flag as CONCERN for human review
- Citations: Every determination must cite specific regulation

### 3. RAG Optimization

**Legal Document Chunking:**
- Respect statutory structure: Keep § 11133(a)(11)(A) as a unit
- Chunk size: 1500 tokens (legal provisions can be long)
- Parent-Child: Child for precise statute match, Parent for full legal context

**Metadata Filtering:**
- Jurisdiction: Federal, state, county (prioritize applicable law)
- Effective date: Filter out superseded regulations
- Regulation type: Detention, placement, etc. (relevance filtering)

**Retrieval Strategy:**
- MultiQueryRetriever: Legal questions phrased multiple ways
- Hybrid search: 50% semantic, 50% keyword (for specific statutory terms like "status offense", "sight and sound")
- Reranking: By authority level (statute > regulation > guidance)

**Update Frequency:**
- Legal changes: Immediately upon enactment (critical)
- Monitor federal register, state legislative updates
- Quarterly review of all content (ensure no outdated regulations)

### 4. Compliance Rule Encoding

**Structured Rules (For High-Stakes Checks):**

In addition to RAG, encode critical rules as structured logic for deterministic checking.

**Example: Status Offense DSO Check:**
```python
def check_dso_compliance(youth_offense):
    status_offenses = [
        "truancy", "runaway", "incorrigibility",
        "curfew violation", "underage drinking",
        "tobacco possession"
    ]

    if youth_offense.lower() in status_offenses:
        return {
            "rule": "JJDPA DSO",
            "status": "VIOLATION",
            "reasoning": f"{youth_offense} is a status offense. JJDPA prohibits detention of status offenders.",
            "citation": "34 U.S.C. § 11133(a)(11)(A)",
            "severity": "CRITICAL"
        }
    else:
        return {
            "rule": "JJDPA DSO",
            "status": "COMPLIANT",
            "reasoning": f"{youth_offense} is a delinquent act, not a status offense. DSO does not apply.",
            "citation": "34 U.S.C. § 11133(a)(11)(A)"
        }
```

**Why Both Structured + RAG:**
- **Structured:** Deterministic, fast, zero false negatives for critical rules
- **RAG:** Handles nuance, new regulations, complex fact patterns

### 5. DMC Tracking

**Disproportionate Minority Contact Monitoring:**

1. **Track by Race/Ethnicity:**
   - Every action logged with youth demographics
   - Calculate rates: % of detentions that are minority youth vs. % of population

2. **Flag Disparities:**
   - If jurisdiction's detention rate for Black youth > 2x white youth: Flag for review
   - Not a "violation" per se, but compliance concern (JJDPA requires addressing DMC)

3. **Dashboard:**
   - Real-time DMC metrics
   - Alerts if ratios spike
   - Drill-down: Which decision points drive disparity? (Detention, prosecution, placement)

**Implementation:**
```python
def check_dmc(state):
    youth_race = state["youth"]["race"]

    # Log for DMC tracking
    log_dmc_decision(
        action_type=state["action_type"],
        youth_race=youth_race,
        decision="detention",  # or other
        timestamp=datetime.now()
    )

    # Check if flagging needed
    current_ratio = calculate_dmc_ratio(state["action_type"], youth_race)
    if current_ratio > DMC_THRESHOLD:
        state["flags"].append({
            "type": "DMC",
            "severity": "MEDIUM",
            "detail": f"{youth_race} youth are detained at {current_ratio}x rate of white youth. JJDPA requires DMC reduction efforts.",
            "citation": "34 U.S.C. § 11133(a)(15)"
        })

    return state
```

### 6. Exception Handling and Overrides

**When Exceptions Appropriate:**
- Rare circumstances where compliance rule doesn't fit
- Example: Jail Removal exception for rural areas without juvenile facilities (limited JJDPA exception)
- Must document thoroughly

**Approval Hierarchy:**
- Staff cannot override CRITICAL flags (e.g., status offense detention)
- HIGH flags: Supervisor approval required
- MEDIUM flags: Staff discretion with documentation

**Documentation Requirements:**
- Extraordinary circumstances justifying exception
- Alternative actions considered
- Mitigating measures (if any)
- Supervisor sign-off

### 7. Integration with Case Management System

**Trigger Points:**
- Detention decision entered → Auto-trigger compliance check
- Placement plan created → Auto-trigger
- Sanction proposed → Auto-trigger
- Transfer to adult court → Auto-trigger

**API Integration:**
- Case management system calls compliance agent API
- Receives compliance report
- Blocks action if CRITICAL flags (system-level enforcement)

**Implementation:**
```python
# FastAPI endpoint
@app.post("/compliance/check")
async def check_compliance(action: ProposedAction):
    # Run LangGraph compliance agent
    result = compliance_graph.invoke({
        "action_type": action.type,
        "youth": action.youth,
        "proposed_action": action.details
    })

    # Return compliance report
    return {
        "flags": result["flags"],
        "recommendation": result["recommendation"],
        "report": result["compliance_report"]
    }
```

### 8. Evaluation and Quality Assurance

**Metrics to Track:**

1. **Detection Accuracy:**
   - False positive rate: % of flags that weren't actual violations (manual review)
   - False negative rate: % of violations missed (audit sampling)
   - Target: <5% false positives, <2% false negatives

2. **Compliance Rate:**
   - % of proposed actions that pass without flags
   - % modified after flags raised
   - Trend over time (improving compliance?)

3. **DMC:**
   - Detention rate ratios by race/ethnicity
   - Trend over time (reducing disparities?)

4. **Staff Response:**
   - Override rate (% of flags where staff proceeded anyway)
   - Average time to resolve flagged actions

**Quality Assurance:**
- Legal counsel reviews 100% of CRITICAL flags
- Supervisor reviews sample of HIGH flags
- Monthly audit: Review all overrides and exceptions
- Quarterly: External compliance audit (sample of cases)

### 9. Security and Audit Requirements

**Data Security:**
- Encryption for compliance audit data
- Access control: Only authorized compliance staff
- Audit logs: Who accessed compliance data when

**Retention:**
- Minimum 5 years (common juvenile justice standard)
- May be longer if under consent decree or litigation

**Federal Reporting:**
- JJDPA compliance data (§ 223 formula grants require annual reporting)
- DMC data (required by JJDPA)
- Export capabilities for OJJDP reporting forms

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals:**
- Build regulatory knowledge base (JJDPA, state law)
- Basic compliance checking (DSO, detention criteria)

**Deliverables:**
1. Vector store with JJDPA statute and state juvenile code
2. Structured DSO check (deterministic)
3. Basic detention criteria evaluation (RAG-based)

**Success Criteria:**
- Correctly identifies status offense violations (100%)
- Flags detention criteria issues (test on 10 scenarios)

### Phase 2: Multi-Rule Evaluation (Weeks 3-4)
**Goals:**
- Expand to all JJDP core protections
- Add proportionality checks for sanctions

**Deliverables:**
1. Sight & Sound, Jail Removal checks
2. Proportionality evaluation for sanctions
3. Compliance report generation

**Success Criteria:**
- Evaluates all 4 JJDPA core protections
- Generates compliance report with citations

### Phase 3: DMC and LangGraph (Week 5)
**Goals:**
- DMC tracking
- Full LangGraph workflow

**Deliverables:**
1. DMC monitoring and flagging
2. LangGraph pipeline (all nodes)
3. State management with checkpointing

**Success Criteria:**
- DMC data logged and monitored
- Complete workflow from proposed action to audit log

### Phase 4: HITL & Integration (Week 6)
**Goals:**
- Human review interface
- Integration with case management system

**Deliverables:**
1. Web UI for compliance report review
2. API endpoint for case management integration
3. Approval workflow for exceptions

**Success Criteria:**
- Staff can review flags and make decisions
- Case management system can trigger checks via API

### Phase 5: Testing & Refinement (Weeks 7-8)
**Goals:**
- Test with historical cases
- Legal counsel review
- Tune prompts and rules

**Deliverables:**
1. Test on 50 historical actions (known violations, known compliant)
2. Legal counsel feedback
3. Quality metrics report

**Success Criteria:**
- <5% false positives, <2% false negatives
- Legal counsel approves logic
- Staff find reports helpful

---

## Success Metrics

### Compliance Metrics
- **Violation Detection Rate:** >98% of actual violations flagged
- **False Positive Rate:** <5% (flags that weren't violations)
- **Compliance Improvement:** Reduction in violations over time (track quarterly)

### DMC Metrics
- **Data Completeness:** 100% of decisions logged with demographics
- **Disparity Reduction:** Track detention rate ratios, goal to reduce disparities
- **Transparency:** DMC data available for review and reporting

### Efficiency Metrics
- **Check Time:** <2 minutes per action (automated check)
- **Staff Review Time:** <10 minutes to review flagged actions (clear reports)

### Legal/Risk Metrics
- **Zero Status Offense Detentions:** Critical JJDPA violation
- **Zero Adult Jail Placements (without valid exception)**
- **Litigation Risk:** No lawsuits for compliance violations (long-term)

---

## Risks and Mitigation

### Risk 1: False Negative (Missed Violation)
**Likelihood:** Low-Medium
**Impact:** Critical (legal liability, harm to youth)
**Mitigation:**
- Layered approach: Structured rules + RAG
- Conservative flagging: When uncertain, flag for review
- Legal counsel review of system logic
- Audit sampling: Manual review to catch missed violations
- Continuous improvement based on audits

### Risk 2: False Positive (Incorrect Flag)
**Likelihood:** Medium
**Impact:** Medium (staff frustration, slows process)
**Mitigation:**
- Clear explanations in flags (staff can understand reasoning)
- Low threshold for CONCERN (vs. VIOLATION) to allow nuance
- Staff can proceed with justification (not hard block for all flags)
- Track false positives, refine prompts/rules

### Risk 3: Outdated Regulations
**Likelihood:** Medium
**Impact:** High (relying on superseded law)
**Mitigation:**
- Metadata filtering by effective date
- Monitoring process for legal changes (legislative updates, new regs)
- Quarterly content review by legal counsel
- Version control for regulatory documents

### Risk 4: Staff Overriding Legitimate Flags
**Likelihood:** Medium
**Impact:** High (defeats purpose, violations occur)
**Mitigation:**
- CRITICAL flags: Hard block, cannot override
- Override tracking and audit
- Supervisor review of all overrides
- Training: Why compliance matters, legal risks

### Risk 5: Complex Fact Patterns (LLM Uncertainty)
**Likelihood:** Medium
**Impact:** Medium (unclear guidance to staff)
**Mitigation:**
- Flag as CONCERN when uncertain (human judgment)
- Provide legal citations for staff to review
- Legal counsel as escalation path
- Continuous learning: Track complex cases, improve prompts

---

## Next Steps After POC Success

1. **Expand Regulatory Coverage:**
   - Add more state-specific regulations
   - Add local court rules
   - Include case law (precedents)

2. **Proactive Monitoring:**
   - Continuous background monitoring (flag patterns suggesting systemic issues)
   - Dashboard: Real-time compliance metrics
   - Predictive: "High risk of violations in this area"

3. **Training Integration:**
   - Use compliance reports as training tool
   - "Why was this flagged?" educational prompts
   - New staff onboarding: Review common violations

4. **Policy Development:**
   - Identify compliance gaps (frequent violations)
   - Suggest policy changes to leadership
   - Draft new policies aligned with best practices

5. **Federal Reporting Automation:**
   - Auto-generate JJDPA compliance reports
   - DMC reports
   - Submit to OJJDP (Office of Juvenile Justice and Delinquency Prevention)

---

## Conclusion

The Compliance and JJDPA Guardrail Agent addresses a critical need: preventing legal violations in a complex regulatory environment. By combining LangChain RAG over legal documents with LangGraph's structured evaluation pipeline, this system can:

- Proactively flag potential violations before they occur
- Ensure youth rights are protected (DSO, detention criteria, proportionality)
- Support DMC reduction efforts (tracking and transparency)
- Reduce legal liability for jurisdictions
- Create complete audit trails for compliance reporting
- Educate staff on regulatory requirements

The architecture prioritizes accuracy with layered checking (structured rules + RAG), legal precision with citations, and human oversight for complex decisions. Success requires high-quality legal content, conservative flagging (err on side of caution), and strong collaboration with legal counsel during development.

**Recommended as fifth POC**, after establishing RAG and multi-agent infrastructure. High complexity but critical value for legal compliance and youth protection. Essential for systems under federal oversight or at risk of litigation.
