# POC #1: Intake Triage Assistant

**Project Type:** LangGraph Multi-Agent System with LangChain RAG
**Complexity:** Medium
**Timeline Estimate:** 4-6 weeks for POC
**Primary Value:** Consistent, thorough intake process with policy compliance

---

## Problem Statement

Intake staff need to conduct structured questioning of youth and families, check eligibility against complex program rules, and create comprehensive initial case summaries. Currently, this process:
- Varies by staff experience and knowledge
- May miss critical eligibility criteria
- Lacks consistent documentation
- Requires manual policy lookup
- Time-consuming (45-90 minutes per intake)

**Goal:** A LangGraph agent that guides intake staff through structured questioning, automatically checks eligibility against program rules, and creates initial case summaries with explicit policy citations.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 INTAKE TRIAGE ASSISTANT                  │
└─────────────────────────────────────────────────────────┘

LAYER 1: Knowledge Base (LangChain RAG)
┌──────────────────────────────────────────────────┐
│ • Local county policies & procedures             │
│ • JJDPA guidance documents                       │
│ • State juvenile justice statutes                │
│ • Prior case patterns & precedents               │
│ • Program eligibility rules                      │
│                                                  │
│ [Document Loaders] → [Text Splitters]           │
│        ↓                                         │
│ [Embeddings] → [Vector Store: ChromaDB/Pinecone]│
└──────────────────────────────────────────────────┘

LAYER 2: LangGraph Multi-Agent Orchestration
┌──────────────────────────────────────────────────┐
│  START                                           │
│    ↓                                             │
│  [Case Intake Node]                              │
│    ├─ Collects basic information                │
│    ├─ Validates required fields                 │
│    └─ Initializes case state                    │
│    ↓                                             │
│  [Structured Questioning Agent]                  │
│    ├─ Generates contextual questions            │
│    ├─ Adapts based on responses                 │
│    ├─ Ensures comprehensive coverage            │
│    └─ Stores responses in state                 │
│    ↓                                             │
│  [Policy Retrieval Agent] ─────────┐            │
│    ├─ Retrieves relevant policies  │            │
│    ├─ Finds similar prior cases    │            │
│    └─ Extracts eligibility rules   │            │
│    ↓                                │            │
│  [Eligibility Checking Agent] ◄────┘            │
│    ├─ Matches youth profile to programs         │
│    ├─ Checks against eligibility criteria       │
│    ├─ Flags potential barriers                  │
│    └─ Cites specific policy passages            │
│    ↓                                             │
│  [Risk Assessment Agent]                         │
│    ├─ Evaluates risk factors                    │
│    ├─ Retrieves risk assessment policies        │
│    ├─ Applies validated risk tools              │
│    └─ Generates risk summary with citations     │
│    ↓                                             │
│  [Case Summary Generator]                        │
│    ├─ Synthesizes all collected information     │
│    ├─ Creates structured case summary           │
│    ├─ Adds policy citations throughout          │
│    └─ Formats for case management system        │
│    ↓                                             │
│  [Human Review & Approval] ◄─── INTERRUPT()     │
│    ├─ Intake officer reviews summary            │
│    ├─ Can edit any section                      │
│    ├─ Can request additional questioning        │
│    └─ Approves before case creation             │
│    ↓                                             │
│  END → Case Created in System                   │
└──────────────────────────────────────────────────┘

LAYER 3: State Management
┌──────────────────────────────────────────────────┐
│ AgentState:                                      │
│  - youth_info: Dict (name, age, demographics)    │
│  - responses: List[QA pairs]                     │
│  - retrieved_policies: List[Document]            │
│  - eligibility_results: Dict[program → status]   │
│  - risk_assessment: Dict                         │
│  - case_summary: str                             │
│  - officer_notes: str                            │
│  - approved: bool                                │
│  - timestamp: datetime                           │
└──────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Knowledge Base (LangChain RAG)

**Document Sources:**
- County-specific intake policies and procedures
- JJDPA federal guidance documents
- State juvenile justice statutes and regulations
- Program eligibility requirements (diversion, detention alternatives, etc.)
- Prior intake case summaries (anonymized patterns)
- Assessment tools and scoring guides

**Document Processing:**
```
Document Loaders:
  - PDFLoader for policy manuals
  - JSONLoader for structured eligibility rules
  - CSVLoader for historical case data
  - WebBaseLoader for online resources (state websites)

Text Splitting Strategy:
  - RecursiveCharacterTextSplitter with semantic chunking
  - chunk_size: 1000 tokens (balance context and precision)
  - chunk_overlap: 200 tokens (maintain context across boundaries)
  - Structure-aware splitting for policy sections

Embeddings:
  - OpenAI text-embedding-3-small (cost-effective)
  - Or sentence-transformers for local/privacy

Vector Store:
  - Development: ChromaDB (local, easy setup)
  - Production: Pinecone or Qdrant (managed, scalable)
  - Metadata: policy_type, effective_date, source, section
```

**Retrieval Strategy:**
- **MultiQueryRetriever** for eligibility questions (multiple phrasings)
- **Parent-Child Retriever** for policy citations (precise match, full context)
- **Metadata filtering** by policy type, date range, jurisdiction

### 2. LangGraph Agent Nodes

#### A. Case Intake Node
**Purpose:** Initialize case and collect basic information

**Logic:**
1. Create new case ID
2. Collect: youth name, DOB, contact info, guardian info
3. Validate required fields
4. Initialize state object
5. Transition to Structured Questioning Agent

**No LLM needed** - simple data collection form

#### B. Structured Questioning Agent
**Purpose:** Guide intake officer through comprehensive questioning

**Logic:**
1. Retrieve intake question templates from knowledge base
2. Use LLM to generate contextual follow-up questions based on responses
3. Track question coverage (ensure all required areas addressed)
4. Store Q&A pairs in state
5. Adaptive questioning:
   - If substance use mentioned → deeper substance use questions
   - If family conflict → family dynamics questions
   - If school issues → educational history questions

**LLM Prompt Template:**
```
You are an expert juvenile justice intake specialist. Based on the following
information already collected, generate the next most important question to ask.

Already Covered: {covered_topics}
Youth Responses So Far: {qa_history}
Required Topics Remaining: {uncovered_topics}

Generate ONE specific, clear question that will help complete a thorough intake
assessment. The question should be appropriate for the youth's age ({age}) and
situation.
```

**Tools:**
- Question template retriever
- Coverage tracker (checklist of required topics)

**Exit Condition:** All required topics covered OR officer manually ends questioning

#### C. Policy Retrieval Agent
**Purpose:** Find relevant policies and prior cases

**Logic:**
1. Extract key facts from collected information (age, charges, risk factors, location)
2. Query vector store for:
   - Relevant eligibility policies
   - Similar prior cases (pattern matching)
   - Applicable assessment tools
3. Rerank results by relevance
4. Store retrieved documents in state

**Retrieval Queries:**
- "Eligibility requirements for {age} year old charged with {offense}"
- "Diversion program criteria for {jurisdiction}"
- "Risk assessment for youth with {risk_factors}"

**Tools:**
- Vector store retriever with metadata filtering
- Reranker (cross-encoder model for relevance scoring)

#### D. Eligibility Checking Agent
**Purpose:** Determine program eligibility with citations

**Logic:**
1. For each program type (diversion, detention alternatives, community programs):
   - Extract eligibility criteria from retrieved policies
   - Match youth profile against criteria
   - Determine: eligible, ineligible, or conditional
   - Cite specific policy passages for determination
2. Identify barriers (age, offense type, prior history)
3. Suggest alternative programs if ineligible

**LLM Prompt Template:**
```
Based on the following youth information and policy excerpt, determine eligibility
for {program_name}.

Youth Profile:
{youth_info}

Policy Excerpt:
{policy_text}

Provide:
1. Eligibility determination: ELIGIBLE / INELIGIBLE / CONDITIONAL
2. Reasoning with specific criteria matched or not met
3. Exact policy citation (section, page)
4. If ineligible, identify the specific barrier(s)
```

**Output Format:**
```json
{
  "program": "Youth Diversion Program",
  "status": "ELIGIBLE",
  "criteria_matched": [
    {"criterion": "Age 12-17", "youth_value": "15", "match": true},
    {"criterion": "First-time offense", "youth_value": "Yes", "match": true},
    {"criterion": "Non-violent offense", "youth_value": "Theft", "match": true}
  ],
  "policy_citation": "County Diversion Policy Manual, Section 3.2, Page 12",
  "confidence": 0.95
}
```

#### E. Risk Assessment Agent
**Purpose:** Conduct validated risk assessment

**Logic:**
1. Retrieve applicable risk assessment tool (YLS/CMI, SAVRY, etc.)
2. Extract risk factors from collected information
3. Apply scoring rules
4. Generate risk level (low, moderate, high)
5. Cite assessment tool and methodology
6. Flag items needing additional information

**Tools:**
- Risk assessment tool retriever
- Scoring calculator
- Risk factor extractor

**Note:** May require human input for subjective items. Use interrupt() to ask officer for clinical judgment items.

#### F. Case Summary Generator
**Purpose:** Create comprehensive, well-structured initial case summary

**Logic:**
1. Synthesize all collected information:
   - Youth demographics and background
   - Presenting issue (referral reason)
   - Intake interview responses
   - Risk assessment results
   - Eligibility determinations
   - Recommended next steps
2. Use Map-Reduce if extensive information (multiple documents)
3. Include citations throughout (policies, prior cases, assessment tools)
4. Format according to case management system requirements

**LLM Prompt Template:**
```
Create a comprehensive initial case summary for juvenile justice case management.

Youth Information: {youth_info}
Intake Responses: {qa_pairs}
Risk Assessment: {risk_results}
Eligibility Results: {eligibility}
Retrieved Policies: {policies}

Structure:
1. Identifying Information
2. Referral Reason and Presenting Issue
3. Background and History
4. Risk and Needs Assessment
5. Eligibility for Programs/Services
6. Recommended Next Steps
7. Citations and References

Use professional juvenile justice terminology. Cite specific policies and assessment
tools. Be objective and factual. Highlight any immediate safety concerns.
```

**Output:** Structured markdown document with clear sections and inline citations

#### G. Human Review & Approval Node
**Purpose:** Intake officer review before case creation

**Logic:**
1. Use `interrupt()` to pause workflow
2. Display case summary to officer
3. Officer can:
   - Approve as-is → Proceed to case creation
   - Edit summary → Update state, resume
   - Request additional questioning → Loop back to Structured Questioning Agent
   - Reject and start over → Clear state, restart
4. Log officer decision and any edits for audit trail

**Implementation:**
```python
def human_review_node(state):
    # Display summary to officer
    interrupt("Please review the case summary and approve or edit.")

    # After human resumes:
    if state["approved"]:
        return "create_case"
    elif state["request_additional_questioning"]:
        return "structured_questioning_agent"
    else:
        return "case_summary_generator"  # Re-generate with edits
```

### 3. State Management

**StateGraph Schema:**
```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class IntakeState(TypedDict):
    # Case identification
    case_id: str
    timestamp: datetime
    intake_officer: str

    # Youth information
    youth_info: Dict[str, Any]  # name, dob, demographics, contact
    guardian_info: Dict[str, Any]

    # Intake process
    responses: List[Dict[str, str]]  # [{"question": "...", "answer": "..."}]
    covered_topics: List[str]
    uncovered_topics: List[str]

    # Retrieved knowledge
    retrieved_policies: List[Dict]  # from vector store
    similar_cases: List[Dict]

    # Assessments
    risk_assessment: Dict[str, Any]
    eligibility_results: List[Dict]

    # Output
    case_summary: str
    officer_notes: str

    # Workflow control
    current_node: str
    approved: bool
    request_additional_questioning: bool
```

**Checkpointing:**
- Use **PostgresSaver** for production (persistent across sessions)
- Checkpoint after each major node
- Allows intake to pause/resume (phone call interruption, shift change)
- Time-travel debugging for supervisory review

### 4. Conditional Routing Logic

**Routing Decisions:**

1. **After Structured Questioning:**
   - If all topics covered → Policy Retrieval Agent
   - If officer ends early but critical topics missing → Warning + offer to continue

2. **After Eligibility Checking:**
   - If high-risk youth → Flag for supervisor review before final summary
   - If eligible for diversion → Include diversion-specific recommendations
   - If ineligible for all programs → Suggest next steps (court, detention, etc.)

3. **After Human Review:**
   - If approved → Create case
   - If edited → Regenerate summary
   - If request more info → Loop back to questioning

**Implementation:**
```python
def route_after_review(state):
    if state["approved"]:
        return "create_case"
    elif state["request_additional_questioning"]:
        return "structured_questioning_agent"
    else:
        return "case_summary_generator"

graph.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "create_case": "create_case_node",
        "structured_questioning_agent": "structured_questioning_agent",
        "case_summary_generator": "case_summary_generator"
    }
)
```

---

## Data Flow

### Input
1. **From Intake Officer:**
   - Youth identifying information (name, DOB, contact)
   - Guardian information
   - Referral source and reason
   - Responses to interview questions

2. **From Knowledge Base:**
   - Policy documents
   - Assessment tools
   - Eligibility criteria
   - Prior case patterns

### Processing
1. **Structured Questioning Phase:**
   - Agent generates contextual questions
   - Officer asks questions, records responses
   - System tracks coverage

2. **Analysis Phase:**
   - Retrieve relevant policies based on case facts
   - Check eligibility against multiple programs
   - Conduct risk assessment

3. **Synthesis Phase:**
   - Generate comprehensive case summary
   - Add policy citations
   - Format for case management system

4. **Review Phase:**
   - Officer reviews and edits
   - Officer approves or requests changes

### Output
1. **Case Summary Document:**
   - Structured sections (identifying info, background, assessment, recommendations)
   - Policy citations throughout
   - Risk level and eligibility determinations

2. **Case Management System Entry:**
   - Structured data exported to existing case management database
   - Attachments (summary PDF, policy references)

3. **Audit Trail:**
   - All questions asked and responses
   - Policies retrieved and applied
   - Officer edits and approvals
   - Timestamp of each step

---

## Technical Implementation Considerations

### 1. LLM Selection
- **Primary:** Claude Sonnet 4.5 (balance of quality and cost)
  - Excellent reasoning for eligibility checking
  - Strong citation capabilities
  - Good at structured outputs
- **Fallback:** GPT-4 (diversity, model availability)
- **Cost Optimization:** Claude Haiku for simple nodes (question generation)

### 2. Prompt Engineering
**Key Principles:**
- **Few-shot examples** for eligibility checking (show format)
- **Chain-of-thought** for complex reasoning ("First, identify the criteria. Then...")
- **Explicit instruction** to cite sources
- **Structured outputs** using JSON schemas where needed
- **Temperature settings:**
  - Low (0.2) for eligibility checking (consistency)
  - Medium (0.5) for question generation (variety)
  - Low (0.3) for summarization (factual)

**Prompt Versioning:**
- Store prompts in configuration files, not hardcoded
- Version control for prompts (Git)
- A/B testing different prompt formulations

### 3. RAG Optimization
**Chunking Strategy:**
- Semantic chunking for policy documents (respect section boundaries)
- Metadata tagging:
  - `policy_type`: "eligibility", "assessment", "procedure"
  - `effective_date`: for filtering outdated policies
  - `jurisdiction`: "county", "state", "federal"
  - `section`: "3.2.1" (for precise citation)

**Retrieval Tuning:**
- Top-k: 5-10 documents per query
- Similarity threshold: 0.7 (balance precision and recall)
- Hybrid search: 70% semantic, 30% keyword (for exact legal terms)
- Reranking: Use cross-encoder for top 10 → top 3

**Evaluation:**
- Manually curate 20-30 test cases
- Measure: retrieval precision, citation accuracy, eligibility correctness
- Iterate on chunking and retrieval parameters

### 4. Human-in-the-Loop Interface
**UI/UX Design:**
- Clear presentation of AI-generated summary
- Side-by-side: summary + source policies (for verification)
- Inline editing capability
- Highlight areas of uncertainty (low confidence scores)
- One-click approval or request changes

**Integration Options:**
- **Web App:** React frontend with FastAPI backend
- **Desktop App:** Electron wrapper for security
- **Case Management Integration:** API to existing system

**Workflow:**
1. Agent completes summary → interrupt()
2. UI displays summary with sections collapsible
3. Officer reviews, clicks sections to expand/edit
4. Officer clicks "Approve" or "Request Additional Information"
5. Graph resumes from checkpoint

### 5. State Persistence
**Checkpointing:**
- PostgreSQL for production (existing case management DB)
- Checkpoint after each agent node
- Enables:
  - Resume after interruption (phone call, break)
  - Supervisor review of intake process
  - Time-travel for training (replay intake)

**Data Retention:**
- Keep checkpoints for 90 days (compliance)
- Archive final case summary permanently
- Anonymize for training data

### 6. Error Handling
**Potential Failures:**
1. **LLM API failures:**
   - Retry with exponential backoff (3 attempts)
   - Fallback to secondary model (GPT-4 if Claude fails)
   - If all fail: alert officer, allow manual summary entry

2. **Vector store unavailable:**
   - Cached frequently-used policies (local fallback)
   - Graceful degradation: continue without RAG (officer references policies manually)

3. **Parsing errors (LLM output):**
   - Validate JSON outputs with Pydantic schemas
   - If invalid: retry with explicit format instruction
   - After 2 retries: fallback to freeform text

4. **User input errors:**
   - Validation at data entry (required fields, format checks)
   - Clear error messages
   - Save partial progress (checkpointing)

### 7. Evaluation and Quality Assurance
**Metrics to Track:**
1. **Efficiency:**
   - Average intake time (target: 30 minutes vs. 60-90 manual)
   - Number of questions asked (consistency)

2. **Quality:**
   - Completeness: % of required topics covered
   - Citation accuracy: % of citations verified by officers
   - Eligibility accuracy: % of determinations agreed with by supervisors

3. **User Satisfaction:**
   - Officer feedback surveys
   - Edit frequency (high edits = low quality)
   - Approval rate (% approved without edits)

4. **Cost:**
   - Average tokens per intake
   - Cost per intake (target: <$0.50)

**Quality Assurance Process:**
1. **Initial Testing:** Supervisors review first 20 intakes completely
2. **Sampling:** 10% random sample ongoing
3. **Feedback Loop:** Weekly review meetings, prompt/agent tuning
4. **Version Control:** Track performance across agent versions

### 8. Security and Privacy
**Data Protection:**
- Encrypt state data at rest (PostgreSQL encryption)
- Encrypt in transit (TLS for API calls)
- PII handling:
  - Anonymize before sending to LLM where possible
  - Use local LLMs for high-sensitivity (Ollama)
  - Data retention policies (purge after 90 days)

**Access Control:**
- Role-based access (only intake officers)
- Audit logs (who accessed what when)
- Case data segregated by jurisdiction

**Compliance:**
- JJDPA confidentiality requirements
- State juvenile record laws
- FERPA (if school records involved)

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals:**
- Set up development environment
- Build RAG pipeline with sample policies
- Create basic LangGraph structure (3 nodes)

**Deliverables:**
1. ChromaDB vector store with 10 sample policy documents
2. Basic LangGraph: Intake → Questioning → Summary
3. Simple CLI for testing

**Success Criteria:**
- Can ingest policy document and retrieve relevant passages
- Can generate 5 contextual questions
- Can produce basic summary

### Phase 2: Multi-Agent Build (Weeks 3-4)
**Goals:**
- Implement all agent nodes
- Add conditional routing
- Integrate RAG into eligibility checking

**Deliverables:**
1. All 7 agent nodes implemented
2. Eligibility checking with policy citations
3. Risk assessment integration
4. State management with checkpointing

**Success Criteria:**
- Complete workflow from intake to summary
- Eligibility checking cites specific policies
- State persists across interruptions

### Phase 3: HITL & Interface (Week 5)
**Goals:**
- Build human-in-the-loop review interface
- Integrate with sample case management system

**Deliverables:**
1. Web UI for case summary review
2. Inline editing capability
3. Approval workflow

**Success Criteria:**
- Officer can review, edit, approve summary
- Edits update state and resume workflow
- Interface intuitive (test with 2-3 officers)

### Phase 4: Testing & Refinement (Week 6)
**Goals:**
- Test with real case scenarios
- Tune prompts and retrieval
- Gather officer feedback

**Deliverables:**
1. Test on 10 historical cases
2. Officer feedback survey
3. Performance metrics report

**Success Criteria:**
- 80% approval rate without major edits
- Average intake time < 40 minutes
- Officer satisfaction score > 4/5

---

## Success Metrics

### Efficiency Metrics
- **Intake Duration:** Target 30-40 minutes (vs. 60-90 manual)
- **Time to Case Creation:** Target same-day (vs. 1-3 days)
- **Officer Time Saved:** 30-40 minutes per intake

### Quality Metrics
- **Topic Coverage:** 100% of required topics addressed
- **Eligibility Accuracy:** >90% agreement with supervisor review
- **Citation Accuracy:** >95% of citations verified as correct
- **Summary Completeness:** >90% require only minor edits

### User Adoption Metrics
- **Approval Rate:** >80% approved without major edits
- **Officer Satisfaction:** >4/5 on surveys
- **Usage Rate:** >80% of intakes use system (after training)

### System Performance
- **Uptime:** >99% during business hours
- **Response Time:** <30 seconds per agent node
- **Cost per Intake:** <$0.50 in LLM/infrastructure costs

---

## Risks and Mitigation

### Risk 1: LLM Hallucination (Incorrect Eligibility)
**Likelihood:** Medium
**Impact:** High (wrong placement decision)
**Mitigation:**
- Require policy citations for all determinations
- Human review mandatory before final decision
- Regular accuracy audits (supervisor sampling)
- Confidence scores (flag low-confidence for extra review)

### Risk 2: Officer Resistance to AI Tool
**Likelihood:** Medium
**Impact:** High (low adoption)
**Mitigation:**
- Involve officers in design (feedback sessions)
- Emphasize AI as assistant, not replacement
- Highlight time savings
- Training and support during rollout
- Quick wins (start with most tedious tasks)

### Risk 3: Data Privacy Breach
**Likelihood:** Low
**Impact:** Critical
**Mitigation:**
- Encryption at rest and in transit
- Access controls and audit logs
- Regular security audits
- Compliance with JJDPA, state laws
- Data minimization (only necessary info to LLM)

### Risk 4: Policy Updates Not Reflected
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Policy version tracking in metadata
- Effective date filtering (only current policies)
- Notification system for policy changes
- Monthly policy refresh process
- Version control for policy documents

### Risk 5: Cost Overruns
**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- Token usage monitoring per intake
- Budgets and alerts
- Prompt optimization for efficiency
- Model selection (Haiku for simple tasks)
- Caching for repeated queries

---

## Next Steps After POC Success

1. **Scale to More Intake Officers:**
   - Train additional staff
   - Gather broader feedback
   - Refine based on diverse use patterns

2. **Expand Knowledge Base:**
   - Add more policy documents
   - Include state-specific variations
   - Add multi-language support

3. **Integration with Case Management:**
   - API to existing case management system
   - Automated case creation
   - Bi-directional data sync

4. **Advanced Features:**
   - Multi-language questioning (Spanish, etc.)
   - Voice-to-text for hands-free data entry
   - Automated scheduling of next steps
   - Family portal for sharing case summaries

5. **Analytics and Continuous Improvement:**
   - Dashboard for intake metrics
   - Trend analysis (common risk factors, program demand)
   - Ongoing prompt tuning based on officer feedback
   - Model fine-tuning on historical cases

---

## Conclusion

The Intake Triage Assistant POC represents a high-value application of LangChain + LangGraph to juvenile justice case management. By combining RAG over policies with multi-agent orchestration and human-in-the-loop approval, this system can:

- Ensure consistent, comprehensive intakes
- Reduce intake time by 30-50%
- Improve policy compliance through citations
- Support officers with AI assistance while maintaining human oversight
- Create audit trails for quality assurance

The architecture is production-ready with checkpointing, error handling, and security considerations. Success requires strong collaboration with intake officers, iterative refinement based on real usage, and commitment to quality assurance.

**Recommended as first POC** due to:
- Clear, measurable value (time savings)
- Focused scope (intake process)
- High impact (affects all cases)
- Foundation for other use cases (RAG infrastructure, LangGraph patterns)
