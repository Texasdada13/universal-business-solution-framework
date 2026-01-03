# POC #2: Case Brief Summarizer and Updater

**Project Type:** LangGraph Document Processing Pipeline with LangChain RAG
**Complexity:** Medium-High
**Timeline Estimate:** 5-7 weeks for POC
**Primary Value:** Time savings on report generation, role-specific communication, source traceability

---

## Problem Statement

Case managers, probation officers, clinicians, court personnel, and families all need regular updates on youth cases, but each requires different information at different detail levels. Currently:
- Staff manually read through all case notes, transcripts, assessments, and reports
- Summarization is time-consuming (2-4 hours per comprehensive update)
- Different stakeholders get inconsistent information
- Sources are not always clearly cited
- Risk of including inappropriate information for specific audiences
- Updates often delayed due to staff workload

**Goal:** An agent that automatically ingests case documents (notes, transcripts, assessments, program reports), produces role-specific briefs with explicit source links, and ensures appropriate information for each audience with mandatory human sign-off.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│           CASE BRIEF SUMMARIZER & UPDATER                 │
└──────────────────────────────────────────────────────────┘

LAYER 1: Document Ingestion (LangChain)
┌───────────────────────────────────────────────────┐
│ INPUT SOURCES:                                    │
│  • Case notes (text, Word docs)                   │
│  • Hearing transcripts (PDF, text)                │
│  • Psychological/clinical assessments (PDF)       │
│  • School reports (PDF, scanned docs with OCR)    │
│  • Program progress reports (structured data)     │
│  • Drug test results (structured data)            │
│  • Family meeting notes (text)                    │
│                                                   │
│ [Document Loaders] → [Text Extraction]           │
│        ↓                                          │
│ [Text Splitters] → [Parent-Child Chunking]       │
│        ↓                                          │
│ [Embeddings] → [Vector Store by Case]            │
└───────────────────────────────────────────────────┘

LAYER 2: LangGraph Processing Pipeline
┌───────────────────────────────────────────────────┐
│  START: Document Upload                          │
│    ↓                                              │
│  [Ingest Node]                                    │
│    ├─ Identify document type                     │
│    ├─ Extract metadata (date, author, type)      │
│    ├─ Store raw document                         │
│    └─ Validate completeness                      │
│    ↓                                              │
│  [De-Identification Node]                        │
│    ├─ Detect PII (names, addresses, SSN)         │
│    ├─ Identify sensitive info (diagnoses, etc.)  │
│    ├─ Tag by sensitivity level                   │
│    └─ Create redacted versions by audience       │
│    ↓                                              │
│  [Summarize Node]                                 │
│    ├─ Extract key facts and events               │
│    ├─ Identify trends and changes                │
│    ├─ Map-Reduce for large documents             │
│    └─ Maintain source citations                  │
│    ↓                                              │
│  [Risk/Needs Checklist Node]                      │
│    ├─ Extract risk indicators                    │
│    ├─ Identify criminogenic needs                │
│    ├─ Track protective factors                   │
│    ├─ Compare to prior assessments (change)      │
│    └─ Flag items needing attention               │
│    ↓                                              │
│  [Role-Specific Brief Generation] ───┐           │
│    ├─ PO Brief (compliance, behavior)            │
│    ├─ Clinician Brief (mental health, trauma)    │
│    ├─ Court Brief (legal, compliance, progress)  │
│    ├─ Family Brief (age-appropriate, positive)   │
│    └─ Each with appropriate de-identification    │
│    ↓                                              │
│  [Human Approval Node] ◄─── INTERRUPT()          │
│    ├─ Staff reviews all briefs                   │
│    ├─ Verifies accuracy and appropriateness      │
│    ├─ Can edit or reject                         │
│    └─ Signs off before distribution              │
│    ↓                                              │
│  [Distribution Node]                              │
│    ├─ Send to appropriate parties                │
│    ├─ Log who received what when                 │
│    └─ Archive with case file                     │
│    ↓                                              │
│  END → Briefs Distributed                        │
└───────────────────────────────────────────────────┘

LAYER 3: State Management
┌───────────────────────────────────────────────────┐
│ SummarizerState:                                  │
│  - case_id: str                                   │
│  - documents: List[Document]                      │
│  - document_metadata: List[Dict]                  │
│  - extracted_facts: List[Dict]                    │
│  - risk_needs_checklist: Dict                     │
│  - sensitivity_tags: Dict[str → level]            │
│  - briefs: Dict[role → brief_text]                │
│  - human_approved: bool                           │
│  - approval_notes: str                            │
│  - distribution_log: List[Dict]                   │
└───────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Document Ingestion (LangChain)

**Document Loaders by Type:**

```python
# Case notes (Word, text)
from langchain.document_loaders import Docx2txtLoader, TextLoader

# Transcripts and assessments (PDF)
from langchain.document_loaders import PyPDFLoader

# Scanned documents (OCR)
from langchain.document_loaders import UnstructuredPDFLoader  # uses Tesseract

# Structured data (program reports, drug tests)
from langchain.document_loaders import JSONLoader, CSVLoader

# Email attachments
from langchain.document_loaders import UnstructuredEmailLoader
```

**Metadata Extraction:**
- Document type: "case_note", "transcript", "assessment", "school_report", etc.
- Date: Creation or event date
- Author: Who wrote it (PO, clinician, teacher, etc.)
- Sensitivity: "public", "staff_only", "clinical_only"
- Case ID: Link to specific case

**Text Splitting Strategy:**
- **Parent-Child Chunking:**
  - Parent chunks: 1500 tokens (full context for summaries)
  - Child chunks: 300 tokens (precise retrieval for citations)
- **Structure-aware splitting:**
  - Respect section headers in assessments
  - Preserve speaker turns in transcripts
  - Maintain date boundaries in chronological notes

**Vector Store Organization:**
- **Separate collection per case** (enables case-specific retrieval)
- Metadata filtering by:
  - `document_type`: Filter to specific document types
  - `date_range`: Recent vs. historical
  - `author_role`: Filter by who wrote it
  - `sensitivity_level`: Filter by audience appropriateness

### 2. LangGraph Pipeline Nodes

#### A. Ingest Node
**Purpose:** Process uploaded documents and prepare for analysis

**Logic:**
1. Accept document upload (batch or individual)
2. Identify document type:
   - File extension (.pdf, .docx, .txt)
   - Content analysis (LLM classification if ambiguous)
   - User-provided tags
3. Extract metadata:
   - Parse filename for date/author if standardized
   - Extract creation date from file metadata
   - Use LLM to extract: author, date, document type from content
4. Validate:
   - Required metadata present
   - Document readable (not corrupted)
   - Belongs to correct case
5. Store raw document in case file system
6. Update state with document info

**Error Handling:**
- OCR failures → Manual upload or flag for staff review
- Metadata extraction failures → Prompt user for manual entry
- Duplicate detection → Ask user if update or separate document

**No LLM needed for most** - rule-based logic for file handling

#### B. De-Identification Node
**Purpose:** Identify and tag sensitive information for role-based redaction

**Logic:**
1. **PII Detection:**
   - Use NER (Named Entity Recognition) model to detect:
     - Names (youth, family members, victims)
     - Addresses
     - Phone numbers
     - SSNs, case numbers
   - Tag locations in text

2. **Sensitivity Classification:**
   - Use LLM to identify sensitive content:
     - Clinical diagnoses (ADHD, PTSD, etc.)
     - Substance use details
     - Family abuse/trauma details
     - Victim information
   - Classify by audience:
     - `public`: General progress info
     - `staff`: Compliance, behavior details
     - `clinical`: Mental health, diagnoses
     - `court_only`: Legal details, victim info

3. **Create Redacted Versions:**
   - Family brief: Replace clinical terms with plain language, redact diagnoses
   - Court brief: Full detail but professional terminology
   - PO brief: Behavioral focus, compliance details
   - Clinical brief: Full mental health details

**LLM Prompt for Sensitivity Tagging:**
```
Analyze the following case document excerpt and identify sensitive information.

Document Excerpt:
{text}

For each sentence or phrase, classify:
1. Sensitivity Level: PUBLIC / STAFF / CLINICAL / COURT_ONLY
2. Reason: Why this classification?
3. Redaction for Family Brief: How should this be rephrased or redacted for families?

Output JSON format for each item.
```

**Example Output:**
```json
{
  "text": "Youth diagnosed with ADHD and PTSD from prior trauma",
  "sensitivity": "CLINICAL",
  "reason": "Contains clinical diagnoses",
  "family_version": "Youth is receiving support for attention and stress management"
}
```

**Tools:**
- spaCy or Hugging Face NER model for PII
- LLM for context-aware sensitivity classification
- Redaction templates by audience

#### C. Summarize Node
**Purpose:** Create comprehensive summary of case developments

**Logic:**
1. **For Single Document:**
   - If < 4000 tokens: Use Stuff method (single prompt)
   - If > 4000 tokens: Use Map-Reduce method

2. **For Multiple Documents (Case Update):**
   - Map-Reduce across all documents:
     - Map: Summarize each document individually
     - Reduce: Combine into chronological narrative
   - Identify themes across documents:
     - Progress areas (school, behavior, family)
     - Setbacks or challenges
     - Emerging needs

3. **Maintain Source Citations:**
   - Every fact linked to source document
   - Format: "Youth completed 10 community service hours (Progress Report, 3/15/24)"

4. **Extract Key Facts:**
   - Events: Court hearings, program completions, incidents
   - Changes: Improved grades, new placement, medication change
   - Current status: Compliance level, program enrollment, living situation

**LLM Prompt for Summarization:**
```
Summarize the following case document, focusing on:
1. Key developments and events
2. Progress or setbacks
3. Current status
4. Any concerns or needs identified

Document:
{document_text}

For EVERY fact, include the source citation in parentheses: (document_name, date)

Use clear, professional juvenile justice terminology. Be objective and factual.
```

**Map-Reduce for Large Case Files:**
- **Map Phase:** Summarize each document (10-20 documents → 10-20 summaries)
- **Reduce Phase:** Combine summaries chronologically, remove redundancy
- **Refinement:** Optional iterative refinement for coherence

#### D. Risk/Needs Checklist Node
**Purpose:** Extract and track risk factors and criminogenic needs

**Logic:**
1. **Extract Risk Indicators:**
   - Retrieve relevant risk assessment framework (YLS/CMI, SAVRY, etc.)
   - Use LLM to identify mentions of risk factors in documents:
     - Prior criminal history
     - Substance use
     - Peer associations
     - Family circumstances
     - Education/employment
     - Attitudes/orientation

2. **Track Changes Over Time:**
   - Compare to prior assessments (retrieve from vector store)
   - Identify: improved, worsened, unchanged
   - Flag significant changes

3. **Identify Criminogenic Needs:**
   - What needs intervention? (substance abuse treatment, tutoring, etc.)
   - What's being addressed currently?
   - What gaps remain?

4. **Protective Factors:**
   - Family support
   - School engagement
   - Positive activities
   - Mentor relationships

**LLM Prompt for Risk Factor Extraction:**
```
Based on the following case documents, identify risk factors and criminogenic needs
using the YLS/CMI framework.

Documents:
{summarized_docs}

For each of the following domains, extract:
1. Risk factors present (with evidence from docs)
2. Current interventions addressing this
3. Change since last assessment (improved/worsened/stable)
4. Source citation for each point

Domains:
- Prior and Current Offenses
- Family Circumstances/Parenting
- Education/Employment
- Peer Relations
- Substance Abuse
- Leisure/Recreation
- Personality/Behavior
- Attitudes/Orientation

Output structured JSON.
```

**Output Format:**
```json
{
  "domain": "Substance Abuse",
  "risk_factors": [
    {
      "factor": "Occasional marijuana use",
      "evidence": "Positive drug test on 3/10/24",
      "source": "Drug Test Results, 3/10/24"
    }
  ],
  "interventions": ["Weekly substance abuse counseling"],
  "change": "IMPROVED",
  "change_evidence": "No positive tests in past 60 days vs. 2 positives in prior 60 days",
  "needs": ["Continue counseling, peer support group"]
}
```

#### E. Role-Specific Brief Generation Node
**Purpose:** Create tailored briefs for each stakeholder

**Role Definitions:**

1. **Probation Officer (PO) Brief:**
   - Focus: Compliance, behavior, program participation
   - Include: Court dates, conditions, violations, progress toward goals
   - Tone: Professional, objective
   - Sensitivity: Staff-level (no deep clinical details)

2. **Clinician Brief:**
   - Focus: Mental health, trauma, therapeutic progress
   - Include: Diagnoses, treatment, medication, behavioral health
   - Tone: Clinical, DSM-aligned
   - Sensitivity: Clinical-level (full mental health details)

3. **Court Brief:**
   - Focus: Legal compliance, case progress, recommendations
   - Include: Charge details, compliance, risk, program completion, next steps
   - Tone: Formal, legal
   - Sensitivity: Court-level (victim info OK, family details limited)

4. **Family Brief:**
   - Focus: Positive progress, next steps, how family can help
   - Include: Achievements, current programs, upcoming events
   - Tone: Supportive, encouraging, plain language
   - Sensitivity: Public-level (no diagnoses, no legal jargon, no victim details)
   - Reading level: 6th-8th grade

**LLM Prompt for Role-Specific Brief (Example: Family):**
```
Create a case update brief for the youth's family.

Case Summary:
{summary}

Risk/Needs Info:
{risk_needs}

Guidelines:
- Use plain, supportive language (6th-8th grade reading level)
- Focus on positive progress and achievements
- Explain next steps in simple terms
- NO clinical diagnoses or jargon
- NO victim details or legal specifics
- Encourage family involvement
- Cite sources for key facts

Structure:
1. Recent Progress (what's going well)
2. Current Activities (school, programs, counseling)
3. Areas for Growth (challenges, with positive framing)
4. Next Steps (upcoming court dates, program milestones)
5. How Families Can Help
```

**Redaction Strategy:**
- Use de-identification tags from earlier node
- Replace or remove content based on role:
  - Family: "Youth is working on managing emotions" vs. Clinical: "Youth diagnosed with PTSD, receiving EMDR therapy"
  - Family: "Youth completed community service hours" vs. Court: "Youth completed 40 hours community service as part of restitution for theft charges"

**Tools:**
- Readability analyzer (Flesch-Kincaid) for family briefs
- Template library for each role
- Citation formatter

#### F. Human Approval Node
**Purpose:** Mandatory staff review before distribution

**Logic:**
1. Use `interrupt()` to pause workflow
2. Display all generated briefs side-by-side
3. Staff reviews for:
   - Accuracy (facts correct?)
   - Appropriateness (right info for each audience?)
   - Tone (professional? supportive?)
   - Completeness (anything missing?)
   - Safety (any info that could cause harm?)

4. Staff can:
   - **Approve:** Proceed to distribution
   - **Edit:** Make corrections, then re-review
   - **Reject:** Specify issues, regenerate
   - **Add notes:** Context for recipients

5. Log approval decision and timestamp

**UI/UX:**
- Tabbed interface (one tab per role)
- Inline editing for each brief
- Highlight areas of concern (low-confidence extractions)
- Source verification: Click citation → see source document excerpt
- Comparison view: Show what changed since last update

**Approval Requirements:**
- Legal: Staff attorney reviews court briefs (if available)
- Clinical: Clinician reviews clinical briefs
- PO: Reviews PO and family briefs
- Supervisor: Sign-off for first 10 cases (training), then sampling

#### G. Distribution Node
**Purpose:** Deliver briefs to appropriate recipients

**Logic:**
1. Route based on role:
   - PO brief → Case management system, PO email
   - Clinician brief → EHR system, clinician email
   - Court brief → Case file, court portal, attorney email
   - Family brief → Family portal, parent email, printed copy

2. Log distribution:
   - Who received what
   - When sent
   - Delivery confirmation (email read receipt, portal viewed)

3. Archive:
   - Store in case file
   - Link to source documents
   - Version control (track updates over time)

**Integration Points:**
- Email API (SendGrid, AWS SES)
- Case management system API
- Court portal API (jurisdiction-specific)
- Family portal (custom or existing)

**No LLM needed** - straightforward routing logic

### 3. State Management

**StateGraph Schema:**
```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class SummarizerState(TypedDict):
    # Case identification
    case_id: str
    timestamp: datetime
    requesting_staff: str

    # Documents
    documents: List[Dict]  # [{path, type, date, author, text}]
    document_metadata: List[Dict]

    # Processing results
    extracted_facts: List[Dict]  # key facts with citations
    sensitivity_tags: Dict[str, str]  # text → sensitivity level
    risk_needs_checklist: Dict[str, Any]

    # Outputs
    briefs: Dict[str, str]  # role → brief_text
    # e.g., {"PO": "...", "Clinician": "...", "Court": "...", "Family": "..."}

    # Approval workflow
    human_approved: bool
    approval_notes: str
    edited_briefs: Dict[str, str]  # if staff edited

    # Distribution
    distribution_log: List[Dict]  # [{role, recipient, timestamp, delivered}]

    # Workflow control
    current_node: str
```

**Checkpointing:**
- PostgresSaver for production
- Checkpoint after each major node (ingest, summarize, generate briefs)
- Enables:
  - Resume if staff interrupted during review
  - Supervisor can review process flow
  - Time-travel to see what AI originally generated vs. staff edits

### 4. Conditional Routing

**Routing Logic:**

1. **After De-Identification:**
   - If high-sensitivity document (court-ordered psych eval) → Flag for supervisor review
   - Otherwise → Proceed to summarization

2. **After Risk/Needs Checklist:**
   - If significant negative change (e.g., risk increased) → Alert supervisor
   - If positive change → Include in family brief prominently

3. **After Human Approval:**
   - If approved → Distribution
   - If edited → Log changes, then distribution
   - If rejected → Loop back to brief generation with staff notes

**Implementation:**
```python
def route_after_approval(state):
    if state["human_approved"]:
        return "distribution"
    else:
        # Rejected - regenerate with staff feedback
        return "role_specific_brief_generation"

graph.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "distribution": "distribution_node",
        "role_specific_brief_generation": "role_specific_brief_generation"
    }
)
```

---

## Data Flow

### Input
1. **Documents:**
   - Case notes from POs
   - Hearing transcripts
   - Psychological/clinical assessments
   - School progress reports
   - Program completion reports
   - Drug test results
   - Family meeting notes

2. **Case Context:**
   - Prior briefs (for comparison)
   - Risk assessment history
   - Case goals and conditions

### Processing
1. **Ingestion:** Load documents, extract metadata
2. **De-Identification:** Tag sensitive info, create role-appropriate versions
3. **Summarization:** Extract key facts, create chronological narrative
4. **Risk/Needs Analysis:** Identify risk factors, track changes
5. **Brief Generation:** Create 4 role-specific briefs
6. **Review:** Staff approves or edits
7. **Distribution:** Send to appropriate stakeholders

### Output
1. **Four Role-Specific Briefs:**
   - PO Brief (compliance focus)
   - Clinician Brief (mental health focus)
   - Court Brief (legal focus)
   - Family Brief (plain language, positive focus)

2. **Metadata:**
   - Source citations for all facts
   - Distribution log (who received what when)
   - Approval audit trail

3. **Archive:**
   - All briefs stored in case file
   - Linked to source documents
   - Versioned (track updates over time)

---

## Technical Implementation Considerations

### 1. LLM Selection
- **Primary:** Claude Sonnet 4.5
  - Excellent summarization quality
  - Strong citation generation
  - Good at role-specific tone adaptation
- **Cost Optimization:** Claude Haiku for simple extractions (metadata, PII detection)
- **Fallback:** GPT-4 for diversity

### 2. Prompt Engineering

**Summarization Prompts:**
- Chain-of-thought: "First identify key events, then organize chronologically, then..."
- Explicit citation instruction: "For EVERY fact, cite the source: (doc_name, date)"
- Temperature: 0.3 (factual, consistent)

**Role-Specific Prompts:**
- Few-shot examples for each role (show desired format and tone)
- Readability constraint for family briefs: "Use 6th-8th grade language"
- Temperature: 0.4 (some variety in phrasing, but consistent facts)

**De-Identification Prompts:**
- Structured output (JSON) for sensitivity tagging
- Examples of clinical vs. plain language
- Temperature: 0.2 (consistency critical)

### 3. RAG Optimization

**Chunking:**
- Parent-Child strategy:
  - Child chunks: 300 tokens (for precise citation retrieval)
  - Parent chunks: 1500 tokens (for summary context)
- Structure-aware: Respect document sections

**Metadata Tagging:**
```python
metadata = {
    "case_id": "12345",
    "doc_type": "clinical_assessment",
    "author": "Dr. Smith",
    "date": "2024-03-15",
    "sensitivity": "clinical",
    "document_id": "uuid-..."
}
```

**Retrieval Strategy:**
- Case-specific collection: Only retrieve from this case's documents
- Date filtering: Recent (last 90 days) vs. historical
- Document type filtering: For risk/needs, prioritize assessments and progress reports

### 4. De-Identification Techniques

**PII Detection:**
- **NER Models:**
  - spaCy `en_core_web_trf` (transformer-based, accurate)
  - Or Hugging Face `dslim/bert-base-NER` (privacy-preserving, local)
- **Regex Patterns:**
  - SSN: `\d{3}-\d{2}-\d{4}`
  - Phone: `\(\d{3}\) \d{3}-\d{4}`
  - Case numbers: jurisdiction-specific patterns

**Sensitivity Classification:**
- LLM-based: Use Claude to classify by context
- Rule-based fallback: Keywords (diagnoses → clinical, "victim" → court_only)

**Redaction Strategies:**
- **Replacement:** "John Doe" → "[Youth]", "123 Main St" → "[Address]"
- **Generalization:** "ADHD diagnosis" → "attention difficulties"
- **Omission:** Remove entirely for family briefs

### 5. Human-in-the-Loop Interface

**UI Components:**
1. **Document Upload:**
   - Drag-and-drop
   - Batch upload support
   - Metadata entry form (if not auto-extracted)

2. **Brief Review:**
   - Tabbed interface (PO / Clinician / Court / Family)
   - Side-by-side: Original docs + Generated brief
   - Inline editing (rich text editor)
   - Citation hover: Mouseover citation → see source excerpt

3. **Approval Controls:**
   - Approve button (with confirmation)
   - Edit mode (unlock for changes)
   - Reject with reason (dropdown + free text)
   - Add notes for recipients

4. **Distribution Settings:**
   - Select recipients (checkboxes)
   - Delivery method (email, portal, print)
   - Schedule send time (immediate or scheduled)

**Technology:**
- React frontend
- FastAPI backend
- WebSocket for real-time updates (processing status)

### 6. Map-Reduce Implementation

**For Large Case Updates (Many Documents):**

**Map Phase:**
```python
# For each document
document_summaries = []
for doc in state["documents"]:
    summary = llm.invoke(
        f"Summarize this document: {doc.text}\nCite key facts with source."
    )
    document_summaries.append(summary)
```

**Reduce Phase:**
```python
# Combine summaries
combined_summary = llm.invoke(
    f"""Combine these document summaries into a chronological case narrative.
    Remove redundancy. Maintain all citations.

    Summaries:
    {document_summaries}
    """
)
```

**Refinement (Optional):**
```python
# Iterative refinement for coherence
refined_summary = llm.invoke(
    f"""Review this case summary and improve clarity and flow.
    Keep all facts and citations unchanged.

    Summary:
    {combined_summary}
    """
)
```

### 7. Citation Management

**Citation Format:**
- Inline: "Youth completed 10 community service hours (Progress Report, 3/15/24, p. 2)"
- Footnotes: "[1] Progress Report, March 15, 2024, page 2"

**Source Linking:**
- Store document IDs with citations
- Enable click-through from brief to source document
- Highlight relevant passage in source

**Verification:**
- Staff can verify citations during review
- Flag if citation seems incorrect (low confidence)
- Provide source document excerpt on hover

### 8. Quality Assurance

**Automated Checks:**
1. **Citation Coverage:**
   - Every major fact has a citation
   - No unsupported claims

2. **Readability (Family Briefs):**
   - Flesch-Kincaid grade level ≤ 8
   - No clinical jargon
   - No legal terms without explanation

3. **Completeness:**
   - All required sections present
   - Key facts from documents included

4. **Consistency:**
   - Facts consistent across briefs (no contradictions)
   - Dates and numbers accurate

**Human Review Metrics:**
- Approval rate (% approved without edits)
- Edit frequency (which sections edited most)
- Rejection reasons (categorize to improve prompts)

### 9. Security and Privacy

**Document Security:**
- Encrypted storage (at rest and in transit)
- Access control: Only case-assigned staff can access
- Audit logs: Who accessed what when

**De-Identification Validation:**
- Secondary check: Run briefs through PII detector before distribution
- Alert if PII detected in family brief (should be redacted)

**Data Retention:**
- Case documents: Retain per state law (often 5+ years)
- Briefs: Archive with case file
- Processing logs: 90 days

**Compliance:**
- JJDPA confidentiality
- HIPAA (for clinical info)
- FERPA (for school records)
- State juvenile record laws

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals:**
- Document ingestion pipeline
- Basic summarization (single document)

**Deliverables:**
1. Document loaders for PDF, Word, text
2. Text splitter with Parent-Child chunking
3. Basic summarization with citations
4. Simple CLI for testing

**Success Criteria:**
- Can ingest 10-page PDF and generate summary
- Citations link to source pages
- Summary captures key facts

### Phase 2: De-Identification (Week 3)
**Goals:**
- PII detection
- Sensitivity classification
- Role-based redaction

**Deliverables:**
1. NER model for PII detection
2. LLM-based sensitivity tagger
3. Redaction logic by role

**Success Criteria:**
- Detects 95%+ of PII (test on sample docs)
- Correctly classifies clinical vs. staff vs. public info
- Family brief has no diagnoses or legal jargon

### Phase 3: Multi-Agent Pipeline (Weeks 4-5)
**Goals:**
- Full LangGraph pipeline
- Risk/Needs checklist extraction
- Role-specific brief generation

**Deliverables:**
1. All 7 nodes implemented
2. State management with checkpointing
3. Four role-specific briefs generated

**Success Criteria:**
- Process 20 documents, generate 4 briefs
- Each brief has appropriate content and tone
- Risk/Needs checklist accurate (manual verification)

### Phase 4: HITL & Interface (Week 6)
**Goals:**
- Build review interface
- Implement approval workflow

**Deliverables:**
1. Web UI for brief review
2. Inline editing
3. Approval/rejection logic

**Success Criteria:**
- Staff can review, edit, approve briefs
- Edits save and resume workflow
- UI intuitive (test with 2-3 staff)

### Phase 5: Testing & Refinement (Week 7)
**Goals:**
- Test with real cases
- Gather staff feedback
- Optimize prompts and quality

**Deliverables:**
1. Process 10 historical cases
2. Staff feedback survey
3. Quality metrics report

**Success Criteria:**
- 80% approval rate without major edits
- Family brief readability ≤ 8th grade
- Staff satisfaction > 4/5

---

## Success Metrics

### Efficiency Metrics
- **Time Savings:** Reduce brief generation from 2-4 hours to 30-60 minutes (staff review time only)
- **Frequency:** Enable more frequent updates (monthly → bi-weekly)
- **Staff Time Saved:** 1.5-3 hours per case update

### Quality Metrics
- **Citation Accuracy:** >95% of citations verified as correct
- **Fact Accuracy:** >95% of facts verified by staff
- **Readability (Family):** Flesch-Kincaid grade level ≤ 8
- **Appropriateness:** <5% of briefs require major redaction/editing for audience

### User Adoption
- **Approval Rate:** >80% approved without major edits
- **Staff Satisfaction:** >4/5 on surveys
- **Usage Rate:** >70% of case updates use system

### Stakeholder Impact
- **Family Engagement:** Track if family portal views increase
- **Court Efficiency:** Judges report briefs helpful (qualitative)
- **Communication Quality:** Stakeholders report better understanding (surveys)

---

## Risks and Mitigation

### Risk 1: Inaccurate Summarization
**Likelihood:** Medium
**Impact:** High (misinformation to stakeholders)
**Mitigation:**
- Mandatory human review before distribution
- Citation verification (staff can check sources)
- Quality metrics (track accuracy over time)
- Conservative summarization (err on side of inclusion)

### Risk 2: Inappropriate Information Disclosure
**Likelihood:** Medium
**Impact:** Critical (privacy breach, harm to youth/family)
**Mitigation:**
- De-identification with secondary check (PII detector on final briefs)
- Human review focuses on appropriateness
- Role-based access control
- Supervisor sign-off for first 20 cases

### Risk 3: De-Identification Failures (PII Leakage)
**Likelihood:** Low-Medium
**Impact:** Critical
**Mitigation:**
- Dual-layer detection (NER + LLM)
- Secondary scan before distribution
- Human review specifically checks for PII
- Audit logs and alerts

### Risk 4: Staff Over-Reliance on AI
**Likelihood:** Medium
**Impact:** Medium (miss important nuances)
**Mitigation:**
- Emphasize AI as assistant, not replacement
- Mandatory review and sign-off
- Training on how to critically evaluate briefs
- Highlight low-confidence areas in UI

### Risk 5: Stakeholder Distrust of AI-Generated Content
**Likelihood:** Medium
**Impact:** Medium (low adoption, resistance)
**Mitigation:**
- Transparency: Clearly label AI-assisted
- Human sign-off prominent ("Reviewed and approved by...")
- Pilot with trusted staff champions
- Feedback mechanisms for recipients

---

## Next Steps After POC Success

1. **Integration with Case Management System:**
   - Automated document ingestion from case files
   - One-click brief generation from case page
   - Bi-directional sync

2. **Expanded Stakeholder Briefs:**
   - Defense attorney briefs
   - Prosecutor briefs
   - Victim advocate briefs (with strict redaction)

3. **Automated Scheduling:**
   - Generate briefs automatically before court hearings
   - Scheduled updates (e.g., monthly family updates)

4. **Multi-Language Support:**
   - Translate family briefs to Spanish, Vietnamese, etc.
   - Cultural adaptation (not just translation)

5. **Trend Analysis:**
   - Track risk factor trends across caseload
   - Identify which programs most effective
   - Dashboard for supervisors

6. **Advanced Features:**
   - Voice narration of family briefs (accessibility)
   - Visual progress timelines
   - Comparison views (this update vs. last update)

---

## Conclusion

The Case Brief Summarizer and Updater addresses a critical pain point: time-consuming, inconsistent case documentation and stakeholder communication. By combining LangChain's document processing with LangGraph's structured pipeline and human-in-the-loop approval, this system can:

- Save 1.5-3 hours per case update (significant staff time savings)
- Ensure consistent, appropriate communication to diverse stakeholders
- Maintain source traceability (builds trust, enables verification)
- Support family engagement (plain language, regular updates)
- Reduce documentation burden on staff

The architecture prioritizes safety with de-identification, mandatory human review, and audit trails. Success requires careful prompt engineering, robust PII detection, and strong staff collaboration during development.

**Recommended as second POC** after Intake Triage Assistant, leveraging shared RAG infrastructure while adding document processing and multi-stakeholder complexity.
