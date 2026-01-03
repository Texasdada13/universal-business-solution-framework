# POC #4: Program Matching and Referral Recommender

**Project Type:** LangGraph Multi-Agent Decision System with LangChain RAG
**Complexity:** High
**Timeline Estimate:** 6-8 weeks for POC
**Primary Value:** Optimal program placement, reduced placement errors, comprehensive eligibility checking

---

## Problem Statement

Matching youth to appropriate programs and services requires considering complex factors: assessed needs (educational, substance use, mental health, family), program eligibility rules, benefit interactions (Medicaid, SNAP), transportation access, scheduling constraints, and safety considerations (conflict checks). Currently:
- Case managers manually review program directories and eligibility criteria
- Time-consuming (1-2 hours per case)
- May miss optimal matches due to incomplete knowledge of all programs
- Eligibility checking prone to errors (complex rules)
- Benefit interactions often overlooked
- Conflict checks (victim proximity, gang affiliations) sometimes missed
- Transportation and scheduling barriers discovered late

**Goal:** An agent that, given a youth's needs and context, suggests appropriate programs with transportation options and scheduling, validates eligibility including benefits interactions, performs conflict checks, and provides override/justification workflow for exceptions.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│      PROGRAM MATCHING & REFERRAL RECOMMENDER              │
└──────────────────────────────────────────────────────────┘

LAYER 1: Knowledge Base (LangChain RAG)
┌───────────────────────────────────────────────────────┐
│ CURATED DIRECTORY:                                    │
│  • Program descriptions (services, approach, hours)   │
│  • Eligibility criteria (age, offense, needs, etc.)   │
│  • Location and transportation routes                 │
│  • Schedule and capacity information                  │
│  • Outcome data (completion rates, recidivism)        │
│  • Benefit interaction rules (Medicaid, SNAP, etc.)   │
│  • Conflict check rules and databases                 │
│  • Cost and funding source information                │
│                                                       │
│ [Document Loaders] → [Structure-Aware Splitting]     │
│        ↓                                              │
│ [Embeddings] → [Vector Store: Programs]              │
│ [Metadata: program_type, location, eligibility]      │
│                                                       │
│ STRUCTURED DATA:                                      │
│  • Program capacity (real-time or updated weekly)     │
│  • Transportation schedules (bus routes, etc.)        │
│  • Youth addresses and gang affiliations (secure)     │
│  • Victim addresses (secure, for conflict checks)     │
└───────────────────────────────────────────────────────┘

LAYER 2: LangGraph Multi-Agent Workflow
┌───────────────────────────────────────────────────────┐
│  START: Youth Profile Input                          │
│    ↓                                                  │
│  [Needs Extraction Agent]                             │
│    ├─ Extract needs from assessment/case notes       │
│    ├─ Categorize: school, substance, mental health,  │
│    │   family, employment, recreation                │
│    ├─ Prioritize based on risk/needs assessment      │
│    └─ Store in state                                 │
│    ↓                                                  │
│  [Program Retrieval Agent]                            │
│    ├─ Query program directory by needs               │
│    ├─ Filter by basic criteria (age, location)       │
│    ├─ Rank by relevance and outcomes                 │
│    └─ Retrieve top 10-15 candidate programs          │
│    ↓                                                  │
│  [Eligibility Checking Agent]                         │
│    ├─ For each program:                              │
│    │   • Match youth profile to criteria             │
│    │   • Check offense restrictions                  │
│    │   • Verify age and residency                    │
│    │   • Cite specific eligibility rules             │
│    └─ Filter to eligible programs                    │
│    ↓                                                  │
│  [Benefits Interaction Agent]                         │
│    ├─ Check Medicaid coverage for clinical programs  │
│    ├─ Verify SNAP compatibility (work requirements)  │
│    ├─ Identify funding conflicts or gaps             │
│    └─ Flag programs that might affect benefits       │
│    ↓                                                  │
│  [Conflict Check Agent]                               │
│    ├─ Check victim proximity (if applicable)         │
│    ├─ Check gang affiliation conflicts               │
│    ├─ Check co-participant restrictions              │
│    │   (e.g., no placement with known associates)    │
│    ├─ Check prior abuse context                      │
│    └─ Flag conflicts with reasoning                  │
│    ↓                                                  │
│  [Transportation & Scheduling Agent]                  │
│    ├─ Calculate distance from youth's home           │
│    ├─ Check public transit availability              │
│    ├─ Verify program schedule fits youth/family      │
│    ├─ Estimate travel time and cost                  │
│    └─ Flag transportation barriers                   │
│    ↓                                                  │
│  [Recommendation Synthesis Agent]                     │
│    ├─ Rank programs by:                              │
│    │   • Need alignment (how well addresses needs)   │
│    │   • Accessibility (transportation, schedule)    │
│    │   • Outcomes (success rate for similar youth)   │
│    │   • Capacity (openings available)               │
│    ├─ Generate top 3-5 recommendations               │
│    ├─ Explain reasoning for each                     │
│    ├─ Cite eligibility, outcomes, logistics          │
│    └─ Flag any concerns or barriers                  │
│    ↓                                                  │
│  [Override/Justification Node] ◄─── INTERRUPT()      │
│    ├─ If conflicts or barriers detected:             │
│    │   • Present to case manager                     │
│    │   • Request override justification if needed    │
│    │   • Document rationale for placement            │
│    └─ Supervisor approval for overrides              │
│    ↓                                                  │
│  [Referral Generation Node]                           │
│    ├─ Create referral documents                      │
│    ├─ Include youth info, needs, recommendations     │
│    ├─ Send to programs (API or email)                │
│    └─ Schedule intake appointments                   │
│    ↓                                                  │
│  END → Referrals Sent                                │
└───────────────────────────────────────────────────────┘

LAYER 3: State Management
┌───────────────────────────────────────────────────────┐
│ MatchingState:                                        │
│  - youth_profile: Dict (demographics, needs, risks)   │
│  - needs: List[Dict] (categorized, prioritized)       │
│  - candidate_programs: List[Program]                  │
│  - eligibility_results: Dict[program → eligible?]     │
│  - benefit_flags: List[Dict] (interactions)           │
│  - conflicts: List[Dict] (safety issues)              │
│  - transportation_analysis: Dict[program → logistics] │
│  - recommendations: List[Program] (ranked)            │
│  - override_requests: List[Dict]                      │
│  - approved: bool                                     │
└───────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Knowledge Base (LangChain RAG)

**Program Directory Content:**

1. **Program Descriptions:**
   - Program name and provider
   - Services offered (counseling, tutoring, job training, recreation, etc.)
   - Approach/model (CBT, restorative justice, mentoring, etc.)
   - Location(s) and hours of operation
   - Capacity (current openings)
   - Contact information

2. **Eligibility Criteria:**
   - Age range
   - Gender (if gender-specific program)
   - Offense restrictions (eligible offenses, excluded offenses)
   - Risk level (low, moderate, high)
   - Specific needs addressed (substance abuse, mental health, education, etc.)
   - Residency requirements
   - Other prerequisites (family consent, school enrollment, etc.)

3. **Logistics:**
   - Address and neighborhood
   - Public transportation access (bus routes, stops)
   - Parking availability
   - Schedule (days, times, duration)
   - Remote/virtual options

4. **Outcomes and Quality:**
   - Completion rate
   - Recidivism rate for graduates
   - Client satisfaction scores
   - Accreditation/certification status

5. **Financial:**
   - Cost (free, sliding scale, insurance)
   - Funding sources (county, grant, Medicaid billable)
   - Benefits accepted (Medicaid, private insurance, SNAP for work programs)

**Structured Data (Databases):**

1. **Capacity Database:**
   - Real-time or weekly updated
   - Program → Current openings, waitlist length

2. **Transportation Database:**
   - Bus routes, schedules
   - Youth addresses (for distance calculation)

3. **Conflict Check Databases (Secure, Restricted Access):**
   - Victim addresses (if applicable)
   - Gang affiliation data (law enforcement shared)
   - Co-defendant/associate lists

4. **Benefits Database:**
   - Youth's current benefits (Medicaid, SNAP, etc.)
   - Benefit interaction rules

**Document Processing:**
```python
Document Loaders:
  - PDFLoader for program brochures
  - JSONLoader for structured program data
  - WebBaseLoader for program websites
  - CSVLoader for program directory spreadsheets

Text Splitting:
  - Structure-aware splitting (keep program entries together)
  - chunk_size: 1000 tokens (full program description)
  - chunk_overlap: 150 tokens

Metadata Tagging:
  - program_name: "Youth Mentoring Program"
  - program_type: "mentoring", "substance_abuse", "education", etc.
  - location: "Downtown", "Westside" (geographic tags)
  - age_range: "12-17"
  - needs_addressed: ["substance_abuse", "peer_relations"]
  - eligibility_level: "low_risk", "moderate_risk", "all"
```

**Embeddings:**
- OpenAI text-embedding-3-small (semantic matching of needs to programs)

**Vector Store:**
- Development: ChromaDB
- Production: Pinecone or Qdrant
- Metadata filtering for efficient retrieval

### 2. LangGraph Agent Nodes

#### A. Needs Extraction Agent
**Purpose:** Identify and prioritize youth's needs from assessment data

**Logic:**
1. Accept input: Youth profile (demographics, risk/needs assessment, case notes)
2. Extract needs across domains:
   - **Education:** School issues, learning disabilities, truancy
   - **Substance Use:** Drug/alcohol use, family substance abuse
   - **Mental Health:** Diagnoses, trauma, behavioral issues
   - **Family:** Family conflict, parenting issues, stability
   - **Peer Relations:** Negative peers, gang involvement, prosocial peers
   - **Employment:** Job readiness, work experience
   - **Recreation:** Constructive use of time, hobbies

3. Categorize by priority:
   - **Critical:** Immediate safety (substance abuse, mental health crisis)
   - **High:** Criminogenic needs (factors linked to offending)
   - **Moderate:** Protective factors to strengthen
   - **Low:** Nice-to-have enhancements

4. Output structured needs list

**LLM Prompt:**
```
Based on the following youth assessment, extract and categorize their needs.

Youth Profile:
{youth_profile}

Risk/Needs Assessment:
{risk_needs_data}

For each need identified, provide:
1. Domain (education, substance_abuse, mental_health, family, peers, employment, recreation)
2. Specific need (e.g., "anger management", "tutoring in math", "substance abuse treatment")
3. Priority (critical, high, moderate, low)
4. Evidence (quote from assessment)

Output as JSON array.
```

**Output Format:**
```json
[
  {
    "domain": "substance_abuse",
    "need": "Outpatient substance abuse treatment for marijuana use",
    "priority": "high",
    "evidence": "Youth reported weekly marijuana use; positive drug test 3/10/24"
  },
  {
    "domain": "education",
    "need": "Tutoring in math and reading (2 grade levels behind)",
    "priority": "high",
    "evidence": "School report indicates 2-grade-level deficit in math and reading"
  }
]
```

#### B. Program Retrieval Agent
**Purpose:** Find candidate programs that address identified needs

**Logic:**
1. For each high/critical need, query vector store:
   - Semantic search: "Substance abuse treatment for adolescents"
   - Metadata filter: age_range matches, needs_addressed contains "substance_abuse"

2. Combine results across needs (union of relevant programs)

3. Rank by:
   - Relevance score (vector similarity)
   - Number of needs addressed (multi-service programs ranked higher)
   - Outcome data (completion rate, recidivism)

4. Retrieve top 10-15 candidate programs

**Retrieval Strategy:**
- **MultiQueryRetriever:** Generate variations of need query
  - "Substance abuse treatment" → "Drug counseling", "Addiction services for teens", "Chemical dependency program"
- **Hybrid Search:** 70% semantic, 30% keyword (for specific program types)

**Tools:**
- Vector store retriever with metadata filtering
- Reranker for final ranking

#### C. Eligibility Checking Agent
**Purpose:** Validate youth meets each program's eligibility criteria

**Logic:**
1. For each candidate program:
   - Retrieve full eligibility criteria from knowledge base
   - Match youth profile against each criterion:
     - Age: Youth 15, program 12-17 → Eligible
     - Offense: Youth theft, program excludes violent → Eligible
     - Risk level: Youth moderate, program accepts all → Eligible
     - Residency: Youth in-county, program county-only → Eligible
     - Prerequisites: Youth in school, program requires enrollment → Eligible

2. Determine: ELIGIBLE, INELIGIBLE, or CONDITIONAL (needs waiver)

3. Cite specific eligibility rules for determination

**LLM Prompt:**
```
Determine if the youth is eligible for this program based on the eligibility criteria.

Youth Profile:
- Age: {age}
- Gender: {gender}
- Offense: {offense}
- Risk Level: {risk_level}
- Residency: {county}
- School Enrollment: {school_status}
- Other: {other_relevant_info}

Program Eligibility Criteria:
{program_eligibility_text}

For each criterion, provide:
1. Criterion name
2. Youth's status relative to criterion
3. Match (yes/no)
4. Citation (section of eligibility rules)

Final Determination: ELIGIBLE / INELIGIBLE / CONDITIONAL (if waiver possible)

If INELIGIBLE, explain the specific barrier(s).
If CONDITIONAL, explain what waiver or exception would be needed.
```

**Output Format:**
```json
{
  "program": "Youth Mentoring Program",
  "determination": "ELIGIBLE",
  "criteria_checked": [
    {
      "criterion": "Age 12-17",
      "youth_status": "15",
      "match": true,
      "citation": "Eligibility Guidelines, Section 2.1"
    },
    {
      "criterion": "Non-violent offense",
      "youth_status": "Theft (non-violent)",
      "match": true,
      "citation": "Eligibility Guidelines, Section 2.3"
    }
  ],
  "barriers": [],
  "waiver_needed": false
}
```

#### D. Benefits Interaction Agent
**Purpose:** Check for interactions with public benefits (Medicaid, SNAP, etc.)

**Logic:**
1. Retrieve youth's current benefits from case data:
   - Medicaid (health coverage)
   - SNAP (food assistance)
   - SSI (disability income)
   - TANF (cash assistance)

2. For each eligible program:
   - **Medicaid:** Is program Medicaid-billable? (If so, no out-of-pocket cost)
   - **SNAP:** Does program have work requirements that might affect SNAP?
   - **SSI/TANF:** Does program income conflict with benefit limits?

3. Check funding conflicts:
   - Can program bill Medicaid AND receive county funding? (Some rules prohibit double-billing)

4. Flag programs that might affect benefits (positive or negative)

**Rules Database (Example):**
```json
{
  "medicaid_billable_services": ["substance_abuse_treatment", "mental_health_counseling"],
  "snap_work_requirements": {
    "job_training_programs": "May count as work activity (positive)",
    "full_time_school": "Exempt from work requirements if under 18"
  },
  "funding_conflicts": {
    "medicaid_billed": "Cannot also receive county mental health funding for same service"
  }
}
```

**LLM Prompt (if complex rules):**
```
Based on the youth's current benefits and the program details, identify any benefit interactions.

Youth Benefits:
- Medicaid: {medicaid_status}
- SNAP: {snap_status}
- SSI: {ssi_status}

Program:
- Name: {program_name}
- Services: {services}
- Funding: {funding_sources}
- Cost: {cost_info}

Identify:
1. Positive interactions (e.g., Medicaid covers program cost)
2. Negative interactions (e.g., program income might affect SSI)
3. Funding conflicts (e.g., can't bill Medicaid AND receive county funds)

Cite relevant benefit rules.
```

**Output:**
```json
{
  "program": "Outpatient Substance Abuse Treatment",
  "interactions": [
    {
      "benefit": "Medicaid",
      "interaction_type": "positive",
      "detail": "Program is Medicaid-billable; no out-of-pocket cost for family",
      "citation": "Medicaid Provider Directory, Code 90832"
    }
  ],
  "funding_conflicts": [],
  "recommendation": "Recommend this program; Medicaid coverage eliminates cost barrier"
}
```

#### E. Conflict Check Agent
**Purpose:** Identify safety conflicts (victim proximity, gang affiliations, etc.)

**Logic:**

1. **Victim Proximity Check** (if victim in case):
   - Retrieve victim's address (if available and appropriate to check)
   - Calculate distance from program location to victim address
   - If < 1 mile (configurable threshold): Flag as conflict
   - Reasoning: "Program within 1 mile of victim's residence; may cause distress or safety concern"

2. **Gang Affiliation Check:**
   - Retrieve youth's gang affiliation data (if available)
   - Check program neighborhood for rival gang territory
   - Check other program participants for rival affiliations (if group program)
   - If conflict: Flag with reasoning

3. **Co-Participant Restrictions:**
   - Check if youth has co-defendants or negative associates
   - Check if those individuals are in program (if group-based)
   - If yes: Flag conflict

4. **Prior Abuse Context:**
   - If youth has history of abuse in specific context (e.g., sports team)
   - Check if program involves similar context
   - If yes: Flag for careful consideration

**Data Sources:**
- Victim database (secure, case-specific)
- Gang intelligence database (law enforcement shared, secure)
- Case management system (co-defendants, associates)

**Implementation:**
```python
def conflict_check_node(state):
    conflicts = []

    for program in state["eligible_programs"]:
        # Victim proximity
        if state["case_has_victim"] and state["victim_address"]:
            distance = calculate_distance(program.address, state["victim_address"])
            if distance < VICTIM_PROXIMITY_THRESHOLD:
                conflicts.append({
                    "program": program.name,
                    "conflict_type": "victim_proximity",
                    "detail": f"Program located {distance} miles from victim's residence",
                    "severity": "high",
                    "recommendation": "Consider alternative program or consult with victim advocate"
                })

        # Gang affiliation
        if state["youth_gang_affiliation"]:
            if program.neighborhood in get_rival_territory(state["youth_gang_affiliation"]):
                conflicts.append({
                    "program": program.name,
                    "conflict_type": "gang_territory",
                    "detail": f"Program in rival gang territory",
                    "severity": "high",
                    "recommendation": "Safety risk; select alternative location"
                })

        # Other checks...

    state["conflicts"] = conflicts
    return state
```

**Output:**
```json
{
  "program": "Westside Community Center",
  "conflicts": [
    {
      "conflict_type": "victim_proximity",
      "detail": "Program located 0.5 miles from victim's residence",
      "severity": "high",
      "recommendation": "Consider alternative program or consult with victim advocate before placement",
      "override_possible": true,
      "override_requires": "Supervisor approval + victim advocate consultation"
    }
  ]
}
```

#### F. Transportation & Scheduling Agent
**Purpose:** Assess transportation feasibility and schedule fit

**Logic:**

1. **Distance Calculation:**
   - Calculate distance from youth's home to each program
   - Flag programs > 10 miles as potential transportation barrier

2. **Public Transit Check:**
   - Query transportation database for bus routes
   - Check if route from youth's home to program exists
   - Calculate travel time
   - Note: Transfers, frequency, hours of operation

3. **Schedule Compatibility:**
   - Check program schedule (days, times)
   - Check youth's school schedule
   - Check family work schedules (if transportation dependent)
   - Identify conflicts or tight windows

4. **Transportation Options:**
   - Public transit (if available)
   - Family transportation (ask if family has vehicle)
   - Program-provided transportation (some programs offer)
   - Ride-share/Uber (cost estimate)

**Tools:**
- Google Maps API (distance, directions)
- Local transit API (bus routes, schedules)

**Output:**
```json
{
  "program": "North Youth Center",
  "logistics": {
    "distance_miles": 5.2,
    "drive_time_minutes": 15,
    "transit_available": true,
    "transit_details": {
      "route": "Bus 45",
      "travel_time_minutes": 35,
      "transfers": 0,
      "frequency": "Every 30 minutes",
      "hours": "6am-10pm"
    },
    "schedule_compatibility": {
      "program_schedule": "Mon/Wed/Fri 4-6pm",
      "youth_school_end": "3pm",
      "compatible": true,
      "notes": "Youth can take bus after school, arrive by 3:45pm"
    },
    "barriers": [],
    "recommendation": "Accessible via public transit; schedule compatible with school"
  }
}
```

#### G. Recommendation Synthesis Agent
**Purpose:** Rank programs and generate final recommendations with reasoning

**Logic:**

1. **Scoring System:**
   - **Need Alignment (40%):** How many/which needs addressed?
     - All critical needs: +40
     - All high needs: +30
     - Some high needs: +20
   - **Accessibility (30%):** Transportation, schedule, cost
     - No barriers: +30
     - Minor barriers (manageable): +20
     - Major barriers: +10
   - **Outcomes (20%):** Completion rate, recidivism data
     - High performers: +20
     - Average: +15
     - Below average or no data: +10
   - **Capacity (10%):** Openings available
     - Immediate openings: +10
     - Waitlist < 1 month: +7
     - Waitlist > 1 month: +5

2. **Deduct for Issues:**
   - Conflicts flagged: -20 points (or removal if critical)
   - Benefit issues: -10 points
   - Conditional eligibility: -5 points

3. **Rank Programs:** Sort by total score

4. **Select Top 3-5:** Present recommendations

5. **Generate Reasoning:**
   - For each recommended program, explain:
     - Why it's a good match (needs addressed)
     - Logistics (how youth will get there, schedule)
     - Outcomes (success rate for similar youth)
     - Any concerns and how to mitigate

**LLM Prompt for Reasoning:**
```
Generate a recommendation summary for this program.

Program: {program_name}
Youth Needs: {needs}
Eligibility: {eligibility_result}
Logistics: {transportation_schedule}
Outcomes: {completion_rate}, {recidivism_rate}
Conflicts: {conflicts}
Score: {total_score}

Explain:
1. Why this program is recommended (which needs it addresses)
2. Logistics (how accessible it is for the youth)
3. Expected outcomes (based on data for similar youth)
4. Any concerns and how to address them

Use professional but accessible language for case managers.
```

**Output:**
```
RECOMMENDATION #1: Outpatient Substance Abuse Treatment at North Clinic

WHY RECOMMENDED:
This program directly addresses the youth's high-priority need for substance abuse
treatment (marijuana use). The cognitive-behavioral approach aligns with best practices
for adolescent substance abuse. The program also includes family sessions, addressing
the family conflict identified in the assessment.

LOGISTICS:
- Location: 5.2 miles from youth's home
- Transportation: Accessible via Bus 45 (35-minute ride, no transfers)
- Schedule: Mon/Wed/Fri 4-6pm (compatible with school schedule)
- Cost: Medicaid-billable (no out-of-pocket cost for family)

EXPECTED OUTCOMES:
Similar youth (age 15-17, moderate risk, marijuana use) have shown:
- 78% completion rate
- 12% recidivism rate at 12 months (well below county average of 23%)

CONCERNS & MITIGATION:
None. Program is immediately accessible and has openings for intake next week.

ELIGIBILITY: ELIGIBLE (Eligibility Guidelines Section 2.1-2.4)
```

#### H. Override/Justification Node (HITL)
**Purpose:** Handle conflicts and exceptions with human judgment

**Logic:**
1. If conflicts or major barriers detected: `interrupt()`
2. Display to case manager:
   - Recommended program
   - Conflict details (victim proximity, gang territory, etc.)
   - Options:
     - **Proceed with program** (override conflict) → Requires justification
     - **Select alternative program** → Show next-ranked options
     - **Request exception** (e.g., waiver for eligibility) → Supervisor approval

3. If override requested:
   - Case manager provides justification
   - Routed to supervisor for approval (if high-severity conflict)

4. Log decision and reasoning for audit trail

**UI Flow:**
```
CONFLICT DETECTED

Program: Westside Community Center
Conflict: Program located 0.5 miles from victim's residence

This proximity may cause distress to the victim or create safety concerns.

OPTIONS:
[ ] Proceed with this program (requires justification and victim advocate consultation)
[ ] Select alternative program (see alternatives below)
[ ] Discuss with supervisor before deciding

If proceeding, please provide justification:
[Text box]

ALTERNATIVE PROGRAMS:
1. North Youth Center (no conflicts, 7.5 miles away)
2. East Side Program (no conflicts, 8.2 miles away)
```

**Implementation:**
```python
def override_justification_node(state):
    if state["conflicts"] and not state.get("override_approved"):
        # Interrupt for human decision
        interrupt("Conflicts detected. Please review and decide on override or alternative.")

        # After human resumes
        if state["override_requested"]:
            # Check if supervisor approval needed
            if state["conflict_severity"] == "high":
                # Route to supervisor
                interrupt("High-severity conflict override requires supervisor approval.")

        # Continue
        return state

    return state
```

#### I. Referral Generation Node
**Purpose:** Create and send referral documents

**Logic:**
1. For each approved program recommendation:
   - Generate referral document:
     - Youth demographics (name, DOB, contact)
     - Reason for referral (needs summary)
     - Case manager contact
     - Requested start date
   - Format according to program's requirements (PDF, online form, etc.)

2. Send referrals:
   - Email to program (if email-based)
   - API call (if program has integration)
   - Generate PDF for fax (if needed)

3. Schedule intake appointments (if program offers online scheduling)

4. Log referrals in case management system

5. Set follow-up reminder (check on referral status in 1 week)

**No LLM needed** - straightforward data formatting and transmission

### 3. State Management

**StateGraph Schema:**
```python
from typing import TypedDict, List, Dict, Optional

class MatchingState(TypedDict):
    # Input
    youth_profile: Dict[str, Any]  # demographics, assessment, case notes
    case_id: str

    # Needs extraction
    needs: List[Dict]  # [{domain, need, priority, evidence}]

    # Program retrieval
    candidate_programs: List[Dict]  # top 10-15 programs
    retrieval_scores: Dict[str, float]  # program_name → relevance score

    # Eligibility
    eligibility_results: Dict[str, Dict]  # program_name → {determination, criteria_checked, barriers}

    # Benefits
    benefit_flags: List[Dict]  # interactions and conflicts

    # Conflicts
    conflicts: List[Dict]  # safety issues

    # Transportation
    transportation_analysis: Dict[str, Dict]  # program_name → logistics

    # Recommendations
    recommendations: List[Dict]  # top 3-5 programs with reasoning
    program_scores: Dict[str, int]  # program_name → total score

    # Override workflow
    override_requests: List[Dict]  # requests for exceptions
    override_justifications: Dict[str, str]  # program_name → justification
    supervisor_approved: bool

    # Referrals
    referrals_generated: List[Dict]
    referrals_sent: bool

    # Workflow control
    current_node: str
    approved: bool
```

**Checkpointing:**
- PostgresSaver for production
- Checkpoint after each major agent node
- Enables supervisory review of matching logic

### 4. Conditional Routing

**Routing Logic:**

1. **After Conflict Check:**
   - If conflicts detected → Override/Justification Node (HITL)
   - Otherwise → Transportation & Scheduling Agent

2. **After Override Node:**
   - If approved (override or alternative selected) → Recommendation Synthesis
   - If not approved (case manager requests different options) → Loop back to Program Retrieval with adjusted criteria

3. **After Recommendations:**
   - If case manager approves → Referral Generation
   - If requests changes → Loop back with feedback

**Implementation:**
```python
def route_after_conflicts(state):
    if state["conflicts"]:
        return "override_justification"
    else:
        return "transportation_scheduling"

graph.add_conditional_edges(
    "conflict_check",
    route_after_conflicts,
    {
        "override_justification": "override_justification_node",
        "transportation_scheduling": "transportation_scheduling_node"
    }
)
```

---

## Data Flow

### Input
1. **Youth Profile:**
   - Demographics (age, gender, address)
   - Assessment data (risk/needs, domains)
   - Case notes (family context, school status)
   - Current benefits (Medicaid, SNAP, etc.)
   - Case specifics (offense, victim, co-defendants if applicable)

### Processing
1. **Needs Extraction:** Categorize and prioritize needs
2. **Program Retrieval:** Find candidate programs addressing needs
3. **Eligibility Checking:** Validate against criteria
4. **Benefits Check:** Identify interactions
5. **Conflict Check:** Safety validation
6. **Transportation Analysis:** Assess accessibility
7. **Recommendation Synthesis:** Rank and explain top programs
8. **Override Workflow (if needed):** Human decision on conflicts
9. **Referral Generation:** Create and send referrals

### Output
1. **Recommendation Report:**
   - Top 3-5 programs with detailed reasoning
   - Eligibility confirmations
   - Logistics details
   - Outcome expectations
   - Concerns and mitigations

2. **Referral Documents:**
   - Generated referrals to programs
   - Sent via email/API/fax
   - Logged in case management system

3. **Audit Trail:**
   - All programs considered
   - Eligibility determinations with citations
   - Conflict checks performed
   - Override justifications (if any)

---

## Technical Implementation Considerations

### 1. LLM Selection
- **Primary:** Claude Sonnet 4.5
  - Strong reasoning for eligibility logic
  - Good at structured outputs (JSON)
  - Citation capabilities
- **Cost Optimization:** Claude Haiku for simple extractions (needs, metadata)
- **Fallback:** GPT-4

### 2. Prompt Engineering

**Needs Extraction:**
- Temperature: 0.3 (factual, consistent)
- Chain-of-thought: "First identify explicit needs, then infer from risk factors..."
- Structured output: JSON schema for needs array

**Eligibility Checking:**
- Temperature: 0.2 (consistency critical)
- Few-shot examples (show eligible, ineligible, conditional cases)
- Explicit instruction: "Cite specific section of eligibility rules"

**Recommendation Reasoning:**
- Temperature: 0.4 (natural language for case managers)
- Examples of good reasoning (clear, concise, actionable)

### 3. RAG Optimization

**Program Directory Indexing:**
- Metadata-rich: program_type, needs_addressed, age_range, location, etc.
- Update frequency: Weekly (for capacity) or Monthly (for program details)
- Version control: Track changes to program offerings

**Retrieval Strategy:**
- MultiQueryRetriever for needs → programs matching
- Metadata filtering for hard constraints (age, location)
- Reranking by outcomes (completion rate, recidivism)

**Hybrid Search:**
- 70% semantic (need → service matching)
- 30% keyword (specific program types, evidence-based models)

### 4. Structured Data Integration

**Databases:**

1. **Program Capacity Database:**
   - Real-time or weekly batch updates
   - API integration if programs have capacity management systems

2. **Transportation Database:**
   - Local transit authority API (bus schedules, routes)
   - Google Maps API (driving distance, directions)

3. **Conflict Databases (Secure):**
   - Victim addresses: Encrypted, access-controlled
   - Gang data: Law enforcement shared, secure
   - Access logging for audit

**API Integration:**
- RESTful APIs for program referrals (if available)
- Standardized data formats (JSON)
- Error handling for API failures (graceful degradation)

### 5. Conflict Check Security

**Data Protection:**
- Victim data: Encrypted at rest and in transit
- Access control: Only authorized case managers
- Audit logs: Who accessed victim data when and why
- Data minimization: Only check proximity if necessary for case

**Victim Notification (Policy Decision):**
- Should victim be notified of youth's program placement?
- Coordinate with victim advocate
- Document in system

### 6. Override Workflow

**Approval Hierarchy:**

1. **Low-severity conflicts:**
   - Case manager can override with justification
   - Logged for supervisory review (sampling)

2. **Medium-severity conflicts:**
   - Case manager requests override
   - Supervisor reviews and approves/denies

3. **High-severity conflicts (victim proximity, safety):**
   - Supervisor approval required
   - Victim advocate consulted
   - Document in detail

**Documentation Requirements:**
- Override justification (why proceeding despite conflict)
- Mitigation plan (how conflict will be managed)
- Approver name and timestamp
- Stored in case file for audit

### 7. Evaluation and Quality Assurance

**Metrics to Track:**

1. **Recommendation Quality:**
   - Acceptance rate (% of recommendations accepted by case managers)
   - Program enrollment (% of referrals that result in enrollment)
   - Completion rate (do recommended programs have good outcomes for youth?)

2. **Efficiency:**
   - Time to generate recommendations (target: < 5 minutes)
   - Time saved vs. manual matching (target: 1-2 hours saved)

3. **Accuracy:**
   - Eligibility accuracy (% of determinations correct - audit sampling)
   - Conflict detection (false positive and false negative rates)

4. **User Satisfaction:**
   - Case manager feedback on recommendations
   - Program staff feedback on referral quality

**Quality Assurance:**
- Supervisor reviews first 20 recommendations completely
- 10% random sample ongoing
- Monthly review: Common override reasons, missed programs, feedback

### 8. Security and Privacy

**Data Protection:**
- Encrypt youth profile data
- Secure conflict databases (victim, gang data)
- Access control by role
- Audit logs

**Compliance:**
- JJDPA confidentiality
- HIPAA (for mental health/substance abuse programs)
- Gang data sharing agreements (law enforcement)

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals:**
- Build program directory (RAG)
- Needs extraction agent

**Deliverables:**
1. Vector store with 20-30 program descriptions
2. Needs extraction from sample assessments
3. Basic program retrieval (needs → programs)

**Success Criteria:**
- Can extract needs from 10 sample assessments
- Retrieve relevant programs for each need category

### Phase 2: Eligibility & Benefits (Weeks 3-4)
**Goals:**
- Eligibility checking logic
- Benefits interaction checking

**Deliverables:**
1. Eligibility agent with criteria matching
2. Benefits interaction agent
3. Structured eligibility rules database

**Success Criteria:**
- 90%+ accuracy on eligibility determinations (manual verification)
- Correctly identifies Medicaid-billable programs

### Phase 3: Conflicts & Transportation (Week 5)
**Goals:**
- Conflict check implementation
- Transportation analysis

**Deliverables:**
1. Conflict check agent (victim, gang, co-participants)
2. Transportation agent with transit API integration

**Success Criteria:**
- Detects test conflicts (victim proximity, gang territory)
- Provides accurate transit directions and times

### Phase 4: Recommendations & Override (Week 6)
**Goals:**
- Recommendation synthesis
- HITL override workflow

**Deliverables:**
1. Recommendation synthesis agent with scoring
2. Override justification interface
3. Full LangGraph workflow

**Success Criteria:**
- Generates ranked recommendations with reasoning
- Override workflow functional (case manager can justify and proceed)

### Phase 5: Referrals & Testing (Weeks 7-8)
**Goals:**
- Referral generation
- End-to-end testing with case managers

**Deliverables:**
1. Referral generation and sending
2. Test with 10 real case scenarios
3. Case manager feedback

**Success Criteria:**
- Referrals generated in correct format
- 80%+ acceptance rate by case managers
- Case manager satisfaction > 4/5

---

## Success Metrics

### Efficiency Metrics
- **Time Savings:** Reduce program matching from 1-2 hours to 10-15 minutes (staff review)
- **Referral Speed:** Generate referrals same-day (vs. often 1-3 days manual)

### Quality Metrics
- **Recommendation Acceptance:** >80% of recommendations accepted by case managers
- **Enrollment Rate:** >70% of referrals result in program enrollment
- **Completion Rate:** Youth in recommended programs complete at rate ≥ county average
- **Eligibility Accuracy:** >95% of eligibility determinations correct (audit)

### Safety Metrics
- **Conflict Detection:** >95% of safety conflicts detected
- **Override Appropriateness:** >90% of overrides deemed appropriate by supervisors (audit)

### User Adoption
- **Usage Rate:** >75% of placements use system (after training)
- **Case Manager Satisfaction:** >4/5 on surveys

---

## Risks and Mitigation

### Risk 1: Incorrect Eligibility Determination
**Likelihood:** Medium
**Impact:** Medium (wrong placement, wasted resources)
**Mitigation:**
- Human review of all recommendations before referral
- Audit sampling of eligibility determinations
- Continuous improvement based on errors
- Citations for all determinations (staff can verify)

### Risk 2: Missed Safety Conflict
**Likelihood:** Low
**Impact:** Critical (safety incident)
**Mitigation:**
- Layered conflict checks (database + LLM reasoning)
- Human review of all conflict flags
- Conservative thresholds (err on side of caution)
- Regular audits of conflict checks
- Incident reporting and review process

### Risk 3: Data Privacy Breach (Victim Info)
**Likelihood:** Low
**Impact:** Critical
**Mitigation:**
- Encryption and access controls
- Audit logs (who accessed victim data when)
- Data minimization (only check if necessary)
- Security audits
- Incident response plan

### Risk 4: Transportation Barriers Overlooked
**Likelihood:** Medium
**Impact:** Medium (enrollment failures, frustration)
**Mitigation:**
- Explicit transportation analysis step
- Ask families directly about transportation availability
- Offer alternatives (remote programs, transportation vouchers)
- Track enrollment failures, identify patterns

### Risk 5: Program Data Outdated
**Likelihood:** Medium
**Impact:** Medium (referrals to full programs, wrong information)
**Mitigation:**
- Regular updates (weekly for capacity, monthly for program details)
- Verification with programs before final referral
- Track referral success rate, flag programs with low enrollment
- Feedback loop from programs

---

## Next Steps After POC Success

1. **Expand Program Directory:**
   - Regional programs (neighboring counties)
   - Specialty programs (trauma-specific, gender-specific, etc.)
   - Community resources (non-justice-involved support)

2. **Outcome Tracking:**
   - Integrate with case management to track program completion
   - Calculate recidivism for youth in each program
   - Machine learning: Predict best program for individual youth

3. **Family Engagement:**
   - Family portal: See recommended programs, provide input
   - Multi-language program descriptions

4. **Advanced Matching:**
   - Peer matching: Match youth to programs with similar demographics (mentor matching)
   - Cultural matching: Culturally-specific programs for youth of color, immigrants, etc.

5. **Integration:**
   - API to program capacity systems (real-time availability)
   - Automated appointment scheduling
   - Progress tracking (program sends updates to case management)

---

## Conclusion

The Program Matching and Referral Recommender addresses a complex, high-stakes decision: placing youth in programs that meet their needs while ensuring safety, accessibility, and eligibility. By combining LangChain RAG over program directories with LangGraph's multi-agent orchestration and human-in-the-loop oversight, this system can:

- Save case managers 1-2 hours per placement
- Ensure comprehensive consideration of all factors (needs, eligibility, benefits, conflicts, transportation)
- Improve placement quality (data-driven recommendations)
- Reduce placement failures (accessibility barriers identified upfront)
- Maintain safety (conflict checks, override justifications)

The architecture prioritizes thoroughness with multi-agent validation, safety with conflict checks, and human judgment with override workflows. Success requires high-quality program data, secure handling of sensitive info (victim, gang data), and strong collaboration with case managers during development.

**Recommended as fourth POC**, after establishing multi-agent patterns with prior use cases. High complexity but high value for optimal youth outcomes.
