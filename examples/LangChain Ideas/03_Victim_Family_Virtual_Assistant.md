# POC #3: Victim and Family Virtual Assistant

**Project Type:** LangGraph Controlled Chatbot with LangChain RAG and Guardrails
**Complexity:** Medium
**Timeline Estimate:** 4-6 weeks for POC
**Primary Value:** 24/7 information access, reduced staff burden, crisis routing, multi-language support

---

## Problem Statement

Families and victims navigating the juvenile justice system have many questions about hearings, rights, diversion options, and community resources. Currently:
- Staff field repetitive questions via phone and email (time-consuming)
- Families often wait hours or days for responses
- Information access limited to business hours
- Language barriers create communication gaps
- Crisis situations may not be identified quickly
- Legal advice requests create liability issues

**Goal:** A controlled chatbot that answers FAQs on hearings, rights, diversion options, and community resources, tuned for reading level and available in multiple languages, with guardrails to prevent legal advice and route crisis situations to human staff.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│         VICTIM & FAMILY VIRTUAL ASSISTANT                 │
└──────────────────────────────────────────────────────────┘

LAYER 1: Knowledge Base (LangChain RAG)
┌───────────────────────────────────────────────────────┐
│ APPROVED CONTENT ONLY:                                │
│  • FAQ documents (hearings, rights, procedures)       │
│  • Diversion program descriptions                     │
│  • Community resource directory                       │
│  • Court process flowcharts                           │
│  • Youth/family rights handbooks                      │
│  • Glossary of legal terms (plain language)           │
│  • Crisis resources (hotlines, emergency services)    │
│                                                       │
│ [Document Loaders] → [Structure-Aware Splitting]     │
│        ↓                                              │
│ [Embeddings] → [Vector Store: Approved Content]      │
│ [Metadata: language, topic, reading_level]           │
└───────────────────────────────────────────────────────┘

LAYER 2: LangGraph Controlled Conversation Flow
┌───────────────────────────────────────────────────────┐
│  START: User Message                                  │
│    ↓                                                  │
│  [Language Detection Node]                            │
│    ├─ Detect user's language                         │
│    ├─ Set conversation language                      │
│    └─ Load language-specific resources               │
│    ↓                                                  │
│  [Guardrail Check Node] ◄─── CRITICAL GATE           │
│    ├─ Crisis keyword detection (suicide, harm)       │
│    ├─ Legal advice request detection                 │
│    ├─ Out-of-scope detection                         │
│    ├─ Inappropriate content detection                │
│    └─ Route based on classification                  │
│    ↓                                                  │
│  CONDITIONAL ROUTING:                                 │
│    ├─ Crisis → [Crisis Response Node]                │
│    ├─ Legal Advice → [Boundary Reinforcement]        │
│    ├─ Out-of-Scope → [Escalation to Human]           │
│    └─ Safe Question → [RAG Response Node]            │
│    ↓                                                  │
│  [RAG Response Node]                                  │
│    ├─ Retrieve relevant approved content             │
│    ├─ Filter by language and reading level           │
│    ├─ Generate response (with citations)             │
│    └─ Apply reading level simplification             │
│    ↓                                                  │
│  [Response Validation Node]                           │
│    ├─ Check for hallucinations (citation coverage)   │
│    ├─ Verify reading level appropriate               │
│    ├─ Ensure no legal advice given                   │
│    └─ Confirm citations present                      │
│    ↓                                                  │
│  [Feedback & Follow-up Node]                          │
│    ├─ Ask if answer helpful                          │
│    ├─ Offer related questions                        │
│    ├─ Provide human contact option                   │
│    └─ Log interaction for analytics                  │
│    ↓                                                  │
│  END → Response to User                               │
│                                                       │
│ PARALLEL PROCESSES:                                   │
│  [Crisis Response Node] → Immediate human alert       │
│  [Escalation Node] → Human staff notification         │
└───────────────────────────────────────────────────────┘

LAYER 3: Guardrails (Safety Layer)
┌───────────────────────────────────────────────────────┐
│ CRISIS KEYWORDS:                                      │
│  • Suicide, self-harm, death, kill                    │
│  • Abuse, violence, threat                            │
│  • Emergency, danger, help now                        │
│  → Immediate routing to crisis resources + human      │
│                                                       │
│ LEGAL ADVICE TRIGGERS:                                │
│  • "Should I plead...", "What should I say..."        │
│  • "Can they make me...", "Do I have to..."          │
│  → Boundary message + referral to attorney            │
│                                                       │
│ OUT-OF-SCOPE:                                         │
│  • Specific case strategy, confidential details       │
│  • Requests for actions (change court date, etc.)    │
│  → Escalation to human staff                          │
│                                                       │
│ SAFETY CHECKS:                                        │
│  • No PII in responses (names, addresses, etc.)       │
│  • No victim identification                           │
│  • Age-appropriate language                           │
└───────────────────────────────────────────────────────┘

LAYER 4: State Management
┌───────────────────────────────────────────────────────┐
│ ConversationState:                                    │
│  - session_id: str                                    │
│  - user_language: str (en, es, vi, etc.)              │
│  - messages: List[Message] (conversation history)     │
│  - flagged_content: List[str] (crisis, legal, etc.)   │
│  - escalated: bool                                    │
│  - human_notified: bool                               │
│  - feedback: List[Dict] (user satisfaction)           │
│  - retrieved_documents: List[Document]                │
└───────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Knowledge Base (LangChain RAG)

**Curated Content Sources:**

1. **FAQs:**
   - What happens at a hearing?
   - How does diversion work?
   - What are my rights as a parent?
   - What is probation?
   - How long does the process take?

2. **Process Guides:**
   - Court process flowcharts
   - Diversion eligibility and process
   - Probation conditions explanations
   - Rights at each stage

3. **Community Resources:**
   - Mental health services
   - Substance abuse treatment
   - Tutoring and education support
   - Job training programs
   - Food assistance, housing support
   - Transportation options

4. **Crisis Resources:**
   - National Suicide Prevention Lifeline
   - Crisis Text Line
   - Local crisis intervention
   - Domestic violence hotlines
   - Child abuse reporting

5. **Legal Information (Not Advice):**
   - Rights handbooks (youth and family)
   - Glossary of legal terms
   - What to expect guides
   - How to access public defender

**Content Curation:**
- All content reviewed and approved by legal and program staff
- Reading level: 6th-8th grade
- Multiple languages: English, Spanish, Vietnamese, (expandable)
- Regular updates (quarterly review)
- Version control (track what content used when)

**Document Processing:**
```python
Document Loaders:
  - PDFLoader for handbooks and guides
  - WebBaseLoader for agency websites
  - MarkdownLoader for FAQs (structured Q&A)
  - JSONLoader for resource directories

Text Splitting:
  - Structure-aware splitting (respect Q&A boundaries, sections)
  - chunk_size: 800 tokens (balance context and precision)
  - chunk_overlap: 150 tokens

Metadata Tagging:
  - topic: "hearing", "rights", "diversion", "resources", "crisis"
  - language: "en", "es", "vi"
  - reading_level: 6, 7, 8 (grade level)
  - content_type: "faq", "guide", "resource", "crisis"
  - last_updated: date
```

**Embeddings:**
- Multilingual embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Or OpenAI embeddings (supports 100+ languages)

**Vector Store:**
- Development: ChromaDB (local)
- Production: Pinecone or Qdrant
- Separate collections by language (faster retrieval)
- Metadata filtering by topic and reading level

### 2. LangGraph Conversation Nodes

#### A. Language Detection Node
**Purpose:** Identify user's language and set conversation context

**Logic:**
1. Use language detection library (langdetect, or LLM-based)
2. Set conversation language in state
3. Load language-specific resources
4. If unsupported language: Politely inform, offer English or Spanish

**Implementation:**
```python
from langdetect import detect

def language_detection_node(state):
    user_message = state["messages"][-1].content

    try:
        detected_lang = detect(user_message)

        # Map to supported languages
        supported = ["en", "es", "vi"]
        if detected_lang in supported:
            state["user_language"] = detected_lang
        else:
            # Fallback to English, inform user
            state["user_language"] = "en"
            state["messages"].append(
                AIMessage(content="I primarily support English, Spanish, and Vietnamese. Continuing in English.")
            )

    except:
        # Detection failed, default to English
        state["user_language"] = "en"

    return state
```

**No LLM needed** - use language detection library

#### B. Guardrail Check Node
**Purpose:** Critical safety gate to detect and route sensitive content

**Logic:**

1. **Crisis Keyword Detection:**
   - Check for crisis keywords: suicide, kill myself, self-harm, abuse, violence, threat
   - Use both keyword matching and LLM-based intent classification
   - If detected → Route to Crisis Response Node

2. **Legal Advice Detection:**
   - Check for legal advice requests: "Should I plead...", "What should I say in court...", "Can they make me..."
   - LLM classifies: Informational vs. Legal advice request
   - If legal advice → Route to Boundary Reinforcement

3. **Out-of-Scope Detection:**
   - Case-specific strategy questions
   - Requests for actions (change court date, contact judge)
   - Confidential case details
   - If out-of-scope → Route to Escalation

4. **Safety Check:**
   - Inappropriate content (threats, harassment)
   - Requests for victim information
   - If unsafe → Block and escalate

**LLM Prompt for Intent Classification:**
```
Classify the following user message into ONE category:

User Message: "{user_message}"

Categories:
1. CRISIS: Indicates immediate danger (suicide, self-harm, abuse, violence)
2. LEGAL_ADVICE: Asks for legal advice or case strategy
3. OUT_OF_SCOPE: Asks for specific actions or case-specific confidential information
4. SAFE_QUESTION: General information about process, rights, or resources

Output only the category name.
```

**Keyword Lists (Multilingual):**
```python
crisis_keywords = {
    "en": ["suicide", "kill myself", "self-harm", "hurt myself", "abuse", "violence"],
    "es": ["suicidio", "matarme", "autolesión", "hacerme daño", "abuso", "violencia"],
    "vi": ["tự tử", "tự sát", "tự làm hại", "lạm dụng", "bạo lực"]
}
```

**Routing Logic:**
```python
def route_from_guardrail(state):
    intent = state["intent_classification"]

    if intent == "CRISIS":
        return "crisis_response"
    elif intent == "LEGAL_ADVICE":
        return "boundary_reinforcement"
    elif intent == "OUT_OF_SCOPE":
        return "escalation"
    else:  # SAFE_QUESTION
        return "rag_response"
```

#### C. Crisis Response Node
**Purpose:** Immediate, empathetic response with crisis resources and human alert

**Logic:**
1. Provide immediate crisis resources:
   - National Suicide Prevention Lifeline: 988
   - Crisis Text Line: Text HOME to 741741
   - Local crisis services (jurisdiction-specific)
2. Empathetic message: "I'm concerned about your safety. Please reach out to these resources immediately."
3. Alert human staff (email/SMS to on-call staff)
4. Log crisis interaction (for follow-up)
5. Do NOT continue conversation on crisis topic (refer to professionals)

**Response Template (Multilingual):**
```
{language_specific_empathy_message}

If you or someone you know is in crisis, please reach out immediately:
- National Suicide Prevention Lifeline: Call or text 988
- Crisis Text Line: Text HOME to 741741
- Local Crisis Team: {jurisdiction_specific_contact}

I've also notified our staff, and someone will follow up with you soon.

Is there anything else I can help you with regarding the juvenile justice process?
```

**Human Alert:**
- Send to on-call staff (email + SMS)
- Include: timestamp, session_id, user message (redacted)
- Priority: URGENT

**No conversational LLM** - use pre-written templates for safety and consistency

#### D. Boundary Reinforcement Node
**Purpose:** Politely decline legal advice requests, refer to attorney

**Logic:**
1. Explain limitation: "I can provide general information about the process, but I can't give legal advice."
2. Explain why: "Legal advice should come from your attorney who knows your specific situation."
3. Provide referral: "If you don't have an attorney, here's how to request a public defender..."
4. Offer alternative: "I can explain general process or rights. What would you like to know?"

**Response Template:**
```
I understand you have important questions about your case. However, I can only provide
general information about the juvenile justice process. I can't give legal advice or
tell you what to do in your specific situation.

For legal advice, please contact:
- Your attorney (if you have one)
- Public Defender's Office: {contact_info}
- Legal Aid: {contact_info}

I can help with:
- General information about court processes
- Your rights as a parent/victim
- Available resources and programs
- What to expect at hearings

What general information can I help you with?
```

**Log for Training:**
- Track legal advice requests (improve detection)
- Review for content gaps (maybe FAQ needs expansion)

#### E. Escalation Node
**Purpose:** Route out-of-scope questions to human staff

**Logic:**
1. Acknowledge question
2. Explain limitation: "This question requires specific information about your case. Let me connect you with staff."
3. Collect contact info (optional)
4. Notify human staff (non-urgent email)
5. Provide expected response time: "Staff typically respond within 1 business day."

**Response Template:**
```
Thank you for your question. This requires information specific to your case,
which I don't have access to.

I've notified our staff, and someone will follow up with you within 1 business day.

If you'd like, you can also contact us directly:
- Phone: {phone_number}
- Email: {email}
- Office hours: {hours}

In the meantime, is there general information I can help you with?
```

**Staff Notification:**
- Email to case assignment staff
- Include: user question, session_id, timestamp
- Priority: NORMAL

#### F. RAG Response Node
**Purpose:** Answer safe, in-scope questions using approved content

**Logic:**
1. **Retrieval:**
   - Query vector store with user question
   - Filter by: language, topic (optional)
   - Top-k: 3-5 documents
   - Retrieve parent chunks for context

2. **Response Generation:**
   - Use LLM to synthesize answer from retrieved documents
   - Maintain reading level (6th-8th grade)
   - Include citations: "According to the Family Handbook, ..."
   - If no relevant content found: "I don't have information on that. Let me connect you with staff."

3. **Store Retrieved Documents:**
   - Save in state for response validation

**LLM Prompt:**
```
You are a helpful assistant for families navigating the juvenile justice system.
Answer the user's question using ONLY the information from the following approved documents.

User Question: "{user_question}"

Approved Documents:
{retrieved_docs}

Guidelines:
- Use simple language (6th-8th grade reading level)
- Be empathetic and supportive
- Cite the source document for key information
- If the documents don't contain an answer, say "I don't have information on that specific question. Let me connect you with staff who can help."
- DO NOT make up information
- DO NOT give legal advice

Language: {user_language}
```

**Temperature:** 0.4 (balance accuracy and natural phrasing)

**Multilingual Handling:**
- If user_language != "en":
  - Retrieve from language-specific collection
  - OR retrieve from English and translate (if multilingual content limited)
  - LLM responds in user's language

#### G. Response Validation Node
**Purpose:** Verify response quality and safety before sending

**Logic:**
1. **Citation Coverage:**
   - Check that main facts have citations
   - If no citations: Flag for review (possible hallucination)

2. **Reading Level:**
   - Use Flesch-Kincaid or similar to check grade level
   - If > 8th grade: Simplify (retry with explicit instruction)

3. **Guardrail Check:**
   - Run response through legal advice detector (make sure no advice slipped through)
   - Check for PII (should be none)

4. **Fallback:**
   - If validation fails multiple times: Escalate to human

**Implementation:**
```python
def response_validation_node(state):
    response = state["messages"][-1].content

    # Check for citations
    has_citations = "according to" in response.lower() or "handbook" in response.lower()

    # Check reading level
    from textstat import flesch_kincaid_grade
    grade_level = flesch_kincaid_grade(response)

    # Validate
    if not has_citations:
        state["flagged_content"].append("No citations")

    if grade_level > 8:
        state["flagged_content"].append("High reading level")

    # If flags, escalate or retry
    if state["flagged_content"]:
        # Retry once with explicit instruction
        if state.get("retry_count", 0) < 1:
            state["retry_count"] = 1
            return "rag_response"  # Loop back to regenerate
        else:
            return "escalation"  # Give up, escalate

    return "feedback_followup"  # Validation passed
```

#### H. Feedback & Follow-up Node
**Purpose:** Gather user feedback and offer continued assistance

**Logic:**
1. Ask if answer was helpful: "Was this information helpful? (Yes/No)"
2. Offer related questions: "You might also want to know: [related question 1], [related question 2]"
3. Provide human contact option: "If you need more help, here's how to reach us: {contact_info}"
4. Log interaction: session_id, question, answer, helpful (Y/N), timestamp

**Response Template:**
```
{ai_generated_answer}

---

Was this information helpful?

You might also want to know:
- {related_question_1}
- {related_question_2}

If you have more questions, feel free to ask! Or, you can contact us directly at {contact_info}.
```

**Analytics:**
- Track helpful rate (% of "Yes" responses)
- Identify common follow-up topics
- Flag low helpful rate for content review

### 3. State Management

**StateGraph Schema:**
```python
from typing import TypedDict, List, Optional
from langchain.schema import BaseMessage

class ConversationState(TypedDict):
    # Session
    session_id: str
    timestamp: datetime
    user_language: str  # en, es, vi

    # Conversation
    messages: List[BaseMessage]  # full history
    current_topic: Optional[str]  # hearing, rights, resources, etc.

    # Guardrails
    intent_classification: str  # CRISIS, LEGAL_ADVICE, OUT_OF_SCOPE, SAFE_QUESTION
    flagged_content: List[str]  # crisis keywords, legal advice attempts

    # Escalation
    escalated: bool
    human_notified: bool
    escalation_reason: str

    # RAG
    retrieved_documents: List[Document]
    retry_count: int  # for validation failures

    # Feedback
    feedback: List[Dict]  # [{question, answer, helpful, timestamp}]

    # Workflow
    current_node: str
```

**Memory Management:**
- **Conversation Buffer Window Memory**: Last 10 messages (balance context and cost)
- Summarize older messages if conversation long (>20 turns)
- Session timeout: 30 minutes of inactivity

**Checkpointing:**
- PostgresSaver for production
- Checkpoint after each user turn
- Enables:
  - Resume conversation if user returns
  - Review flagged conversations
  - Analytics on conversation flows

### 4. Conditional Routing

**Primary Routing (From Guardrail Check):**
```python
def route_from_guardrail(state):
    intent = state["intent_classification"]

    if intent == "CRISIS":
        return "crisis_response"
    elif intent == "LEGAL_ADVICE":
        return "boundary_reinforcement"
    elif intent == "OUT_OF_SCOPE":
        return "escalation"
    else:  # SAFE_QUESTION
        return "rag_response"

graph.add_conditional_edges(
    "guardrail_check",
    route_from_guardrail,
    {
        "crisis_response": "crisis_response_node",
        "boundary_reinforcement": "boundary_reinforcement_node",
        "escalation": "escalation_node",
        "rag_response": "rag_response_node"
    }
)
```

**Validation Routing:**
```python
def route_from_validation(state):
    if state["flagged_content"] and state.get("retry_count", 0) < 1:
        return "rag_response"  # Retry
    elif state["flagged_content"]:
        return "escalation"  # Failed validation
    else:
        return "feedback_followup"  # Success

graph.add_conditional_edges(
    "response_validation",
    route_from_validation,
    {
        "rag_response": "rag_response_node",
        "escalation": "escalation_node",
        "feedback_followup": "feedback_followup_node"
    }
)
```

---

## Data Flow

### Input
1. **User Message:**
   - Text input from web chat, SMS, or voice (transcribed)
   - Language: English, Spanish, Vietnamese, or other

2. **Conversation Context:**
   - Prior messages in session (last 10)
   - User language preference

### Processing
1. **Language Detection:** Identify user's language
2. **Guardrail Check:** Classify intent (crisis, legal advice, out-of-scope, safe)
3. **Routing:** Send to appropriate node based on classification
4. **RAG Response (if safe):** Retrieve approved content, generate answer
5. **Validation:** Check citations, reading level, safety
6. **Feedback:** Gather user satisfaction, offer follow-up

### Output
1. **To User:**
   - Empathetic, helpful response
   - Citations to approved sources
   - Related questions or next steps
   - Contact info for human staff (if needed)

2. **To Staff (if escalated):**
   - Email/SMS alert
   - User question and context
   - Priority level (URGENT for crisis, NORMAL for out-of-scope)

3. **Analytics:**
   - Conversation logs (anonymized)
   - Helpful rate, common topics
   - Escalation reasons

---

## Technical Implementation Considerations

### 1. LLM Selection
- **Primary:** Claude Sonnet 4.5 or Haiku
  - Strong at following instructions (important for guardrails)
  - Good multilingual support
  - Cost-effective (Haiku for simple responses)
- **Fallback:** GPT-3.5 Turbo (cost optimization for high volume)

### 2. Prompt Engineering

**Guardrail Prompts:**
- Temperature: 0.1 (consistency critical for safety)
- Explicit categories: Show examples of each intent type
- Few-shot examples for edge cases

**Response Generation:**
- Temperature: 0.4 (balance natural language and accuracy)
- System message emphasizes: empathy, reading level, citations, no advice
- Few-shot examples of good responses (helpful, cited, simple language)

**Reading Level Control:**
```
Use simple language that a 6th-8th grader can understand.
- Short sentences (10-15 words)
- Common words, not jargon
- Explain legal terms if used (e.g., "probation (supervised release)")
```

### 3. RAG Optimization

**Retrieval Strategy:**
- Hybrid search: 70% semantic, 30% keyword (for legal terms)
- Top-k: 3-5 documents
- Metadata filtering:
  - language = user_language
  - Optional: topic (if user's question clearly about hearing, filter to hearing docs)

**Chunking:**
- Structure-aware: Keep Q&A pairs together
- chunk_size: 800 tokens (FAQ + answer)
- chunk_overlap: 150 tokens

**Evaluation:**
- Curate 50 test questions (diverse topics, languages)
- Measure: retrieval precision (correct docs retrieved), answer accuracy, citation presence

### 4. Multilingual Support

**Approach 1: Separate Collections**
- Pros: Faster retrieval, better accuracy
- Cons: Requires translating all content
- Best for: 2-3 primary languages

**Approach 2: Cross-Lingual Retrieval**
- Multilingual embeddings (paraphrase-multilingual-mpnet)
- Retrieve English content, LLM translates
- Pros: Less content duplication
- Cons: Translation quality varies

**Recommended:**
- Start with Approach 1 for English and Spanish (primary languages)
- Add Vietnamese if significant user base
- Expand to others as needed

**Translation Quality Assurance:**
- Native speaker review of all translated content
- Cultural adaptation (not just literal translation)
- Test with native speakers

### 5. Guardrail Implementation

**Layered Approach:**

1. **Keyword Matching** (Fast, catches obvious cases):
   - Crisis keywords: suicide, kill, harm
   - Legal keywords: should I, can they make me

2. **LLM Classification** (Nuanced, catches edge cases):
   - Few-shot examples of each category
   - Confidence score (low confidence → escalate)

3. **Human Review** (Audit):
   - Sample 10% of conversations
   - Review all escalated conversations
   - Update keywords and prompts based on findings

**Testing:**
- Red team testing: Try to bypass guardrails (adversarial prompts)
- Edge cases: Implied crisis, subtle legal advice requests
- Continuously update based on real usage

### 6. Crisis Routing

**Immediate Human Alert:**
```python
def send_crisis_alert(session_id, user_message_excerpt):
    # Email
    send_email(
        to="oncall@jjsystem.org",
        subject="URGENT: Crisis Detected in Virtual Assistant",
        body=f"Session {session_id} flagged for crisis keywords. Excerpt: {user_message_excerpt[:100]}"
    )

    # SMS (for faster response)
    send_sms(
        to=ONCALL_PHONE,
        message=f"URGENT: Crisis in chatbot session {session_id}. Check email."
    )

    # Log
    log_crisis_event(session_id, timestamp, user_message_excerpt)
```

**Follow-up Protocol:**
- On-call staff reviews within 15 minutes
- Attempts to contact user (if contact info available)
- Coordinates with local crisis team if needed
- Logs outcome

### 7. User Interface

**Web Chat Widget:**
- Embed on juvenile justice agency website
- Mobile-responsive
- Accessibility: WCAG 2.1 AA compliant
- Features:
  - Language selector (EN / ES / VI)
  - Typing indicators
  - Suggested questions (common FAQs)
  - "Talk to a person" button (always visible)
  - Disclaimer: "This is an automated assistant. For emergencies, call 911."

**SMS Interface (Optional):**
- Users text questions to dedicated number
- Responses via SMS (keep under 160 chars or split)
- Link to web chat for longer conversations

**Voice Interface (Future):**
- Phone hotline with voice bot
- Speech-to-text → Chatbot → Text-to-speech
- Accessibility for low-literacy users

### 8. Content Management

**Content Update Process:**
1. Staff propose new FAQ or update
2. Legal review (ensure no legal advice)
3. Plain language editor (reading level check)
4. Translate to all supported languages
5. Add to knowledge base with metadata
6. Version control (Git for content files)

**Quarterly Review:**
- Review analytics: What questions most common? What escalated?
- Identify content gaps
- Update outdated information (policy changes)
- Re-embed and redeploy

**Content Format:**
```markdown
## Question: What happens at a juvenile court hearing?

**Reading Level:** 7
**Topic:** hearing
**Language:** en

A juvenile court hearing is a meeting where a judge listens to your case.
Here's what usually happens:

1. The judge calls your name
2. Your attorney (lawyer) and the prosecutor (county attorney) speak
3. The judge may ask you questions
4. The judge decides what happens next

You have the right to have your parent or guardian with you. You also
have the right to an attorney.

**Source:** Family Handbook, Section 3, Page 12
```

### 9. Analytics and Monitoring

**Key Metrics:**

1. **Usage:**
   - Conversations per day
   - Questions per conversation
   - Language distribution

2. **Quality:**
   - Helpful rate (% of positive feedback)
   - Escalation rate (% of conversations escalated)
   - Crisis detection rate

3. **Performance:**
   - Response time (avg time to generate response)
   - Retrieval accuracy (manually sampled)
   - Reading level compliance (% of responses ≤ 8th grade)

4. **Safety:**
   - False positive rate (crisis alerts that weren't crises)
   - False negative rate (missed crises - audit sampling)
   - Legal advice boundary adherence (manual review)

**Dashboard:**
- Real-time: Conversations active, escalations today, crisis alerts
- Daily: Conversations, helpful rate, top topics
- Weekly: Trends, content gaps identified, escalation reasons

**Alerts:**
- Spike in crisis flags (possible systemic issue)
- Low helpful rate (< 70% for the day)
- High escalation rate (> 30%)

### 10. Security and Privacy

**Data Protection:**
- Encrypt conversation logs (at rest and in transit)
- PII Minimization: Don't ask for names, case numbers
- Session-based: No persistent user accounts (anonymous)
- Data retention: 90 days for conversations, indefinitely for analytics (anonymized)

**Access Control:**
- Staff access to conversation logs (audit purposes)
- Role-based: Only supervisors can view flagged conversations
- Audit logs: Who accessed what when

**Compliance:**
- COPPA (if youth under 13 might use): Require parent consent
- State privacy laws
- Terms of service: Clear disclaimer (not legal advice, staff may review)

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goals:**
- Build RAG pipeline with approved FAQs
- Basic chatbot (no guardrails yet)

**Deliverables:**
1. Vector store with 20 FAQs (English)
2. Simple chatbot (question → RAG → response)
3. Web chat interface (basic)

**Success Criteria:**
- Can answer 15/20 test questions accurately
- Responses cite sources
- Reading level ≤ 8th grade

### Phase 2: Guardrails (Week 3)
**Goals:**
- Implement guardrail checks
- Crisis routing, legal advice boundaries

**Deliverables:**
1. Guardrail check node (all 4 categories)
2. Crisis response node with human alert
3. Boundary reinforcement node

**Success Criteria:**
- Detects 95% of test crisis keywords
- Detects 90% of legal advice requests
- Crisis alerts sent within 5 seconds

### Phase 3: LangGraph Integration (Week 4)
**Goals:**
- Full LangGraph workflow
- State management, conditional routing

**Deliverables:**
1. All nodes implemented
2. Conditional routing logic
3. State checkpointing

**Success Criteria:**
- Complete conversation flow (greeting → question → answer → feedback)
- Escalations route correctly
- State persists across session

### Phase 4: Multilingual Support (Week 5)
**Goals:**
- Add Spanish language support
- Test with bilingual users

**Deliverables:**
1. Translate 20 FAQs to Spanish
2. Multilingual embeddings
3. Language detection

**Success Criteria:**
- Spanish queries answered in Spanish
- Retrieval accuracy ≥ 85% for Spanish
- Native speaker approves translations

### Phase 5: Testing & Refinement (Week 6)
**Goals:**
- User testing with families
- Refine based on feedback

**Deliverables:**
1. Test with 10 family volunteers
2. Gather feedback surveys
3. Performance metrics report

**Success Criteria:**
- Helpful rate > 75%
- User satisfaction > 4/5
- No safety incidents (legal advice given, crisis missed)

---

## Success Metrics

### Efficiency Metrics
- **Staff Time Saved:** Reduce phone/email FAQs by 50%
- **Response Time:** Instant (vs. hours/days for staff)
- **Availability:** 24/7 access

### Quality Metrics
- **Helpful Rate:** >75% of users mark answers as helpful
- **Accuracy:** >90% of answers factually correct (manual audit)
- **Citation Rate:** >95% of responses include citations
- **Reading Level:** >95% of responses ≤ 8th grade

### Safety Metrics
- **Crisis Detection:** >95% of crisis keywords detected
- **Legal Advice Boundary:** Zero instances of actual legal advice given (audit)
- **False Positive Rate (Crisis):** <10% (manageable staff burden)

### User Adoption
- **Usage:** 100+ conversations per month (adjust based on population)
- **Return Users:** >30% of users have multi-turn conversations
- **Escalation Rate:** <20% (most questions answered without human)

---

## Risks and Mitigation

### Risk 1: Crisis Missed (False Negative)
**Likelihood:** Low
**Impact:** Critical (harm to individual)
**Mitigation:**
- Layered detection (keywords + LLM)
- Regular audit of conversations (sample for missed crises)
- Prominent "Talk to a person" button
- Clear disclaimer: "For emergencies, call 911"
- Continuous improvement of detection based on audits

### Risk 2: Legal Advice Given (Liability)
**Likelihood:** Low-Medium
**Impact:** High (legal liability, harm to case)
**Mitigation:**
- Strong boundary enforcement (detection + boundary message)
- Explicit prompts: "DO NOT give legal advice"
- Manual audit of 100% of boundary reinforcement conversations
- Clear disclaimer at start of chat
- Legal review of all content before adding to knowledge base

### Risk 3: Hallucination (Incorrect Information)
**Likelihood:** Medium
**Impact:** Medium (misinformation, confusion)
**Mitigation:**
- RAG over approved content only (no general knowledge)
- Citation requirement (forces grounding)
- Response validation (check for citations)
- Fallback: "I don't have information on that" if no relevant docs
- Manual quality audits

### Risk 4: Low Adoption (Users Don't Trust Bot)
**Likelihood:** Medium
**Impact:** Medium (waste of investment)
**Mitigation:**
- Clear branding: "Information Assistant" not "replace human staff"
- "Talk to a person" always available
- Human staff endorsement ("We use this tool to help you get answers faster")
- Pilot with small group, gather feedback, iterate
- Measure and showcase value (response time, helpful rate)

### Risk 5: Multilingual Quality Issues
**Likelihood:** Medium
**Impact:** Medium (misinformation, confusion for non-English speakers)
**Mitigation:**
- Native speaker review of all translations
- Cultural adaptation, not just translation
- Test with native speakers before launch
- Higher manual audit rate for non-English conversations initially

---

## Next Steps After POC Success

1. **Expand Language Support:**
   - Add Vietnamese, Mandarin, Arabic, etc. based on demographics
   - Voice interface for low-literacy users

2. **Proactive Features:**
   - Send reminders (upcoming court dates, program deadlines)
   - Personalized resource recommendations based on conversation

3. **Integration:**
   - Link to case management system (if user provides case number, personalized info)
   - Integrate with family portal (persistent history)

4. **Advanced Analytics:**
   - Identify systemic issues (many questions about same confusing process)
   - Feed insights to policy staff (what families struggle to understand)

5. **Expansion to Other Stakeholders:**
   - Version for victims (different content, stricter boundaries)
   - Version for community partners (referral sources)

---

## Conclusion

The Victim and Family Virtual Assistant fills a critical gap: accessible, 24/7 information for families navigating a confusing system. By combining LangChain RAG over approved content with LangGraph's guardrails and routing, this system can:

- Provide instant, accurate answers to common questions
- Reduce burden on staff (fewer repetitive phone calls)
- Improve family engagement and understanding
- Ensure safety with crisis detection and legal advice boundaries
- Support multiple languages (equity and access)

The architecture prioritizes safety with multi-layered guardrails, human escalation paths, and strict content curation. Success requires careful prompt engineering, rigorous testing (especially for safety), and ongoing monitoring.

**Recommended as third POC**, after establishing RAG infrastructure with prior use cases. Lower complexity than case management, but high value for family engagement and equity.
