# Research: Taxonomy of Text-Rewriting / Editing Tasks for Writing Assistants

## Summary

The NLP community has converged on **~8–12 top-level task families** for text rewriting, well-documented across four major benchmarks: **IteraTeR** (Grammarly/UMN), **CoEdIT** (Grammarly), **EditEval** (Meta AI), and **RewriteLM / OpenRewriteEval** (Google Research). Below these top-level families sit dozens of fine-grained sub-tasks, most comprehensively catalogued by **StylePTB** (CMU), which defines 21 atomic style constructs across lexical, syntactic, semantic, and thematic dimensions. These families and sub-tasks provide a solid seed set for building rewriting task taxonomies for writing assistants.

---

## Task Families (Consolidated Taxonomy)

### 1. Grammar & Fluency (Proofreading)
Fix spelling, grammar, punctuation, subject-verb agreement errors.
- **Seed datasets:** JFLEG (Napoles et al. 2017), BEA-2019 GEC shared task, IteraTeR "Fluency" category
- **Example instructions:** "Fix the grammar in this paragraph", "Correct any spelling mistakes"
- **Sub-tasks:** spelling correction, punctuation fix, tense consistency, article correction, subject-verb agreement

### 2. Clarity & Readability (Copyediting)
Improve word choice, remove ambiguity, restructure confusing sentences.
- **Seed datasets:** IteraTeR "Clarity" category (most frequent edit type across all domains)
- **Example instructions:** "Make this easier to understand", "Remove ambiguity", "Clarify the main point"
- **Sub-tasks:** word-choice improvement, sentence restructuring, removing redundancy, disambiguation

### 3. Coherence & Flow
Improve logical flow between sentences, fix dangling references, reorder paragraphs.
- **Seed datasets:** IteraTeR "Coherence" category, CoEdIT coherence subset
- **Example instructions:** "Make this paragraph more coherent", "Improve the logical flow", "Fix the transition between paragraphs"
- **Sub-tasks:** transition improvement, reference resolution, logical reordering, paragraph restructuring

### 4. Simplification
Reduce reading level, shorten sentences, replace jargon with plain language.
- **Seed datasets:** TurkCorpus (Xu et al. 2016), ASSET (Alva-Manchego et al. 2020), EditEval "TRK" + "AST" splits
- **Example instructions:** "Simplify this for a general audience", "Rewrite at a 6th-grade reading level"
- **Sub-tasks:** lexical simplification, sentence splitting, jargon removal, passive-to-active voice

### 5. Concision / Compression
Remove filler, tighten prose, shorten without losing meaning.
- **Seed datasets:** OpenRewriteEval D_Shorten, Google Sentence Compression
- **Example instructions:** "Make this more concise", "Cut this in half", "Remove unnecessary words"
- **Sub-tasks:** filler-word removal, sentence compression, redundancy elimination, bullet-pointing

### 6. Elaboration / Expansion
Add detail, examples, or supporting information.
- **Seed datasets:** OpenRewriteEval D_Elaborate, FRUIT (Iv et al. 2022), StylePTB "Information Addition"
- **Example instructions:** "Expand on this idea", "Add supporting examples", "Elaborate with more detail"
- **Sub-tasks:** example addition, explanation expansion, context enrichment, detail injection

### 7. Formality / Register Transfer
Shift between formal ↔ informal, professional ↔ casual register.
- **Seed datasets:** GYAFC (Rao & Tetreault 2018), OpenRewriteEval D_Formality, CoEdIT formality subset
- **Example instructions:** "Make this more formal", "Rewrite in a casual/conversational tone", "Write this as a professional email"
- **Sub-tasks:** contraction expansion/insertion, vocabulary register shift, pronoun formality, hedging addition/removal

### 8. Tone & Style Transfer
Change emotional tone (friendly, assertive, empathetic, urgent) or writing style.
- **Seed datasets:** Politeness Transfer (Wang et al. 2022), CoEdIT style subset, IteraTeR "Style" category
- **Example instructions:** "Make this sound more friendly", "Rewrite with a more assertive tone", "Make this more empathetic"
- **Sub-tasks:** sentiment shift, politeness transfer, urgency injection, humor addition, confidence boosting

### 9. Paraphrasing
Restate the same meaning with different words and structure.
- **Seed datasets:** OpenRewriteEval D_Paraphrase, MRPC, QQP, ParaNMT
- **Example instructions:** "Rephrase this in your own words", "Say the same thing differently", "Paraphrase this paragraph"
- **Sub-tasks:** lexical paraphrase, structural paraphrase, back-translation paraphrase

### 10. Neutralization / De-biasing
Remove subjective bias, POV language, or loaded framing.
- **Seed datasets:** WNC (Pryzant et al. 2020), EditEval neutralization split
- **Example instructions:** "Make this more neutral/objective", "Remove the biased language", "Write from a neutral point of view"
- **Sub-tasks:** hedge insertion, opinion-marker removal, balanced framing, weasel-word removal

### 11. Summarization
Condense longer text into a shorter summary preserving key information.
- **Seed datasets:** CNN/DailyMail, XSum, Multi-News, BillSum
- **Example instructions:** "Summarize this in 2 sentences", "Write a TL;DR", "Extract the key points"
- **Sub-tasks:** extractive summary, abstractive summary, headline generation, key-point extraction

### 12. Humanization / De-AI-ification
Rewrite AI-generated text to sound more natural, varied, and human-authored.
- **Seed datasets:** No major public benchmark yet; commercial tools (ProofreaderPro, HumanText.pro) offer this
- **Example instructions:** "Make this sound less robotic", "Rewrite so it doesn't sound AI-generated", "Add natural variation"
- **Sub-tasks:** cliché elimination, rhythm variation, hedging naturalization, reducing over-structured prose, removing "magic adverbs" and tricolon patterns

---

## Fine-Grained Atomic Transfers (from StylePTB)

StylePTB (Lyu et al., NAACL 2021) defines **21 atomic style constructs** across 4 categories, with 59,767 paired sentences:

| Category | Transfer | Count |
|----------|---------|-------|
| **Lexical** | Noun synonym replacement | 5,948 |
| | Noun antonym replacement | 2,227 |
| | Verb synonym replacement | 2,574 |
| | Verb antonym replacement | 1,284 |
| | ADJ synonym replacement | 434 |
| | ADJ antonym replacement | 1,146 |
| | Most frequent synonym replacement | 4,722 |
| | Least frequent synonym replacement | 7,112 |
| **Syntax** | To future tense | 7,272 |
| | To present tense | 4,365 |
| | To past tense | 4,422 |
| | Active ↔ passive voice | 2,808 |
| | PP front ↔ back (prepositional phrase reorder) | 467 |
| **Semantics** | ADJ/ADV removal | 4,863 |
| | PP removal | 4,767 |
| | Substatement removal | 1,345 |
| | Information addition | 2,114 |
| **Thematics** | Verb/action emphasis | 1,201 |
| | Adjective emphasis | 696 |

Key finding: **72% of GYAFC formality transfers** and **82% of Yelp sentiment transfers** can be decomposed into compositions of these 21 atomic transfers.

---

## Composite / Multi-Edit Tasks

Real-world editing often chains multiple atomic operations. Key examples from the literature:

- **GEC + Paraphrase + Simplification** — CoEdIT-Composite showed this works well with instruction tuning
- **Tense change + Voice change** — StylePTB compositional benchmark
- **Tense change + PP removal** — StylePTB compositional benchmark
- **Formalize + Shorten** — Common in professional writing (emails → executive summaries)
- **Clarify + Expand** — Common in educational writing
- **Neutralize + Formalize** — Common in Wikipedia and journalistic editing

---

## Seed Instruction Ideas (for dataset construction)

### Proofreading
- "Fix the grammar and spelling errors"
- "Correct any punctuation mistakes"
- "Fix the subject-verb agreement issues"

### Clarity
- "Make this clearer and easier to understand"  
- "Remove the ambiguity in the second sentence"
- "Rewrite this so a non-expert can understand it"

### Concision
- "Make this 50% shorter without losing key info"
- "Remove filler words and tighten the prose"
- "Rewrite as bullet points"

### Formality
- "Rewrite this email in a professional tone"
- "Make this more casual and conversational"
- "Convert this informal note to formal business writing"

### Tone
- "Make this sound more encouraging and supportive"
- "Rewrite with more urgency"
- "Make this less aggressive and more diplomatic"

### Simplification
- "Simplify this for an 8th-grade reading level"
- "Replace technical jargon with plain language"
- "Break long sentences into shorter ones"

### Elaboration
- "Expand this into a full paragraph with examples"
- "Add more context and supporting details"
- "Flesh out the argument with evidence"

### Paraphrase
- "Say the same thing in completely different words"
- "Rephrase this without changing the meaning"
- "Rewrite this from scratch while keeping the core message"

### Neutralization
- "Remove any opinion or bias from this paragraph"
- "Rewrite from a neutral, encyclopedic perspective"
- "Replace loaded language with neutral alternatives"

### Humanization
- "Make this sound like a real person wrote it"
- "Add natural voice and personality"
- "Remove the AI-sounding patterns"

---

## Sources

### Kept
- **IteraTeR** (Grammarly, ACL 2022) — [Blog](https://www.grammarly.com/blog/engineering/introducing-iterater/) | [Paper](https://arxiv.org/abs/2203.03802) — Primary taxonomy of edit intentions: Fluency, Clarity, Coherence, Style. Multi-domain (Wikipedia, ArXiv, Wikinews). Foundation for CoEdIT.
- **CoEdIT** (Grammarly, EMNLP 2023) — [Blog](https://www.grammarly.com/blog/engineering/coedit-text-editing/) | [Paper](https://arxiv.org/abs/2305.09857) — Instruction-tuned text editing model covering GEC, coherence, clarity, simplification, paraphrasing, formality, neutralization. 60× smaller than GPT-3 but outperforms it.
- **EditEval** (Meta AI, CoNLL 2024) — [Paper](https://arxiv.org/abs/2209.13331) | [GitHub](https://github.com/facebookresearch/EditEval) — Instruction-based benchmark spanning fluency (JFLEG), simplification (TurkCorpus, ASSET), neutralization (WNC), information update (FRUIT, WikiFactCheck).
- **RewriteLM / OpenRewriteEval** (Google Research, AAAI 2024) — [Paper](https://arxiv.org/html/2305.15685v2) — Cross-sentence rewriting benchmark with 6 subtasks: Formality, Paraphrase, Shorten, Elaborate, MixedWiki, MixedOthers. 1,629 human-annotated examples.
- **StylePTB** (CMU, NAACL 2021) — [Paper](https://aclanthology.org/2021.naacl-main.171/) — 21 fine-grained atomic style constructs across lexical/syntax/semantic/thematic dimensions. 59,767 sentence pairs + 35,887 compositional pairs.
- **GYAFC** (Rao & Tetreault, NAACL 2018) — [Paper](https://aclanthology.org/N18-1012.pdf) — Gold standard for formality style transfer (formal ↔ informal).
- **Human-AI Collaborative Taxonomy** (Lee et al., 2024) — [Paper](https://arxiv.org/abs/2406.18675) — Domain-specific writing assistant taxonomy construction; identified verbosity, clarity, coherence, and style as top professional editing needs.

### Dropped
- Grammarly.com product page — marketing copy, no research value
- editGPT product page — no taxonomy detail
- ProWritingAid features page — product feature list, not research
- Microsoft 365 AI writing page — product marketing

---

## Gaps

1. **Humanization/De-AI benchmarks:** No established academic benchmark exists for detecting and removing AI-generated writing patterns. This is a growing commercial need but under-researched in NLP.
2. **Domain-specific taxonomies:** Most benchmarks cover general writing. Profession-specific editing (legal, medical, marketing) has distinct requirements that current taxonomies don't capture well (Lee et al. 2024 flagged this).
3. **Multi-paragraph / document-level editing:** Most benchmarks operate at sentence or paragraph level. Document-level coherence editing, restructuring, and organizational revision are under-represented.
4. **Evaluation metrics for subjective tasks:** Tone transfer, humanization, and style tasks lack reliable automatic evaluation — human evaluation remains the gold standard but doesn't scale.
5. **Composite task benchmarks:** While StylePTB explored compositions, there's no large-scale benchmark for real-world multi-edit chains (e.g., "formalize + shorten + add citations").

### Suggested next steps
- Build a humanization benchmark by collecting AI-generated text → human-edited pairs
- Extend IteraTeR/CoEdIT taxonomy to include domain-specific sub-categories
- Create composite-task evaluation sets that chain 2–3 atomic operations
