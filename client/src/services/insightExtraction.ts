/**
 * Unified Insight Extraction Service
 * ====================================
 * A single AI-powered pipeline that handles insight extraction from both
 * interview transcripts and imported content. Replaces the previous fragmented
 * approach where interviews used extractInsightsAI() and imports used three
 * separate rule-based functions (extractInsightsFromChatgpt, extractInsightsFromUrl,
 * extractInsightsFromText).
 *
 * All source types flow through the same pipeline:
 *   content → (optional chunking) → AI extraction → standardized output
 *   with a rule-based fallback if the AI is unavailable.
 */

import { callMiniMax, isApiKeyConfigured } from './minimax'

// ============================================
// Types
// ============================================

export type SourceType = 'interview' | 'import_url' | 'import_text' | 'import_chatgpt' | 'import_file' | 'note_redistill'

export interface ExtractionContext {
  /** The content to extract insights from */
  content: string
  /** The type of source content */
  sourceType: SourceType
  /** Optional topic title for context */
  topicTitle?: string
  /** Optional topic description for context */
  topicDescription?: string
  /** User's name for personalization */
  userName?: string
  /** User's occupation for personalization */
  occupation?: string
  /** Whether this is a mini session (interview only) */
  isMiniSession?: boolean
  /** Existing verified insights for deduplication */
  existingVerifiedInsights?: Array<{ content: string; confidenceScore: number }>
}

export type NoteFormat = 'full_analysis' | 'brief_summary' | 'decision_framework'

export interface ExtractedInsight {
  /** The insight content text */
  content: string
  /** Confidence score 0-100 */
  confidenceScore: number
  /** Suggested category for the insight */
  category: string
  /** How the insight was extracted */
  extractionMethod: 'ai' | 'fallback'
  /** Which distilled note format this insight came from. Only set on the
   *  note_redistill path; undefined everywhere else. */
  sourceFormat?: NoteFormat
  /** Verbatim sentence/quote from the source document that the insight was
   *  derived from. Helps verify provenance. */
  quotedText?: string
}

// ============================================
// Content Chunking for Large Imports
// ============================================

/** Maximum characters per chunk to stay within token limits (~4 chars/token, target ~3000 tokens of content) */
const MAX_CHUNK_SIZE = 12000

/** Maximum insights per chunk */
const MAX_INSIGHTS_PER_CHUNK = 15

/** Maximum total insights across all chunks */
const MAX_TOTAL_INSIGHTS = 30

/**
 * Split large content into processable chunks, breaking at sentence boundaries.
 */
function chunkContent(content: string): string[] {
  if (content.length <= MAX_CHUNK_SIZE) {
    return [content]
  }

  const chunks: string[] = []
  let remaining = content

  while (remaining.length > 0) {
    if (remaining.length <= MAX_CHUNK_SIZE) {
      chunks.push(remaining)
      break
    }

    // Find a good break point near the limit (sentence or paragraph boundary)
    let breakPoint = MAX_CHUNK_SIZE

    // Try paragraph break first
    const paragraphBreak = remaining.lastIndexOf('\n\n', MAX_CHUNK_SIZE)
    if (paragraphBreak > MAX_CHUNK_SIZE * 0.5) {
      breakPoint = paragraphBreak + 2
    } else {
      // Try sentence break
      const sentenceBreak = remaining.lastIndexOf('. ', MAX_CHUNK_SIZE)
      if (sentenceBreak > MAX_CHUNK_SIZE * 0.5) {
        breakPoint = sentenceBreak + 2
      }
    }

    chunks.push(remaining.substring(0, breakPoint).trim())
    remaining = remaining.substring(breakPoint).trim()
  }

  return chunks
}

// ============================================
// AI-Powered Extraction
// ============================================

/**
 * Build the system prompt based on source type.
 */
function buildSystemPrompt(ctx: ExtractionContext): string {
  const basePrompt = `You are a personal knowledge analyst for me.md, a system that builds verified personal context from AI-guided interviews. Your job is to semantically identify genuine personal insights from content — not keyword-match, but deeply understand what the user is revealing about themselves.

Output ONLY a valid JSON array with no markdown code fences, no explanation, and no commentary.`

  const userContext = ctx.userName
    ? `\nThe user's name is ${ctx.userName}${ctx.occupation ? `, occupation: ${ctx.occupation}` : ''}.`
    : ''

  return basePrompt + userContext
}

/**
 * Build the user prompt adapted to the source type.
 */
function buildUserPrompt(ctx: ExtractionContext, contentChunk: string): string {
  const deduplicationSection = ctx.existingVerifiedInsights && ctx.existingVerifiedInsights.length > 0
    ? `\n## Existing Verified Insights (DO NOT duplicate these)
${ctx.existingVerifiedInsights.map(i => `- "${i.content}" (confidence: ${i.confidenceScore})`).join('\n')}
\nAvoid extracting insights that are semantically equivalent to any of the above.\n`
    : ''

  const topicContext = ctx.topicTitle
    ? ` about "${ctx.topicTitle}"${ctx.topicDescription ? ` (${ctx.topicDescription})` : ''}`
    : ''

  // Adapt extraction instructions based on source type
  let sourceInstructions: string
  let insightRange: string

  switch (ctx.sourceType) {
    case 'interview':
      insightRange = ctx.isMiniSession ? '2-5' : '3-10'
      sourceInstructions = `Extract self-knowledge insights ONLY from statements marked **User:** in the transcript below. Statements marked **Interviewer:** are questions, prompts, reflections, or methodological scaffolding from the AI — they are NOT insights about the user. Skip them entirely.

For example, an interviewer turn such as "I'm here to help you articulate your thoughts on this topic" must NOT be extracted; only the user's reply ("I've been thinking about…") is a valid source.

Extract self-knowledge insights from the following interview session${topicContext}.
${ctx.isMiniSession ? '\nNote: This was a quick mini-session with shorter, more direct answers. Adjust expectations accordingly — even brief self-descriptions can be meaningful insights.\n' : ''}
## Conversation Transcript

${contentChunk}`
      break

    case 'import_chatgpt':
      insightRange = '5-15'
      sourceInstructions = `Extract self-knowledge insights from the following ChatGPT memory export${topicContext}. This content represents structured personal data the user previously shared with ChatGPT. Treat it as high-quality personal knowledge since the user intentionally stored this.

## ChatGPT Memory Content

${contentChunk}`
      break

    case 'import_url':
      insightRange = '3-10'
      sourceInstructions = `Extract self-knowledge insights from the following web page content${topicContext}. This is content the user chose to import, so look for personal relevance — the user likely identifies with or values aspects of this content. Focus on extracting insights that reveal the user's interests, values, or self-identification.

## Web Page Content

${contentChunk}`
      break

    case 'import_text':
    case 'import_file':
      insightRange = '3-12'
      sourceInstructions = `Extract self-knowledge insights from the following ${ctx.sourceType === 'import_file' ? 'uploaded file' : 'text'} content${topicContext}. This is content the user chose to import into their personal knowledge system, so it likely contains personally meaningful information.

## Imported Content

${contentChunk}`
      break

    case 'note_redistill':
      insightRange = ctx.isMiniSession ? '2-5' : '3-10'
      sourceInstructions = `Extract self-knowledge insights from the following distilled session notes${topicContext}. These are previously-generated AI summaries from an interview — they have already been pre-cleaned of headings, section titles, and interview scaffolding.

Each document below is one of three formats of the same session (Full Analysis, Brief Summary, Decision Framework). Read all three together for full context.

Skip any residual noise — bullet labels, "Based on the conversation..." intros, generic filler like "Consider exploring...", and any templated prompts. Only the user's actual statements, examples, principles, and quotes are insight sources.

## Distilled Notes

${contentChunk}`
      break

    default:
      insightRange = '3-10'
      sourceInstructions = `Extract self-knowledge insights from the following content${topicContext}.

## Content

${contentChunk}`
  }

  return `${sourceInstructions}
${deduplicationSection}
## Instructions

Extract ${insightRange} distinct, genuine self-knowledge insights — statements that capture something true and specific about the user.

**Frame each insight in FIRST PERSON**, as if YOU (the user) are stating it about yourself. Begin insights with "I…" — for example: "I value autonomy over stability when making career decisions", NOT "Values autonomy over stability…" (third person). First-person framing is required so the insight reads as direct self-knowledge that any AI assistant can adopt as the user's voice.

Each insight should be:
- A clear, first-person declarative statement beginning with "I" (e.g., "I value autonomy over stability when making career decisions")
- Specific and grounded in what the content actually reveals (not generic truisms)
- Semantically meaningful — capturing genuine personal knowledge, not surface-level keywords
- Useful as portable context for other AI tools to understand and act like the user

Also categorize each insight into one of these categories:
- "identity" — core traits, personality, self-concept
- "skills" — abilities, expertise, professional competencies
- "experiences" — life events, memories, journeys
- "perspectives" — beliefs, opinions, approaches, communication/decision style
- "goals" — aspirations, plans, desired futures

Avoid extracting:
- Generic statements that could apply to anyone (e.g., "I want to be happy")
- Third-person statements about the user (always use first-person "I…")
- Simple restatements of questions or prompts
- Interviewer prompts, reflections, or meta-commentary (anything from **Interviewer:** turns)
- Vague or purely emotional reactions without substance

## Confidence Scoring

Evaluate each insight's confidenceScore (50-95) based on THREE dimensions:

**Conviction** (How strongly/emphatically was this expressed?):
- Low (50-60): Hedged, tentative, or inferred
- Medium (61-75): Stated clearly but without emphasis
- High (76-95): Emphatic, repeated, or emotionally charged

**Specificity** (How precise and detailed is the insight?):
- Low: Broad generalization ("I like helping people")
- Medium: Somewhat specific ("I prefer mentoring junior developers")
- High: Highly specific with context

**Consistency** (Is it reinforced across multiple statements or just mentioned once?):
- Low: Mentioned once in passing
- Medium: Referenced in 2+ related statements
- High: A recurring theme throughout the content

Output format (JSON array only, no wrapping):
[
  { "content": "Insight statement here", "confidenceScore": 75, "category": "identity" }
]`
}

/**
 * Call Claude API for insight extraction from a single chunk.
 */
async function callClaudeForInsights(
  systemPrompt: string,
  userPrompt: string
): Promise<Array<{ content: string; confidenceScore: number; category: string }> | null> {
  if (!isApiKeyConfigured()) return null

  try {
    console.log('[me.md:insight-extraction] Calling Claude API for unified insight extraction')
    const responseText = await callMiniMax({
      messages: [{ role: 'user', content: userPrompt }],
      system: systemPrompt,
      maxTokens: 4096,
    })

    if (!responseText || responseText.trim().length === 0) {
      console.warn('[me.md:insight-extraction] Claude returned empty response.')
      return null
    }

    console.log(`[me.md:insight-extraction] Response received (${responseText.length} chars)`)

    // Clean up markdown code fences
    let cleaned = responseText.trim()
    if (cleaned.startsWith('```json')) {
      cleaned = cleaned.slice(7)
    } else if (cleaned.startsWith('```')) {
      cleaned = cleaned.slice(3)
    }
    if (cleaned.endsWith('```')) {
      cleaned = cleaned.slice(0, -3)
    }
    cleaned = cleaned.trim()

    const parsed = JSON.parse(cleaned)
    if (!Array.isArray(parsed)) return null

    // Validate and normalize structure
    const validCategories = new Set(['identity', 'skills', 'experiences', 'perspectives', 'goals'])

    return parsed
      .filter((item: unknown) => {
        if (typeof item !== 'object' || item === null) return false
        const obj = item as Record<string, unknown>
        return typeof obj.content === 'string' && typeof obj.confidenceScore === 'number'
      })
      .map((item: { content: string; confidenceScore: number; category?: string }) => ({
        content: item.content.substring(0, 500),
        confidenceScore: Math.min(Math.max(item.confidenceScore, 50), 95),
        category: (typeof item.category === 'string' && validCategories.has(item.category))
          ? item.category
          : 'identity',
      }))
      .slice(0, MAX_INSIGHTS_PER_CHUNK)
  } catch (error: unknown) {
    const err = error as { message?: string }
    console.warn(`[me.md:insight-extraction] Failed to parse AI extraction result: ${err.message || 'Unknown error'}`)
    return null
  }
}

// ============================================
// Per-Format LLM Extraction (note_redistill path)
// ============================================

/**
 * Per-format LLM call used by the note_redistill path. Same endpoint and
 * parsing as callClaudeForInsights, but the expected schema includes an
 * optional `quotedText` field that records the verbatim sentence the
 * insight was derived from.
 */
async function callClaudeForSectionInsights(
  systemPrompt: string,
  userPrompt: string,
): Promise<Array<{ content: string; confidenceScore: number; category: string; quotedText?: string }> | null> {
  if (!isApiKeyConfigured()) return null

  try {
    console.log('[me.md:insight-extraction] Calling Claude API for per-section insight extraction')
    const responseText = await callMiniMax({
      messages: [{ role: 'user', content: userPrompt }],
      system: systemPrompt,
      maxTokens: 2048,
    })

    if (!responseText || responseText.trim().length === 0) {
      console.warn('[me.md:insight-extraction] Claude returned empty section response.')
      return null
    }

    console.log(`[me.md:insight-extraction] Section response received (${responseText.length} chars)`)

    let cleaned = responseText.trim()
    if (cleaned.startsWith('```json')) {
      cleaned = cleaned.slice(7)
    } else if (cleaned.startsWith('```')) {
      cleaned = cleaned.slice(3)
    }
    if (cleaned.endsWith('```')) {
      cleaned = cleaned.slice(0, -3)
    }
    cleaned = cleaned.trim()

    const parsed = JSON.parse(cleaned)
    if (!Array.isArray(parsed)) return null

    const validCategories = new Set(['identity', 'skills', 'experiences', 'perspectives', 'goals'])

    return parsed
      .filter((item: unknown) => {
        if (typeof item !== 'object' || item === null) return false
        const obj = item as Record<string, unknown>
        return typeof obj.content === 'string' && typeof obj.confidenceScore === 'number'
      })
      .map((item: { content: string; confidenceScore: number; category?: string; quotedText?: unknown }) => ({
        content: item.content.substring(0, 500),
        confidenceScore: Math.min(Math.max(item.confidenceScore, 50), 95),
        category: (typeof item.category === 'string' && validCategories.has(item.category))
          ? item.category
          : 'identity',
        quotedText: typeof item.quotedText === 'string' ? item.quotedText.substring(0, 500) : undefined,
      }))
      .slice(0, MAX_INSIGHTS_PER_CHUNK)
  } catch (error: unknown) {
    const err = error as { message?: string }
    console.warn(`[me.md:insight-extraction] Failed to parse section AI result: ${err.message || 'Unknown error'}`)
    return null
  }
}

const SECTION_PROMPTS: Record<NoteFormat, string> = {
  full_analysis: `You are extracting insights from the FULL ANALYSIS format of a previously-distilled interview session.

Full Analysis is the longest and most detailed of the three formats. It typically contains:
- ## Core Principles — direct user quotes revealing beliefs, values, and priorities
- ## Mental Models & Frameworks — the user's reasoning patterns and how they think
- ## Key Examples — concrete stories that illustrate the user's perspective

Skip any meta-section, intro line, or instruction prompt that slipped through cleaning. Only the user's actual statements, quotes, examples, and principles are insight sources.`,

  brief_summary: `You are extracting insights from the BRIEF SUMMARY format of a previously-distilled interview session.

Brief Summary is the shortest format. It typically contains:
- ## Key Takeaways — the most important insight statements, already filtered
- ## One Thing to Remember — a single highlighted quote

This format is already highly curated. Most insights you extract will be direct restatements of lines already in the document. Quote them verbatim.`,

  decision_framework: `You are extracting insights from the DECISION FRAMEWORK format of a previously-distilled interview session.

Decision Framework focuses on how the user makes decisions. It typically contains:
- ## Guiding Principles — the rules and beliefs that drive their choices

Skip templated sections like "Decision Criteria", "Red Flags", "Green Lights" — those are scaffolding, not insights. Pull from the user's own statements about what guides their choices.`,
}

/**
 * Run a single per-format extraction with a focused LLM call.
 * Returns insights tagged with the source format.
 */
async function extractInsightsFromNoteSection(
  section: { format: NoteFormat; label: string; content: string },
  baseCtx: ExtractionContext,
): Promise<ExtractedInsight[]> {
  const systemPrompt = buildSystemPrompt(baseCtx)
  const userPrompt = `${SECTION_PROMPTS[section.format]}

## ${section.label}

${section.content}

## Output Format

For each insight, return a JSON object with these fields:
- "content" (string): a clear, declarative first-person statement about the user
- "confidenceScore" (number, 50-95): how strongly the source text supports this
- "category" (string): one of "identity", "skills", "experiences", "perspectives", "goals"
- "quotedText" (string, optional): the verbatim sentence or short passage from the document above that this insight was derived from

Skip any sentence that is templated, instructional, or meta-commentary. Only the user's actual statements belong as insights.

Output ONLY a valid JSON array, no markdown fences, no commentary:
[
  { "content": "...", "confidenceScore": 75, "category": "identity", "quotedText": "..." }
]`

  const result = await callClaudeForSectionInsights(systemPrompt, userPrompt)

  if (result) {
    return result.map((item) => ({
      ...item,
      sourceFormat: section.format,
      extractionMethod: 'ai' as const,
    }))
  }

  // Fallback for this section: rule-based extraction on the cleaned text
  const fallbackResults = extractInsightsFallback({
    ...baseCtx,
    content: section.content,
  })
  return fallbackResults.map((insight) => ({
    ...insight,
    sourceFormat: section.format,
  }))
}

/**
 * Splits a note_redistill context into the three note formats (full_analysis,
 * brief_summary, decision_framework) and runs one focused extraction per
 * format in parallel. Each resulting insight is tagged with its sourceFormat
 * for downstream persistence.
 *
 * The input `ctx.content` is expected to already be in the form returned by
 * buildNoteContentForExtraction — sections delimited by [Full Analysis],
 * [Brief Summary], [Decision Framework] headers.
 */
async function extractInsightsFromNoteSections(ctx: ExtractionContext): Promise<ExtractedInsight[]> {
  console.log(`[me.md:insight-extraction] Starting per-format extraction (sourceType=note_redistill)`)

  // Reconstruct the per-format sections from the labeled content blob.
  const sections: Array<{ format: NoteFormat; label: string; content: string }> = []
  const lines = ctx.content.split('\n')
  let currentLabel: string | null = null
  let currentLines: string[] = []
  let currentFormat: NoteFormat | null = null

  const labelToFormat: Record<string, NoteFormat> = {
    'Full Analysis': 'full_analysis',
    'Brief Summary': 'brief_summary',
    'Decision Framework': 'decision_framework',
  }

  const flush = () => {
    if (currentLabel && currentFormat && currentLines.length > 0) {
      const body = currentLines.join('\n').trim()
      if (body.length > 0) sections.push({ format: currentFormat, label: currentLabel, content: body })
    }
    currentLabel = null
    currentLines = []
    currentFormat = null
  }

  for (const line of lines) {
    const headerMatch = line.match(/^\[(Full Analysis|Brief Summary|Decision Framework)\]$/)
    if (headerMatch) {
      flush()
      currentLabel = headerMatch[1]
      currentFormat = labelToFormat[currentLabel]
      continue
    }
    if (currentLabel) currentLines.push(line)
  }
  flush()

  if (sections.length === 0) {
    console.warn('[me.md:insight-extraction] note_redistill: no labeled sections found in content — falling back to single-pass extraction.')
    // Degenerate case: content wasn't properly labeled. Run the unified pipeline as a safety net.
    return extractInsightsUnifiedPath(ctx)
  }

  console.log(`[me.md:insight-extraction] note_redistill: running extraction across ${sections.length} format(s) in parallel`)

  const sectionResults = await Promise.all(
    sections.map((section) => extractInsightsFromNoteSection(section, ctx)),
  )

  const allInsights: ExtractedInsight[] = []
  for (const result of sectionResults) allInsights.push(...result)

  // Apply confidence penalty to any fallback-tagged insights so reviewers
  // know to scrutinize them.
  const penalized = allInsights.map((insight) =>
    insight.extractionMethod === 'fallback'
      ? applyFallbackConfidencePenalty([insight])[0]
      : insight,
  )

  const deduplicated = deduplicateInsights(penalized, ctx.existingVerifiedInsights)
  console.log(
    `[me.md:insight-extraction] note_redistill extraction complete: ${deduplicated.length} insights ` +
      `(by format: full_analysis=${deduplicated.filter((i) => i.sourceFormat === 'full_analysis').length}, ` +
      `brief_summary=${deduplicated.filter((i) => i.sourceFormat === 'brief_summary').length}, ` +
      `decision_framework=${deduplicated.filter((i) => i.sourceFormat === 'decision_framework').length})`,
  )
  return deduplicated.slice(0, MAX_TOTAL_INSIGHTS)
}

// ============================================
// Rule-Based Fallback Extraction
// ============================================

/**
 * Score a statement for insight-worthiness (rule-based fallback).
 */
function scoreStatement(statement: string, sourceType: SourceType): number {
  const lower = statement.toLowerCase()
  let score: number

  // Base scores differ by source type
  switch (sourceType) {
    case 'interview':
      score = 50
      break
    case 'import_chatgpt':
      score = 45 // ChatGPT memories are already curated personal data
      break
    case 'import_url':
      score = 40 // Web content may not be personal
      break
    default:
      score = 40
  }

  // Strong personal statements
  if (/\b(i am|i believe|i value|i always|i never|i think|i feel|my|i prefer|i tend to)\b/i.test(lower)) {
    score += 15
  }

  // Reasoning/understanding markers
  if (/\b(because|reason|learned|realized|understand|important|matters)\b/i.test(lower)) {
    score += 10
  }

  // Core trait indicators
  if (/\b(core|fundamental|deeply|who i am|trait|personality|character|principle|philosophy)\b/i.test(lower)) {
    score += 10
  }

  // Preference indicators
  if (/\b(prefer|like|enjoy|love|dislike|hate|comfortable|style|approach)\b/i.test(lower)) {
    score += 8
  }

  // Length bonus
  if (statement.length > 60) {
    score += 5
  }

  // Belief/principle keywords (interview-specific boost)
  if (sourceType === 'interview' && /\b(believe|think|feel|value|important|always|never|principle)\b/i.test(lower)) {
    score += 5
  }

  return Math.min(score, 95)
}

/**
 * Categorize a statement by keyword analysis.
 */
function categorizeStatement(statement: string): string {
  const lower = statement.toLowerCase()

  if (/\b(skill|expert|experience|professional|work|career|project|competent|proficien)\b/.test(lower)) return 'skills'
  if (/\b(goal|aspir|dream|plan|future|want to|aim|ambition)\b/.test(lower)) return 'goals'
  if (/\b(learn|grew|journey|story|memory|remember|once|when i was)\b/.test(lower)) return 'experiences'
  if (/\b(think|believe|approach|perspective|opinion|view|prefer|style|method)\b/.test(lower)) return 'perspectives'

  return 'identity'
}

/**
 * Get score threshold based on source type.
 */
function getScoreThreshold(sourceType: SourceType, isMiniSession?: boolean): number {
  switch (sourceType) {
    case 'interview':
      return isMiniSession ? 45 : 55
    case 'import_chatgpt':
      return 45
    case 'import_url':
      return 55
    case 'import_text':
    case 'import_file':
      return 48
    default:
      return 50
  }
}

/**
 * Get max insights based on source type.
 */
function getMaxInsights(sourceType: SourceType): number {
  switch (sourceType) {
    case 'interview':
      return 10
    case 'import_chatgpt':
      return 30
    case 'import_url':
      return 20
    case 'import_text':
    case 'import_file':
      return 25
    default:
      return 15
  }
}

/**
 * Extract insights using rule-based pattern matching (fallback when AI unavailable).
 */
function extractInsightsFallback(ctx: ExtractionContext): ExtractedInsight[] {
  const results: ExtractedInsight[] = []
  const threshold = getScoreThreshold(ctx.sourceType, ctx.isMiniSession)
  const maxInsights = getMaxInsights(ctx.sourceType)
  const minLength = ctx.sourceType === 'interview' && ctx.isMiniSession ? 15 : 20

  // Special handling for ChatGPT structured sections
  if (ctx.sourceType === 'import_chatgpt') {
    // Try to detect structured sections in the content
    const sectionCategoryMap: Record<string, string> = {
      'personal background': 'identity',
      'communication style': 'perspectives',
      'values & beliefs': 'identity',
      'interests & hobbies': 'experiences',
      'professional life': 'skills',
      'decision-making style': 'perspectives',
      'strengths & weaknesses': 'skills',
      'goals & aspirations': 'goals',
      'preferences': 'perspectives',
      'personality traits': 'identity',
    }

    // Try to parse sections from the content
    const sectionRegex = /^##?\s*(.+)$/gm
    let match
    const sectionPositions: Array<{ name: string; start: number }> = []

    while ((match = sectionRegex.exec(ctx.content)) !== null) {
      sectionPositions.push({ name: match[1].trim(), start: match.index + match[0].length })
    }

    if (sectionPositions.length > 0) {
      for (let i = 0; i < sectionPositions.length; i++) {
        const sectionName = sectionPositions[i].name
        const sectionStart = sectionPositions[i].start
        const sectionEnd = i + 1 < sectionPositions.length ? sectionPositions[i + 1].start : ctx.content.length
        const sectionContent = ctx.content.substring(sectionStart, sectionEnd)
        const category = sectionCategoryMap[sectionName.toLowerCase()] || 'identity'

        const statements = sectionContent
          .split(/[.!?\n]+/)
          .map(s => s.replace(/^[-*•]\s*/, '').trim())
          .filter(s => s.length > 20 && s.length < 500)

        for (const statement of statements) {
          const score = scoreStatement(statement, ctx.sourceType)
          if (score >= threshold) {
            results.push({
              content: statement,
              confidenceScore: score,
              category,
              extractionMethod: 'fallback',
            })
          }
        }
      }

      // If we found section-based results, return them
      if (results.length > 0) {
        return deduplicateInsights(results, ctx.existingVerifiedInsights).slice(0, maxInsights)
      }
    }
  }

  // Generic extraction: split content into statements
  const statements = ctx.content
    .split(/[.!?\n]+/)
    .map(s => s.replace(/^[-*•]\s*/, '').trim())
    .filter(s => s.length > minLength && s.length < 500)

  // For interview mini sessions, if no sentences found, try whole message chunks
  if (statements.length === 0 && ctx.sourceType === 'interview' && ctx.isMiniSession) {
    const paragraphs = ctx.content.split(/\n+/).filter(p => p.trim().length > minLength)
    for (const p of paragraphs) {
      statements.push(p.trim())
    }
  }

  for (const statement of statements) {
    const score = scoreStatement(statement, ctx.sourceType)
    if (score >= threshold) {
      results.push({
        content: statement.substring(0, 500),
        confidenceScore: Math.min(score, 95),
        category: categorizeStatement(statement),
        extractionMethod: 'fallback',
      })
    }
  }

  return deduplicateInsights(results, ctx.existingVerifiedInsights).slice(0, maxInsights)
}

// ============================================
// Deduplication
// ============================================

/**
 * Remove duplicate insights (case-insensitive) and filter out
 * insights that are semantically too similar to existing verified insights.
 */
function deduplicateInsights(
  insights: ExtractedInsight[],
  existingVerified?: Array<{ content: string; confidenceScore: number }>
): ExtractedInsight[] {
  // Remove exact duplicates (case-insensitive)
  const seen = new Set<string>()
  const unique = insights.filter(insight => {
    const key = insight.content.toLowerCase().trim()
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })

  // If we have existing verified insights, filter out near-duplicates
  if (existingVerified && existingVerified.length > 0) {
    const existingLower = existingVerified.map(i => i.content.toLowerCase().trim())
    return unique.filter(insight => {
      const lowerContent = insight.content.toLowerCase().trim()
      // Check for high overlap with existing insights
      return !existingLower.some(existing => {
        // Exact match
        if (existing === lowerContent) return true
        // One contains the other (substring match for significant overlap)
        if (existing.length > 30 && lowerContent.length > 30) {
          if (existing.includes(lowerContent) || lowerContent.includes(existing)) return true
        }
        return false
      })
    })
  }

  return unique
}

// ============================================
// Main Extraction Pipeline
// ============================================

/** Maximum number of AI retry attempts before falling back to rule-based extraction */
const AI_RETRY_ATTEMPTS = 1

/**
 * Attempt a single AI call with retry logic.
 * Returns null only if both the initial call and retry fail.
 */
async function callClaudeForInsightsWithRetry(
  systemPrompt: string,
  userPrompt: string
): Promise<Array<{ content: string; confidenceScore: number; category: string }> | null> {
  // First attempt
  const firstResult = await callClaudeForInsights(systemPrompt, userPrompt)
  if (firstResult) return firstResult

  // Retry logic
  for (let attempt = 1; attempt <= AI_RETRY_ATTEMPTS; attempt++) {
    console.warn(`[me.md:insight-extraction] AI extraction attempt failed, retrying (${attempt}/${AI_RETRY_ATTEMPTS})...`)
    // Brief delay before retry (500ms * attempt)
    await new Promise(resolve => setTimeout(resolve, 500 * attempt))
    const retryResult = await callClaudeForInsights(systemPrompt, userPrompt)
    if (retryResult) {
      console.log(`[me.md:insight-extraction] AI extraction succeeded on retry attempt ${attempt}`)
      return retryResult
    }
  }

  return null
}

/**
 * Apply confidence penalty to fallback-generated insights.
 * Fallback insights are lower quality, so we reduce their confidence scores
 * and cap them to signal that human review is especially important.
 */
function applyFallbackConfidencePenalty(insights: ExtractedInsight[]): ExtractedInsight[] {
  const FALLBACK_CONFIDENCE_PENALTY = 15
  const FALLBACK_MAX_CONFIDENCE = 70

  return insights.map(insight => ({
    ...insight,
    confidenceScore: Math.min(
      Math.max(insight.confidenceScore - FALLBACK_CONFIDENCE_PENALTY, 30),
      FALLBACK_MAX_CONFIDENCE
    ),
  }))
}

/**
 * Returns true when the given content string looks like a genuine user
 * insight — a specific, declarative statement about the user. Returns false
 * when it still reads like meta scaffolding (headings, section labels, AI
 * self-references, interviewer prompts, planning monologues, or templated
 * META section bodies).
 *
 * Used as a final defensive filter at the boundary of the extraction
 * pipeline so leak-through meta never reaches the verification queue —
 * either from a malformed prompt or from a model that echoed the scaffolding.
 */
export function isValidInsightContent(content: string): boolean {
  if (typeof content !== 'string') return false
  const trimmed = content.trim()
  if (trimmed.length < 25 || trimmed.length > 600) return false

  // Transcript markers (check on raw trimmed — the leading-** strip below
  // would otherwise eat them).
  if (/\*\*user:\*\*|\*\*interviewer:\*\*/i.test(trimmed)) return false

  // Strip markdown formatting markers (** / __) at the edges so wrapped
  // headers like "**Core Principles (3-5):**" expose the underlying header
  // for the prefix-based meta rules below. We deliberately do NOT strip
  // leading `#` — that would defeat the heading rule that fires next.
  const normalized = trimmed
    .replace(/^\s*(?:\*\*+|__+)/, '')
    .replace(/(?:\*\*+|__+)\s*$/, '')
    .trim()
  if (normalized.length < 25) return false

  const lower = normalized.toLowerCase()

  // Markdown headings and section labels — never an insight
  if (/^#+\s/.test(normalized)) return false
  if (/[\s]#{2,}\s/.test(normalized)) return false
  if (/\[(?:full analysis|brief summary|decision framework)\]/i.test(normalized)) return false

  // Self-referential intros the LLM sometimes echoes from the prompt
  if (/^(this analysis|this document|this section|this format|this summary|this framework|this note)\b/i.test(normalized)) return false
  if (/^(based on the (conversation|analysis|document|section|format|notes|transcript))\b/i.test(normalized)) return false
  if (/^(here (is|are))\b/i.test(normalized)) return false
  if (/^(below (is|are))\b/i.test(normalized)) return false
  if (/^(the following|these (insights|notes|takeaways|patterns|principles))\b/i.test(normalized)) return false
  if (/^(in (summary|this|the))\b/i.test(normalized)) return false
  if (/^(the (session|conversation|interview))\b/i.test(normalized)) return false
  if (/^(this (session|conversation|interview))\b/i.test(normalized)) return false

  // LLM planning/intent monologues — the model thinking out loud about
  // what it's going to do, never an insight about the user.
  if (/^(i (need|should|must|will|want to|can|am going to))\b/i.test(normalized)) return false
  if (/^(i'?m going to|i'?ll)\b/i.test(normalized)) return false

  // Markdown list markers at the very start (single * / - / +). Bold (**)
  // and emphasis continuation have already been handled by the decoration
  // strip. Catches content like "*i am poor, others are rich".
  if (/^[+\-*]\s|^\*[^*\s]/.test(normalized)) return false

  // Blockquote marker at the very start ("> autonomy matters…").
  if (/^\s*>\s?/.test(normalized)) return false

  // LLM thinking/reasoning trace tags — Qwen-style  块 / 泄漏 of internal
  // monologues. Checks inside the string to also catch embedded traces.
  if (/<\s*think|<\s*\/\s*think\s*>|<\|[^>|]+\|>/i.test(normalized)) return false

  // Descriptive narratives about the session rather than statements about
  // the user. Catches openings like "The user revealed…", "The participant
  // shared…", "The interviewee was…". The trailing verb list avoids
  // false-positives like "The user-friendly approach…" or
  // "The user's primary values…". An optional "(Name)" between the
  // subject and verb is allowed so "The user (Dérik) revealed…" still
  // fires.
  if (
    /^the (user|participant|interviewee)(?:\s*\([^)]+\))?\s+(is|was|has|had|revealed|showed|shared|said|expressed|mentioned|demonstrated|exhibited|displayed|seemed|appeared|tends?|seems?)\b/i.test(
      normalized,
    )
  ) {
    return false
  }

  // Unbalanced quotes or parens are a strong signal of truncated output —
  // the LLM's content ran out mid-token and dropped the closing delimiter.
  // Single quotes are NOT counted here — apostrophes inside words
  // (don't, it's, user's) would false-positive. The double-quote + paren
  // checks cover the common LLM-truncation patterns.
  const doubleQuoteMatches = normalized.match(/"/g)
  if (doubleQuoteMatches && doubleQuoteMatches.length % 2 !== 0) return false
  const openParens = (normalized.match(/\(/g) || []).length
  const closeParens = (normalized.match(/\)/g) || []).length
  if (openParens !== closeParens) return false

  // Output/format labels
  if (/^(output|format|section|document|note)\s*:/i.test(normalized)) return false

  // Interviewer-style meta (Clear-Language / Socratic scaffolding)
  if (/^(i'?m here|i see|let'?s|let me|i'?d like|let us)\b/i.test(normalized)) return false

  // Header patterns (only on short content so natural sentences aren't
  // false-positive-flagged):
  //   - "Core Principles (3-5):" / "Tensions:" — ends with a colon
  //   - "Core Principles (3-5)" — parenthetical quantifier like (3-5), (N), (optional)
  if (normalized.length < 100) {
    // Colon-terminated is almost always a header, never a complete insight
    if (/[:;]\s*$/.test(normalized)) return false
    // Parenthetical quantifier / placeholder is the model echoing the prompt
    if (/\(\s*(?:\d+\s*-\s*\d+|\d+\+?|n|optional|example(?:s)?|if applicable)\s*\)/i.test(normalized)) return false

    const META_HEADERS = [
      'key concepts that appeared',
      'further exploration is needed',
      'areas of tension or uncertainty',
      'decision-making patterns observed',
      'no specific examples',
      'no explicit red flags',
      'no explicit guiding principles',
      'decision criteria',
      'red flags',
      'green lights',
      'one thing to remember',
      'mental models & frameworks',
      'questions for further exploration',
      'tl;dr',
      'key takeaways',
    ]
    for (const h of META_HEADERS) {
      if (lower.startsWith(h)) return false
    }
  }

  // JSON-shaped output — the LLM occasionally returns raw JSON when the
  // prompt asks for one
  if (normalized.startsWith('{') || normalized.startsWith('[')) return false

  return true
}

/**
 * Extract insights from content using the unified pipeline.
 *
 * This is the main entry point for all insight extraction. It:
 * 1. Chunks large content if needed
 * 2. Attempts AI-powered extraction with Claude (with retry)
 * 3. Falls back to rule-based extraction if AI is unavailable (with logging/flagging)
 * 4. Deduplicates results and applies consistent formatting
 * 5. Filters out anything that still reads like meta scaffolding (headings,
 *    self-references, interviewer prompts, templated headers)
 *
 * @param ctx - Extraction context with content, source type, and optional metadata
 * @returns Array of standardized extracted insights
 */
export async function extractInsights(ctx: ExtractionContext): Promise<ExtractedInsight[]> {
  if (!ctx.content || ctx.content.trim().length === 0) {
    return []
  }

  const raw =
    ctx.sourceType === 'note_redistill'
      ? await extractInsightsFromNoteSections(ctx)
      : await extractInsightsUnifiedPath(ctx)

  // Final defensive filter: drop any insight that still looks like meta.
  // This is intentionally applied at the boundary so every pipeline
  // (re-extract, distill, import_url, import_file, …) gets the guard.
  const validated: ExtractedInsight[] = []
  for (const insight of raw) {
    if (isValidInsightContent(insight.content)) {
      validated.push(insight)
      continue
    }
    console.warn(
      `[me.md:insight-extraction] Dropping meta-like insight from extraction pipeline ` +
        `(sourceType=${ctx.sourceType}, topic="${ctx.topicTitle || 'unknown'}"): ` +
        `"${insight.content.slice(0, 80)}"`,
    )
  }

  // Drop near-duplicate insights within this single extraction. The model
  // sometimes returns the same idea twice in different words ("I value
  // autonomy over stability in my career decisions" / "I prioritize
  // autonomy rather than stability in my career"). Token-set Jaccard
  // catches the obvious cases; anything subtler falls to the verified
  // dedup pass that runs at insertion time.
  return dedupeExtractedInsights(validated)
}

/**
 * Tokenize an insight for similarity comparison: lowercase, split on
 * whitespace and punctuation, drop short stop-words.
 */
function tokenizeForSimilarity(content: string): Set<string> {
  const STOP_WORDS = new Set([
    'a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'to', 'for',
    'with', 'as', 'at', 'by', 'is', 'it', 'i', 'my', 'me', 'that', 'this',
  ])
  const tokens = content
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(t => t.length > 1 && !STOP_WORDS.has(t))
  return new Set(tokens)
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  let intersection = 0
  for (const t of a) {
    if (b.has(t)) intersection++
  }
  const union = a.size + b.size - intersection
  return union === 0 ? 0 : intersection / union
}

/**
 * Drop insights that are near-duplicates of each other within a single
 * extraction batch. Keeps the higher-confidence one when two collide.
 * Threshold tuned so genuinely-distinct insights (different topics or
 * claims) survive but obvious rephrasings collapse.
 */
const SEMANTIC_DEDUP_THRESHOLD = 0.7

export function dedupeExtractedInsights(
  insights: ExtractedInsight[],
): ExtractedInsight[] {
  if (insights.length <= 1) return insights

  const tokenSets = insights.map(i => tokenizeForSimilarity(i.content))
  const kept: ExtractedInsight[] = []
  const keptTokens: Set<string>[] = []

  for (let i = 0; i < insights.length; i++) {
    let isDup = false
    for (let j = 0; j < kept.length; j++) {
      const sim = jaccardSimilarity(tokenSets[i], keptTokens[j])
      if (sim >= SEMANTIC_DEDUP_THRESHOLD) {
        isDup = true
        if (insights[i].confidenceScore > kept[j].confidenceScore) {
          kept[j] = insights[i]
          keptTokens[j] = tokenSets[i]
        }
        break
      }
    }
    if (!isDup) {
      kept.push(insights[i])
      keptTokens.push(tokenSets[i])
    }
  }
  return kept
}

/**
 * The standard multi-chunk extraction path (interview, import_text, etc.).
 * Also the safety-net fallback for note_redistill when the labeled content
 * is malformed.
 */
async function extractInsightsUnifiedPath(ctx: ExtractionContext): Promise<ExtractedInsight[]> {
  console.log(`[me.md:insight-extraction] Starting unified extraction: sourceType=${ctx.sourceType}, contentLength=${ctx.content.length}`)

  // For interviews, format the content as a conversation transcript if it isn't already
  const processedContent = ctx.content

  // Chunk large content
  const chunks = chunkContent(processedContent)
  console.log(`[me.md:insight-extraction] Content split into ${chunks.length} chunk(s)`)

  // Track whether fallback was used for any chunk
  let fallbackUsedForChunks = 0
  let aiSuccessForChunks = 0

  // Try AI extraction first
  if (isApiKeyConfigured()) {
    try {
      const systemPrompt = buildSystemPrompt(ctx)
      const allInsights: ExtractedInsight[] = []

      for (const chunk of chunks) {
        const userPrompt = buildUserPrompt(ctx, chunk)
        // Use retry-enabled version: tries once, then retries up to AI_RETRY_ATTEMPTS times
        const aiResult = await callClaudeForInsightsWithRetry(systemPrompt, userPrompt)

        if (aiResult) {
          aiSuccessForChunks++
          for (const item of aiResult) {
            allInsights.push({
              ...item,
              extractionMethod: 'ai',
            })
          }
        } else {
          // AI failed even after retry for this chunk — use fallback with logging
          fallbackUsedForChunks++
          console.warn(`[me.md:insight-extraction] FALLBACK ACTIVATED for chunk (sourceType=${ctx.sourceType}, topic="${ctx.topicTitle || 'unknown'}"): AI extraction failed after ${AI_RETRY_ATTEMPTS + 1} attempt(s). Using rule-based extraction — insights may be lower quality.`)
          const fallbackResults = extractInsightsFallback({
            ...ctx,
            content: chunk,
          })
          // Apply confidence penalty to fallback insights
          const penalizedResults = applyFallbackConfidencePenalty(fallbackResults)
          allInsights.push(...penalizedResults)
        }

        // Stop if we have enough insights
        if (allInsights.length >= MAX_TOTAL_INSIGHTS) break
      }

      // Log summary of extraction methods used
      if (fallbackUsedForChunks > 0) {
        console.warn(`[me.md:insight-extraction] EXTRACTION SUMMARY: ${aiSuccessForChunks} chunk(s) via AI, ${fallbackUsedForChunks} chunk(s) via fallback. Fallback insights have reduced confidence scores and are flagged with extractionMethod='fallback'.`)
      }

      if (allInsights.length > 0) {
        const deduplicated = deduplicateInsights(allInsights, ctx.existingVerifiedInsights)
        console.log(`[me.md:insight-extraction] Extraction complete: ${deduplicated.length} insights (${deduplicated.filter(i => i.extractionMethod === 'ai').length} AI, ${deduplicated.filter(i => i.extractionMethod === 'fallback').length} fallback)`)
        return deduplicated.slice(0, MAX_TOTAL_INSIGHTS)
      }
    } catch (error: unknown) {
      const err = error as { message?: string }
      console.warn(`[me.md:insight-extraction] AI extraction pipeline failed entirely, using full fallback: ${err.message || 'Unknown error'}`)
    }
  } else {
    console.warn(`[me.md:insight-extraction] AI is not available (no API key configured). All insights will use rule-based fallback.`)
  }

  // Fallback to rule-based extraction for ALL content
  console.warn(`[me.md:insight-extraction] FULL FALLBACK ACTIVATED for sourceType=${ctx.sourceType}, topic="${ctx.topicTitle || 'unknown'}": All insights generated via rule-based extraction.`)
  const fallbackResults = extractInsightsFallback(ctx)
  // Apply confidence penalty to all fallback insights
  const penalizedResults = applyFallbackConfidencePenalty(fallbackResults)
  console.log(`[me.md:insight-extraction] Fallback extraction complete: ${penalizedResults.length} insights (all flagged as fallback with reduced confidence)`)
  return penalizedResults
}

/**
 * Helper to format interview messages into a conversation transcript string.
 * Use this when preparing interview content for the unified extraction service.
 */
export function formatInterviewTranscript(
  userMessages: Array<{ role: string; content: string }>,
  assistantMessages: Array<{ role: string; content: string }>
): string {
  const lines: string[] = []
  const maxMessages = Math.max(userMessages.length, assistantMessages.length)
  for (let i = 0; i < maxMessages; i++) {
    if (i < assistantMessages.length) {
      lines.push(`**Interviewer:** ${assistantMessages[i].content}`)
    }
    if (i < userMessages.length) {
      lines.push(`**User:** ${userMessages[i].content}`)
    }
  }
  return lines.join('\n\n')
}

// ============================================
// Note Content Cleaner (for note_redistill)
// ============================================

/**
 * Section headings in distilled notes whose body is META — instructional
 * prompts, interview scaffolding, or templated filler — rather than
 * genuine user-derived insight content. Drop these sections entirely.
 */
const NOTE_META_SECTION_NAMES: ReadonlySet<string> = new Set([
  'Context',
  'TL;DR',
  'Decision Context',
  'Decision Criteria',
  'Red Flags',
  'Green Lights',
  'Open Questions & Tensions',
  'Questions for Further Exploration',
])

/**
 * Lines that read like instructional filler rather than insight content —
 * intros that introduce a list, or open prompts addressed to the reader.
 * Drop standalone occurrences (only when the trimmed line is short).
 */
const NOTE_INSTRUCTIONAL_PHRASES: readonly string[] = [
  'Based on the conversation, the following',
  'Key concepts that appeared frequently',
  'Further exploration is needed',
  'No specific examples were shared',
  'No explicit',
  'Consider exploring',
  'You\'ve previously explored topics like',
  'I see you\'ve already explored topics like',
]

interface NoteLikeContent {
  contentFullAnalysis?: string | null
  contentBriefSummary?: string | null
  contentDecisionFramework?: string | null
}

/**
 * Strip one markdown document: drop all heading lines, drop the bodies of
 * META sections, and drop short instructional intros. Returns plain body.
 */
function stripOneNoteDoc(doc: string | null | undefined): string {
  if (!doc) return ''

  const lines = doc.split('\n')
  const out: string[] = []
  let inMetaSection = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const headingMatch = line.match(/^#{1,6}\s+(.+?)\s*$/)

    if (headingMatch) {
      const headingText = headingMatch[1].trim()
      // Strip the top-level document title (e.g., "Full Analysis: Career Goals")
      inMetaSection = NOTE_META_SECTION_NAMES.has(headingText)
      continue
    }

    if (inMetaSection) continue

    const trimmed = line.trim()
    if (trimmed.length === 0) {
      // Collapse consecutive blank lines
      if (out.length > 0 && out[out.length - 1].trim().length === 0) continue
      out.push('')
      continue
    }

    // Drop short standalone instructional intros
    if (trimmed.length < 120) {
      const lower = trimmed.toLowerCase()
      const isInstructional = NOTE_INSTRUCTIONAL_PHRASES.some(p => lower.startsWith(p.toLowerCase()))
      if (isInstructional) continue
    }

    out.push(line)
  }

  // Collapse trailing blank lines
  while (out.length > 0 && out[out.length - 1].trim().length === 0) out.pop()
  return out.join('\n')
}

/**
 * Build a cleaned extraction input from a distilled session note.
 *
 * Joins the three note formats (Full Analysis, Brief Summary, Decision
 * Framework) after stripping headings, document titles, META sections
 * (Context, TL;DR, Decision Context/Criteria, Red/Green Flags, Open
 * Questions & Tensions, Questions for Further Exploration), and short
 * instructional intros.
 *
 * Use this instead of the raw interview transcript when re-running
 * extraction — the transcript inherently contains conversational
 * scaffolding, no matter how the prompt is worded.
 */
export function buildNoteContentForExtraction(note: NoteLikeContent): string {
  const parts = [
    ['Full Analysis', stripOneNoteDoc(note.contentFullAnalysis)],
    ['Brief Summary', stripOneNoteDoc(note.contentBriefSummary)],
    ['Decision Framework', stripOneNoteDoc(note.contentDecisionFramework)],
  ]
    .filter(([, body]) => body.length > 0)
    .map(([label, body]) => `[${label}]\n${body}`)

  return parts.join('\n\n')
}

/**
 * Like buildNoteContentForExtraction but returns the three formats as
 * separate, non-empty sections. Used by extractInsightsFromNoteSections to
 * drive one focused LLM call per format with clean source attribution.
 */
export function splitNoteIntoSections(
  note: NoteLikeContent,
): Array<{ format: NoteFormat; label: string; content: string }> {
  const sections: Array<{ format: NoteFormat; label: string; content: string }> = []

  const full = stripOneNoteDoc(note.contentFullAnalysis)
  if (full) sections.push({ format: 'full_analysis', label: 'Full Analysis', content: full })

  const brief = stripOneNoteDoc(note.contentBriefSummary)
  if (brief) sections.push({ format: 'brief_summary', label: 'Brief Summary', content: brief })

  const framework = stripOneNoteDoc(note.contentDecisionFramework)
  if (framework) sections.push({ format: 'decision_framework', label: 'Decision Framework', content: framework })

  return sections
}
