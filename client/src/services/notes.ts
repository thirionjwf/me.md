import { eq, and, desc, ne } from 'drizzle-orm'
import { scheduleSave } from '@/db/persistence'
import { LOCAL_USER_ID } from '@/contexts/UserContext'
import { notes, sessions, topics, messages, insights, conceptNodes, users } from '@/db/schema'
import {
  generateFullAnalysisAI,
  generateBriefSummaryAI,
  generateDecisionFrameworkAI,
  generateJsonContentAI,
  type DistillationContext,
} from './ai'
import { extractInsights, formatInterviewTranscript, type ExtractionContext, buildNoteContentForExtraction } from './insightExtraction'
import { getSession } from './sessions'
import { deleteInsight, clearAllInsights } from './insights'
type Db = any // Drizzle sql.js instance

// ============================================
// Distillation Generation Functions (fallback)
// ============================================

interface MessageData {
  role: string
  content: string
  isBookmarked?: boolean | number | null
}

function generateFullAnalysis(
  topicTitle: string,
  topicDescription: string | null,
  userMessages: MessageData[],
  assistantMessages: MessageData[]
): string {
  const userContent = userMessages.map(m => m.content).join('\n\n')

  const keyQuotes = userMessages
    .filter(m => m.content.length > 30)
    .slice(0, 5)
    .map(m => {
      const sentences = m.content.split(/[.!?]+/).filter(s => s.trim().length > 20)
      return sentences[0]?.trim() || m.content.substring(0, 150)
    })

  const allWords = userContent.toLowerCase().split(/\s+/)
  const meaningfulWords = allWords.filter(w => w.length > 5)
  const wordFreq: Record<string, number> = {}
  meaningfulWords.forEach(w => {
    const clean = w.replace(/[^a-z]/g, '')
    if (clean.length > 5) {
      wordFreq[clean] = (wordFreq[clean] || 0) + 1
    }
  })
  const topConcepts = Object.entries(wordFreq)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8)
    .map(([word]) => word)

  let analysis = `# Full Analysis: ${topicTitle}\n\n`

  analysis += `## Context\n\n`
  analysis += `This analysis distills insights from an interview session about **${topicTitle}**`
  if (topicDescription) {
    analysis += ` — ${topicDescription}`
  }
  analysis += `.\n\n`
  analysis += `The session covered ${userMessages.length} user responses across ${userMessages.length + assistantMessages.length} total exchanges.\n\n`

  analysis += `## Core Principles\n\n`
  if (keyQuotes.length > 0) {
    analysis += `Based on the conversation, the following core principles emerged:\n\n`
    keyQuotes.slice(0, 3).forEach((quote, i) => {
      analysis += `${i + 1}. > "${quote}"\n\n`
    })
  } else {
    analysis += `Further exploration is needed to identify core principles in this area.\n\n`
  }

  analysis += `## Mental Models & Frameworks\n\n`
  if (topConcepts.length > 0) {
    analysis += `Key concepts that appeared frequently in the discussion:\n\n`
    topConcepts.forEach(concept => {
      analysis += `- **${concept}**: Referenced in the context of ${topicTitle}\n`
    })
    analysis += `\n`
  }

  const frameworkPatterns = userMessages
    .filter(m => m.content.includes('when') || m.content.includes('because') || m.content.includes('always') || m.content.includes('usually') || m.content.includes('tend to'))
    .slice(0, 3)

  if (frameworkPatterns.length > 0) {
    analysis += `Decision-making patterns observed:\n\n`
    frameworkPatterns.forEach(m => {
      const excerpt = m.content.substring(0, 200).trim()
      analysis += `- "${excerpt}${m.content.length > 200 ? '...' : ''}"\n`
    })
    analysis += `\n`
  }

  analysis += `## Key Examples\n\n`
  const exampleMessages = userMessages.filter(m =>
    m.content.includes('example') || m.content.includes('time when') ||
    m.content.includes('instance') || m.content.includes('remember') ||
    m.content.includes('experience') || m.content.length > 100
  ).slice(0, 3)

  if (exampleMessages.length > 0) {
    exampleMessages.forEach((m, i) => {
      const excerpt = m.content.substring(0, 300).trim()
      analysis += `### Example ${i + 1}\n`
      analysis += `> "${excerpt}${m.content.length > 300 ? '...' : ''}"\n\n`
    })
  } else {
    analysis += `No specific examples were shared during this session. Consider exploring concrete experiences in a follow-up session.\n\n`
  }

  analysis += `## Open Questions & Tensions\n\n`

  const uncertainMessages = userMessages.filter(m =>
    m.content.includes('not sure') || m.content.includes("don't know") ||
    m.content.includes('maybe') || m.content.includes('complicated') ||
    m.content.includes('but') || m.content.includes('however') ||
    m.content.includes('on the other hand') || m.content.includes('tension')
  ).slice(0, 3)

  if (uncertainMessages.length > 0) {
    analysis += `Areas of tension or uncertainty identified:\n\n`
    uncertainMessages.forEach(m => {
      const excerpt = m.content.substring(0, 200).trim()
      analysis += `- "${excerpt}${m.content.length > 200 ? '...' : ''}"\n`
    })
    analysis += `\n`
  }

  analysis += `### Questions for Further Exploration\n\n`
  analysis += `- How does this perspective on ${topicTitle} connect to other areas of life?\n`
  analysis += `- What would change if circumstances were different?\n`
  analysis += `- Are there counterexamples that challenge the principles identified above?\n`

  return analysis
}

function generateBriefSummary(
  topicTitle: string,
  userMessages: MessageData[],
  _assistantMessages: MessageData[]
): string {
  let summary = `# Brief Summary: ${topicTitle}\n\n`

  summary += `## TL;DR\n\n`
  summary += `Interview session covering ${topicTitle} with ${userMessages.length} responses. `

  if (userMessages.length > 0) {
    const firstResponse = userMessages[0].content.substring(0, 150).trim()
    summary += `The conversation began with: "${firstResponse}${userMessages[0].content.length > 150 ? '...' : ''}"\n\n`
  }

  summary += `## Key Takeaways\n\n`
  const takeaways = userMessages
    .filter(m => m.content.length > 30)
    .slice(0, 5)
    .map(m => {
      const sentences = m.content.split(/[.!?]+/).filter(s => s.trim().length > 15)
      return sentences[0]?.trim() || m.content.substring(0, 100).trim()
    })

  takeaways.forEach((takeaway, i) => {
    summary += `${i + 1}. ${takeaway}\n`
  })
  summary += `\n`

  summary += `## One Thing to Remember\n\n`
  const keyMessage = [...userMessages].sort((a, b) => b.content.length - a.content.length)[0]
  if (keyMessage) {
    const excerpt = keyMessage.content.substring(0, 200).trim()
    summary += `> "${excerpt}${keyMessage.content.length > 200 ? '...' : ''}"\n`
  }

  return summary
}

function generateDecisionFramework(
  topicTitle: string,
  userMessages: MessageData[],
  _assistantMessages: MessageData[]
): string {
  let framework = `# Decision Framework: ${topicTitle}\n\n`

  framework += `## Decision Context\n\n`
  framework += `This framework synthesizes decision-making patterns from an interview about **${topicTitle}**.\n\n`

  framework += `## Guiding Principles\n\n`
  const principles = userMessages
    .filter(m => m.content.includes('important') || m.content.includes('believe') ||
      m.content.includes('value') || m.content.includes('always') || m.content.includes('principle'))
    .slice(0, 4)

  if (principles.length > 0) {
    principles.forEach((m, i) => {
      const excerpt = m.content.substring(0, 150).trim()
      framework += `${i + 1}. "${excerpt}${m.content.length > 150 ? '...' : ''}"\n`
    })
  } else {
    framework += `- Explore further to identify explicit guiding principles\n`
  }
  framework += `\n`

  framework += `## Decision Criteria\n\n`
  framework += `When making decisions about ${topicTitle}, consider:\n\n`
  framework += `- Does it align with the core principles above?\n`
  framework += `- How does past experience inform this choice?\n`
  framework += `- What are the potential trade-offs?\n\n`

  framework += `## Red Flags\n\n`
  const concerns = userMessages.filter(m =>
    m.content.includes('worry') || m.content.includes('concern') ||
    m.content.includes('avoid') || m.content.includes('risk') || m.content.includes('problem')
  ).slice(0, 3)

  if (concerns.length > 0) {
    concerns.forEach(m => {
      const excerpt = m.content.substring(0, 150).trim()
      framework += `- "${excerpt}${m.content.length > 150 ? '...' : ''}"\n`
    })
  } else {
    framework += `- No explicit red flags identified in this session\n`
  }
  framework += `\n`

  framework += `## Green Lights\n\n`
  const positives = userMessages.filter(m =>
    m.content.includes('love') || m.content.includes('enjoy') ||
    m.content.includes('excited') || m.content.includes('passion') || m.content.includes('great')
  ).slice(0, 3)

  if (positives.length > 0) {
    positives.forEach(m => {
      const excerpt = m.content.substring(0, 150).trim()
      framework += `- "${excerpt}${m.content.length > 150 ? '...' : ''}"\n`
    })
  } else {
    framework += `- Further sessions can help identify positive signals\n`
  }

  return framework
}

function generateJsonContent(
  topicTitle: string,
  userMessages: MessageData[],
  assistantMessages: MessageData[]
): string {
  const principles = userMessages
    .filter(m => m.content.length > 50)
    .slice(0, 5)
    .map(m => {
      const sentences = m.content.split(/[.!?]+/).filter(s => s.trim().length > 15)
      return sentences[0]?.trim() || m.content.substring(0, 100).trim()
    })

  const frameworkMessages = userMessages.filter(m =>
    /\b(when|because|always|usually|tend to|approach|strategy|method|process|framework|model|pattern|rule|guideline)\b/i.test(m.content)
  )
  const frameworks = frameworkMessages
    .slice(0, 4)
    .map(m => {
      const sentences = m.content.split(/[.!?]+/).filter(s => s.trim().length > 15)
      return sentences[0]?.trim() || m.content.substring(0, 200).trim()
    })

  const examples = userMessages
    .filter(m => m.content.length > 100 ||
      /\b(example|instance|time when|remember|experience|once|story)\b/i.test(m.content))
    .slice(0, 3)
    .map(m => m.content.substring(0, 300).trim())

  const decisionMessages = userMessages.filter(m =>
    /\b(decide|decided|decision|chose|choose|choice|option|alternative|trade-?off|weigh|consider|prefer|priority|prioritize)\b/i.test(m.content)
  )
  const decisions = decisionMessages
    .slice(0, 4)
    .map(m => {
      const sentences = m.content.split(/[.!?]+/).filter(s => s.trim().length > 15)
      return sentences[0]?.trim() || m.content.substring(0, 200).trim()
    })

  const data = {
    topic: topicTitle,
    sessionDate: new Date().toISOString(),
    messageCount: userMessages.length + assistantMessages.length,
    userResponseCount: userMessages.length,
    principles,
    frameworks,
    examples,
    decisions,
    tags: extractTags(userMessages),
  }

  return JSON.stringify(data, null, 2)
}

function extractTags(userMessages: MessageData[]): string[] {
  const allText = userMessages.map(m => m.content).join(' ').toLowerCase()
  const tags: string[] = []

  const tagPatterns: [string, RegExp][] = [
    ['values', /value|believe|principle|important/i],
    ['experience', /experience|memory|remember|time when/i],
    ['decision-making', /decide|choice|decision|chose/i],
    ['growth', /learn|grow|change|develop/i],
    ['relationships', /relationship|people|friend|family|colleague/i],
    ['career', /work|job|career|professional/i],
    ['creativity', /creative|create|design|build|make/i],
    ['leadership', /lead|manage|team|guide/i],
    ['communication', /communicate|talk|express|write/i],
    ['goals', /goal|aim|target|aspire|want to/i],
  ]

  tagPatterns.forEach(([tag, pattern]) => {
    if (pattern.test(allText)) {
      tags.push(tag)
    }
  })

  return tags.slice(0, 6)
}

// ============================================
// Service Functions
// ============================================

/**
 * Distill a session into a note with AI-powered analysis.
 * Extracts insights and creates concept nodes for mini sessions.
 */
export async function distillSession(
  db: Db,
  sessionId: string,
  format: string = 'full_analysis'
) {
  const userId = LOCAL_USER_ID

  // Verify session belongs to user
  const session = db.select().from(sessions).where(
    and(eq(sessions.id, sessionId), eq(sessions.userId, userId))
  ).get()

  if (!session) {
    throw new Error('Session not found')
  }

  // Check if note already exists for this session
  const existingNote = db.select().from(notes).where(
    eq(notes.sessionId, sessionId)
  ).get()

  if (existingNote) {
    return { note: existingNote, alreadyExists: true }
  }

  // Get topic info
  const topic = db.select().from(topics).where(eq(topics.id, session.topicId)).get()

  if (!topic) {
    throw new Error('Topic not found')
  }

  // Get all messages for this session
  const sessionMessages = db.select().from(messages)
    .where(eq(messages.sessionId, sessionId))
    .orderBy(messages.createdAt)
    .all()

  if (sessionMessages.length < 2) {
    throw new Error('Session needs at least one exchange before distillation')
  }

  // Mark session as completed
  db.update(sessions).set({
    status: 'completed',
    completedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }).where(eq(sessions.id, sessionId)).run()

  // Update topic status to extracted
  db.update(topics).set({
    status: 'extracted',
    updatedAt: new Date().toISOString(),
  }).where(eq(topics.id, session.topicId)).run()

  // Generate distillation in all formats (AI-powered with fallback)
  const userMsgs = sessionMessages.filter((m: any) => m.role === 'user')
  const assistantMsgs = sessionMessages.filter((m: any) => m.role === 'assistant')

  // Gather user profile context for richer AI prompts
  const userProfile = db.select().from(users).where(eq(users.id, userId)).get()
  const distillCtx: DistillationContext = {
    topicTitle: topic.title,
    topicDescription: topic.description,
    userMessages: userMsgs,
    assistantMessages: assistantMsgs,
    userName: userProfile?.name || undefined,
    occupation: userProfile?.occupation || undefined,
    isMiniSession: !!session.isMiniSession,
  }

  // Run all 4 AI distillation calls in parallel, falling back to regex-based versions
  const [aiFullAnalysis, aiBriefSummary, aiDecisionFramework, aiJsonContent] = await Promise.all([
    generateFullAnalysisAI(distillCtx).catch(() => null),
    generateBriefSummaryAI(distillCtx).catch(() => null),
    generateDecisionFrameworkAI(distillCtx).catch(() => null),
    generateJsonContentAI(distillCtx).catch(() => null),
  ])

  const fullAnalysis = aiFullAnalysis || generateFullAnalysis(topic.title, topic.description, userMsgs, assistantMsgs)
  const briefSummary = aiBriefSummary || generateBriefSummary(topic.title, userMsgs, assistantMsgs)
  const decisionFramework = aiDecisionFramework || generateDecisionFramework(topic.title, userMsgs, assistantMsgs)
  const jsonContent = aiJsonContent || generateJsonContent(topic.title, userMsgs, assistantMsgs)

  // Create note
  const noteId = crypto.randomUUID()
  const selectedFormat = format || 'full_analysis'

  const newNote = db.insert(notes).values({
    id: noteId,
    sessionId,
    topicId: session.topicId,
    userId,
    title: `Session Notes: ${topic.title}`,
    contentFullAnalysis: fullAnalysis,
    contentBriefSummary: briefSummary,
    contentDecisionFramework: decisionFramework,
    contentJson: jsonContent,
    selectedFormat,
  }).returning().get()

  // Extract insights using the unified insight extraction service
  const existingVerified = db.select({
    content: insights.content,
    confidenceScore: insights.confidenceScore,
  }).from(insights).where(
    and(eq(insights.userId, userId), eq(insights.verificationStatus, 'verified'))
  ).all().map((i: any) => ({
    content: i.content,
    confidenceScore: i.confidenceScore ?? 50,
  }))

  const transcript = formatInterviewTranscript(userMsgs, assistantMsgs)
  const extractionCtx: ExtractionContext = {
    content: transcript,
    sourceType: 'interview',
    topicTitle: topic.title,
    topicDescription: topic.description || undefined,
    userName: userProfile?.name || undefined,
    occupation: userProfile?.occupation || undefined,
    isMiniSession: !!session.isMiniSession,
    existingVerifiedInsights: existingVerified,
  }

  const extractedInsights = await extractInsights(extractionCtx)
  const savedInsights = []

  for (const insight of extractedInsights) {
    const insightId = crypto.randomUUID()
    const saved = db.insert(insights).values({
      id: insightId,
      noteId,
      topicId: session.topicId,
      userId,
      content: insight.content,
      confidenceScore: insight.confidenceScore,
      extractionMethod: insight.extractionMethod || 'ai',
      verificationStatus: 'unverified',
      sourceSessionId: sessionId,
    }).returning().get()
    savedInsights.push(saved)
  }

  // For mini sessions, also create concept nodes in the knowledge graph for each insight
  if (session.isMiniSession) {
    for (const saved of savedInsights) {
      const nodeId = crypto.randomUUID()
      db.insert(conceptNodes).values({
        id: nodeId,
        userId,
        topicId: session.topicId,
        insightId: saved.id,
        label: saved.content.substring(0, 60),
        weight: (saved.confidenceScore ?? 50) / 100,
      }).run()
    }
  }

  // Multi-bucket cross-topic extraction: score relevance to other topics
  const otherTopics = db.select().from(topics).where(
    and(eq(topics.userId, userId), ne(topics.id, session.topicId))
  ).all()

  const suggestedConnections: Array<{ targetTopicId: string; topicTitle: string; relevanceScore: number }> = []

  if (otherTopics.length > 0) {
    const contentSummary = userMsgs
      .map((m: any) => m.content)
      .join(' ')
      .toLowerCase()

    const contentWords = contentSummary
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w: string) => w.length > 3)

    const stopWords = new Set(['that', 'this', 'with', 'from', 'have', 'been', 'they', 'will', 'would', 'could', 'should', 'what', 'when', 'where', 'which', 'their', 'about', 'more', 'some', 'very', 'just', 'also', 'than', 'them', 'into', 'most', 'only', 'your', 'like', 'then', 'make', 'over', 'such', 'much', 'know', 'think', 'really', 'things', 'because', 'something'])
    const meaningfulWords = contentWords.filter((w: string) => !stopWords.has(w))

    const wordFreqMap = new Map<string, number>()
    for (const w of meaningfulWords) {
      wordFreqMap.set(w, (wordFreqMap.get(w) || 0) + 1)
    }

    const topKeywords = [...wordFreqMap.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([word]) => word)

    for (const otherTopic of otherTopics) {
      const topicText = `${otherTopic.title} ${otherTopic.description || ''}`.toLowerCase()
      let topicTags: string[] = []
      if (otherTopic.tags) {
        try {
          let parsed = JSON.parse(otherTopic.tags as string)
          if (typeof parsed === 'string') parsed = JSON.parse(parsed)
          topicTags = Array.isArray(parsed) ? parsed : []
        } catch { topicTags = [] }
      }
      const topicTagsLower = topicTags.map((t: string) => t.toLowerCase())

      let score = 0

      for (const keyword of topKeywords) {
        if (topicText.includes(keyword)) {
          score += 5
        }
      }

      for (const tag of topicTagsLower) {
        if (contentSummary.includes(tag)) {
          score += 10
        }
        for (const keyword of topKeywords) {
          if (tag.includes(keyword) || keyword.includes(tag)) {
            score += 8
          }
        }
      }

      const titleWords = topicText.split(/\s+/).filter((w: string) => w.length > 3 && !stopWords.has(w))
      for (const tw of titleWords) {
        if (contentSummary.includes(tw)) {
          score += 7
        }
      }

      score = Math.min(score, 100)

      if (score >= 15) {
        suggestedConnections.push({
          targetTopicId: otherTopic.id,
          topicTitle: otherTopic.title,
          relevanceScore: score,
        })
      }
    }

    suggestedConnections.sort((a, b) => b.relevanceScore - a.relevanceScore)
  }

  // Return the updated session too
  const updatedSession = db.select().from(sessions).where(eq(sessions.id, sessionId)).get()

  scheduleSave()

  return {
    note: newNote,
    insights: savedInsights,
    session: updatedSession,
    suggestedConnections,
  }
}

/**
 * Re-run AI insight extraction against the messages of an already-distilled
 * session. Deletes only unverified insights from that session, then inserts
 * the new set. Verified and rejected insights are preserved.
 *
 * Use this to clean up extraction-quality issues (e.g. interviewer meta-text
 * leaking into insights) without redoing the interview.
 */
export async function reExtractSessionInsights(
  db: Db,
  sessionId: string,
): Promise<{
  sessionId: string
  topicTitle: string
  deleted: number
  inserted: number
}> {
  const { session, topic } = getSession(db, sessionId)

  // Re-extraction mines the DISTILLED NOTE (which is the AI's own synthesis
  // of the interview) — not the raw transcript. The transcript contains
  // interviewer scaffolding that the extractor can't always resist pulling
  // into insights, even with strong prompting. The distilled note is
  // pre-curated and is a much cleaner input.
  const { note } = getNoteForSession(db, sessionId)

  const cleanedContent = buildNoteContentForExtraction({
    contentFullAnalysis: note.contentFullAnalysis,
    contentBriefSummary: note.contentBriefSummary,
    contentDecisionFramework: note.contentDecisionFramework,
  })

  if (cleanedContent.trim().length === 0) {
    throw new Error('Distilled note is empty — cannot re-extract')
  }

  // Gather user profile context for richer AI prompts (matches distillSession)
  const userProfile = db.select().from(users).where(eq(users.id, LOCAL_USER_ID)).get()

  // Verified insights for deduplication — from OTHER sessions
  // (we're about to delete this session's own unverified insights anyway)
  const existingVerified = db
    .select({ content: insights.content, confidenceScore: insights.confidenceScore })
    .from(insights)
    .where(
      and(
        eq(insights.userId, LOCAL_USER_ID),
        eq(insights.verificationStatus, 'verified'),
      ),
    )
    .all()
    .map((i: any) => ({ content: i.content, confidenceScore: i.confidenceScore ?? 50 }))

  const extractionCtx: ExtractionContext = {
    content: cleanedContent,
    sourceType: 'note_redistill',
    topicTitle: topic?.title,
    topicDescription: topic?.description || undefined,
    userName: userProfile?.name || undefined,
    occupation: userProfile?.occupation || undefined,
    isMiniSession: !!session.isMiniSession,
    existingVerifiedInsights: existingVerified,
  }

  // Delete only unverified insights from this session (verified/rejected preserved).
  // Match via sourceSessionId OR via noteId → notes.sessionId, since older
  // insights may have a null sourceSessionId even when correctly attached.
  const noteIdsForSession = db
    .select({ id: notes.id })
    .from(notes)
    .where(eq(notes.sessionId, sessionId))
    .all()
    .map((n: any) => n.id)
  const noteIdSet = new Set(noteIdsForSession)

  const staleCandidates = db
    .select({
      id: insights.id,
      sourceSessionId: insights.sourceSessionId,
      noteId: insights.noteId,
    })
    .from(insights)
    .where(eq(insights.verificationStatus, 'unverified'))
    .all()

  const staleInsightIds = staleCandidates
    .filter((i: any) => i.sourceSessionId === sessionId || (i.noteId && noteIdSet.has(i.noteId)))
    .map((i: any) => i.id)

  let deleted = 0
  for (const id of staleInsightIds) {
    try {
      deleteInsight(db, id)
      deleted++
    } catch (err) {
      console.warn(`[me.md:notes] Failed to delete stale insight ${id}:`, err)
    }
  }

  const extractedInsights = await extractInsights(extractionCtx)

  let inserted = 0
  for (const insight of extractedInsights) {
    const insightId = crypto.randomUUID()
    // Encode source format into extractionMethod so per-format provenance
    // survives in the DB without a schema migration. e.g. "ai:full_analysis",
    // "ai:brief_summary", "ai:decision_framework". Fallback insights lose
    // the suffix since their source format is irrelevant for review.
    const extractionMethod = insight.sourceFormat
      ? `${insight.extractionMethod || 'ai'}:${insight.sourceFormat}`
      : (insight.extractionMethod || 'ai')
    db.insert(insights)
      .values({
        id: insightId,
        noteId: note.id,
        topicId: session.topicId,
        userId: LOCAL_USER_ID,
        content: insight.content,
        confidenceScore: insight.confidenceScore,
        extractionMethod,
        verificationStatus: 'unverified',
        sourceSessionId: sessionId,
      })
      .run()
    inserted++
  }

  scheduleSave()

  return {
    sessionId,
    topicTitle: topic?.title || 'Unknown Topic',
    deleted,
    inserted,
  }
}

/**
 * Re-run extraction across every session that has a distilled note.
 *
 * Sessions without a distilled note are skipped — the extractor needs the
 * note as input. Per-session errors are collected and the batch continues.
 *
 * The `onProgress` callback fires twice per session:
 * - `'start'` before the per-session work begins, so the UI can announce
 *   "Extracting from {topic} (3 / 16)…" while the LLM call is in flight.
 * - `'complete'` after the session's work finishes (success or error).
 */
export async function reExtractAllSessions(
  db: Db,
  onProgress?: (
    phase: 'start' | 'complete',
    index: number,
    total: number,
    title: string,
  ) => void,
): Promise<{
  sessionsProcessed: number
  totalDeleted: number
  totalInserted: number
  errors: Array<{ sessionId: string; topicTitle: string; message: string }>
}> {
  const allSessions = db
    .select({ id: sessions.id })
    .from(sessions)
    .where(eq(sessions.userId, LOCAL_USER_ID))
    .all()

  // Process every session that has a distilled note. Sessions that already
  // have only verified insights are still re-mined — the new extraction
  // produces a fresh batch of unverified candidates for review, while the
  // verified set is preserved inside reExtractSessionInsights.
  const allNotes = db
    .select({ id: notes.id, sessionId: notes.sessionId })
    .from(notes)
    .where(eq(notes.userId, LOCAL_USER_ID))
    .all()
  const sessionIdsWithNotes = new Set<string>(
    allNotes.map((n: any) => n.sessionId as string),
  )

  const candidateIds = allSessions
    .map((s: any) => s.id as string)
    .filter((sid: string) => sessionIdsWithNotes.has(sid))

  if (candidateIds.length === 0) {
    console.warn(
      `[me.md:notes] reExtractAllSessions: ${allSessions.length} session(s) for user, but ` +
        `none have a distilled note — nothing to re-extract.`,
    )
    onProgress?.('complete', 0, 0, '')
    return { sessionsProcessed: 0, totalDeleted: 0, totalInserted: 0, errors: [] }
  }

  let totalDeleted = 0
  let totalInserted = 0
  const errors: Array<{ sessionId: string; topicTitle: string; message: string }> = []
  let sessionsProcessed = 0

  for (let i = 0; i < candidateIds.length; i++) {
    const sid = candidateIds[i]

    // Pre-look up the topic title so the UI can show "starting X" before
    // the slow AI extraction call rather than waiting for it to finish.
    let upcomingTitle = 'Unknown Topic'
    try {
      const { topic } = getSession(db, sid)
      upcomingTitle = topic?.title || 'Unknown Topic'
    } catch {
      // session may have been deleted mid-batch; surface as Unknown
    }
    onProgress?.('start', i, candidateIds.length, upcomingTitle)

    let currentTitle = upcomingTitle
    try {
      const result = await reExtractSessionInsights(db, sid)
      currentTitle = result.topicTitle
      totalDeleted += result.deleted
      totalInserted += result.inserted
      sessionsProcessed++
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      try {
        const { topic } = getSession(db, sid)
        currentTitle = topic?.title || upcomingTitle
      } catch {
        currentTitle = upcomingTitle
      }
      console.warn(`[me.md:notes] Re-extraction failed for session ${sid}:`, err)
      errors.push({ sessionId: sid, topicTitle: currentTitle, message })
    }

    onProgress?.('complete', i + 1, candidateIds.length, currentTitle)
  }

  return {
    sessionsProcessed,
    totalDeleted,
    totalInserted,
    errors,
  }
}

/**
 * Regenerate note content in a specific format.
 */
export async function regenerateNote(
  db: Db,
  sessionId: string,
  format: string,
  regenerateContent: boolean = false
) {
  const userId = LOCAL_USER_ID

  if (!format || !['full_analysis', 'brief_summary', 'decision_framework', 'json'].includes(format)) {
    throw new Error('Invalid format. Must be: full_analysis, brief_summary, decision_framework, or json')
  }

  // Find existing note
  const note = db.select().from(notes).where(
    and(eq(notes.sessionId, sessionId), eq(notes.userId, userId))
  ).get()

  if (!note) {
    throw new Error('Note not found. Distill session first.')
  }

  // If regenerateContent is true, regenerate the specific format from session messages
  if (regenerateContent) {
    const session = db.select().from(sessions).where(
      and(eq(sessions.id, sessionId), eq(sessions.userId, userId))
    ).get()

    if (!session) {
      throw new Error('Session not found')
    }

    const topic = db.select().from(topics).where(eq(topics.id, session.topicId)).get()
    if (!topic) {
      throw new Error('Topic not found')
    }

    const sessionMessages = db.select().from(messages)
      .where(eq(messages.sessionId, sessionId))
      .orderBy(messages.createdAt)
      .all()

    const userMsgs = sessionMessages.filter((m: any) => m.role === 'user')
    const assistantMsgs = sessionMessages.filter((m: any) => m.role === 'assistant')

    const regenUserProfile = db.select().from(users).where(eq(users.id, userId)).get()
    const regenCtx: DistillationContext = {
      topicTitle: topic.title,
      topicDescription: topic.description,
      userMessages: userMsgs,
      assistantMessages: assistantMsgs,
      userName: regenUserProfile?.name || undefined,
      occupation: regenUserProfile?.occupation || undefined,
    }

    const updateData: Record<string, string> = {
      selectedFormat: format,
      updatedAt: new Date().toISOString(),
    }

    switch (format) {
      case 'full_analysis': {
        const aiResult = await generateFullAnalysisAI(regenCtx).catch(() => null)
        updateData.contentFullAnalysis = aiResult || generateFullAnalysis(topic.title, topic.description, userMsgs, assistantMsgs)
        break
      }
      case 'brief_summary': {
        const aiResult = await generateBriefSummaryAI(regenCtx).catch(() => null)
        updateData.contentBriefSummary = aiResult || generateBriefSummary(topic.title, userMsgs, assistantMsgs)
        break
      }
      case 'decision_framework': {
        const aiResult = await generateDecisionFrameworkAI(regenCtx).catch(() => null)
        updateData.contentDecisionFramework = aiResult || generateDecisionFramework(topic.title, userMsgs, assistantMsgs)
        break
      }
      case 'json': {
        const aiResult = await generateJsonContentAI(regenCtx).catch(() => null)
        updateData.contentJson = aiResult || generateJsonContent(topic.title, userMsgs, assistantMsgs)
        break
      }
    }

    const updated = db.update(notes).set(updateData).where(eq(notes.id, note.id)).returning().get()
    scheduleSave()
    return { note: updated, regenerated: true }
  }

  // Just update selected format (no content regeneration)
  const updated = db.update(notes).set({
    selectedFormat: format,
    updatedAt: new Date().toISOString(),
  }).where(eq(notes.id, note.id)).returning().get()

  scheduleSave()
  return { note: updated }
}

/**
 * Get all notes for the local user, enriched with topic titles.
 */
export function getNotes(db: Db) {
  const userId = LOCAL_USER_ID

  const userNotes = db.select().from(notes)
    .where(eq(notes.userId, userId))
    .orderBy(desc(notes.createdAt))
    .all()

  const enrichedNotes = userNotes.map((n: any) => {
    const topic = n.topicId ? db.select().from(topics).where(eq(topics.id, n.topicId)).get() : null
    return {
      ...n,
      topicTitle: topic?.title || 'Unknown Topic',
    }
  })

  return { notes: enrichedNotes }
}

/**
 * Get a specific note by ID with related insights.
 */
export function getNote(db: Db, id: string) {
  const userId = LOCAL_USER_ID

  const note = db.select().from(notes).where(
    and(eq(notes.id, id), eq(notes.userId, userId))
  ).get()

  if (!note) {
    throw new Error('Note not found')
  }

  const noteInsights = db.select().from(insights)
    .where(eq(insights.noteId, id))
    .all()

  return { note, insights: noteInsights }
}

/**
 * Get note for a specific session with related insights.
 */
export function getNoteForSession(db: Db, sessionId: string) {
  const userId = LOCAL_USER_ID

  const note = db.select().from(notes).where(
    and(eq(notes.sessionId, sessionId), eq(notes.userId, userId))
  ).get()

  if (!note) {
    throw new Error('No note found for this session')
  }

  const noteInsights = db.select().from(insights)
    .where(eq(insights.noteId, note.id))
    .all()

  return { note, insights: noteInsights }
}

/**
 * Update a note's content and/or format.
 */
export function updateNote(
  db: Db,
  id: string,
  data: {
    contentFullAnalysis?: string
    contentBriefSummary?: string
    contentDecisionFramework?: string
    contentJson?: string
    selectedFormat?: string
    title?: string
  }
) {
  const userId = LOCAL_USER_ID

  const note = db.select().from(notes).where(
    and(eq(notes.id, id), eq(notes.userId, userId))
  ).get()

  if (!note) {
    throw new Error('Note not found')
  }

  const updated = db.update(notes).set({
    contentFullAnalysis: data.contentFullAnalysis !== undefined ? data.contentFullAnalysis : note.contentFullAnalysis,
    contentBriefSummary: data.contentBriefSummary !== undefined ? data.contentBriefSummary : note.contentBriefSummary,
    contentDecisionFramework: data.contentDecisionFramework !== undefined ? data.contentDecisionFramework : note.contentDecisionFramework,
    contentJson: data.contentJson !== undefined ? data.contentJson : note.contentJson,
    selectedFormat: data.selectedFormat !== undefined ? data.selectedFormat : note.selectedFormat,
    title: data.title !== undefined ? data.title : note.title,
    updatedAt: new Date().toISOString(),
  }).where(eq(notes.id, id)).returning().get()

  scheduleSave()
  return { note: updated }
}

/**
 * Export a note as markdown content.
 */
export function exportNoteMarkdown(db: Db, id: string, format: string = 'full_analysis') {
  const userId = LOCAL_USER_ID

  const note = db.select().from(notes).where(
    and(eq(notes.id, id), eq(notes.userId, userId))
  ).get()

  if (!note) {
    throw new Error('Note not found')
  }

  const topic = note.topicId ? db.select().from(topics).where(eq(topics.id, note.topicId)).get() : null
  const topicTitle = topic?.title || 'Unknown Topic'

  let content: string
  let formatLabel: string
  switch (format) {
    case 'brief_summary':
      content = note.contentBriefSummary || 'No content available'
      formatLabel = 'Brief Summary'
      break
    case 'decision_framework':
      content = note.contentDecisionFramework || 'No content available'
      formatLabel = 'Decision Framework'
      break
    case 'json':
      content = note.contentJson || '{}'
      formatLabel = 'JSON Data'
      break
    default:
      content = note.contentFullAnalysis || 'No content available'
      formatLabel = 'Full Analysis'
  }

  const title = note.title || 'Untitled Note'
  const createdDate = note.createdAt ? new Date(note.createdAt).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : ''

  let markdownContent: string
  if (format === 'json') {
    markdownContent = `# ${title}\n\n**Topic:** ${topicTitle}  \n**Format:** ${formatLabel}  \n**Date:** ${createdDate}\n\n---\n\n\`\`\`json\n${content}\n\`\`\`\n`
  } else {
    markdownContent = `# ${title}\n\n**Topic:** ${topicTitle}  \n**Format:** ${formatLabel}  \n**Date:** ${createdDate}\n\n---\n\n${content}\n`
  }

  const safeTitle = title.replace(/[^a-zA-Z0-9-_ ]/g, '').replace(/\s+/g, '_').substring(0, 50)

  return { markdownContent, filename: `${safeTitle}.md` }
}

/**
 * Destructive full reset: delete ALL insights + notes + concept graph
 * artifacts for the local user, then re-distill every session from raw
 * transcript, then re-extract insights using the current prompt (which
 * produces first-person, dedup-aware output).
 *
 * `distillSession` runs insight extraction itself, so a separate
 * re-extract phase is unnecessary.
 *
 * Phase callbacks fire in this order:
 *   - 'clear' once with index=0 (clearing is a single step)
 *   - 'distill' once per session as it begins
 *   - 'distill' completes implicitly when the next session starts
 *
 * Per-session errors are captured and the batch continues. The caller
 * can show a final summary with `errors` populated.
 */
export async function clearAndRedistillAll(
  db: Db,
  onProgress?: (
    phase: 'clear' | 'distill',
    index: number,
    total: number,
    title: string,
  ) => void,
): Promise<{
  insightsDeleted: number
  notesRegenerated: number
  sessionsProcessed: number
  totalInserted: number
  errors: Array<{ sessionId: string; topicTitle: string; message: string }>
}> {
  const userId = LOCAL_USER_ID
  const errors: Array<{ sessionId: string; topicTitle: string; message: string }> = []

  // Phase 1: clear all insights + notes + concept graph for the user.
  onProgress?.('clear', 0, 0, 'Clearing insights and notes')

  const { deleted: insightsDeleted } = clearAllInsights(db)

  // Delete all notes for the user. Cascade handles linked insights
  // (already gone) but better-sqlite3 is happy with the explicit delete.
  db.delete(notes).where(eq(notes.userId, userId)).run()

  // Concept nodes & edges are orphaned now (their insights are gone).
  // Wipe them all so the knowledge graph starts fresh.
  db.delete(conceptNodes).where(eq(conceptNodes.userId, userId)).run()
  // Edges don't have userId — they're per-knowledge-graph and edges
  // reference concept nodes. After clearing all nodes, edges are
  // orphaned but harmless. Skip the edge cleanup; the graph view
  // ignores edges that point to missing nodes.

  // Phase 2: distill every session in turn.
  const allSessions = db.select({
    id: sessions.id,
    topicId: sessions.topicId,
  }).from(sessions)
    .where(eq(sessions.userId, userId))
    .all()

  const total = allSessions.length
  let notesRegenerated = 0
  let totalInserted = 0

  for (let i = 0; i < total; i++) {
    const session = allSessions[i]
    const topicTitle = db.select({ title: topics.title })
      .from(topics)
      .where(eq(topics.id, session.topicId))
      .get()?.title || 'Untitled'

    onProgress?.('distill', i, total, topicTitle)

    try {
      // distillSession creates the note (since we cleared all notes above)
      // and runs initial insight extraction against the new note.
      await distillSession(db, session.id)
      notesRegenerated += 1

      // Count insights inserted for this session by querying what was just added.
      const sessionInsights = db.select({ id: insights.id })
        .from(insights)
        .where(eq(insights.sourceSessionId, session.id))
        .all()
      totalInserted += sessionInsights.length
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      errors.push({ sessionId: session.id, topicTitle, message })
      console.error(
        `[me.md:clear-redistill] Failed to distill session ${session.id} (${topicTitle}):`,
        message,
      )
    }
  }

  scheduleSave()

  return {
    insightsDeleted,
    notesRegenerated,
    sessionsProcessed: total - errors.length,
    totalInserted,
    errors,
  }
}
