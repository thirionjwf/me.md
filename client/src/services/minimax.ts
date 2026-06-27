// MiniMax client (OpenAI-compatible mode).
//
// Token Plan endpoint: https://api.minimax.io/v1/chat/completions
// Auth: Authorization: Bearer <key>
// Stream format: OpenAI-style SSE — data: {"choices":[{"delta":{"content":"..."}}]}

const MINIMAX_BASE = '/v1'
const DEFAULT_MODEL = 'MiniMax-M3'

export class MiniMaxError extends Error {
  constructor(
    message: string,
    public status?: number,
    public type?: string
  ) {
    super(message)
    this.name = 'MiniMaxError'
  }
}

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface CallOptions {
  messages: Message[]
  system?: string
  model?: string
  maxTokens?: number
  signal?: AbortSignal
}

function getApiKey(): string {
  const key = localStorage.getItem('memd_api_key')
  if (!key) throw new MiniMaxError('API key not configured. Set it in Settings.')
  return key
}

function buildHeaders(apiKey: string): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
  }
}

function buildBody(options: CallOptions, stream = false): string {
  const messages: Message[] = []
  if (options.system) {
    messages.push({ role: 'system', content: options.system })
  }
  for (const m of options.messages) {
    messages.push({ role: m.role, content: m.content })
  }

  return JSON.stringify({
    model: options.model ?? DEFAULT_MODEL,
    max_tokens: options.maxTokens ?? 1024,
    messages,
    stream,
  })
}

function extractContent(data: any): string {
  return data?.choices?.[0]?.message?.content ?? ''
}

export async function callMiniMax(options: CallOptions): Promise<string> {
  const apiKey = getApiKey()
  const res = await fetch(`${MINIMAX_BASE}/chat/completions`, {
    method: 'POST',
    headers: buildHeaders(apiKey),
    body: buildBody(options),
    signal: options.signal,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new MiniMaxError(
      body.error?.message ?? `API error ${res.status}`,
      res.status,
      body.error?.type,
    )
  }

  const data = await res.json()
  return extractContent(data)
}

export async function* streamMiniMax(
  options: CallOptions
): AsyncGenerator<string, string, undefined> {
  const apiKey = getApiKey()
  const res = await fetch(`${MINIMAX_BASE}/chat/completions`, {
    method: 'POST',
    headers: buildHeaders(apiKey),
    body: buildBody(options, true),
    signal: options.signal,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new MiniMaxError(
      body.error?.message ?? `API error ${res.status}`,
      res.status,
      body.error?.type,
    )
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let fullText = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const data = trimmed.slice(5).trim()
      if (data === '[DONE]') continue

      try {
        const event = JSON.parse(data)
        const chunk = event?.choices?.[0]?.delta?.content
        if (typeof chunk === 'string') {
          fullText += chunk
          yield chunk
        }
      } catch {
        // Skip non-JSON lines
      }
    }
  }

  return fullText
}

export function isApiKeyConfigured(): boolean {
  const key = localStorage.getItem('memd_api_key')
  return !!key && key.trim() !== ''
}