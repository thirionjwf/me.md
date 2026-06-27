import { describe, it, expect, vi, beforeEach } from 'vitest'
import { callMiniMax } from '../minimax'

describe('minimax client', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn().mockReturnValue('test-api-key'),
      setItem: vi.fn(),
    })
  })

  it('sends Authorization Bearer header and OpenAI-format body', async () => {
    let capturedUrl: string | undefined
    let capturedInit: any
    const mockFetch = vi.fn().mockImplementation((url: string, init: any) => {
      capturedUrl = url
      capturedInit = init
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'Hello!' } }],
          usage: { prompt_tokens: 10, completion_tokens: 5 },
        }),
      })
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await callMiniMax({
      messages: [{ role: 'user', content: 'Hi' }],
      system: 'You are helpful.',
    })

    expect(result).toBe('Hello!')
    expect(capturedUrl).toBe('/v1/chat/completions')
    expect(capturedInit.headers).toMatchObject({
      'Authorization': 'Bearer test-api-key',
      'Content-Type': 'application/json',
    })

    const body = JSON.parse(capturedInit.body)
    expect(body.model).toBe('MiniMax-M3')
    expect(body.stream).toBe(false)
    // system becomes the first message in OpenAI format
    expect(body.messages).toEqual([
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi' },
    ])
  })

  it('parses OpenAI choices[0].message.content from the response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        choices: [{ message: { content: 'response text' } }],
      }),
    }))

    const result = await callMiniMax({ messages: [{ role: 'user', content: 'Hi' }] })
    expect(result).toBe('response text')
  })

  it('throws when API key is missing', async () => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn().mockReturnValue(null),
    })

    await expect(callMiniMax({
      messages: [{ role: 'user', content: 'Hi' }],
    })).rejects.toThrow('API key not configured')
  })
})