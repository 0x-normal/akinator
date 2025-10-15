// server.js — Akinator backend (Fireworks) with anti-repeat + hint flow
require('dotenv').config();
const express = require('express');
const path = require('path');

// If Node < 18, uncomment to polyfill fetch
// global.fetch = global.fetch || ((...args) => import('node-fetch').then(({ default: f }) => f(...args)));

const app = express();
app.use(express.json({ limit: '2mb' }));

// ===== Helpers for anti-repeat =====
const STOP = new Set('the a an is are was were do does did has have had of to in on for with without about from at by as be been being character person human someone somebody male female man woman guy girl boy child adult old young age big small same similar repeat repeated repeating any'.split(/\s+/));
function norm(s) { return String(s).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim(); }
function tokens(s) { return norm(s).split(' ').filter(w => w && !STOP.has(w)); }
function jaccard(a, b) {
  const A = new Set(a), B = new Set(b); let inter = 0;
  for (const x of A) if (B.has(x)) inter++;
  const union = A.size + B.size - inter; return union ? inter / union : 0;
}
function isRepeat(question, history) {
  const tq = tokens(question);
  for (const h of history) {
    const t = tokens(h.q);
    const sim = jaccard(tq, t);
    if (sim >= 0.7) return true; // near-duplicate
    const a = norm(question), b = norm(h.q);
    if (a.includes(b) || b.includes(a)) return true; // containment
  }
  return false;
}

// ===== Prompt builders =====
function buildSystem(domain) {
  return [
    'You are an expert reasoning agent playing a 20 Questions (Akinator-style) game.',
    'Your goal is to identify the hidden target by asking strategic yes/no questions.',
    `Domain: ${domain}. Ask ONE question per turn.`,
    'Use deductive reasoning based on all previous questions and answers. Each new question must narrow down the space of possibilities.',
    'Ask questions that explore new dimensions (occupation, nationality, time period, traits, media type, etc.).',
    'Avoid repeating or rephrasing previous questions.',
    'When you have enough information (confidence ≥ 0.75), propose a guess.',
    'If forced final, give your best guess.',
    'Output ONLY strict JSON (no prose):',
    '{"type":"ask","question":"<short yes/no question>"} OR',
    '{"type":"guess","guess":"<one candidate>","confidence":0.0} OR',
    '{"type":"final","guess":"<best candidate>","confidence":0.0}',
    'Never include explanations, markdown, or comments.'
  ].join(' ');
}

function buildUser({ domain, history, turns, forceFinal, hint }) {
  // turn structured Q&A into clear logical context
  const structuredFacts = history.map((x, i) => {
    const q = x.q.toLowerCase();
    const a = x.a.toLowerCase();
    return `Q${i + 1}: ${q} → ${a}`;
  }).join('\n') || '(no previous questions)';

  const reasoningIntro = [
    `We are playing 20 Questions. Domain: ${domain}.`,
    'Here are all facts discovered so far:',
    structuredFacts,
    '',
    'Based on these facts, reason logically to choose the next most informative question.',
    'Your question should reduce uncertainty the most and move closer to a confident guess.',
    'Focus on attributes not yet covered (e.g. nationality, occupation, historical era, public role, media type, field of work, notable achievements).',
    forceFinal
      ? 'forceFinal=true → make your best final guess.'
      : 'forceFinal=false → ask the next question logically.',
    hint ? `Player hint: ${hint}` : '(no hint)',
    'Return ONLY strict JSON.'
  ].join('\n');

  return reasoningIntro;
}

function extractJson(text) {
  if (!text) return null;
  text = String(text)
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/```(?:json)?/gi, '')
    .replace(/```/g, '')
    .trim();

  // Try to find the largest JSON-like block
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1) return null;

  const jsonCandidate = text.slice(start, end + 1);

  try {
    return JSON.parse(jsonCandidate);
  } catch (err) {
    // Attempt a cleanup if commas or quotes are malformed
    const cleaned = jsonCandidate
      .replace(/,\s*([}\]])/g, '$1')
      .replace(/:\s*'([^']*)'/g, (_, p1) => `: "${p1}"`)
      .replace(/“|”/g, '"');
    try {
      return JSON.parse(cleaned);
    } catch {
      console.error("Failed to parse JSON:", text);
      return null;
    }
  }
}


async function callFireworks(messages) {
  const resp = await fetch("https://api.fireworks.ai/inference/v1/chat/completions", {
    method: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.FIREWORKS_API_KEY}`,
    },
    body: JSON.stringify({
      model: "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new",
      max_tokens: 4096,
      top_p: 1,
      top_k: 40,
      presence_penalty: 0,
      frequency_penalty: 0,
      temperature: 0.6,
      messages,
    }),
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data?.error || "Fireworks error");

  const raw = data?.choices?.[0]?.message?.content || "";
let obj = extractJson(raw);

if (!obj || !obj.type) {
  console.error("Fireworks raw output (unparsed):", raw);
  // Try asking the model to reformat strictly
  const retry = await fetch("https://api.fireworks.ai/inference/v1/chat/completions", {
    method: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.FIREWORKS_API_KEY}`,
    },
    body: JSON.stringify({
      model: "accounts/sentientfoundation-serverless/models/dobby-mini-unhinged-plus-llama-3-1-8b",
      max_tokens: 256,
      temperature: 0,
      messages: [
        { role: "system", content: "Return only valid JSON, no text outside it." },
        { role: "user", content: `Reformat this text to valid JSON only: ${raw}` },
      ],
    }),
  });
  const retryData = await retry.json().catch(() => ({}));
  const retryRaw = retryData?.choices?.[0]?.message?.content || "";
  obj = extractJson(retryRaw);
}

if (!obj || !obj.type) throw new Error("Parse error");

  return obj;
}


app.post('/api/step', async (req, res) => {
  try {
    const { domain = 'character', history = [], turns = history.length, forceFinal = false, hint = '' } = req.body || {};
    if (!process.env.FIREWORKS_API_KEY) return res.status(500).json({ error: 'Server missing FIREWORKS_API_KEY' });

    let messages = [
      { role: 'system', content: buildSystem(domain) },
      { role: 'user', content: buildUser({ domain, history, turns, forceFinal, hint }) },
    ];

    let obj; let attempts = 0;
    while (attempts < 3) {
      attempts++;
      obj = await callFireworks(messages);
      if (obj.type === 'ask') {
        obj.question = String(obj.question || '').slice(0, 200);
        if (!obj.question) throw new Error('Missing question');
        if (isRepeat(obj.question, history)) {
          // Append a hard constraint and try again
          messages.push({ role: 'system', content: `You repeated: "${obj.question}". Ask a different, non-redundant question focusing on a new attribute. Absolutely avoid re-asking about the same topic.` });
          continue;
        }
      } else if (obj.type === 'guess' || obj.type === 'final') {
        obj.guess = String(obj.guess || '').slice(0, 120);
        obj.confidence = Number.isFinite(obj.confidence) ? Math.max(0, Math.min(1, Number(obj.confidence))) : 0;
        if (!obj.guess) throw new Error('Missing guess');
      } else {
        throw new Error('Invalid type');
      }
      break;
    }

    res.json(obj);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || 'Server error' });
  }
});

// Serve static last so /api/* never falls through
app.use(express.static(path.join(__dirname, 'public')));
app.use('/api', (_req, res) => res.status(404).json({ error: 'API route not found' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Akinator v2 running: http://localhost:${PORT}`));
