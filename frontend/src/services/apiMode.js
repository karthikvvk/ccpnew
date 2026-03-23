/**
 * apiMode.js  —  Fast demo pipeline via cloud APIs
 * ─────────────────────────────────────────────────
 * Supports two providers:
 *   "groq"   → Whisper-large-v3 STT + LLaMA-3 translation + PlayAI TTS  (FREE, fastest)
 *   "openai" → Whisper STT + GPT-4o-mini translation + TTS-1              (paid)
 *
 * Provider is selected by which key is set in settings.json:
 *   api_keys.groq   → uses Groq  (preferred for demo)
 *   api_keys.openai → uses OpenAI (fallback)
 * ─────────────────────────────────────────────────
 */

const BACKEND = '/api/v1';

const LANG_NAMES = {
  TA: 'Tamil', JP: 'Japanese', DE: 'German', EN: 'English',
  ES: 'Spanish', FR: 'French', ZH: 'Chinese', KO: 'Korean',
  HI: 'Hindi', AR: 'Arabic',
};

// Groq PlayAI TTS voice map
const GROQ_TTS_VOICES = {
  TA: 'Fritz-PlayAI',   // deep male, closest for South Asian feel
  JP: 'Aria-PlayAI',
  DE: 'Atlas-PlayAI',
  EN: 'Fritz-PlayAI',
  ES: 'Celeste-PlayAI',
  FR: 'Aria-PlayAI',
  ZH: 'Atlas-PlayAI',
  KO: 'Aria-PlayAI',
  HI: 'Fritz-PlayAI',
  AR: 'Fritz-PlayAI',
};

// OpenAI TTS voice map
const OPENAI_TTS_VOICES = {
  TA: 'onyx', JP: 'nova', DE: 'alloy', EN: 'alloy',
  ES: 'shimmer', FR: 'nova', ZH: 'echo', KO: 'echo',
  HI: 'onyx', AR: 'onyx',
};

// ─── Load keys from backend /api/v1/settings ────────────────────

export async function loadKeys() {
  const res = await fetch(`${BACKEND}/settings`);
  if (!res.ok) throw new Error('Could not load settings from backend');
  const data = await res.json();
  const groq   = data?.api_keys?.groq   || '';
  const openai = data?.api_keys?.openai || '';
  if (!groq && !openai)
    throw new Error('No API key found. Set api_keys.groq or api_keys.openai in settings.json');
  const provider = groq ? 'groq' : 'openai';
  return { groq, openai, provider };
}

// ─── Step 1: Extract audio (backend ffmpeg) ──────────────────────

export async function extractAudio(videoFile, onProgress) {
  onProgress?.('📽️ Extracting audio from video...');
  const form = new FormData();
  form.append('video', videoFile);
  const res = await fetch(`${BACKEND}/extract-audio`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Audio extraction failed: ${await res.text()}`);
  return await res.blob(); // WAV blob
}

// ─── Step 2: Whisper STT ─────────────────────────────────────────

export async function transcribeAudio(audioBlob, keys, onProgress) {
  onProgress?.('🎙️ Transcribing speech with Whisper...');

  const { provider, groq: groqKey, openai: openaiKey } = keys;
  const endpoint = provider === 'groq'
    ? 'https://api.groq.com/openai/v1/audio/transcriptions'
    : 'https://api.openai.com/v1/audio/transcriptions';
  const apiKey   = provider === 'groq' ? groqKey : openaiKey;
  const model    = provider === 'groq' ? 'whisper-large-v3-turbo' : 'whisper-1';

  const form = new FormData();
  form.append('file', audioBlob, 'audio.wav');
  form.append('model', model);
  form.append('response_format', 'verbose_json');

  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`Whisper STT failed (${provider}): ${err?.error?.message || res.statusText}`);
  }
  return await res.json(); // { text, segments: [{start, end, text}] }
}

// ─── Step 3: Translate ───────────────────────────────────────────

export async function translateSegments(whisperResult, targetLangCode, keys, onProgress) {
  onProgress?.('🌐 Translating...');

  const { provider, groq: groqKey, openai: openaiKey } = keys;
  const endpoint = provider === 'groq'
    ? 'https://api.groq.com/openai/v1/chat/completions'
    : 'https://api.openai.com/v1/chat/completions';
  const apiKey = provider === 'groq' ? groqKey : openaiKey;
  const model  = provider === 'groq' ? 'llama-3.1-8b-instant' : 'gpt-4o-mini';

  const targetLang = LANG_NAMES[targetLangCode] || targetLangCode;
  const segments   = whisperResult.segments || [];
  const numbered   = segments.map((s, i) => `[${i}] ${s.text}`).join('\n');

  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: 'system',
          content: `You are a professional dubbing translator. Translate each numbered line into ${targetLang}.
Keep the same [N] numbering. Preserve tone and pace. Return ONLY the numbered translated lines.`,
        },
        { role: 'user', content: numbered },
      ],
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`Translation failed (${provider}): ${err?.error?.message || res.statusText}`);
  }

  const data = await res.json();
  const raw  = data.choices[0].message.content;
  const lines = raw.split('\n').filter(Boolean);

  const translated = segments.map((seg, i) => {
    const line = lines.find(l => l.startsWith(`[${i}]`));
    const text = line ? line.replace(/^\[\d+\]\s*/, '').trim() : seg.text;
    return { ...seg, translated: text };
  });

  return {
    segments: translated,
    fullText: translated.map(s => s.translated).join(' '),
  };
}

// ─── Step 4: TTS ─────────────────────────────────────────────────

export async function synthesizeSpeech(text, targetLangCode, keys, videoDuration, onProgress) {
  onProgress?.('🔊 Generating dubbed speech (edge-tts, speed-matched)...');

  const form = new FormData();
  form.append('text', text);
  form.append('lang', targetLangCode);
  if (videoDuration > 0) form.append('video_duration', videoDuration.toFixed(3));

  const res = await fetch(`${BACKEND}/tts`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`TTS failed: ${err}`);
  }
  return await res.blob();
}

// ─── Helper: probe video file duration in the browser ────────────

export function getVideoDuration(videoFile) {
  return new Promise((resolve) => {
    const url  = URL.createObjectURL(videoFile);
    const vid  = document.createElement('video');
    vid.preload = 'metadata';
    vid.onloadedmetadata = () => {
      const d = vid.duration;
      URL.revokeObjectURL(url);
      resolve(isFinite(d) ? d : 0);
    };
    vid.onerror = () => { URL.revokeObjectURL(url); resolve(0); };
    vid.src = url;
  });
}



// ─── Step 5: Mux audio into video (backend ffmpeg) ───────────────

export async function muxAudio(videoFile, audioBlob, jobId, onProgress) {
  onProgress?.('🎬 Merging dubbed audio into video...');
  const form = new FormData();
  form.append('video', videoFile);
  form.append('audio', audioBlob, 'dubbed_audio.mp3');
  if (jobId) form.append('job_id', jobId);

  const res = await fetch(`${BACKEND}/mux-audio`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Video mux failed: ${await res.text()}`);

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ─── MAIN PIPELINE ───────────────────────────────────────────────

export async function runApiModePipeline(videoFile, targetLangCode, onProgress) {
  const keys = await loadKeys();
  onProgress?.(`⚡ Using provider: ${keys.provider.toUpperCase()}`);

  // Probe video duration in browser (used for atempo speed matching)
  const videoDuration = await getVideoDuration(videoFile);
  if (videoDuration > 0) {
    onProgress?.(`⏱️ Video duration: ${videoDuration.toFixed(1)}s — will speed-match TTS`);
  }

  const audioWav   = await extractAudio(videoFile, onProgress);
  const whisper    = await transcribeAudio(audioWav, keys, onProgress);
  const translated = await translateSegments(whisper, targetLangCode, keys, onProgress);
  const speech     = await synthesizeSpeech(translated.fullText, targetLangCode, keys, videoDuration, onProgress);

  const jobId    = `api-${Date.now()}`;
  const videoUrl = await muxAudio(videoFile, speech, jobId, onProgress);

  onProgress?.('✅ Done!');
  return { videoUrl, jobId, transcript: whisper, translation: translated };
}


export default {
  runApiModePipeline,
  loadKeys,
  extractAudio,
  transcribeAudio,
  translateSegments,
  synthesizeSpeech,
  muxAudio,
};
