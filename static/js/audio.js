// Audio layer: neural TTS via the local server (Edge neural voices) with an
// in-memory URL cache, a single shared player, and a Web Speech API fallback
// so audio keeps working even fully offline.

import { api } from './api.js';
import { state } from './store.js';

const urlCache = new Map(); // cacheKey -> audio URL
const player = new Audio();
let playToken = 0;
const stateListeners = new Set();

export function onAudioState(fn) {
  stateListeners.add(fn);
  return () => stateListeners.delete(fn);
}

function notify(playing) {
  for (const fn of stateListeners) fn(playing);
}

player.addEventListener('ended', () => notify(false));
player.addEventListener('pause', () => notify(false));
player.addEventListener('play', () => notify(true));

function voiceFor(lang) {
  return state.settings.voices[lang] || null;
}

function cacheKey(text, lang, voice, rate) {
  return `${lang}|${voice || 'default'}|${rate}|${text}`;
}

async function fetchUrl(text, lang, { slow = false, voice = null } = {}) {
  const rate = slow ? 'slow' : state.settings.ttsRate;
  const voiceId = voice || voiceFor(lang);
  const key = cacheKey(text, lang, voiceId, rate);
  if (urlCache.has(key)) return urlCache.get(key);
  const result = await api.tts({ text, language: lang, voice: voiceId, rate });
  urlCache.set(key, result.url);
  return result.url;
}

function speakWithBrowser(text, lang) {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) return resolve(false);
    const utterance = new SpeechSynthesisUtterance(text);
    const locales = { fr: 'fr-FR', es: 'es-ES', de: 'de-DE', it: 'it-IT', pt: 'pt-PT', nl: 'nl-NL', ru: 'ru-RU', zh: 'zh-CN' };
    utterance.lang = locales[lang] || lang;
    utterance.rate = state.settings.ttsRate === 'slow' ? 0.7 : state.settings.ttsRate === 'fast' ? 1.1 : 0.9;
    utterance.onend = () => { notify(false); resolve(true); };
    utterance.onerror = () => { notify(false); resolve(false); };
    notify(true);
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  });
}

/** Speak text aloud. Resolves when playback finishes (or fails silently).
 *  `voice` overrides the settings voice - used for dialogue speakers. */
export async function speak(text, { lang, slow = false, voice = null } = {}) {
  const language = lang || state.settings.language;
  const clean = (text || '').trim();
  if (!clean) return;
  const token = ++playToken;
  try {
    const url = await fetchUrl(clean, language, { slow, voice });
    if (token !== playToken) return; // superseded by a newer request
    stopBrowserSpeech();
    player.src = url;
    await player.play();
    await new Promise((resolve) => {
      const done = () => { cleanup(); resolve(); };
      const cleanup = () => {
        player.removeEventListener('ended', done);
        player.removeEventListener('pause', done);
      };
      player.addEventListener('ended', done);
      player.addEventListener('pause', done);
    });
  } catch {
    if (token !== playToken) return;
    await speakWithBrowser(clean, language);
  }
}

function stopBrowserSpeech() {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
}

export function stopAudio() {
  playToken += 1;
  player.pause();
  player.currentTime = 0;
  stopBrowserSpeech();
  notify(false);
}

/** Warm the TTS cache for upcoming items (fire and forget). */
export function preload(texts, lang) {
  const language = lang || state.settings.language;
  for (const text of texts.slice(0, 6)) {
    const clean = (text || '').trim();
    if (clean) fetchUrl(clean, language).catch(() => {});
  }
}

/** Play a short UI feedback tone (correct / wrong), respecting settings. */
let audioCtx = null;
export function feedbackTone(kind) {
  if (!state.settings.soundEffects) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain).connect(audioCtx.destination);
    const now = audioCtx.currentTime;
    if (kind === 'correct') {
      osc.frequency.setValueAtTime(660, now);
      osc.frequency.setValueAtTime(880, now + 0.09);
    } else {
      osc.frequency.setValueAtTime(220, now);
      osc.frequency.setValueAtTime(180, now + 0.1);
    }
    gain.gain.setValueAtTime(0.06, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);
    osc.start(now);
    osc.stop(now + 0.25);
  } catch { /* audio context unavailable - ignore */ }
}
