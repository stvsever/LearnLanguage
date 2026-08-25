// Central state: settings, decks, cards, stats, stories - persisted to localStorage.
//
// All learner data lives in the browser. The backend only generates content.

import { newSrsState, isDue } from './srs.js';

const STORAGE_KEY = 'learnlanguage.v2';
const DAY_MS = 86400000;

export const DEFAULT_SETTINGS = {
  language: 'fr',
  level: 'A2',
  theme: 'system',
  voices: {},            // languageCode -> voice id
  ttsRate: 'study',      // slow | study | natural | fast
  autoplayAudio: true,
  soundEffects: true,
  newPerDay: 10,
  maxReviewsPerDay: 100,
  targetRetention: 0.9,
  strictAccents: false,
  typoTolerance: true,
  showPronunciation: true,
  showExamples: true,
  accentToolbar: true,
  tourDone: false,
  model: null,           // OpenRouter model override; null = server default
};

const SUPPORTED_LANGUAGES = ['fr', 'es', 'ru', 'zh'];

function freshState() {
  return {
    version: 2,
    createdAt: Date.now(),
    settings: { ...DEFAULT_SETTINGS },
    decks: {},        // lang -> { cards: {id: card}, topics: [] }
    stats: {},        // lang -> { days: { 'YYYY-MM-DD': {reviews, correct, newCards, timeMs} } }
    compositions: {}, // lang -> [compositionPack]
    grammar: {},      // lang -> { featureId: {seen, lastSeen} }
  };
}

export let state = load();

function migrate(parsed) {
  // v2.0 stored generated stories under `stories` in a sentences/sentences_en
  // shape; v2.1 generalizes them to compositions with segments.
  if (parsed.stories && !parsed.compositions) {
    parsed.compositions = {};
    for (const [lang, items] of Object.entries(parsed.stories)) {
      parsed.compositions[lang] = (items || []).map((story) => story.segments ? story : ({
        language: story.language || lang,
        format: 'story',
        title: story.title || 'Untitled',
        level: story.level || '',
        scene: '',
        participants: [],
        segments: (story.sentences || []).map((text, i) => ({
          speaker: '', text, text_en: story.sentences_en?.[i] || '',
        })),
        glossary: story.glossary || [],
        grammar_spotlights: [],
        questions: story.questions || [],
      }));
    }
    delete parsed.stories;
  }
  if (!parsed.compositions) parsed.compositions = {};
  if (!parsed.grammar) parsed.grammar = {};
  if (!SUPPORTED_LANGUAGES.includes(parsed.settings.language)) parsed.settings.language = 'fr';
  return parsed;
}

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return freshState();
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.version !== 2) return freshState();
    parsed.settings = { ...DEFAULT_SETTINGS, ...(parsed.settings || {}) };
    return migrate(parsed);
  } catch {
    return freshState();
  }
}

let saveTimer = null;
export function persist(immediate = false) {
  if (saveTimer) clearTimeout(saveTimer);
  const write = () => {
    saveTimer = null;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (err) {
      console.error('Persist failed', err);
    }
  };
  if (immediate) write();
  else saveTimer = setTimeout(write, 400);
}

window.addEventListener('beforeunload', () => persist(true));

// -- pub/sub -----------------------------------------------------------------
const listeners = new Map();
export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}
export function emit(event, payload) {
  for (const fn of listeners.get(event) || []) fn(payload);
}

// -- settings ----------------------------------------------------------------
export function updateSettings(patch) {
  Object.assign(state.settings, patch);
  persist();
  emit('settings', state.settings);
}

export function currentLanguage() { return state.settings.language; }

// -- decks / cards -----------------------------------------------------------
export function deck(lang = currentLanguage()) {
  if (!state.decks[lang]) state.decks[lang] = { cards: {}, topics: [] };
  return state.decks[lang];
}

export function cards(lang = currentLanguage()) {
  return Object.values(deck(lang).cards);
}

function cardId(target) {
  return target.toLowerCase().replace(/\s+/g, ' ').trim();
}

/** Add lesson items as cards; returns the number of genuinely new cards. */
export function addCards(items, topic, lang = currentLanguage()) {
  const d = deck(lang);
  let added = 0;
  const now = Date.now();
  for (const item of items) {
    const id = cardId(item.target);
    if (!id || d.cards[id]) continue;
    d.cards[id] = {
      id,
      target: item.target.trim(),
      english: (item.english || '').trim(),
      pronunciation: (item.pronunciation || '').trim(),
      example: (item.example || '').trim(),
      exampleEn: (item.example_en || item.exampleEn || '').trim(),
      note: (item.note || '').trim(),
      tags: item.tags || [],
      topic: topic || '',
      addedAt: now,
      introduced: false, // becomes true once studied in Learn
      suspended: false,
      state: 'new',
      srs: newSrsState(now),
    };
    added += 1;
  }
  if (topic && !d.topics.includes(topic)) d.topics.unshift(topic);
  d.topics = d.topics.slice(0, 20);
  persist();
  emit('deck', lang);
  return added;
}

export function removeCard(id, lang = currentLanguage()) {
  delete deck(lang).cards[id];
  persist();
  emit('deck', lang);
}

export function dueCards(lang = currentLanguage(), now = Date.now()) {
  return cards(lang)
    .filter((c) => isDue(c, now))
    .sort((a, b) => a.srs.due - b.srs.due);
}

export function newCards(lang = currentLanguage()) {
  return cards(lang).filter((c) => c.state === 'new' && !c.suspended)
    .sort((a, b) => a.addedAt - b.addedAt);
}

export function learnedCards(lang = currentLanguage()) {
  return cards(lang).filter((c) => c.state !== 'new' && !c.suspended);
}

export function deckCounts(lang = currentLanguage()) {
  const all = cards(lang);
  const counts = { total: all.length, new: 0, learning: 0, review: 0, mature: 0, due: 0 };
  const now = Date.now();
  for (const c of all) {
    if (c.suspended) continue;
    if (c.state === 'new') counts.new += 1;
    else if (c.state === 'learning' || c.state === 'relearning') counts.learning += 1;
    else if (c.srs.S >= 21) counts.mature += 1;
    else counts.review += 1;
    if (isDue(c, now)) counts.due += 1;
  }
  return counts;
}

// -- daily stats -------------------------------------------------------------
export function todayKey(ts = Date.now()) {
  const d = new Date(ts);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function statDay(lang = currentLanguage(), key = todayKey()) {
  if (!state.stats[lang]) state.stats[lang] = { days: {} };
  const days = state.stats[lang].days;
  if (!days[key]) days[key] = { reviews: 0, correct: 0, newCards: 0, timeMs: 0, listening: 0, speaking: 0 };
  return days[key];
}

export function recordReview({ correct, ms = 0, mode = 'review' }, lang = currentLanguage()) {
  const day = statDay(lang);
  day.reviews += 1;
  if (correct) day.correct += 1;
  day.timeMs += Math.min(ms, 120000);
  if (mode === 'listen') day.listening += 1;
  if (mode === 'speak') day.speaking += 1;
  persist();
  emit('stats', lang);
}

export function recordNewCard(lang = currentLanguage()) {
  statDay(lang).newCards += 1;
  persist();
  emit('stats', lang);
}

export function recordTime(ms, lang = currentLanguage()) {
  statDay(lang).timeMs += Math.min(ms, 600000);
  persist();
}

export function newCardsIntroducedToday(lang = currentLanguage()) {
  return statDay(lang).newCards;
}

export function reviewsDoneToday(lang = currentLanguage()) {
  return statDay(lang).reviews;
}

export function streak(lang = currentLanguage()) {
  const days = state.stats[lang]?.days || {};
  let count = 0;
  let cursor = Date.now();
  // today counts if active; otherwise the streak may still be alive from yesterday
  if (!days[todayKey(cursor)]?.reviews && !days[todayKey(cursor)]?.newCards) cursor -= DAY_MS;
  while (true) {
    const record = days[todayKey(cursor)];
    if (record && (record.reviews > 0 || record.newCards > 0)) {
      count += 1;
      cursor -= DAY_MS;
    } else break;
  }
  return count;
}

export function accuracyOverDays(daysBack = 30, lang = currentLanguage()) {
  const days = state.stats[lang]?.days || {};
  let reviews = 0, correct = 0;
  for (let i = 0; i < daysBack; i++) {
    const rec = days[todayKey(Date.now() - i * DAY_MS)];
    if (rec) { reviews += rec.reviews; correct += rec.correct; }
  }
  return { reviews, correct, rate: reviews ? correct / reviews : null };
}

// -- compositions ------------------------------------------------------------
export function saveComposition(pack, lang = currentLanguage()) {
  if (!state.compositions[lang]) state.compositions[lang] = [];
  state.compositions[lang] = [pack, ...state.compositions[lang].filter((c) => c.title !== pack.title)].slice(0, 20);
  recordGrammarFeatures((pack.grammar_spotlights || []).map((s) => s.feature), lang);
  persist();
}

export function compositions(lang = currentLanguage()) {
  return state.compositions[lang] || [];
}

export function removeComposition(title, lang = currentLanguage()) {
  state.compositions[lang] = (state.compositions[lang] || []).filter((c) => c.title !== title);
  persist();
}

// -- grammar coverage --------------------------------------------------------
export function recordGrammarFeatures(featureIds, lang = currentLanguage()) {
  if (!featureIds?.length) return;
  if (!state.grammar[lang]) state.grammar[lang] = {};
  const record = state.grammar[lang];
  for (const id of featureIds) {
    if (!record[id]) record[id] = { seen: 0, lastSeen: null };
    record[id].seen += 1;
    record[id].lastSeen = Date.now();
  }
  persist();
  emit('grammar', lang);
}

export function grammarSeen(lang = currentLanguage()) {
  return state.grammar[lang] || {};
}

// -- data management ---------------------------------------------------------
export function exportData() {
  return JSON.stringify({ ...state, exportedAt: new Date().toISOString() }, null, 2);
}

export function importData(json) {
  const parsed = JSON.parse(json);
  if (!parsed || parsed.version !== 2 || !parsed.settings || !parsed.decks) {
    throw new Error('Not a valid LearnLanguage export file.');
  }
  delete parsed.exportedAt;
  parsed.settings = { ...DEFAULT_SETTINGS, ...parsed.settings };
  state = migrate(parsed);
  persist(true);
  emit('imported');
}

export function resetProgress(lang = currentLanguage()) {
  delete state.decks[lang];
  delete state.stats[lang];
  delete state.compositions[lang];
  delete state.grammar[lang];
  persist(true);
  emit('deck', lang);
}

export function resetAll() {
  state = freshState();
  persist(true);
  emit('imported');
}
