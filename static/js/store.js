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
  adaptiveTesting: true,    // let adaptive.js choose exercise difficulty
  topicLevelFilter: 'all',  // Topics library: 'all' or a CEFR code
  tourDone: false,
  model: null,           // OpenRouter model override; null = server default
};

const SUPPORTED_LANGUAGES = ['fr', 'es', 'ru', 'zh'];
const SUPPORTED_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];

function freshState() {
  return {
    version: 2,
    createdAt: Date.now(),
    settings: { ...DEFAULT_SETTINGS },
    decks: {},        // lang -> { cards: {id: card}, topics: [] }
    stats: {},        // lang -> { days: { 'YYYY-MM-DD': {reviews, correct, newCards, timeMs} } }
    compositions: {}, // lang -> [compositionPack]
    grammar: {},      // lang -> { featureId: {seen, lastSeen} }
    ability: {},      // lang -> { score, samples, updatedAt, recent } (adaptive.js)
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
  // v2.1 identified compositions by title, which broke as soon as two pieces
  // shared one. Backfill stable ids for anything saved before v2.2.
  for (const list of Object.values(parsed.compositions)) {
    (list || []).forEach((pack, index) => {
      if (pack && !pack.id) pack.id = `legacy-${index}-${(pack.title || '').slice(0, 24)}`;
    });
  }
  if (!parsed.grammar) parsed.grammar = {};
  // v2.2 adds the adaptive ability estimate.
  if (!parsed.ability || typeof parsed.ability !== 'object') parsed.ability = {};
  if (!SUPPORTED_LANGUAGES.includes(parsed.settings.language)) parsed.settings.language = 'fr';
  // A level outside the scale silently broke the grammar focus and generation
  // defaults, so it is normalised on the way in rather than trusted.
  parsed.settings.level = normalizeLevelCode(parsed.settings.level);
  if (parsed.settings.topicLevelFilter !== 'all'
      && !SUPPORTED_LEVELS.includes(parsed.settings.topicLevelFilter)) {
    parsed.settings.topicLevelFilter = 'all';
  }
  return parsed;
}

function normalizeLevelCode(value) {
  const raw = String(value || '').trim().toUpperCase();
  return SUPPORTED_LEVELS.includes(raw) ? raw : DEFAULT_SETTINGS.level;
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
  const next = { ...patch };
  if ('level' in next) next.level = normalizeLevelCode(next.level);
  Object.assign(state.settings, next);
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

/**
 * Add lesson or curriculum items as cards.
 *
 * Defensive by design: a single malformed item from a model response must never
 * take down a whole lesson, so bad entries are counted and skipped rather than
 * thrown. Returns a report the caller can show the learner.
 *
 * @returns {{added: number, duplicates: number, skipped: number, ids: string[]}}
 */
export function addCards(items, topic, lang = currentLanguage(), meta = {}) {
  const d = deck(lang);
  const report = { added: 0, duplicates: 0, skipped: 0, ids: [] };
  const now = Date.now();
  for (const item of Array.isArray(items) ? items : []) {
    const target = typeof item?.target === 'string' ? item.target.trim() : '';
    const english = typeof item?.english === 'string' ? item.english.trim() : '';
    if (!target || !english) { report.skipped += 1; continue; }
    const id = cardId(target);
    if (!id) { report.skipped += 1; continue; }
    if (d.cards[id]) {
      report.duplicates += 1;
      // Late-arriving provenance still improves library progress tracking.
      if (!d.cards[id].unit && (item.unit || meta.unit)) d.cards[id].unit = item.unit || meta.unit;
      continue;
    }
    d.cards[id] = {
      id,
      target,
      english,
      pronunciation: str(item.pronunciation),
      example: str(item.example),
      exampleEn: str(item.example_en ?? item.exampleEn),
      note: str(item.note),
      tags: Array.isArray(item.tags) ? item.tags.filter((t) => typeof t === 'string') : [],
      topic: topic || '',
      unit: item.unit || meta.unit || '',
      level: item.level || meta.level || '',
      addedAt: now,
      introduced: false, // becomes true once studied in Learn
      suspended: false,
      state: 'new',
      srs: newSrsState(now),
    };
    report.added += 1;
    report.ids.push(id);
  }
  if (topic && !d.topics.includes(topic)) d.topics.unshift(topic);
  d.topics = d.topics.slice(0, 20);
  persist();
  emit('deck', lang);
  return report;
}

function str(value) {
  return typeof value === 'string' ? value.trim() : '';
}

/** Per-unit deck coverage, for the Topics library progress bars. */
export function unitCoverage(lang = currentLanguage()) {
  const map = {};
  for (const card of cards(lang)) {
    if (!card.unit) continue;
    if (!map[card.unit]) map[card.unit] = { inDeck: 0, learned: 0, mature: 0 };
    map[card.unit].inDeck += 1;
    if (card.state !== 'new') map[card.unit].learned += 1;
    if (card.srs?.S >= 21) map[card.unit].mature += 1;
  }
  return map;
}

/** Targets already in the deck, as a Set of card ids, for fast lookups. */
export function deckIndex(lang = currentLanguage()) {
  return new Set(Object.keys(deck(lang).cards));
}

export function normalizeTarget(target) {
  return cardId(String(target || ''));
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
let compositionSeq = 0;

/** Stable per-piece id so two compositions may legitimately share a title. */
function compositionId() {
  compositionSeq += 1;
  return `c${Date.now().toString(36)}${compositionSeq.toString(36)}`;
}

export function saveComposition(pack, lang = currentLanguage()) {
  if (!state.compositions[lang]) state.compositions[lang] = [];
  const stored = { ...pack, id: pack.id || compositionId(), savedAt: pack.savedAt || Date.now() };
  state.compositions[lang] = [stored, ...state.compositions[lang].filter((c) => c.id !== stored.id)].slice(0, 40);
  recordGrammarFeatures((stored.grammar_spotlights || []).map((s) => s.feature), lang);
  persist();
  emit('compositions', lang);
  return stored;
}

export function compositions(lang = currentLanguage()) {
  return state.compositions[lang] || [];
}

export function removeComposition(id, lang = currentLanguage()) {
  state.compositions[lang] = (state.compositions[lang] || []).filter((c) => c.id !== id);
  persist();
  emit('compositions', lang);
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
    throw new Error('Not a valid AI-Studio for Learning Languages export file.');
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
  if (state.ability) delete state.ability[lang];
  persist(true);
  emit('deck', lang);
  emit('ability', lang);
}

export function resetAll() {
  state = freshState();
  persist(true);
  emit('imported');
}
