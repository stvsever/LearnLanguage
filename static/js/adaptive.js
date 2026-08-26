// Adaptive testing: choose how hard each retrieval should be, automatically.
//
// The scheduler (srs.js) decides WHEN a card comes back. This decides HOW it
// comes back. Those are different questions, and answering the second one well
// is what keeps a session at the edge of ability instead of drifting into
// either boredom or failure.
//
// The model is deliberately small and legible:
//
//  1. Every exercise mode carries a difficulty on a 0..1 scale, ordered by how
//     much retrieval effort it demands (recognising a meaning is cheap, typing
//     a sentence from audio is not).
//  2. One ability estimate per language, also 0..1, updated after every graded
//     answer by an Elo-style step: the surprise of the outcome moves the
//     estimate, and the step size shrinks as evidence accumulates.
//  3. Before each item, the predicted success rate of every mode is computed
//     from ability minus difficulty. The hardest mode still predicted to
//     succeed at the target rate wins.
//
// The target rate sits at 80% (desirable difficulty), nudged by the learner's
// retention preference, so one dial influences both scheduling and testing
// without the two being conflated. See targetSuccess().
//
// With adaptive testing switched off, callers fall back to their fixed
// rotations and nothing here runs except the bookkeeping.

import { state, persist, emit, currentLanguage } from './store.js';

/**
 * Exercise modes, ordered by retrieval cost.
 *
 * `difficulty` is where a learner of equal ability has a coin-flip chance. The
 * values are CALIBRATED against the selection rule, not picked by feel: a mode
 * only unlocks once ability exceeds its difficulty by ln(4)/SLOPE, so the four
 * review modes have to be spaced to unlock across the range of ability a real
 * learner reaches (roughly 0.3 to 0.9). Spacing them any wider makes the top
 * rungs mathematically unreachable, which is exactly the bug this replaced.
 * `adaptive.test.mjs` asserts every mode is still reachable.
 */
export const MODES = {
  recognize: { difficulty: 0.18, label: 'Recognise meaning', kind: 'choice' },
  discriminate: { difficulty: 0.30, label: 'Hear and choose', kind: 'choice' },
  cloze: { difficulty: 0.42, label: 'Fill the gap in context', kind: 'typing' },
  produce: { difficulty: 0.58, label: 'Type from meaning', kind: 'typing' },
  dictation: { difficulty: 0.74, label: 'Type what you hear', kind: 'typing' },
};

// Review offers these; Listen has its own two-mode ladder.
const REVIEW_LADDER = ['recognize', 'cloze', 'produce', 'dictation'];

const START_ABILITY = 0.35;   // a fresh learner is assumed to be near the bottom
const MAX_STEP = 0.09;        // how far one answer can move the estimate at most
const MIN_STEP = 0.012;       // ... and once there is plenty of evidence
const STEP_HALF_LIFE = 24;    // answers until the step size is halfway to MIN
const SLOPE = 10;             // logistic steepness of ability vs difficulty
const RECENT_WINDOW = 60;     // samples kept for the readout and the trend

// -- state --------------------------------------------------------------------

function slice(lang) {
  if (!state.ability) state.ability = {};
  if (!state.ability[lang]) {
    state.ability[lang] = { score: START_ABILITY, samples: 0, updatedAt: null, recent: [] };
  }
  const record = state.ability[lang];
  // Guard against a hand-edited or partially migrated import.
  if (!Number.isFinite(record.score)) record.score = START_ABILITY;
  record.score = clamp01(record.score);
  if (!Number.isFinite(record.samples) || record.samples < 0) record.samples = 0;
  if (!Array.isArray(record.recent)) record.recent = [];
  return record;
}

export function ability(lang = currentLanguage()) {
  return slice(lang).score;
}

export function abilityRecord(lang = currentLanguage()) {
  const record = slice(lang);
  return { ...record, recent: [...record.recent] };
}

export function isEnabled() {
  return state.settings.adaptiveTesting !== false;
}

/**
 * Success rate the selector aims for.
 *
 * This is NOT the retention target: retention is how much you should still
 * remember when a card comes back, and aiming a live exercise at 90% would
 * park everyone on the easiest rung forever. Desirable difficulty sits around
 * 80%, so the retention preference nudges a 0.8 baseline rather than setting
 * it, keeping one dial in charge of both without conflating them.
 */
export function targetSuccess() {
  const retention = Number(state.settings.targetRetention);
  const preference = Number.isFinite(retention) ? retention : 0.9;
  return clamp(0.8 + (preference - 0.9) * 0.8, 0.68, 0.9);
}

/** Ability at which a mode starts being selected, for calibration and tests. */
export function unlockAbility(mode, target = targetSuccess()) {
  const difficulty = MODES[mode]?.difficulty ?? 0.5;
  return difficulty + Math.log(target / (1 - target)) / SLOPE;
}

// -- the maths ----------------------------------------------------------------

function clamp(value, low, high) { return Math.min(high, Math.max(low, value)); }
function clamp01(value) { return clamp(value, 0, 1); }

/** Probability that a learner of this ability passes an item of this difficulty. */
export function predictSuccess(abilityScore, difficulty) {
  return 1 / (1 + Math.exp(-SLOPE * (abilityScore - difficulty)));
}

/** Step size shrinks with evidence, so early answers move the estimate fast. */
export function stepSize(samples) {
  const decay = Math.pow(0.5, Math.max(0, samples) / STEP_HALF_LIFE);
  return MIN_STEP + (MAX_STEP - MIN_STEP) * decay;
}

/** Graded verdicts map onto a partial-credit outcome. */
export function outcomeValue(verdict) {
  if (verdict === true) return 1;
  if (verdict === false) return 0;
  switch (verdict) {
    case 'correct': return 1;
    case 'accents': return 0.9;
    case 'almost': return 0.5;
    case 'wrong': return 0;
    default: return 0;
  }
}

/**
 * How hard this particular card is, on top of the mode.
 *
 * A card the learner keeps forgetting is harder than the mode alone implies,
 * and the FSRS difficulty parameter already tracks exactly that. Bounded so a
 * single leech cannot dominate the estimate.
 */
export function cardPenalty(card) {
  if (!card?.srs) return 0;
  const fromDifficulty = ((card.srs.D || 5) - 5) / 5 * 0.08; // D is 1..10
  const fromLapses = Math.min(card.srs.lapses || 0, 4) * 0.02;
  return clamp(fromDifficulty + fromLapses, -0.08, 0.16);
}

// -- recording ----------------------------------------------------------------

/**
 * Record one graded attempt and move the ability estimate.
 * Runs whether or not adaptive selection is switched on, so turning the toggle
 * on later starts from a real estimate rather than from zero.
 *
 * @returns {{before: number, after: number, expected: number}}
 */
export function recordAttempt({ mode, verdict, card, lang = currentLanguage() }) {
  const record = slice(lang);
  const difficulty = clamp01((MODES[mode]?.difficulty ?? 0.5) + cardPenalty(card));
  const observed = outcomeValue(verdict);
  const expected = predictSuccess(record.score, difficulty);
  const before = record.score;

  record.score = clamp01(record.score + stepSize(record.samples) * (observed - expected));
  record.samples += 1;
  record.updatedAt = Date.now();
  record.recent.push({ t: Date.now(), mode, difficulty: round3(difficulty), observed, score: round3(record.score) });
  if (record.recent.length > RECENT_WINDOW) record.recent.splice(0, record.recent.length - RECENT_WINDOW);

  persist();
  emit('ability', lang);
  return { before, after: record.score, expected };
}

function round3(value) { return Math.round(value * 1000) / 1000; }

// -- selection ----------------------------------------------------------------

/**
 * Pick the retrieval mode for one review.
 *
 * Returns the hardest mode still predicted to succeed at the target rate; if
 * none clears the bar, the easiest available mode, because a learner who is
 * failing everything needs the floor, not the ceiling.
 */
export function selectReviewMode(card, lang = currentLanguage(), options = {}) {
  const available = (options.available || REVIEW_LADDER).filter((mode) => MODES[mode]);
  if (!available.length) return 'produce';

  // The first couple of retrievals after learning stay easy on purpose: the
  // point is a successful recall, not a measurement.
  if ((card?.srs?.reps ?? 0) < 2) return available[0];

  const score = options.ability ?? ability(lang);
  const penalty = cardPenalty(card);
  const target = options.target ?? targetSuccess();

  const ranked = [...available].sort((a, b) => MODES[a].difficulty - MODES[b].difficulty);
  let chosen = ranked[0];
  for (const mode of ranked) {
    if (predictSuccess(score, clamp01(MODES[mode].difficulty + penalty)) >= target) chosen = mode;
  }
  return chosen;
}

/** What the selector would say for every mode, for the Settings readout. */
export function modeForecast(lang = currentLanguage()) {
  const score = ability(lang);
  const target = targetSuccess();
  return REVIEW_LADDER.map((mode) => ({
    mode,
    label: MODES[mode].label,
    difficulty: MODES[mode].difficulty,
    predicted: predictSuccess(score, MODES[mode].difficulty),
    selected: predictSuccess(score, MODES[mode].difficulty) >= target,
  }));
}

/**
 * The encoding ladder for a brand-new item.
 * A capable learner does not need the recognition rung; going straight from
 * presentation to production is the stronger encoding event.
 */
export function learnLadder(lang = currentLanguage()) {
  if (!isEnabled()) return ['present', 'recognize', 'produce'];
  const score = ability(lang);
  const record = slice(lang);
  if (record.samples < 12) return ['present', 'recognize', 'produce'];
  // The recognition rung earns its place until typed production is reliable.
  if (predictSuccess(score, MODES.produce.difficulty) >= targetSuccess()) return ['present', 'produce'];
  return ['present', 'recognize', 'produce'];
}

/** Listening lab: which drill, on what material, against how many distractors. */
export function listenPlan(lang = currentLanguage(), requested = null) {
  const score = ability(lang);
  if (!isEnabled()) {
    return { mode: requested || 'dictation', distractors: 3, preferSentence: true, adaptive: false };
  }
  const mode = requested || (predictSuccess(score, MODES.dictation.difficulty) >= targetSuccess()
    ? 'dictation' : 'discrimination');
  return {
    mode,
    // Harder means more confusable options and longer material to hold in mind.
    distractors: score >= 0.6 ? 5 : score >= 0.4 ? 4 : 3,
    // A learner at the starting estimate transcribes single items; whole
    // sentences only once they are clear of the bottom of the scale.
    preferSentence: score >= 0.45,
    adaptive: true,
  };
}

// -- what the estimate means --------------------------------------------------

// Band edges sit on the mode unlock points, so the label always matches what
// the learner is actually being asked to do.
const BANDS = [
  { max: 0.32, key: 'finding-feet', label: 'Finding your feet', hint: 'Recognition first, production once meanings stick.' },
  { max: 0.56, key: 'building', label: 'Building', hint: 'Recognition, with gap-filling in context unlocking next.' },
  { max: 0.72, key: 'steady', label: 'Steady', hint: 'Filling gaps in context, with typed production next up.' },
  { max: 0.88, key: 'strong', label: 'Strong', hint: 'Typed production by default; dictation is within reach.' },
  { max: 1.01, key: 'sharp', label: 'Sharp', hint: 'The hardest retrievals, including typing from audio alone.' },
];

export function abilityBand(score = ability()) {
  return BANDS.find((band) => score < band.max) || BANDS[BANDS.length - 1];
}

/** Recent accuracy, for the readout. Null until there is anything to report. */
export function recentAccuracy(lang = currentLanguage(), window = 20) {
  const recent = slice(lang).recent.slice(-window);
  if (!recent.length) return null;
  return recent.reduce((sum, sample) => sum + sample.observed, 0) / recent.length;
}

/**
 * Whether the learner's CEFR setting still matches the evidence.
 *
 * Deliberately conservative: it needs a real sample, a clear ability signal,
 * and sustained recent accuracy before it says anything, and it never changes
 * the setting on its own. The learner decides; this only points.
 */
export function levelRecommendation(lang = currentLanguage(), currentLevel = state.settings.level) {
  const record = slice(lang);
  if (record.samples < 25) {
    return { action: 'hold', reason: `${25 - record.samples} more answers before there is enough evidence to judge.` };
  }
  const accuracy = recentAccuracy(lang, 20);
  const order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];
  const index = order.indexOf(String(currentLevel || 'A2').toUpperCase());
  if (index === -1) return { action: 'hold', reason: 'Set a level in Settings first.' };

  if (record.score >= unlockAbility('produce') && accuracy >= 0.85 && index < order.length - 1) {
    return {
      action: 'up',
      suggested: order[index + 1],
      reason: `You are clearing the hardest retrievals at ${Math.round(accuracy * 100)}% accuracy. New material can be pitched a level higher.`,
    };
  }
  if (record.score <= 0.3 && accuracy <= 0.6 && index > 0) {
    return {
      action: 'down',
      suggested: order[index - 1],
      reason: `Recent accuracy is ${Math.round(accuracy * 100)}%. A level down would rebuild the base faster than pushing on.`,
    };
  }
  return {
    action: 'hold',
    reason: `Ability and accuracy both sit where ${currentLevel} expects them. Nothing to change.`,
  };
}

/** Wipe the estimate for one language (used by the Settings reset). */
export function resetAbility(lang = currentLanguage()) {
  if (state.ability) delete state.ability[lang];
  persist();
  emit('ability', lang);
}
