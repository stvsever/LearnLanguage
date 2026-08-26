// Unit tests for the adaptive testing engine and the level model.
//
// The load-bearing test here is `every review mode is reachable`: the first
// calibration of this engine spaced the mode difficulties so widely that typed
// production and dictation could never clear the selection target at any
// attainable ability, which would have silently pinned every learner to the
// recognition rung forever. That class of bug is invisible in the interface
// and obvious in arithmetic, so it is pinned in arithmetic.
//
// Run with: node --test tests/adaptive.test.mjs

import test from 'node:test';
import assert from 'node:assert/strict';

const memory = new Map();
globalThis.localStorage = {
  getItem: (k) => (memory.has(k) ? memory.get(k) : null),
  setItem: (k, v) => memory.set(k, String(v)),
  removeItem: (k) => memory.delete(k),
  clear: () => memory.clear(),
};
globalThis.window = { addEventListener() {} };

const store = await import('../static/js/store.js');
const adaptive = await import('../static/js/adaptive.js');
const levels = await import('../static/js/levels.js');

const LANG = 'fr';

function reset(settings = {}) {
  adaptive.resetAbility(LANG);
  store.updateSettings({ adaptiveTesting: true, targetRetention: 0.9, level: 'A2', ...settings });
}

function card({ reps = 5, D = 5, lapses = 0, example = 'un exemple' } = {}) {
  return { target: 'x', english: 'x', example, srs: { reps, D, lapses, S: 10, lastReview: Date.now() } };
}

/** Feed the engine a run of answers at one mode. */
function answer(times, { mode = 'produce', verdict = 'correct' } = {}) {
  for (let i = 0; i < times; i++) adaptive.recordAttempt({ mode, verdict, card: card(), lang: LANG });
}

// -- calibration --------------------------------------------------------------

test('every review mode is reachable within the attainable ability range', () => {
  reset();
  const target = adaptive.targetSuccess();
  for (const mode of ['recognize', 'cloze', 'produce', 'dictation']) {
    const unlock = adaptive.unlockAbility(mode, target);
    assert.ok(unlock < 0.95,
      `${mode} unlocks at ${unlock.toFixed(3)}, which no learner reaches`);
    assert.ok(adaptive.predictSuccess(unlock + 0.001, adaptive.MODES[mode].difficulty) >= target,
      `${mode} does not actually clear the target at its own unlock point`);
  }
});

test('mode unlock points are strictly increasing and spread across the range', () => {
  reset();
  const ladder = ['recognize', 'cloze', 'produce', 'dictation'];
  const unlocks = ladder.map((mode) => adaptive.unlockAbility(mode));
  for (let i = 1; i < unlocks.length; i++) {
    assert.ok(unlocks[i] > unlocks[i - 1], `${ladder[i]} must be harder than ${ladder[i - 1]}`);
    assert.ok(unlocks[i] - unlocks[i - 1] >= 0.08, `${ladder[i]} unlocks too close to ${ladder[i - 1]}`);
  }
  assert.ok(unlocks[0] < 0.4, 'the easiest mode must be available almost immediately');
});

test('the selection target is desirable difficulty, not the retention setting', () => {
  reset({ targetRetention: 0.9 });
  assert.equal(Math.round(adaptive.targetSuccess() * 100), 80);
  reset({ targetRetention: 0.97 });
  assert.ok(adaptive.targetSuccess() > 0.8, 'a higher retention preference asks for more certainty');
  reset({ targetRetention: 0.8 });
  assert.ok(adaptive.targetSuccess() < 0.8, 'a lower retention preference accepts more risk');
});

// -- the estimate -------------------------------------------------------------

test('correct answers raise the estimate, wrong answers lower it', () => {
  reset();
  const start = adaptive.ability(LANG);
  answer(10, { verdict: 'correct' });
  const afterWins = adaptive.ability(LANG);
  assert.ok(afterWins > start, 'ten correct answers should raise ability');

  answer(10, { verdict: 'wrong' });
  assert.ok(adaptive.ability(LANG) < afterWins, 'ten wrong answers should lower it again');
});

test('the estimate stays inside 0..1 under sustained extremes', () => {
  reset();
  answer(200, { mode: 'dictation', verdict: 'correct' });
  assert.ok(adaptive.ability(LANG) <= 1 && adaptive.ability(LANG) > 0.8);
  answer(400, { mode: 'recognize', verdict: 'wrong' });
  assert.ok(adaptive.ability(LANG) >= 0 && adaptive.ability(LANG) < 0.3);
});

test('the step size shrinks as evidence accumulates', () => {
  assert.ok(adaptive.stepSize(0) > adaptive.stepSize(25));
  assert.ok(adaptive.stepSize(25) > adaptive.stepSize(200));
  assert.ok(adaptive.stepSize(10000) > 0, 'it never reaches zero');
});

test('an easy win moves the estimate less than a hard win', () => {
  reset();
  const easy = adaptive.recordAttempt({ mode: 'recognize', verdict: 'correct', card: card(), lang: LANG });
  reset();
  const hard = adaptive.recordAttempt({ mode: 'dictation', verdict: 'correct', card: card(), lang: LANG });
  assert.ok(hard.after - hard.before > easy.after - easy.before,
    'succeeding at something hard is more informative');
});

test('partial credit lands between correct and wrong', () => {
  assert.equal(adaptive.outcomeValue('correct'), 1);
  assert.equal(adaptive.outcomeValue('wrong'), 0);
  assert.ok(adaptive.outcomeValue('almost') > 0 && adaptive.outcomeValue('almost') < 1);
  assert.ok(adaptive.outcomeValue('accents') > adaptive.outcomeValue('almost'));
  assert.equal(adaptive.outcomeValue(true), 1);
  assert.equal(adaptive.outcomeValue(false), 0);
  assert.equal(adaptive.outcomeValue('nonsense'), 0, 'unknown verdicts must not credit the learner');
});

test('a leech counts as harder than the mode alone', () => {
  const easy = adaptive.cardPenalty(card({ D: 3, lapses: 0 }));
  const leech = adaptive.cardPenalty(card({ D: 9, lapses: 6 }));
  assert.ok(leech > easy);
  assert.ok(leech <= 0.16, 'one bad card cannot dominate the estimate');
  assert.equal(adaptive.cardPenalty(null), 0);
  assert.equal(adaptive.cardPenalty({}), 0);
});

test('samples are kept bounded', () => {
  reset();
  answer(120);
  assert.ok(adaptive.abilityRecord(LANG).recent.length <= 60);
  assert.equal(adaptive.abilityRecord(LANG).samples, 120, 'the count itself is not truncated');
});

// -- selection ----------------------------------------------------------------

test('selection climbs the ladder as ability grows', () => {
  reset();
  const seen = [];
  for (const score of [0.2, 0.35, 0.6, 0.75, 0.95]) {
    seen.push(adaptive.selectReviewMode(card(), LANG, { ability: score }));
  }
  const rank = (m) => adaptive.MODES[m].difficulty;
  for (let i = 1; i < seen.length; i++) {
    assert.ok(rank(seen[i]) >= rank(seen[i - 1]), `mode went backwards: ${seen}`);
  }
  assert.equal(seen[0], 'recognize', 'a struggling learner gets the floor');
  assert.equal(seen[seen.length - 1], 'dictation', 'a strong learner gets the ceiling');
});

test('a card without a usable example never gets a cloze', () => {
  reset();
  const chosen = adaptive.selectReviewMode(card(), LANG, {
    ability: 0.62, available: ['recognize', 'produce', 'dictation'],
  });
  assert.notEqual(chosen, 'cloze');
});

test('the first retrievals after learning stay easy on purpose', () => {
  reset();
  assert.equal(adaptive.selectReviewMode(card({ reps: 0 }), LANG, { ability: 0.95 }), 'recognize');
  assert.equal(adaptive.selectReviewMode(card({ reps: 1 }), LANG, { ability: 0.95 }), 'recognize');
  assert.notEqual(adaptive.selectReviewMode(card({ reps: 2 }), LANG, { ability: 0.95 }), 'recognize');
});

test('a leech is given an easier retrieval than a comfortable card', () => {
  reset();
  const opts = { ability: 0.73 };
  const easy = adaptive.selectReviewMode(card({ D: 3, lapses: 0 }), LANG, opts);
  const leech = adaptive.selectReviewMode(card({ D: 10, lapses: 6 }), LANG, opts);
  assert.ok(adaptive.MODES[leech].difficulty <= adaptive.MODES[easy].difficulty);
});

test('selection never returns a mode that was not offered', () => {
  reset();
  for (const score of [0, 0.3, 0.5, 0.8, 1]) {
    const available = ['recognize', 'produce'];
    assert.ok(available.includes(adaptive.selectReviewMode(card(), LANG, { ability: score, available })));
  }
  assert.equal(adaptive.selectReviewMode(card(), LANG, { available: [] }), 'produce', 'a safe default');
});

// -- integration surfaces -----------------------------------------------------

test('the learn ladder drops the recognition rung only once production is reliable', () => {
  reset();
  assert.deepEqual(adaptive.learnLadder(LANG), ['present', 'recognize', 'produce'],
    'a new learner keeps all three rungs');

  answer(60, { mode: 'dictation', verdict: 'correct' });
  assert.deepEqual(adaptive.learnLadder(LANG), ['present', 'produce'],
    'a strong learner goes straight to production');

  store.updateSettings({ adaptiveTesting: false });
  assert.deepEqual(adaptive.learnLadder(LANG), ['present', 'recognize', 'produce'],
    'with adaptive off the ladder is fixed regardless of ability');
});

test('the listen plan hardens with ability and honours an explicit request', () => {
  reset();
  const weak = adaptive.listenPlan(LANG);
  assert.equal(weak.mode, 'discrimination');
  assert.equal(weak.preferSentence, false);

  answer(60, { mode: 'dictation', verdict: 'correct' });
  const strong = adaptive.listenPlan(LANG);
  assert.equal(strong.mode, 'dictation');
  assert.ok(strong.distractors > weak.distractors);
  assert.equal(strong.preferSentence, true);

  assert.equal(adaptive.listenPlan(LANG, 'discrimination').mode, 'discrimination',
    'an explicit choice is never overridden');
});

test('the ability band always matches what the selector is doing', () => {
  reset();
  for (const score of [0.1, 0.35, 0.6, 0.8, 0.95]) {
    const band = adaptive.abilityBand(score);
    assert.ok(band && band.label && band.hint, `no band for ${score}`);
  }
  assert.equal(adaptive.abilityBand(0.05).key, 'finding-feet');
  assert.equal(adaptive.abilityBand(0.99).key, 'sharp');
});

test('the forecast marks exactly the modes the selector would pick', () => {
  reset();
  answer(40, { mode: 'produce', verdict: 'correct' });
  const forecast = adaptive.modeForecast(LANG);
  const chosen = adaptive.selectReviewMode(card(), LANG);
  const hardestSelected = forecast.filter((r) => r.selected).pop();
  assert.equal(hardestSelected?.mode ?? 'recognize', chosen);
});

// -- level recommendation ------------------------------------------------------

test('no level advice is offered before there is evidence', () => {
  reset();
  answer(5);
  assert.equal(adaptive.levelRecommendation(LANG, 'A2').action, 'hold');
});

test('sustained success suggests moving up, and never past C2', () => {
  reset();
  answer(60, { mode: 'dictation', verdict: 'correct' });
  const up = adaptive.levelRecommendation(LANG, 'A2');
  assert.equal(up.action, 'up');
  assert.equal(up.suggested, 'B1');
  assert.equal(adaptive.levelRecommendation(LANG, 'C2').action, 'hold', 'there is nothing above C2');
});

test('sustained failure suggests dropping down, and never below A1', () => {
  reset();
  answer(60, { mode: 'recognize', verdict: 'wrong' });
  const down = adaptive.levelRecommendation(LANG, 'B1');
  assert.equal(down.action, 'down');
  assert.equal(down.suggested, 'A2');
  assert.equal(adaptive.levelRecommendation(LANG, 'A1').action, 'hold', 'there is nothing below A1');
});

test('a nonsense level is refused rather than guessed at', () => {
  reset();
  answer(40);
  assert.equal(adaptive.levelRecommendation(LANG, 'Z9').action, 'hold');
});

// -- the level model -----------------------------------------------------------

test('every level has a name and a blurb', () => {
  for (const code of levels.LEVEL_ORDER) {
    assert.ok(levels.levelName(code), `${code} has no name`);
    assert.ok(levels.levelBlurb(code), `${code} has no blurb`);
    assert.equal(levels.levelLabel(code), `${code} ${levels.levelName(code)}`);
  }
});

test('levels are ordered and bad input is normalised, not thrown', () => {
  assert.equal(levels.levelIndex('A1'), 0);
  assert.ok(levels.levelIndex('C2') > levels.levelIndex('B1'));
  assert.equal(levels.normalizeLevel('b1'), 'B1');
  assert.equal(levels.normalizeLevel('nonsense'), 'A2');
  assert.equal(levels.normalizeLevel(null), 'A2');
  assert.equal(levels.normalizeLevel(''), 'A2');
});

test('stepping stays inside the scale', () => {
  assert.equal(levels.levelStep('A1', -1), 'A1');
  assert.equal(levels.levelStep('C2', 1), 'C2');
  assert.equal(levels.levelStep('B1', 1), 'B2');
  assert.equal(levels.levelStep('B1', -1), 'A2');
});

test('relation to the learner reads correctly in both directions', () => {
  assert.equal(levels.levelRelation('A1', 'B1').key, 'review');
  assert.equal(levels.levelRelation('B1', 'B1').key, 'at');
  assert.equal(levels.levelRelation('B2', 'B1').key, 'stretch');
  assert.equal(levels.levelRelation('C1', 'A2').key, 'ahead');
});

// -- the toggle ----------------------------------------------------------------

test('the estimate keeps updating while adaptive selection is off', () => {
  reset();
  store.updateSettings({ adaptiveTesting: false });
  assert.equal(adaptive.isEnabled(), false);
  const before = adaptive.ability(LANG);
  answer(15, { verdict: 'correct' });
  assert.ok(adaptive.ability(LANG) > before,
    'switching the toggle on later should start from real evidence');
});

test('resetting one language leaves the others alone', () => {
  reset();
  answer(20, { verdict: 'correct' });
  adaptive.recordAttempt({ mode: 'produce', verdict: 'correct', card: card(), lang: 'es' });
  const spanish = adaptive.ability('es');
  adaptive.resetAbility(LANG);
  assert.equal(adaptive.abilityRecord(LANG).samples, 0);
  assert.equal(adaptive.ability('es'), spanish);
});

test('a corrupted stored estimate is repaired rather than propagated', () => {
  reset();
  store.state.ability[LANG] = { score: NaN, samples: -5, recent: 'not an array' };
  const record = adaptive.abilityRecord(LANG);
  assert.ok(Number.isFinite(record.score));
  assert.equal(record.samples, 0);
  assert.deepEqual(record.recent, []);
});
