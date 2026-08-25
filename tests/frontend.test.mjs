// Unit tests for the pure frontend engines (FSRS scheduler + grading).
// Run with: node --test tests/frontend.test.mjs

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Rating, newSrsState, schedule, retrievability, intervalForRetention,
  previewIntervals, isDue, isMature,
} from '../static/js/srs.js';
import {
  normalize, stripDiacritics, editDistance, grade, suggestedRating,
  diffExpected, similarity,
} from '../static/js/grading.js';

const DAY_MS = 86400000;

function makeCard(now = Date.now()) {
  return { state: 'new', suspended: false, srs: newSrsState(now) };
}

// -- FSRS --------------------------------------------------------------------

test('new card walks learning steps and graduates', () => {
  const now = Date.now();
  const card = makeCard(now);

  schedule(card, Rating.GOOD, now, { applyFuzz: false });
  assert.equal(card.state, 'learning');
  assert.ok(card.srs.due > now && card.srs.due <= now + 11 * 60000, 'second step due within minutes');

  schedule(card, Rating.GOOD, card.srs.due, { applyFuzz: false });
  assert.equal(card.state, 'review');
  assert.ok(card.srs.S > 0, 'graduated with a stability');
  assert.ok(card.srs.due - card.srs.lastReview >= DAY_MS, 'first review interval >= 1 day');
});

test('Easy graduates immediately with a longer interval than Good', () => {
  const now = Date.now();
  const easyCard = makeCard(now);
  schedule(easyCard, Rating.EASY, now, { applyFuzz: false });
  assert.equal(easyCard.state, 'review');

  const goodCard = makeCard(now);
  schedule(goodCard, Rating.GOOD, now, { applyFuzz: false });
  schedule(goodCard, Rating.GOOD, goodCard.srs.due, { applyFuzz: false });
  assert.ok(easyCard.srs.S > goodCard.srs.S, 'easy stability exceeds good stability');
});

test('successful review grows stability; lapse shrinks it and relearns', () => {
  const now = Date.now();
  const card = makeCard(now);
  schedule(card, Rating.EASY, now, { applyFuzz: false });
  const s1 = card.srs.S;

  const reviewTime = card.srs.due;
  schedule(card, Rating.GOOD, reviewTime, { applyFuzz: false });
  assert.ok(card.srs.S > s1, 'stability grew after successful recall');

  const s2 = card.srs.S;
  schedule(card, Rating.AGAIN, card.srs.due, { applyFuzz: false });
  assert.equal(card.state, 'relearning');
  assert.equal(card.srs.lapses, 1);
  assert.ok(card.srs.S < s2, 'stability shrank after lapse');

  schedule(card, Rating.GOOD, card.srs.due, { applyFuzz: false });
  assert.equal(card.state, 'review');
});

test('higher target retention means shorter intervals', () => {
  assert.ok(intervalForRetention(10, 0.95) < intervalForRetention(10, 0.85));
});

test('retrievability decays over time', () => {
  const r0 = retrievability(0, 5);
  const r10 = retrievability(10, 5);
  assert.equal(r0, 1);
  assert.ok(r10 < r0 && r10 > 0);
});

test('previewIntervals orders Again < Hard <= Good <= Easy for review cards', () => {
  const now = Date.now();
  const card = makeCard(now);
  schedule(card, Rating.EASY, now, { applyFuzz: false });
  schedule(card, Rating.GOOD, card.srs.due, { applyFuzz: false });
  const p = previewIntervals(card, card.srs.due, 0.9);
  assert.ok(p[1] < p[2], 'again < hard');
  assert.ok(p[2] <= p[3], 'hard <= good');
  assert.ok(p[3] <= p[4], 'good <= easy');
});

test('isDue / isMature basics', () => {
  const now = Date.now();
  const card = makeCard(now);
  assert.equal(isDue(card, now), false, 'new cards are never due');
  schedule(card, Rating.GOOD, now, { applyFuzz: false });
  assert.equal(isDue(card, card.srs.due + 1), true);
  card.state = 'review';
  card.srs.S = 30;
  assert.equal(isMature(card), true);
});

// -- Grading -----------------------------------------------------------------

test('normalize handles case, whitespace, apostrophes, punctuation', () => {
  assert.equal(normalize('  Bonjour  !'), 'bonjour');
  assert.equal(normalize('L’EAU'), "l'eau");
  assert.equal(normalize('Où est la gare ?'), 'où est la gare');
});

test('stripDiacritics removes accents only', () => {
  assert.equal(stripDiacritics('éèêàçîïôûœ'), 'eeeaciiouœ'.replace('œ', 'œ'));
  assert.equal(stripDiacritics('déjà'), 'deja');
});

test('editDistance includes transpositions', () => {
  assert.equal(editDistance('bonjour', 'bonjour'), 0);
  assert.equal(editDistance('bonjuor', 'bonjour'), 1, 'transposition costs 1');
  assert.equal(editDistance('chat', 'chien'), 3);
});

test('grade verdicts: correct, accents, almost, wrong', () => {
  const opts = { strictAccents: false, typoTolerance: true };
  assert.equal(grade('déjà', 'déjà', opts).verdict, 'correct');
  assert.equal(grade('deja', 'déjà', opts).verdict, 'accents');
  assert.equal(grade('aujourdhui', "aujourd'hui", opts).verdict, 'almost');
  assert.equal(grade('bonjour', 'merci', opts).verdict, 'wrong');
  assert.equal(grade('', 'merci', opts).verdict, 'wrong');
});

test('strict accents demotes accent slips', () => {
  assert.equal(grade('deja', 'déjà', { strictAccents: true }).verdict, 'almost');
});

test('short words get no typo budget', () => {
  assert.equal(grade('pein', 'pain', { typoTolerance: true }).verdict, 'wrong');
  assert.equal(grade('bilette', 'billette', { typoTolerance: true }).verdict, 'almost');
});

test('suggestedRating maps verdicts to FSRS ratings', () => {
  assert.equal(suggestedRating('correct'), 3);
  assert.equal(suggestedRating('accents'), 3);
  assert.equal(suggestedRating('almost'), 2);
  assert.equal(suggestedRating('wrong'), 1);
});

test('diffExpected marks missed characters', () => {
  const ops = diffExpected('bonjor', 'bonjour');
  const missed = ops.filter((o) => !o.ok).map((o) => o.ch);
  assert.deepEqual(missed, ['u']);
  assert.equal(ops.map((o) => o.ch).join(''), 'bonjour');
});

test('similarity is 1 for equal, lower for different', () => {
  assert.equal(similarity('Bonjour', 'bonjour !'), 1);
  assert.ok(similarity('je voudrais un café', 'je voudrais un thé') > 0.6);
  assert.ok(similarity('bonjour', 'xyz') < 0.3);
});
