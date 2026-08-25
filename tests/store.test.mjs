// Unit tests for the deck store, the layer that turns generated or curated
// items into cards. A single malformed item used to be able to throw here and
// take down a whole lesson after it had already been paid for, so these tests
// pin the defensive behaviour down.
//
// Run with: node --test tests/store.test.mjs

import test from 'node:test';
import assert from 'node:assert/strict';

// store.js is a browser module: give it the two globals it touches before it
// is imported, so it can run under plain node.
const memory = new Map();
globalThis.localStorage = {
  getItem: (k) => (memory.has(k) ? memory.get(k) : null),
  setItem: (k, v) => memory.set(k, String(v)),
  removeItem: (k) => memory.delete(k),
  clear: () => memory.clear(),
};
globalThis.window = { addEventListener() {} };

const store = await import('../static/js/store.js');

function freshDeck(lang = 'fr') {
  store.resetProgress(lang);
  return lang;
}

// -- addCards ----------------------------------------------------------------

test('addCards reports what it actually stored', () => {
  const lang = freshDeck();
  const report = store.addCards([
    { target: 'bonjour', english: 'hello' },
    { target: 'merci', english: 'thank you' },
  ], 'Greetings', lang);

  assert.equal(report.added, 2);
  assert.equal(report.duplicates, 0);
  assert.equal(report.skipped, 0);
  assert.deepEqual(report.ids, ['bonjour', 'merci']);
  assert.equal(store.cards(lang).length, 2);
});

test('malformed items are skipped, never thrown, and the rest still land', () => {
  const lang = freshDeck();
  const report = store.addCards([
    { target: 'bonjour', english: 'hello' },
    { english: 'no target at all' },
    { target: 'sans anglais' },
    { target: '   ', english: 'blank target' },
    null,
    undefined,
    'not an object',
    { target: 42, english: 'wrong type' },
    { target: 'merci', english: 'thank you' },
  ], 'Mixed', lang);

  assert.equal(report.added, 2, 'the two good items were stored');
  assert.equal(report.skipped, 7, 'every bad entry was counted, not thrown');
  assert.equal(store.cards(lang).length, 2);
});

test('addCards survives a non-array payload', () => {
  const lang = freshDeck();
  for (const payload of [null, undefined, {}, 'nope', 7]) {
    const report = store.addCards(payload, 'Bad payload', lang);
    assert.equal(report.added, 0);
  }
  assert.equal(store.cards(lang).length, 0);
});

test('duplicates are counted separately and do not overwrite the original', () => {
  const lang = freshDeck();
  store.addCards([{ target: 'bonjour', english: 'hello' }], 'First', lang);
  const report = store.addCards([
    { target: 'Bonjour', english: 'a different gloss' },
    { target: 'salut', english: 'hi' },
  ], 'Second', lang);

  assert.equal(report.added, 1);
  assert.equal(report.duplicates, 1, 'case-insensitive match counts as duplicate');
  assert.equal(store.deck(lang).cards.bonjour.english, 'hello', 'original gloss kept');
});

test('a duplicate still back-fills missing unit provenance', () => {
  const lang = freshDeck();
  store.addCards([{ target: 'bonjour', english: 'hello' }], 'Free lesson', lang);
  assert.equal(store.deck(lang).cards.bonjour.unit, '');

  store.addCards([{ target: 'bonjour', english: 'hello', unit: 'greetings' }], 'Greetings', lang);
  assert.equal(store.deck(lang).cards.bonjour.unit, 'greetings',
    'library progress should recognise a card added earlier from elsewhere');
});

test('items carry snake_case and camelCase example fields alike', () => {
  const lang = freshDeck();
  store.addCards([
    { target: 'a', english: 'a', example: 'x', example_en: 'from snake_case' },
    { target: 'b', english: 'b', example: 'y', exampleEn: 'from camelCase' },
  ], 'Fields', lang);
  assert.equal(store.deck(lang).cards.a.exampleEn, 'from snake_case');
  assert.equal(store.deck(lang).cards.b.exampleEn, 'from camelCase');
});

test('tags survive only when they are strings', () => {
  const lang = freshDeck();
  store.addCards([{ target: 'a', english: 'a', tags: ['noun', 7, null, 'trap'] }], 'Tags', lang);
  assert.deepEqual(store.deck(lang).cards.a.tags, ['noun', 'trap']);
  store.addCards([{ target: 'b', english: 'b', tags: 'not an array' }], 'Tags', lang);
  assert.deepEqual(store.deck(lang).cards.b.tags, []);
});

// -- unit coverage ------------------------------------------------------------

test('unitCoverage counts deck, learned, and mature per unit', () => {
  const lang = freshDeck();
  store.addCards([
    { target: 'un cafe', english: 'a coffee', unit: 'cafe-bar' },
    { target: 'une biere', english: 'a beer', unit: 'cafe-bar' },
    { target: 'la gare', english: 'the station', unit: 'directions' },
    { target: 'orphan', english: 'no unit' },
  ], 'Mixed', lang);

  let coverage = store.unitCoverage(lang);
  assert.equal(coverage['cafe-bar'].inDeck, 2);
  assert.equal(coverage['cafe-bar'].learned, 0);
  assert.equal(coverage.directions.inDeck, 1);
  assert.ok(!('undefined' in coverage), 'cards without a unit are not bucketed');

  const card = store.deck(lang).cards['un cafe'];
  card.state = 'review';
  card.srs.S = 40;
  coverage = store.unitCoverage(lang);
  assert.equal(coverage['cafe-bar'].learned, 1);
  assert.equal(coverage['cafe-bar'].mature, 1);
});

test('deckIndex and normalizeTarget agree on card identity', () => {
  const lang = freshDeck();
  store.addCards([{ target: '  Un   Café ', english: 'a coffee' }], 'Spacing', lang);
  const index = store.deckIndex(lang);
  assert.ok(index.has(store.normalizeTarget('un café')));
  assert.ok(index.has(store.normalizeTarget('UN    CAFÉ')));
  assert.ok(!index.has(store.normalizeTarget('un cafe')), 'accents are part of identity');
});

// -- compositions -------------------------------------------------------------

test('compositions get stable ids so two pieces may share a title', () => {
  const lang = freshDeck();
  const a = store.saveComposition({ title: 'Le marché', segments: [], language: lang }, lang);
  const b = store.saveComposition({ title: 'Le marché', segments: [], language: lang }, lang);

  assert.notEqual(a.id, b.id);
  assert.equal(store.compositions(lang).length, 2, 'the same title must not evict the earlier piece');

  store.removeComposition(a.id, lang);
  const left = store.compositions(lang);
  assert.equal(left.length, 1);
  assert.equal(left[0].id, b.id, 'only the targeted piece was removed');
});

test('resetProgress clears one language and leaves the others alone', () => {
  freshDeck('fr');
  freshDeck('es');
  store.addCards([{ target: 'bonjour', english: 'hello' }], 'fr', 'fr');
  store.addCards([{ target: 'hola', english: 'hello' }], 'es', 'es');

  store.resetProgress('fr');
  assert.equal(store.cards('fr').length, 0);
  assert.equal(store.cards('es').length, 1);
});
