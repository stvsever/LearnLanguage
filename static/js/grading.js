// Answer grading: normalization, edit distance, verdicts, and visual diffs.
//
// Design goals:
// - Diacritics matter for learning French, but a missing accent should not be
//   punished like a wrong word: verdict 'accents' scores as correct-with-note
//   unless the learner enables strict accent mode.
// - Small typos (Damerau-Levenshtein distance 1-2 depending on length) grade
//   as 'almost' so the scheduler can rate them Hard instead of Again.

const APOSTROPHES = /[’ʼ`´]/g;
const PUNCT_EDGES = /^[\s.,;:!?¿¡«»"()\-\u2013\u2014]+|[\s.,;:!?¿¡«»"()\-\u2013\u2014]+$/g;

export function stripDiacritics(text) {
  return text.normalize('NFD').replace(/[̀-ͯ]/g, '').normalize('NFC');
}

export function normalize(text, { keepDiacritics = true } = {}) {
  let out = String(text ?? '')
    .replace(APOSTROPHES, "'")
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
    .replace(PUNCT_EDGES, '');
  // Uniform internal punctuation spacing (French "?" spacing etc.)
  out = out.replace(/\s+([?!;:])/g, '$1').replace(/\s*'\s*/g, "'");
  if (!keepDiacritics) out = stripDiacritics(out);
  return out;
}

/** Damerau-Levenshtein distance (optimal string alignment). */
export function editDistance(a, b) {
  const m = a.length, n = b.length;
  if (!m) return n;
  if (!n) return m;
  const d = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i][0] = i;
  for (let j = 0; j <= n; j++) d[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
      }
    }
  }
  return d[m][n];
}

function typoBudget(length) {
  if (length <= 4) return 0;
  if (length <= 8) return 1;
  return 2;
}

/**
 * Grade a typed answer against the expected form.
 * Returns { verdict: 'correct'|'accents'|'almost'|'wrong', distance }.
 */
export function grade(answer, expected, { strictAccents = false, typoTolerance = true } = {}) {
  const a = normalize(answer);
  const e = normalize(expected);
  if (!a) return { verdict: 'wrong', distance: e.length };
  if (a === e) return { verdict: 'correct', distance: 0 };

  const aBase = stripDiacritics(a);
  const eBase = stripDiacritics(e);
  if (aBase === eBase) {
    return { verdict: strictAccents ? 'almost' : 'accents', distance: editDistance(a, e) };
  }
  const distance = editDistance(aBase, eBase);
  if (typoTolerance && distance <= typoBudget(eBase.length)) {
    return { verdict: 'almost', distance };
  }
  return { verdict: 'wrong', distance };
}

/** Map a grading verdict to a suggested FSRS rating. */
export function suggestedRating(verdict) {
  if (verdict === 'correct') return 3; // Good
  if (verdict === 'accents') return 3;
  if (verdict === 'almost') return 2;  // Hard
  return 1;                            // Again
}

// -- Character diff (LCS) for feedback rendering -----------------------------

function lcsMatrix(a, b) {
  const m = a.length, n = b.length;
  const table = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      table[i][j] = a[i - 1] === b[j - 1]
        ? table[i - 1][j - 1] + 1
        : Math.max(table[i - 1][j], table[i][j - 1]);
    }
  }
  return table;
}

/**
 * Diff the expected string against the answer.
 * Returns a list of ops over the *expected* string: {ch, ok} where ok means
 * the learner produced this character (in order).
 */
export function diffExpected(answer, expected) {
  const a = normalize(answer);
  const e = normalize(expected);
  const table = lcsMatrix(a, e);
  const ops = [];
  let i = a.length, j = e.length;
  while (i > 0 && j > 0) {
    if (a[i - 1] === e[j - 1]) {
      ops.unshift({ ch: e[j - 1], ok: true });
      i--; j--;
    } else if (table[i - 1][j] >= table[i][j - 1]) {
      i--; // extra char typed by learner - not shown on expected diff
    } else {
      ops.unshift({ ch: e[j - 1], ok: false });
      j--;
    }
  }
  while (j > 0) { ops.unshift({ ch: e[j - 1], ok: false }); j--; }
  return ops;
}

/** Similarity in [0,1] for speech-transcript scoring. */
export function similarity(a, b) {
  const x = normalize(a, { keepDiacritics: false });
  const y = normalize(b, { keepDiacritics: false });
  if (!x && !y) return 1;
  if (!x || !y) return 0;
  const distance = editDistance(x, y);
  return Math.max(0, 1 - distance / Math.max(x.length, y.length));
}
