// Shared exercise components used by the Learn, Review, and Listen views:
// audio buttons, typing input with accent toolbar, choice grids, verdict
// feedback with character diffs, and the FSRS grading bar.

import { el, icon, fmtInterval } from './ui.js';
import { speak } from './audio.js';
import { diffExpected } from './grading.js';
import { previewIntervals, Rating } from './srs.js';
import { state } from './store.js';

export const ACCENT_CHARS = {
  fr: ['é', 'è', 'ê', 'ë', 'à', 'â', 'ç', 'î', 'ï', 'ô', 'û', 'ù', 'œ'],
  es: ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡'],
  de: ['ä', 'ö', 'ü', 'ß'],
  it: ['à', 'è', 'é', 'ì', 'ò', 'ù'],
  pt: ['ã', 'õ', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô', 'ç'],
  nl: ['é', 'ë', 'ï', 'ö', 'ü'],
  ru: [], zh: [],
};

export function audioButton(text, { lang, slow = false, kind = 'ghost', label } = {}) {
  const btn = el('button', {
    class: `btn btn-${kind} btn-audio`,
    type: 'button',
    title: slow ? 'Play slowly' : 'Play audio',
    onclick: async (e) => {
      e.stopPropagation();
      btn.classList.add('playing');
      await speak(text, { lang, slow });
      btn.classList.remove('playing');
    },
  }, icon(slow ? 'turtle' : 'volume', 18), label ? el('span', {}, label) : null);
  return btn;
}

export function typingInput({ placeholder = 'Type your answer…', lang, onSubmit, large = false }) {
  const input = el('input', {
    class: `type-input${large ? ' large' : ''}`,
    type: 'text',
    autocomplete: 'off', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false',
    placeholder,
    onkeydown: (e) => {
      if (e.key === 'Enter') { e.preventDefault(); onSubmit(input.value); }
    },
  });
  const children = [input];
  const accents = ACCENT_CHARS[lang] || [];
  if (accents.length && state.settings.accentToolbar) {
    children.push(el('div', { class: 'accent-bar' },
      accents.map((ch) => el('button', {
        class: 'accent-key', type: 'button', tabindex: '-1',
        onmousedown: (e) => e.preventDefault(), // keep input focus
        onclick: () => {
          const start = input.selectionStart ?? input.value.length;
          input.setRangeText(ch, start, input.selectionEnd ?? start, 'end');
          input.focus();
        },
      }, ch))));
  }
  const root = el('div', { class: 'typing-block' }, children);
  return { root, input, focus: () => input.focus() };
}

export function choiceGrid({ choices, correctIndex, onPick }) {
  let answered = false;
  const buttons = choices.map((choice, index) => el('button', {
    class: 'choice-btn', type: 'button',
    onclick: () => {
      if (answered) return;
      answered = true;
      buttons.forEach((b, i) => {
        b.disabled = true;
        if (i === correctIndex) b.classList.add('is-correct');
      });
      if (index !== correctIndex) buttons[index].classList.add('is-wrong');
      onPick(index === correctIndex, index);
    },
  }, el('span', { class: 'choice-key' }, String(index + 1)), el('span', {}, choice)));

  const root = el('div', { class: 'choice-grid' }, buttons);
  root.keyHandler = (e) => {
    const n = Number(e.key);
    if (n >= 1 && n <= buttons.length && !answered) buttons[n - 1].click();
  };
  return root;
}

function diffLine(answer, expected) {
  const ops = diffExpected(answer, expected);
  if (ops.every((op) => op.ok)) return null;
  return el('div', { class: 'diff-line', title: 'Highlighted characters were missing or wrong' },
    ops.map((op) => el('span', { class: op.ok ? 'd-ok' : 'd-bad' }, op.ch)));
}

export function verdictPanel({ verdict, answer, expected, card, lang }) {
  const labels = {
    correct: ['Correct', 'check', 'ok'],
    accents: ['Correct - watch the accents', 'check', 'accents'],
    almost: ['Almost - small slip', 'penLine', 'almost'],
    wrong: ['Not quite', 'x', 'wrong'],
  };
  const [text, iconName, cls] = labels[verdict] || labels.wrong;
  const rows = [
    el('div', { class: `verdict-head v-${cls}` }, icon(iconName, 18), el('strong', {}, text)),
    el('div', { class: 'verdict-answer' },
      el('span', { class: 'answer-target' }, expected),
      audioButton(expected, { lang })),
  ];
  if (verdict !== 'correct' && answer) {
    const diff = diffLine(answer, expected);
    if (diff) rows.push(diff);
  }
  if (card?.pronunciation && state.settings.showPronunciation) {
    rows.push(el('div', { class: 'verdict-ipa' }, card.pronunciation));
  }
  if (card?.example) {
    rows.push(el('div', { class: 'verdict-example' },
      el('span', {}, card.example),
      card.exampleEn ? el('small', {}, card.exampleEn) : null));
  }
  if (card?.note) rows.push(el('div', { class: 'verdict-note' }, icon('lightbulb', 14), el('span', {}, card.note)));
  return el('div', { class: 'verdict-panel' }, rows);
}

export function gradeBar({ card, suggested, onGrade }) {
  const previews = previewIntervals(card, Date.now(), state.settings.targetRetention);
  const defs = [
    [Rating.AGAIN, 'Again', 'g-again'],
    [Rating.HARD, 'Hard', 'g-hard'],
    [Rating.GOOD, 'Good', 'g-good'],
    [Rating.EASY, 'Easy', 'g-easy'],
  ];
  const root = el('div', { class: 'grade-bar' },
    defs.map(([rating, label, cls]) => el('button', {
      class: `grade-btn ${cls}${rating === suggested ? ' suggested' : ''}`,
      type: 'button',
      onclick: () => onGrade(rating),
    },
      el('span', { class: 'grade-key' }, String(rating)),
      el('strong', {}, label),
      el('small', {}, fmtInterval(previews[rating])))));
  root.keyHandler = (e) => {
    const n = Number(e.key);
    if (n >= 1 && n <= 4) { onGrade(n); return true; }
    if (e.key === 'Enter' && suggested) { onGrade(suggested); return true; }
    return false;
  };
  return root;
}

export function progressBar(current, total) {
  const pct = total ? Math.min(100, Math.round((current / total) * 100)) : 0;
  return el('div', { class: 'session-progress' },
    el('div', { class: 'progress-track' }, el('span', { style: { width: `${pct}%` } })),
    el('span', { class: 'progress-count' }, `${current} / ${total}`));
}

/** Pick distractor strings for an MCQ from the deck, most-similar-first. */
export function pickDistractors(pool, correct, count, key = (c) => c) {
  const target = key(correct);
  const candidates = pool
    .filter((c) => key(c) !== target && key(c))
    .map((c) => ({ value: key(c), score: crudeSimilarity(key(c), target) }))
    .sort((a, b) => b.score - a.score);
  const unique = [];
  for (const cand of candidates) {
    if (!unique.includes(cand.value)) unique.push(cand.value);
    if (unique.length >= count * 3) break;
  }
  // mix: half similar, half random for less guessable options
  const similar = unique.slice(0, count);
  return similar.slice(0, count);
}

function crudeSimilarity(a, b) {
  const setA = new Set(a.toLowerCase());
  const setB = new Set(b.toLowerCase());
  let common = 0;
  for (const ch of setA) if (setB.has(ch)) common += 1;
  const lengthCloseness = 1 - Math.abs(a.length - b.length) / Math.max(a.length, b.length, 1);
  return common / Math.max(setA.size, setB.size, 1) + lengthCloseness + Math.random() * 0.4;
}
