// Listen view: dedicated auditory training.
//  - Dictation: hear a sentence or item, type it exactly (phoneme->grapheme mapping).
//  - Discrimination: hear one item, pick it among orthographically similar ones
//    (sharpens phonemic categories the L1 ear collapses).

import { el, icon, shuffled, sample } from '../ui.js';
import { state, learnedCards, cards, recordReview, currentLanguage } from '../store.js';
import { speak, feedbackTone, stopAudio } from '../audio.js';
import { grade } from '../grading.js';
import { typingInput, verdictPanel, audioButton, choiceGrid, progressBar } from '../exercises.js';
import { ctx } from '../context.js';
import { listenPlan, recordAttempt, isEnabled as adaptiveOn, abilityBand, ability } from '../adaptive.js';

const ROUND_LENGTH = 8;
let session = null;
let keyHandler = null;

export function render(container) {
  cleanup();
  session = null;
  const lang = currentLanguage();
  const pool = learnedCards(lang);
  const ready = pool.length >= 4;
  const plan = listenPlan(lang);
  const recommended = plan.mode;

  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('headphones', 28)),
        el('h2', {}, 'Listening lab'),
        el('p', { class: 'muted' }, ready
          ? 'Train your ear on the words you already know. Dictation builds sound-to-spelling mapping; discrimination sharpens similar-sounding words.'
          : 'Learn at least 4 items first - listening drills reuse your own deck so every rep also reinforces vocabulary.'),
        ready && adaptiveOn() ? el('p', { class: 'muted small adaptive-note' },
          icon('zap', 13),
          el('span', {}, `Adaptive testing suggests ${recommended === 'dictation' ? 'dictation' : 'discrimination'} right now, with ${plan.distractors} options and ${plan.preferSentence ? 'full sentences' : 'single items'}. Either button still works.`)) : null,
        el('div', { class: 'row gap center' },
          el('button', {
            class: `btn btn-lg ${recommended === 'dictation' ? 'btn-primary' : 'btn-soft'}`,
            disabled: !ready || undefined, onclick: () => start(container, 'dictation'),
          }, icon('penLine', 18), 'Dictation'),
          el('button', {
            class: `btn btn-lg ${recommended === 'discrimination' ? 'btn-primary' : 'btn-soft'}`,
            disabled: !ready || undefined, onclick: () => start(container, 'discrimination'),
          }, icon('ear', 18), 'Discrimination')),
        !ready ? el('button', { class: 'btn btn-ghost', style: { marginTop: '10px' }, onclick: () => ctx.navigate('learn') }, 'Go to Learn') : null)));
}

export function cleanup() {
  if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
  stopAudio();
}

function start(container, mode) {
  const lang = currentLanguage();
  const pool = shuffled(learnedCards(lang));
  const items = pool.slice(0, ROUND_LENGTH);
  // The plan sets how hard the round is: sentence or single item for
  // dictation, and how many confusable options for discrimination.
  session = { mode, lang, items, index: 0, correct: 0, startedAt: Date.now(), plan: listenPlan(lang, mode) };
  step(container);
}

function step(container) {
  cleanup();
  if (session.index >= session.items.length) { summary(container); return; }
  const card = session.items[session.index];
  const stage = el('div', { class: 'exercise-stage' });

  container.replaceChildren(
    el('div', { class: 'view-inner narrow session' },
      el('div', { class: 'session-head' },
        progressBar(session.index, session.items.length),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => render(container) }, 'End')),
      stage));

  if (session.mode === 'dictation') dictation(stage, card, container);
  else discrimination(stage, card, container);
}

function useSentence(card) {
  // Prefer the example sentence when it's not too long - richer signal - but
  // only once the learner can hold that much; below that, single items.
  if (!session?.plan?.preferSentence) return card.target;
  return card.example && card.example.length <= 90 ? card.example : card.target;
}

function dictation(stage, card, container) {
  const text = useSentence(card);
  let answered = false;
  const typing = typingInput({
    lang: session.lang,
    placeholder: 'Type exactly what you hear…',
    large: text.length > 30,
    onSubmit: (value) => {
      if (answered) return;
      answered = true;
      typing.input.disabled = true;
      const result = grade(value, text, { strictAccents: state.settings.strictAccents, typoTolerance: true });
      const ok = result.verdict === 'correct' || result.verdict === 'accents';
      if (ok) session.correct += 1;
      feedbackTone(ok ? 'correct' : 'wrong');
      recordAttempt({ mode: 'dictation', verdict: result.verdict, card, lang: session.lang });
      recordReview({ correct: ok, ms: 0, mode: 'listen' }, session.lang);
      stage.querySelector('.quiz-card').append(
        verdictPanel({ verdict: result.verdict, answer: value, expected: text, card: text === card.target ? card : null, lang: session.lang }),
        el('button', { class: 'btn btn-primary continue-btn', onclick: () => { session.index += 1; step(container); } },
          'Continue', icon('arrowRight', 16)));
      keyHandler = (e) => { if (e.key === 'Enter') { e.preventDefault(); session.index += 1; step(container); } };
      // Deferred so the submitting Enter doesn't immediately advance.
      setTimeout(() => { if (keyHandler) document.addEventListener('keydown', keyHandler); }, 0);
    },
  });

  stage.append(el('div', { class: 'quiz-card' },
    el('span', { class: 'phase-tag' }, 'Dictation'),
    el('div', { class: 'listen-controls' },
      audioButton(text, { lang: session.lang, kind: 'soft', label: 'Play' }),
      audioButton(text, { lang: session.lang, slow: true, kind: 'ghost', label: 'Slow' })),
    typing.root));
  speak(text, { lang: session.lang });
  typing.focus();
}

function discrimination(stage, card, container) {
  const pool = cards(session.lang).filter((c) => c.id !== card.id);
  // Most orthographically confusable first
  const scored = pool
    .map((c) => ({ c, s: confusability(c.target, card.target) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, 8);
  const wanted = Math.max(2, (session.plan?.distractors || 4) - 1);
  const distractors = sample(scored, Math.min(wanted, scored.length)).map((x) => x.c.target);
  const choices = shuffled([card.target, ...distractors]);
  const correctIndex = choices.indexOf(card.target);

  const grid = choiceGrid({
    choices, correctIndex,
    onPick: (correct) => {
      if (correct) session.correct += 1;
      feedbackTone(correct ? 'correct' : 'wrong');
      recordAttempt({ mode: 'discriminate', verdict: correct, card, lang: session.lang });
      recordReview({ correct, ms: 0, mode: 'listen' }, session.lang);
      setTimeout(() => { session.index += 1; step(container); }, correct ? 600 : 1400);
    },
  });

  stage.append(el('div', { class: 'quiz-card' },
    el('span', { class: 'phase-tag' }, 'What did you hear?'),
    el('div', { class: 'listen-controls' },
      audioButton(card.target, { lang: session.lang, kind: 'soft', label: 'Replay' }),
      audioButton(card.target, { lang: session.lang, slow: true, kind: 'ghost', label: 'Slow' })),
    grid));
  speak(card.target, { lang: session.lang });
  keyHandler = (e) => grid.keyHandler?.(e);
  document.addEventListener('keydown', keyHandler);
}

function confusability(a, b) {
  const x = a.toLowerCase(), y = b.toLowerCase();
  let score = 0;
  if (x[0] === y[0]) score += 2;
  if (x.slice(-2) === y.slice(-2)) score += 2;
  score += 3 - Math.min(3, Math.abs(x.length - y.length));
  const setX = new Set(x), setY = new Set(y);
  let common = 0;
  for (const ch of setX) if (setY.has(ch)) common += 1;
  score += common / Math.max(setX.size, setY.size);
  return score + Math.random();
}

function summary(container) {
  const { correct, items, lang } = session;
  const band = abilityBand(ability(lang));
  session = null;
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, `${correct} / ${items.length} correct`),
        el('p', { class: 'muted' }, correct === items.length
          ? 'Perfect ear. Try a faster voice speed in Settings for a harder challenge.'
          : 'Missed items stay in your review pipeline - the scheduler will bring them back.'),
        adaptiveOn() ? el('p', { class: 'muted small' },
          `Adaptive testing now reads your level as ${band.label.toLowerCase()}.`) : null,
        el('div', { class: 'row gap center' },
          el('button', { class: 'btn btn-primary', onclick: () => render(container) }, 'Another round'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('dashboard') }, 'Dashboard')))));
}
