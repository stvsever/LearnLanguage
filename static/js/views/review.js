// Review view: the FSRS queue with interleaved retrieval modalities.
//
// Exercise type adapts to card maturity (retrieval difficulty grows with
// memory strength - a desirable difficulty):
//   early reps  -> recognition (target -> meaning)
//   young cards -> typed production (meaning -> target)
//   mature      -> listening transcription or cloze-in-context.

import { el, icon, toast, shuffled, sample } from '../ui.js';
import {
  state, cards, dueCards, recordReview, recordTime, reviewsDoneToday,
  currentLanguage, persist,
} from '../store.js';
import { speak, preload, feedbackTone, stopAudio } from '../audio.js';
import { grade, suggestedRating } from '../grading.js';
import { schedule, Rating } from '../srs.js';
import { typingInput, verdictPanel, choiceGrid, audioButton, progressBar, gradeBar } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';
import { isEnabled as adaptiveOn, selectReviewMode, recordAttempt, MODES } from '../adaptive.js';

let session = null;
let keyHandler = null;

export function render(container) {
  cleanup();
  const lang = currentLanguage();

  if (session && session.lang === lang && session.index < session.queue.length) {
    renderCard(container);
    return;
  }
  session = null;

  const capLeft = Math.max(0, state.settings.maxReviewsPerDay - reviewsDoneToday(lang));
  const queue = dueCards(lang).slice(0, capLeft);

  if (queue.length === 0) {
    renderEmpty(container, capLeft);
    return;
  }

  session = {
    lang, queue: shuffleLearningLast(queue), index: 0,
    startedAt: Date.now(), cardStart: Date.now(),
    results: [],
  };
  preload(session.queue.slice(0, 4).map((c) => c.target), lang);
  renderCard(container);
}

export function cleanup() {
  if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
  stopAudio();
}

function shuffleLearningLast(queue) {
  // Interleave: overdue review cards first (most at risk), learning steps mixed in.
  const learning = queue.filter((c) => c.state === 'learning' || c.state === 'relearning');
  const review = queue.filter((c) => c.state === 'review');
  return [...shuffled(review), ...shuffled(learning)];
}

/**
 * Which retrieval this card gets.
 *
 * With adaptive testing on, adaptive.js picks the hardest mode the learner is
 * still predicted to pass, from the modes this card can actually support (a
 * card whose target does not appear literally in its example cannot be clozed).
 * With it off, the previous fixed rotation applies unchanged.
 */
function pickMode(card) {
  const available = ['recognize'];
  if (clozeSentence(card)) available.push('cloze');
  available.push('produce', 'dictation');

  if (adaptiveOn()) return selectReviewMode(card, session.lang, { available });

  if (card.srs.reps < 2) return 'recognize';
  const rotation = available.includes('cloze')
    ? ['produce', 'dictation', 'produce', 'cloze']
    : ['produce', 'dictation'];
  return rotation[(card.srs.reps - 2) % rotation.length];
}

/** The cloze display for a card, or null when the item is not literally there. */
function clozeSentence(card) {
  if (!card.example) return null;
  const pattern = new RegExp(escapeRegExp(card.target), 'i');
  return pattern.test(card.example) ? card.example.replace(pattern, '_____') : null;
}

function renderEmpty(container, capLeft) {
  const nextDue = cards(currentLanguage())
    .filter((c) => c.state !== 'new' && !c.suspended)
    .sort((a, b) => a.srs.due - b.srs.due)[0];
  let hint = 'Nothing scheduled. Learn new items, or pick a topic from the library.';
  if (capLeft === 0) hint = 'You reached today\'s review cap - adjustable in Settings.';
  else if (nextDue) {
    const minutes = Math.max(1, Math.round((nextDue.srs.due - Date.now()) / 60000));
    hint = minutes < 90
      ? `Next review due in ~${minutes} min.`
      : `Next review due ${new Date(nextDue.srs.due).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })}.`;
  }
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, 'Queue clear'),
        el('p', { class: 'muted' }, hint),
        el('div', { class: 'row gap center wrap' },
          el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('learn') }, icon('sparkles', 16), 'Learn new items'),
          el('button', { class: 'btn btn-soft', onclick: () => ctx.navigate('topics') }, icon('layers', 16), 'Browse topics'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('compose') }, icon('book', 16), 'Compose')))));
}

function renderCard(container) {
  cleanup();
  const { queue, index } = session;
  if (index >= queue.length) { renderSummary(container); return; }
  const card = queue[index];
  session.cardStart = Date.now();
  const mode = pickMode(card);
  const stage = el('div', { class: 'exercise-stage' });

  container.replaceChildren(
    el('div', { class: 'view-inner narrow session' },
      el('div', { class: 'session-head' },
        progressBar(index, queue.length),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => endSession(container) }, 'End session')),
      stage));

  session.mode = mode;
  const finish = (verdict, answer) => showVerdict(stage, container, card, verdict, answer);

  if (mode === 'recognize') renderRecognize(stage, card, finish);
  else if (mode === 'dictation') renderListen(stage, card, finish);
  else if (mode === 'cloze') renderCloze(stage, card, finish);
  else renderProduce(stage, card, finish);
}

// -- exercise renderers ------------------------------------------------------
function renderRecognize(stage, card, finish) {
  const pool = cards(session.lang).filter((c) => c.id !== card.id && c.english);
  const distractors = sample(pool, 3).map((c) => c.english);
  while (distractors.length < 3) distractors.push(['maybe', 'the ticket', 'to look for'][distractors.length]);
  const choices = shuffled([card.english, ...distractors]);
  const correctIndex = choices.indexOf(card.english);
  const grid = choiceGrid({
    choices, correctIndex,
    onPick: (correct) => setTimeout(() => finish(correct ? 'correct' : 'wrong', null), correct ? 500 : 1100),
  });
  stage.append(el('div', { class: 'quiz-card' },
    modeTag('recognize'),
    el('h2', { class: 'quiz-prompt' }, card.target),
    el('div', { class: 'row gap center' }, audioButton(card.target, { lang: session.lang })),
    grid));
  if (state.settings.autoplayAudio) speak(card.target, { lang: session.lang });
  keyHandler = (e) => grid.keyHandler?.(e);
  document.addEventListener('keydown', keyHandler);
}

function renderProduce(stage, card, finish) {
  const typing = makeTyping(card, finish, `Translate to ${languageProfile(session.lang)?.display || 'target'}…`);
  stage.append(el('div', { class: 'quiz-card' },
    modeTag('produce'),
    el('h2', { class: 'quiz-prompt' }, card.english),
    typing.root));
  typing.focus();
}

function renderListen(stage, card, finish) {
  const typing = makeTyping(card, finish, 'Type what you hear…');
  stage.append(el('div', { class: 'quiz-card' },
    modeTag('dictation'),
    el('div', { class: 'listen-controls' },
      audioButton(card.target, { lang: session.lang, kind: 'soft', label: 'Play' }),
      audioButton(card.target, { lang: session.lang, slow: true, kind: 'ghost', label: 'Slow' })),
    typing.root));
  speak(card.target, { lang: session.lang });
  typing.focus();
}

function renderCloze(stage, card, finish) {
  const display = clozeSentence(card);
  if (!display) {
    // Item not literally in the example (inflection) - fall back to production.
    session.mode = 'produce';
    renderProduce(stage, card, finish);
    return;
  }
  const typing = makeTyping(card, finish, 'Fill in the blank…');
  stage.append(el('div', { class: 'quiz-card' },
    modeTag('cloze'),
    el('h2', { class: 'quiz-prompt cloze' }, display),
    card.exampleEn ? el('p', { class: 'muted small center' }, card.exampleEn) : null,
    el('p', { class: 'muted small center' }, `(${card.english})`),
    typing.root));
  typing.focus();
}

function makeTyping(card, finish, placeholder) {
  let answered = false;
  const typing = typingInput({
    lang: session.lang,
    placeholder,
    onSubmit: (value) => {
      if (answered) return;
      answered = true;
      typing.input.disabled = true;
      const result = grade(value, card.target, {
        strictAccents: state.settings.strictAccents,
        typoTolerance: state.settings.typoTolerance,
      });
      finish(result.verdict, value);
    },
  });
  return typing;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Names the retrieval, and says so when adaptive testing chose it. */
function modeTag(mode) {
  const label = MODES[mode]?.label || mode;
  return el('span', {
    class: `phase-tag${adaptiveOn() ? ' adaptive' : ''}`,
    title: adaptiveOn() ? 'Chosen by adaptive testing from your current ability' : '',
  }, adaptiveOn() ? icon('zap', 11) : null, label);
}

// -- verdict + grading -------------------------------------------------------
function showVerdict(stage, container, card, verdict, answer) {
  cleanup();
  feedbackTone(verdict === 'wrong' ? 'wrong' : 'correct');
  const suggested = suggestedRating(verdict);
  const quiz = stage.querySelector('.quiz-card') || stage;
  quiz.querySelectorAll('.typing-block').forEach((n) => n.classList.add('answered'));

  const bar = el('div', {});
  quiz.append(
    verdictPanel({ verdict, answer, expected: card.target, card, lang: session.lang }),
    bar);

  const gradeBarEl = gradeBar({ card, suggested, onGrade: (rating) => applyGrade(container, card, verdict, rating) });
  bar.append(gradeBarEl);

  if (verdict !== 'correct') speak(card.target, { lang: session.lang });
  keyHandler = (e) => {
    if (gradeBarEl.keyHandler?.(e)) e.preventDefault();
  };
  // Deferred: the Enter that submitted the answer is still bubbling and must
  // not instantly trigger the grade bar.
  setTimeout(() => { if (keyHandler) document.addEventListener('keydown', keyHandler); }, 0);
}

function applyGrade(container, card, verdict, rating) {
  // The estimate is fed whether or not adaptive selection is on, so switching
  // the toggle on later starts from real evidence rather than from scratch.
  recordAttempt({ mode: session.mode || 'produce', verdict, card, lang: session.lang });
  schedule(card, rating, Date.now(), { targetRetention: state.settings.targetRetention });
  const ms = Date.now() - session.cardStart;
  // A dictation review really is listening practice, so the study mix says so.
  recordReview({
    correct: rating >= Rating.GOOD, ms,
    mode: session.mode === 'dictation' ? 'listen' : 'review',
  }, session.lang);
  session.results.push({ id: card.id, target: card.target, rating, verdict, mode: session.mode });
  persist();
  session.index += 1;
  const upcoming = session.queue[session.index + 1];
  if (upcoming) preload([upcoming.target], session.lang);
  renderCard(container);
}

function endSession(container) {
  if (session?.results.length) renderSummary(container);
  else { session = null; ctx.navigate('dashboard'); }
}

function renderSummary(container) {
  cleanup();
  const results = session.results;
  const lang = session.lang;
  session = null;
  const correct = results.filter((r) => r.rating >= Rating.GOOD).length;
  const accuracy = results.length ? Math.round((correct / results.length) * 100) : 0;
  const lapses = results.filter((r) => r.rating === Rating.AGAIN);
  const modeCounts = results.reduce((acc, r) => {
    if (r.mode) acc[r.mode] = (acc[r.mode] || 0) + 1;
    return acc;
  }, {});

  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, 'Session complete'),
        el('div', { class: 'summary-stats' },
          summaryStat(String(results.length), 'reviews'),
          summaryStat(`${accuracy}%`, 'accuracy'),
          summaryStat(String(lapses.length), 'lapses')),
        adaptiveOn() && Object.keys(modeCounts).length ? el('div', { class: 'mode-mix' },
          el('h4', {}, 'Retrievals used'),
          el('div', { class: 'row gap wrap center' },
            Object.entries(modeCounts)
              .sort((a, b) => (MODES[a[0]]?.difficulty || 0) - (MODES[b[0]]?.difficulty || 0))
              .map(([mode, count]) => el('span', { class: 'chip mode-chip' },
                `${MODES[mode]?.label || mode} · ${count}`)))) : null,
        lapses.length ? el('div', { class: 'lapse-list' },
          el('h4', {}, 'Back in the queue soon:'),
          lapses.slice(0, 6).map((r) => el('span', { class: 'chip' }, r.target))) : null,
        el('div', { class: 'row gap center', style: { marginTop: '18px' } },
          el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('dashboard') }, 'Dashboard'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('learn') }, icon('sparkles', 16), 'Learn new')))));
}

function summaryStat(value, label) {
  return el('div', { class: 'summary-stat' }, el('strong', {}, value), el('span', {}, label));
}
