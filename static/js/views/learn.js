// Learn view: acquire new items through a guided encoding ladder.
//
// Each new item passes three phases built on how memories form:
//  1. Present  - dual coding: orthography + IPA + audio + example in context.
//  2. Recognize - low-effort retrieval (meaning MCQ) right after encoding.
//  3. Produce  - effortful typed recall, the strongest encoding event.
// Completed items enter the FSRS pipeline and resurface in Review.

import { el, icon, toast, shuffled, sample } from '../ui.js';
import {
  state, cards, newCards, addCards, recordNewCard, recordTime,
  newCardsIntroducedToday, currentLanguage, persist, recordGrammarFeatures,
} from '../store.js';
import { api, ApiError } from '../api.js';
import { speak, preload, feedbackTone } from '../audio.js';
import { grade } from '../grading.js';
import { schedule, Rating } from '../srs.js';
import { typingInput, verdictPanel, choiceGrid, audioButton, progressBar } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';

let session = null; // { queue, index, phase, startedAt }
let keyHandler = null;

export function render(container) {
  cleanup();
  const lang = currentLanguage();
  const remaining = Math.max(0, state.settings.newPerDay - newCardsIntroducedToday(lang));
  const queue = newCards(lang).slice(0, remaining);

  if (session && session.lang === lang && session.index < session.queue.length) {
    renderSession(container);
    return;
  }
  session = null;

  if (queue.length > 0) {
    renderStart(container, queue);
  } else {
    renderGeneratePanel(container, {
      title: newCards(lang).length > 0 ? 'Daily new-card goal reached' : 'Add learning material',
      sub: newCards(lang).length > 0
        ? `You've introduced ${newCardsIntroducedToday(lang)} new cards today. Raise the daily limit in Settings, or add material for tomorrow.`
        : 'Generate a lesson on any topic - the model builds high-frequency, level-appropriate items with examples and pronunciation.',
    });
  }
}

export function cleanup() {
  if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
}

// -- start screen ------------------------------------------------------------
function renderStart(container, queue) {
  const lang = currentLanguage();
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('sparkles', 28)),
        el('h2', {}, `${queue.length} new item${queue.length === 1 ? '' : 's'} ready`),
        el('p', { class: 'muted' },
          'Each item is presented with audio and context, then tested immediately - first recognition, then typed recall. Immediate retrieval is what locks new words in.'),
        el('div', { class: 'row gap center' },
          el('button', {
            class: 'btn btn-primary btn-lg', id: 'startLearnBtn',
            onclick: () => startSession(container, queue),
          }, icon('play', 18), `Start learning`),
          el('button', { class: 'btn btn-ghost', onclick: () => renderGeneratePanel(container, { title: 'Add learning material', sub: 'Generate another lesson on any topic.', backTo: () => render(container) }) },
            icon('plus', 16), 'Add material'))),
    ));
}

function startSession(container, queue) {
  session = { queue, index: 0, phase: 0, lang: currentLanguage(), startedAt: Date.now(), phaseStart: Date.now() };
  preload(queue.slice(0, 4).map((c) => c.target), session.lang);
  renderSession(container);
}

// -- session -----------------------------------------------------------------
function renderSession(container) {
  cleanup();
  const { queue, index } = session;
  if (index >= queue.length) { renderDone(container); return; }
  const card = queue[index];
  const stage = el('div', { class: 'exercise-stage' });

  container.replaceChildren(
    el('div', { class: 'view-inner narrow session' },
      el('div', { class: 'session-head' },
        progressBar(index, queue.length),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => { session = null; render(container); } }, 'End session')),
      stage));

  if (session.phase === 0) renderPresent(stage, card, container);
  else if (session.phase === 1) renderRecognize(stage, card, container);
  else renderProduce(stage, card, container);
}

function nextPhase(container) {
  session.phase += 1;
  session.phaseStart = Date.now();
  if (session.phase > 2) {
    const card = session.queue[session.index];
    finishItem(card);
    session.index += 1;
    session.phase = 0;
    const upcoming = session.queue[session.index + 1];
    if (upcoming) preload([upcoming.target], session.lang);
  }
  renderSession(container);
}

function finishItem(card) {
  card.introduced = true;
  const verdictOk = card._learnFailed !== true;
  schedule(card, verdictOk ? Rating.GOOD : Rating.AGAIN, Date.now(), { targetRetention: state.settings.targetRetention });
  delete card._learnFailed;
  recordNewCard(session.lang);
  recordTime(Date.now() - session.phaseStart, session.lang);
  persist();
}

// Phase 1: presentation with full encoding cues.
function renderPresent(stage, card, container) {
  const profile = languageProfile(session.lang);
  stage.append(
    el('div', { class: 'present-card', style: { fontFamily: profile?.fontStack || 'inherit' } },
      el('span', { class: 'phase-tag' }, 'New item'),
      el('h1', { class: 'present-target' }, card.target),
      state.settings.showPronunciation && card.pronunciation ? el('div', { class: 'present-ipa' }, card.pronunciation) : null,
      el('div', { class: 'present-english' }, card.english),
      card.example ? el('div', { class: 'present-example' },
        el('p', {}, card.example),
        card.exampleEn ? el('small', {}, card.exampleEn) : null,
        audioButton(card.example, { lang: session.lang })) : null,
      card.note ? el('div', { class: 'present-note' }, icon('lightbulb', 15), el('span', {}, card.note)) : null,
      el('div', { class: 'row gap center', style: { marginTop: '8px' } },
        audioButton(card.target, { lang: session.lang, kind: 'soft', label: 'Listen' }),
        audioButton(card.target, { lang: session.lang, slow: true, kind: 'ghost', label: 'Slow' })),
      el('button', { class: 'btn btn-primary btn-lg continue-btn', onclick: () => nextPhase(container) },
        'Continue', icon('arrowRight', 18))));

  if (state.settings.autoplayAudio) speak(card.target, { lang: session.lang });
  keyHandler = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
      e.preventDefault();
      nextPhase(container);
    }
    if (e.key === 'r' || e.key === 'R') speak(card.target, { lang: session.lang });
  };
  document.addEventListener('keydown', keyHandler);
}

// Phase 2: recognition MCQ (target -> meaning).
function renderRecognize(stage, card, container) {
  const pool = cards(session.lang).filter((c) => c.id !== card.id && c.english);
  const distractors = sample(pool, 3).map((c) => c.english);
  while (distractors.length < 3) distractors.push(['the station', 'to wait', 'already', 'the corner'][distractors.length]);
  const choices = shuffled([card.english, ...distractors]);
  const correctIndex = choices.indexOf(card.english);

  const grid = choiceGrid({
    choices,
    correctIndex,
    onPick: (correct) => {
      feedbackTone(correct ? 'correct' : 'wrong');
      if (!correct) card._learnFailed = true;
      setTimeout(() => nextPhase(container), correct ? 700 : 1600);
    },
  });

  stage.append(
    el('div', { class: 'quiz-card' },
      el('span', { class: 'phase-tag' }, 'Recognize'),
      el('h2', { class: 'quiz-prompt' }, card.target),
      el('div', { class: 'row gap center' }, audioButton(card.target, { lang: session.lang })),
      el('p', { class: 'muted small center' }, 'What does it mean?'),
      grid));

  if (state.settings.autoplayAudio) speak(card.target, { lang: session.lang });
  keyHandler = (e) => grid.keyHandler?.(e);
  document.addEventListener('keydown', keyHandler);
}

// Phase 3: typed production (meaning -> target).
function renderProduce(stage, card, container) {
  let answered = false;
  const typing = typingInput({
    lang: session.lang,
    placeholder: `Type it in ${languageProfile(session.lang)?.display || 'the target language'}…`,
    onSubmit: (value) => {
      if (answered) return;
      answered = true;
      const result = grade(value, card.target, {
        strictAccents: state.settings.strictAccents,
        typoTolerance: state.settings.typoTolerance,
      });
      feedbackTone(result.verdict === 'wrong' ? 'wrong' : 'correct');
      if (result.verdict === 'wrong') card._learnFailed = true;
      typing.input.disabled = true;
      stage.querySelector('.quiz-card').append(
        verdictPanel({ verdict: result.verdict, answer: value, expected: card.target, card, lang: session.lang }),
        el('button', { class: 'btn btn-primary continue-btn', onclick: () => nextPhase(container) }, 'Continue', icon('arrowRight', 16)));
      speak(card.target, { lang: session.lang });
      keyHandler = (e) => { if (e.key === 'Enter') { e.preventDefault(); nextPhase(container); } };
      // Deferred so the submitting Enter doesn't immediately advance.
      setTimeout(() => { if (keyHandler) document.addEventListener('keydown', keyHandler); }, 0);
    },
  });

  stage.append(
    el('div', { class: 'quiz-card' },
      el('span', { class: 'phase-tag' }, 'Produce'),
      el('h2', { class: 'quiz-prompt' }, card.english),
      card.exampleEn ? el('p', { class: 'muted small center' }, `“${card.exampleEn}”`) : null,
      typing.root));
  typing.focus();

  keyHandler = null; // Enter handled by the input itself
}

function renderDone(container) {
  const total = session.queue.length;
  const lang = session.lang;
  session = null;
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, `${total} item${total === 1 ? '' : 's'} learned`),
        el('p', { class: 'muted' }, 'They are now in your spaced-repetition pipeline and will come up for review in a few minutes - the first retrieval soon after learning matters most.'),
        el('div', { class: 'row gap center' },
          el('button', { class: 'btn btn-primary btn-lg', onclick: () => ctx.navigate('review') }, icon('refresh', 18), 'Go to reviews'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('dashboard') }, 'Dashboard')))));
}

// -- generation panel --------------------------------------------------------
function renderGeneratePanel(container, { title, sub, backTo }) {
  const lang = currentLanguage();
  const profile = languageProfile(lang);
  const offline = ctx.config?.provider === 'offline';
  const hasSeed = ctx.config?.seedLanguages?.includes(lang);

  const topicInput = el('textarea', {
    class: 'input', rows: 2, maxlength: 280, id: 'topicInput',
    placeholder: `e.g. ordering at a café, apartment hunting, small talk about work…`,
  });
  const levelSelect = el('select', { class: 'input' },
    (ctx.config?.levels || ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']).map((lv) =>
      el('option', { value: lv, selected: lv === state.settings.level || undefined }, lv)));
  const countSelect = el('select', { class: 'input' },
    [8, 12, 16, 20, 24].map((n) => el('option', { value: n, selected: n === 12 || undefined }, `${n} items`)));

  const generateBtn = el('button', {
    class: 'btn btn-primary', id: 'generateLessonBtn',
    onclick: async () => {
      generateBtn.disabled = true;
      generateBtn.replaceChildren(el('span', { class: 'spinner' }), 'Generating…');
      try {
        const known = cards(lang).map((c) => c.target).slice(-120);
        const pack = await api.lesson({
          topic: topicInput.value.trim(),
          language: lang,
          level: levelSelect.value,
          count: Number(countSelect.value),
          knownWords: known,
        });
        const added = addCards(pack.items, pack.topic || topicInput.value.trim(), lang);
        recordGrammarFeatures(pack.grammar_features || [], lang);
        if (pack.notice) toast(pack.notice, 'info', 6000);
        toast(`${added} new card${added === 1 ? '' : 's'} added`, 'success');
        render(container);
      } catch (err) {
        toast(err instanceof ApiError && err.status === 503
          ? 'No LLM configured - add OPENROUTER_API_KEY to .env (see Settings → About).'
          : `Generation failed: ${err.message}`, 'error', 6000);
        generateBtn.disabled = false;
        generateBtn.replaceChildren(icon('sparkles', 16), 'Generate lesson');
      }
    },
  }, icon('sparkles', 16), 'Generate lesson');

  const seedBtn = hasSeed ? el('button', {
    class: 'btn btn-soft',
    onclick: async () => {
      try {
        const pack = await api.lesson({ language: lang, seed: true });
        const added = addCards(pack.items, pack.topic, lang);
        toast(added > 0 ? `Starter deck loaded - ${added} cards` : 'Starter deck already in your deck', added > 0 ? 'success' : 'info');
        render(container);
      } catch (err) {
        toast(`Could not load starter deck: ${err.message}`, 'error');
      }
    },
  }, icon('layers', 16), `${profile?.display || ''} starter deck`) : null;

  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card generate-card', id: 'generatePanel' },
        el('h2', {}, title),
        el('p', { class: 'muted' }, sub),
        offline ? el('div', { class: 'notice' },
          icon('zap', 16),
          el('span', {}, 'Running offline - no API key found. The built-in starter content works fully; add OPENROUTER_API_KEY to .env for unlimited generation.')) : null,
        el('label', { class: 'field' }, el('span', {}, 'Topic'), topicInput),
        el('div', { class: 'row gap' },
          el('label', { class: 'field grow' }, el('span', {}, 'Level (CEFR)'), levelSelect),
          el('label', { class: 'field grow' }, el('span', {}, 'Amount'), countSelect)),
        el('div', { class: 'row gap wrap', style: { marginTop: '6px' } },
          generateBtn, seedBtn,
          backTo ? el('button', { class: 'btn btn-ghost', onclick: backTo }, 'Back') : null))));
}
