// Learn view: acquire new items through a guided encoding ladder.
//
// Each new item passes the rungs of a ladder built on how memories form:
//  1. Present  - dual coding: orthography + IPA + audio + example in context.
//  2. Recognize - low-effort retrieval (meaning MCQ) right after encoding.
//  3. Produce  - effortful typed recall, the strongest encoding event.
// With adaptive testing on, the recognition rung drops away once the learner
// is reliably producing (see adaptive.js). Completed items enter the FSRS
// pipeline and resurface in Review.
//
// Material comes from two sources, both first-class: the curated Topics
// library (works offline) and AI generation on any topic you can describe.

import { el, icon, toast, shuffled, sample, progressSteps, fmtInt } from '../ui.js';
import { keySetupCard } from '../keysetup.js';
import {
  state, cards, newCards, addCards, recordNewCard, recordTime,
  newCardsIntroducedToday, currentLanguage, persist, recordGrammarFeatures,
  updateSettings, deckCounts,
} from '../store.js';
import { api, ApiError } from '../api.js';
import { speak, preload, feedbackTone } from '../audio.js';
import { grade } from '../grading.js';
import { schedule, Rating } from '../srs.js';
import { typingInput, verdictPanel, choiceGrid, audioButton, progressBar } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';
import { learnLadder, recordAttempt, isEnabled as adaptiveOn } from '../adaptive.js';
import { allLevels, levelLabel } from '../levels.js';

let session = null; // { queue, index, phase, startedAt }
let keyHandler = null;

export function render(container) {
  cleanup();
  const lang = currentLanguage();

  if (session && session.lang === lang && session.index < session.queue.length) {
    renderSession(container);
    return;
  }
  session = null;

  const pending = newCards(lang);
  const introduced = newCardsIntroducedToday(lang);
  const remaining = Math.max(0, state.settings.newPerDay - introduced);
  const queue = pending.slice(0, remaining);

  if (queue.length > 0) {
    renderStart(container, queue, pending.length);
  } else if (pending.length > 0) {
    // Cards are waiting but today's budget is spent. Say so plainly and offer
    // both honest options instead of silently showing the generator again.
    renderCapReached(container, pending, introduced);
  } else {
    renderSourcePanel(container, {
      title: 'Add learning material',
      sub: 'Pick a curated topic from the library, or describe any subject and let the model build a lesson for it.',
    });
  }
}

export function cleanup() {
  if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
}

// -- start screen ------------------------------------------------------------
function renderStart(container, queue, pendingTotal) {
  const extra = pendingTotal - queue.length;
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('sparkles', 28)),
        el('h2', {}, `${queue.length} new item${queue.length === 1 ? '' : 's'} ready`),
        el('p', { class: 'muted' }, ladderBlurb()),
        extra > 0 ? el('p', { class: 'muted small' },
          `${extra} more are queued for the coming days (daily limit: ${state.settings.newPerDay}).`) : null,
        el('div', { class: 'row gap center wrap' },
          el('button', {
            class: 'btn btn-primary btn-lg', id: 'startLearnBtn',
            onclick: () => startSession(container, queue),
          }, icon('play', 18), 'Start learning'),
          el('button', {
            class: 'btn btn-ghost',
            onclick: () => renderSourcePanel(container, {
              title: 'Add more material',
              sub: 'Browse the curated library or generate a lesson on any topic.',
              backTo: () => render(container),
            }),
          }, icon('plus', 16), 'Add material')))));
}

/** Describes the ladder the next session will actually use. */
function ladderBlurb() {
  const ladder = learnLadder(currentLanguage());
  if (!adaptiveOn()) {
    return 'Each item is presented with audio and context, then tested immediately: first recognition, then typed recall. Immediate retrieval is what locks new words in.';
  }
  return ladder.includes('recognize')
    ? 'Each item is presented with audio and context, then tested immediately: first recognition, then typed recall. Adaptive testing will drop the recognition step once you are reliably producing.'
    : 'Adaptive testing has you going straight from presentation to typed recall, the strongest encoding event, because your recognition is already secure.';
}

function renderCapReached(container, pending, introduced) {
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, 'Daily new-card goal reached'),
        el('p', { class: 'muted' },
          `You introduced ${introduced} new item${introduced === 1 ? '' : 's'} today, which is your limit. ${fmtInt(pending.length)} more are waiting in the deck. Stopping here is the sustainable choice: every new card generates future reviews.`),
        el('div', { class: 'row gap center wrap' },
          el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('review') }, icon('refresh', 16), 'Go to reviews'),
          el('button', {
            class: 'btn btn-soft',
            onclick: () => {
              const bonus = Math.min(5, pending.length);
              updateSettings({ newPerDay: state.settings.newPerDay + bonus });
              toast(`Daily limit raised to ${state.settings.newPerDay}`, 'info');
              render(container);
            },
          }, icon('plus', 16), 'Learn 5 more anyway'),
          el('button', {
            class: 'btn btn-ghost',
            onclick: () => renderSourcePanel(container, {
              title: 'Add material for tomorrow',
              sub: 'Queue up the next topic now; it will be waiting when the daily budget resets.',
              backTo: () => render(container),
            }),
          }, 'Add material')))));
}

function startSession(container, queue) {
  const lang = currentLanguage();
  session = {
    queue, index: 0, phase: 0, lang,
    // Adaptive testing decides how many rungs the encoding ladder needs: a
    // learner who reliably produces does not gain from the recognition rung.
    ladder: learnLadder(lang),
    startedAt: Date.now(), phaseStart: Date.now(),
  };
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

  const rung = session.ladder[session.phase];
  if (rung === 'present') renderPresent(stage, card, container);
  else if (rung === 'recognize') renderRecognize(stage, card, container);
  else renderProduce(stage, card, container);
}

function nextPhase(container) {
  session.phase += 1;
  session.phaseStart = Date.now();
  if (session.phase >= session.ladder.length) {
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
      recordAttempt({ mode: 'recognize', verdict: correct, card, lang: session.lang });
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
      recordAttempt({ mode: 'produce', verdict: result.verdict, card, lang: session.lang });
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
  session = null;
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, `${total} item${total === 1 ? '' : 's'} learned`),
        el('p', { class: 'muted' }, 'They are now in your spaced-repetition pipeline and will come up for review in a few minutes. The first retrieval soon after learning matters most.'),
        el('div', { class: 'row gap center wrap' },
          el('button', { class: 'btn btn-primary btn-lg', onclick: () => ctx.navigate('review') }, icon('refresh', 18), 'Go to reviews'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('topics') }, icon('layers', 16), 'More topics')))));
}

// -- material sources: library first, generation second ----------------------
function renderSourcePanel(container, { title, sub, backTo }) {
  const lang = currentLanguage();
  const profile = languageProfile(lang);
  const offline = (ctx.config?.provider || 'offline') === 'offline';
  const stock = ctx.config?.curriculum?.[lang];
  const hasLibrary = Boolean(stock?.items);
  const counts = deckCounts(lang);

  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'source-head' },
        el('h2', {}, title),
        el('p', { class: 'muted' }, sub)),

      hasLibrary ? el('section', { class: 'card source-card primary-source' },
        el('div', { class: 'source-card-head' },
          el('span', { class: 'source-icon' }, icon('layers', 22)),
          el('div', {},
            el('h3', {}, 'Browse the topic library'),
            el('p', { class: 'muted small' },
              `${fmtInt(stock.items)} curated ${profile?.display || ''} items in ${stock.units} topics across ${stock.domains} areas of life, with pronunciation, examples, and usage traps. No AI key needed.`))),
        el('div', { class: 'row gap wrap' },
          el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('topics') },
            icon('layers', 16), 'Open the library'),
          counts.total === 0 ? el('button', {
            class: 'btn btn-soft',
            onclick: (e) => loadStarter(e.currentTarget, container, lang),
          }, icon('zap', 16), 'Quick start: 24 core items') : null)) : null,

      el('section', { class: 'card source-card' },
        el('div', { class: 'source-card-head' },
          el('span', { class: 'source-icon' }, icon('sparkles', 22)),
          el('div', {},
            el('h3', {}, 'Generate a lesson'),
            el('p', { class: 'muted small' },
              offline
                ? 'Connect a key to build a lesson on any subject you can describe.'
                : 'Describe any subject; the model writes high-frequency, level-appropriate items with examples and pronunciation.'))),
        offline ? keySetupCard({ onConnected: () => renderSourcePanel(container, { title, sub, backTo }) }) : null,
        generateForm(container, lang, offline)),

      backTo ? el('div', { class: 'row center' },
        el('button', { class: 'btn btn-ghost', onclick: backTo }, icon('arrowLeft', 16), 'Back')) : null));
}

function generateForm(container, lang, offline) {
  const prefill = ctx.learnTopic || '';
  ctx.learnTopic = null;

  const topicInput = el('textarea', {
    class: 'input', rows: 2, maxlength: 280, id: 'topicInput',
    placeholder: 'e.g. ordering at a cafe, apartment hunting, small talk about work…',
  });
  if (prefill) topicInput.value = prefill;

  const levelSelect = el('select', { class: 'input' },
    allLevels().map((lv) =>
      el('option', {
        value: lv.code, title: lv.blurb,
        selected: lv.code === state.settings.level || undefined,
      }, `${lv.code} · ${lv.name}`)));
  const countSelect = el('select', { class: 'input' },
    [8, 12, 16, 20, 24].map((n) => el('option', { value: n, selected: n === 12 || undefined }, `${n} items`)));

  const progressHost = el('div', { class: 'progress-host' });
  const resultHost = el('div', {});

  const generateBtn = el('button', {
    class: 'btn btn-primary', id: 'generateLessonBtn', disabled: offline || undefined,
    onclick: async () => {
      const topic = topicInput.value.trim();
      if (!topic) { topicInput.focus(); toast('Describe a topic first', 'info'); return; }
      resultHost.replaceChildren();
      generateBtn.disabled = true;
      generateBtn.replaceChildren(el('span', { class: 'spinner' }), 'Working');
      const model = (state.settings.model || ctx.config?.model || 'the model').split('/').pop();
      const progress = progressSteps([
        `Sending your topic to ${model}`,
        `Writing ${countSelect.value} items at ${levelSelect.value} with your grammar targets`,
        'Checking structure, examples, and pronunciation',
        'Adding cards to your deck',
      ], [1400, 14000, 5000]);
      progressHost.replaceChildren(progress.root);

      const restore = () => {
        generateBtn.disabled = false;
        generateBtn.replaceChildren(icon('sparkles', 16), 'Generate lesson');
      };

      let pack;
      try {
        pack = await api.lesson({
          topic,
          language: lang,
          level: levelSelect.value,
          count: Number(countSelect.value),
          knownWords: cards(lang).map((c) => c.target).slice(-120),
        });
      } catch (err) {
        progress.fail(err instanceof ApiError && err.status === 503
          ? 'No AI key configured yet. Connect one above, or use the topic library.'
          : `Generation failed: ${err.message}`);
        resultHost.replaceChildren(failureCard(container, err, topic));
        restore();
        return;
      }

      // Storing the pack is a separate failure domain from generating it: a
      // malformed item must never swallow a lesson that already arrived.
      try {
        const report = addCards(pack.items, pack.topic || topic, lang, { level: pack.level });
        recordGrammarFeatures(pack.grammar_features || [], lang);
        progress.finish();
        if (pack.notice) toast(pack.notice, 'info', 6000);
        resultHost.replaceChildren(resultCard(container, report, pack, topic));
        restore();
      } catch (err) {
        console.error('Saving generated cards failed', err);
        progress.fail(`The lesson arrived but could not be saved: ${err.message}`);
        resultHost.replaceChildren(failureCard(container, err, topic, pack));
        restore();
      }
    },
  }, icon('sparkles', 16), 'Generate lesson');

  return el('div', {},
    el('label', { class: 'field' }, el('span', {}, 'Topic'), topicInput),
    el('div', { class: 'row gap' },
      el('label', { class: 'field grow' }, el('span', {}, 'Level (CEFR)'), levelSelect),
      el('label', { class: 'field grow' }, el('span', {}, 'Amount'), countSelect)),
    el('div', { class: 'row gap wrap', style: { marginTop: '6px' } }, generateBtn),
    progressHost,
    resultHost);
}

/** What actually landed in the deck, stated plainly. Never a silent redirect. */
function resultCard(container, report, pack, topic) {
  const lines = [];
  if (report.added) lines.push(`${report.added} new card${report.added === 1 ? '' : 's'} added`);
  if (report.duplicates) lines.push(`${report.duplicates} already in your deck`);
  if (report.skipped) lines.push(`${report.skipped} incomplete item${report.skipped === 1 ? '' : 's'} skipped`);

  return el('div', { class: `result-card${report.added ? ' ok' : ' warn'}` },
    el('div', { class: 'result-head' },
      icon(report.added ? 'check' : 'lightbulb', 18),
      el('strong', {}, report.added
        ? `“${pack.topic || topic}” is in your deck`
        : 'Nothing new to add')),
    el('p', { class: 'muted small' },
      [lines.join(' · ') || 'No items came back.', pack.level ? `pitched at ${levelLabel(pack.level)}` : null]
        .filter(Boolean).join(' · ')),
    !report.added && report.duplicates
      ? el('p', { class: 'muted small' }, 'Try a narrower topic, a higher level, or the topic library for material you do not have yet.')
      : null,
    el('div', { class: 'row gap wrap' },
      report.added ? el('button', {
        class: 'btn btn-primary btn-sm', onclick: () => render(container),
      }, icon('play', 15), 'Start learning them') : null,
      el('button', { class: 'btn btn-ghost btn-sm', onclick: () => ctx.navigate('topics') },
        icon('layers', 15), 'Topic library')));
}

function failureCard(container, err, topic, pack) {
  return el('div', { class: 'result-card error' },
    el('div', { class: 'result-head' }, icon('x', 18), el('strong', {}, 'That did not work')),
    el('p', { class: 'muted small' }, err?.message || String(err)),
    pack ? el('p', { class: 'muted small' }, `The model returned ${pack.items?.length || 0} items, but they could not be stored.`) : null,
    el('div', { class: 'row gap wrap' },
      el('button', { class: 'btn btn-soft btn-sm', onclick: () => document.querySelector('#generateLessonBtn')?.click() },
        icon('refresh', 15), 'Try again'),
      el('button', { class: 'btn btn-ghost btn-sm', onclick: () => ctx.navigate('topics') },
        icon('layers', 15), 'Use the curated library instead')),
    topic ? el('p', { class: 'muted small' }, `Topic kept: “${topic}”`) : null);
}

async function loadStarter(btn, container, lang) {
  btn.disabled = true;
  btn.replaceChildren(el('span', { class: 'spinner' }), 'Loading');
  try {
    const pack = await api.lesson({ language: lang, seed: true, count: 24 });
    const report = addCards(pack.items, pack.topic, lang, { level: pack.level });
    toast(report.added
      ? `Starter set loaded: ${report.added} cards`
      : 'Those starter cards are already in your deck', report.added ? 'success' : 'info');
    render(container);
  } catch (err) {
    toast(`Could not load the starter set: ${err.message}`, 'error');
    btn.disabled = false;
    btn.replaceChildren(icon('zap', 16), 'Quick start: 24 core items');
  }
}
