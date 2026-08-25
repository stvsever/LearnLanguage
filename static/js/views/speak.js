// Speak view: pronunciation practice via the browser's free speech recognition.
// Listen -> imitate -> the recognizer transcribes you -> word-level comparison.
// Producing output (and noticing the gap) is a core driver of acquisition.

import { el, icon, shuffled } from '../ui.js';
import { learnedCards, recordReview, currentLanguage } from '../store.js';
import { speak, stopAudio, feedbackTone } from '../audio.js';
import { similarity, normalize } from '../grading.js';
import { audioButton, progressBar } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';
import { speechSupported, listenOnce } from '../speech.js';

const ROUND_LENGTH = 6;
let session = null;
let activeRecognition = null;

export function render(container) {
  cleanup();
  session = null;
  const lang = currentLanguage();
  const supported = speechSupported();
  const pool = learnedCards(lang).filter((c) => c.example || c.target.split(' ').length >= 1);
  const ready = pool.length >= 3;

  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('mic', 28)),
        el('h2', {}, 'Speaking lab'),
        !supported
          ? el('p', { class: 'muted' }, 'Speech recognition is not available in this browser. Chrome and Edge support it - or simply shadow the audio aloud in any view.')
          : el('p', { class: 'muted' }, ready
            ? 'Hear a phrase, say it aloud, and get instant word-by-word feedback from free on-device speech recognition.'
            : 'Learn a few items first - speaking drills use your own deck.'),
        el('div', { class: 'row gap center' },
          el('button', {
            class: 'btn btn-primary btn-lg',
            disabled: (!supported || !ready) || undefined,
            onclick: () => start(container),
          }, icon('mic', 18), 'Start speaking'),
          !ready && supported ? el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('learn') }, 'Go to Learn') : null))));
}

export function cleanup() {
  activeRecognition?.stop();
  activeRecognition = null;
  stopAudio();
}

function start(container) {
  const lang = currentLanguage();
  const pool = shuffled(learnedCards(lang));
  const items = pool.slice(0, ROUND_LENGTH).map((card) => ({
    card,
    text: card.example && card.example.length <= 80 ? card.example : card.target,
  }));
  session = { lang, items, index: 0, scores: [] };
  step(container);
}

function step(container) {
  cleanup();
  if (session.index >= session.items.length) { summary(container); return; }
  const { card, text } = session.items[session.index];
  const locale = languageProfile(session.lang)?.recognitionLocale || 'fr-FR';

  const feedbackZone = el('div', {});
  const micBtn = el('button', {
    class: 'btn btn-primary btn-lg mic-btn',
    onclick: () => record(micBtn, feedbackZone, text, card, container, locale),
  }, icon('mic', 20), 'Hold on… tap to speak');
  micBtn.replaceChildren(icon('mic', 20), 'Tap to speak');

  container.replaceChildren(
    el('div', { class: 'view-inner narrow session' },
      el('div', { class: 'session-head' },
        progressBar(session.index, session.items.length),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => render(container) }, 'End')),
      el('div', { class: 'exercise-stage' },
        el('div', { class: 'quiz-card' },
          el('span', { class: 'phase-tag' }, 'Repeat aloud'),
          el('h2', { class: 'quiz-prompt speak-prompt' }, text),
          card.pronunciation && text === card.target ? el('div', { class: 'present-ipa' }, card.pronunciation) : null,
          el('div', { class: 'row gap center' },
            audioButton(text, { lang: session.lang, kind: 'soft', label: 'Listen' }),
            audioButton(text, { lang: session.lang, slow: true, kind: 'ghost', label: 'Slow' })),
          el('div', { class: 'row center', style: { marginTop: '14px' } }, micBtn),
          feedbackZone))));

  speak(text, { lang: session.lang });
}

function record(micBtn, feedbackZone, text, card, container, locale) {
  stopAudio();
  micBtn.disabled = true;
  micBtn.classList.add('recording');
  micBtn.replaceChildren(el('span', { class: 'pulse-dot' }), 'Listening…');
  feedbackZone.replaceChildren();

  activeRecognition = listenOnce(locale, {
    onInterim: (interim) => {
      feedbackZone.replaceChildren(el('p', { class: 'muted small center interim' }, `“${interim}”`));
    },
  });

  activeRecognition.promise.then((transcript) => {
    showResult(feedbackZone, micBtn, transcript, text, container);
  }).catch((err) => {
    micBtn.disabled = false;
    micBtn.classList.remove('recording');
    micBtn.replaceChildren(icon('mic', 20), 'Try again');
    const message = err.message === 'no_speech'
      ? 'No speech detected - speak a little louder, closer to the mic.'
      : err.message === 'not-allowed'
        ? 'Microphone access was blocked. Allow it in your browser settings.'
        : 'Recognition failed - tap to try again.';
    feedbackZone.replaceChildren(el('p', { class: 'muted small center' }, message));
  });
}

function showResult(feedbackZone, micBtn, transcript, expected, container) {
  const score = similarity(transcript, expected);
  const pct = Math.round(score * 100);
  const ok = score >= 0.8;
  feedbackTone(ok ? 'correct' : 'wrong');
  recordReview({ correct: ok, ms: 0, mode: 'speak' }, session.lang);
  session.scores.push(score);

  const expectedWords = normalize(expected, { keepDiacritics: false }).split(' ');
  const saidWords = new Set(normalize(transcript, { keepDiacritics: false }).split(' '));
  const wordRow = el('div', { class: 'word-compare' },
    expectedWords.map((w) => el('span', { class: saidWords.has(w) ? 'w-ok' : 'w-miss' }, w)));

  micBtn.disabled = false;
  micBtn.classList.remove('recording');
  micBtn.replaceChildren(icon('mic', 20), 'Retry');

  feedbackZone.replaceChildren(
    el('div', { class: `speak-score ${ok ? 'good' : 'meh'}` },
      el('strong', {}, `${pct}%`),
      el('span', {}, ok ? 'Sounded great' : 'Close - listen again and mind the missing words')),
    el('p', { class: 'muted small center' }, `Heard: “${transcript}”`),
    wordRow,
    el('button', { class: 'btn btn-primary continue-btn', onclick: () => { session.index += 1; step(container); } },
      'Continue', icon('arrowRight', 16)));
}

function summary(container) {
  const scores = session.scores;
  const avg = scores.length ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) : 0;
  session = null;
  container.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon success' }, icon('check', 28)),
        el('h2', {}, `Average score ${avg}%`),
        el('p', { class: 'muted' }, avg >= 85
          ? 'Excellent articulation. Try a longer text in Compose next.'
          : 'Recognition scores are a proxy, not a judge - repeat the tricky phrases after the slow audio and the score will follow.'),
        el('div', { class: 'row gap center' },
          el('button', { class: 'btn btn-primary', onclick: () => render(container) }, 'Another round'),
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('dashboard') }, 'Dashboard')))));
}
