// First-run walkthrough: a spotlight overlay that dims everything except the
// element being explained, with a step card, progress dots, and keyboard nav.

import { el, icon } from './ui.js';
import { state, updateSettings } from './store.js';
import { ctx } from './context.js';

const STEPS = [
  {
    target: null,
    title: 'Bienvenue 👋',
    body: 'AI-Studio for Learning Languages is a science-based tutor: spaced repetition, active recall, listening, speaking, and real reading - all in one loop, all stored privately in your browser.',
  },
  {
    target: '[data-tour="nav"]',
    title: 'One loop, eight rooms',
    body: 'Learn introduces new words; Review brings them back right before you\'d forget. Listen and Speak train your ear and mouth. Compose writes anything you describe at your level, Grammar maps the language, Progress keeps the score.',
  },
  {
    target: '[data-tour="lang"]',
    title: 'Your language',
    body: 'French is set up by default. Switch anytime - every language keeps its own deck, schedule, and stats.',
  },
  {
    target: '[data-tour="actions"]', view: 'dashboard',
    title: 'The daily plan',
    body: 'The dashboard always tells you the single best next action: clear due reviews first, then learn new items, then get input. Consistency is the whole trick.',
  },
  {
    target: '[data-tour="streak"]',
    title: 'Streak & time',
    body: 'Memories consolidate during sleep, so daily contact beats weekend marathons. The flame keeps you honest.',
  },
  {
    target: '[data-tour="settings"]',
    title: 'Make it yours',
    body: 'Voices, speech speed, daily limits, target retention, accent strictness, themes, data export - everything is adjustable in Settings.',
  },
  {
    target: null,
    title: 'C\'est parti !',
    body: 'Head to Learn for the starter deck or a lesson on any topic - or open Compose and describe a scene you\'d actually enjoy reading. Two minutes from now you\'ll be practicing.',
  },
];

let active = null;

export function maybeStartTour() {
  if (!state.settings.tourDone) startTour(true);
}

export function startTour(force = false) {
  if (active) return;
  if (!force && state.settings.tourDone) return;
  ctx.navigate('dashboard');

  const overlay = el('div', { class: 'tour-overlay' });
  const hole = el('div', { class: 'tour-hole' });
  const card = el('div', { class: 'tour-card' });
  overlay.append(hole, card);
  document.body.append(overlay);
  active = { overlay, hole, card, step: 0 };

  const onKey = (e) => {
    if (e.key === 'Escape') finish();
    if (e.key === 'ArrowRight' || e.key === 'Enter') next();
    if (e.key === 'ArrowLeft') back();
  };
  const onResize = () => show(active.step, false);
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', onResize);
  active.teardown = () => {
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', onResize);
  };

  function finish() {
    updateSettings({ tourDone: true });
    active.teardown();
    overlay.classList.remove('show');
    setTimeout(() => overlay.remove(), 250);
    active = null;
  }

  function next() {
    if (active.step >= STEPS.length - 1) finish();
    else show(active.step + 1);
  }

  function back() {
    if (active.step > 0) show(active.step - 1);
  }

  function show(index, animate = true) {
    active.step = index;
    const step = STEPS[index];
    if (step.view) ctx.navigate(step.view);

    // allow the view to render before measuring
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const target = step.target ? document.querySelector(step.target) : null;
      if (target) {
        hole.classList.remove('centered');
        const rect = target.getBoundingClientRect();
        const pad = 8;
        Object.assign(hole.style, {
          left: `${rect.left - pad}px`,
          top: `${rect.top - pad}px`,
          width: `${rect.width + pad * 2}px`,
          height: `${rect.height + pad * 2}px`,
        });
        positionCard(rect);
      } else {
        // No target: keep the full-screen dim (the hole's box-shadow) with a
        // zero-size hole, and center the card.
        hole.classList.add('centered');
        Object.assign(hole.style, {
          left: '50%', top: '38%', width: '0px', height: '0px',
        });
        Object.assign(card.style, { left: '50%', top: '42%', transform: 'translate(-50%, -50%)' });
      }
      renderCard(step, index);
      if (animate) card.classList.remove('pop');
      requestAnimationFrame(() => card.classList.add('pop'));
    }));
  }

  function positionCard(rect) {
    card.style.transform = 'none';
    const cardWidth = 340;
    const margin = 16;
    let left = rect.left + rect.width / 2 - cardWidth / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - cardWidth - margin));
    const below = rect.bottom + 18;
    const estimatedHeight = 210;
    if (below + estimatedHeight < window.innerHeight - margin) {
      card.style.top = `${below}px`;
    } else {
      card.style.top = `${Math.max(margin, rect.top - estimatedHeight - 18)}px`;
    }
    // If the target is a tall sidebar, place the card to its right instead
    if (rect.height > window.innerHeight * 0.6 && rect.width < 300) {
      card.style.top = `${Math.max(margin, rect.top + 40)}px`;
      left = rect.right + 18;
    }
    card.style.left = `${left}px`;
  }

  function renderCard(step, index) {
    card.replaceChildren(
      el('div', { class: 'tour-step-count' }, `${index + 1} / ${STEPS.length}`),
      el('h3', {}, step.title),
      el('p', {}, step.body),
      el('div', { class: 'tour-dots' },
        STEPS.map((_, i) => el('span', { class: `dot${i === index ? ' active' : ''}` }))),
      el('div', { class: 'tour-actions' },
        el('button', { class: 'btn btn-ghost btn-sm', onclick: finish }, index === STEPS.length - 1 ? '' : 'Skip'),
        el('div', { class: 'row gap' },
          index > 0 ? el('button', { class: 'btn btn-soft btn-sm', onclick: back }, icon('arrowLeft', 14), 'Back') : null,
          el('button', { class: 'btn btn-primary btn-sm', onclick: next },
            index === STEPS.length - 1 ? 'Start learning' : 'Next', icon('arrowRight', 14)))));
  }

  requestAnimationFrame(() => overlay.classList.add('show'));
  show(0, true);
}
