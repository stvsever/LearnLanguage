// Grammar view: the language's structural map.
//
// Renders the grammar-theoretical architecture served by the backend:
// typology overview, the six pillars (the mental model), a CEFR roadmap of
// features with examples, transfer challenges, and phonology traps.
// Features show how often generated content has surfaced them, and every
// feature can be sent straight to Compose for targeted practice.

import { el, icon } from '../ui.js';
import { state, grammarSeen, currentLanguage } from '../store.js';
import { api } from '../api.js';
import { speak } from '../audio.js';
import { ctx, languageProfile } from '../context.js';

const profileCache = new Map(); // lang -> grammar profile

export function render(container) {
  const lang = currentLanguage();
  container.replaceChildren(
    el('div', { class: 'view-inner' },
      el('div', { class: 'grammar-loading' }, el('span', { class: 'spinner dark' }), ' Loading grammar map…')));

  loadProfile(lang).then((profile) => {
    if (currentLanguage() !== lang) return;
    if (!profile) {
      container.replaceChildren(el('div', { class: 'view-inner narrow' },
        el('div', { class: 'card start-card' },
          el('h2', {}, 'Grammar map unavailable'),
          el('p', { class: 'muted' }, 'Could not load the grammar profile - is the server running?'))));
      return;
    }
    renderProfile(container, profile, lang);
  });
}

export function cleanup() {}

async function loadProfile(lang) {
  if (!profileCache.has(lang)) {
    try {
      profileCache.set(lang, await api.grammar(lang));
    } catch {
      return null;
    }
  }
  return profileCache.get(lang);
}

function renderProfile(container, profile, lang) {
  const display = languageProfile(lang)?.display || lang;
  const seen = grammarSeen(lang);
  const userLevel = state.settings.level;
  const focus = ctx.grammarFocus || null;
  ctx.grammarFocus = null;

  container.replaceChildren(
    el('div', { class: 'view-inner' },
      el('header', { class: 'grammar-hero' },
        el('div', {},
          el('h1', {}, `How ${display} works`),
          el('p', { class: 'muted grammar-overview' }, profile.overview)),
        el('div', { class: 'typology-list' },
          profile.typology.map((t) => el('div', { class: 'typology-item' }, icon('check', 14), el('span', {}, t))))),

      el('section', { class: 'pillar-grid' },
        profile.pillars.map((pillar) => el('article', { class: 'pillar-card' },
          el('strong', {}, pillar.title),
          el('p', {}, pillar.summary)))),

      el('section', { class: 'card roadmap-card' },
        el('div', { class: 'roadmap-head' },
          el('h3', {}, 'The roadmap'),
          el('p', { class: 'muted small' },
            `CEFR-staged structures. Counters show how often your generated lessons and compositions have used each one. Your level: ${userLevel}.`)),
        el('div', { class: 'roadmap' },
          Object.entries(profile.roadmap).map(([level, features]) =>
            levelBlock(level, features, seen, userLevel, lang, focus)))),

      el('div', { class: 'dash-grid' },
        el('section', { class: 'card' },
          el('h3', {}, `Hard parts for English speakers`),
          el('ul', { class: 'clean-list' }, profile.challenges.map((c) => el('li', {}, c)))),
        el('section', { class: 'card' },
          el('h3', {}, 'Sound system'),
          el('ul', { class: 'clean-list' }, profile.phonology.map((p) => el('li', {}, p)))))));

  if (focus) {
    const target = container.querySelector(`[data-feature="${focus}"]`);
    if (target) {
      target.scrollIntoView({ block: 'center', behavior: 'smooth' });
      target.classList.add('flash');
      setTimeout(() => target.classList.remove('flash'), 2400);
    }
  }
}

function levelBlock(level, features, seen, userLevel, lang, focus) {
  const isCurrent = level === userLevel;
  const covered = features.filter((f) => seen[f.id]?.seen).length;
  const containsFocus = focus && features.some((f) => f.id === focus);
  const details = el('details', {
    class: `roadmap-level${isCurrent ? ' current' : ''}`,
    open: (isCurrent || containsFocus) || undefined,
  },
    el('summary', {},
      el('span', { class: `level-chip big${isCurrent ? ' active' : ''}` }, level),
      el('span', { class: 'roadmap-level-label' }, isCurrent ? 'Your level' : ''),
      el('span', { class: 'roadmap-coverage' },
        el('span', { class: 'coverage-track' },
          el('span', { style: { width: `${Math.round((covered / features.length) * 100)}%` } })),
        `${covered}/${features.length} met`)),
    el('div', { class: 'feature-list' },
      features.map((f) => featureRow(f, seen[f.id], lang))));
  return details;
}

function featureRow(feature, seenRecord, lang) {
  return el('div', { class: 'feature-row', dataset: { feature: feature.id } },
    el('div', { class: 'feature-main' },
      el('div', { class: 'feature-title' },
        el('strong', {}, feature.name),
        seenRecord?.seen ? el('span', { class: 'seen-badge', title: 'Times surfaced in your generated content' },
          icon('zap', 11), String(seenRecord.seen)) : null),
      el('p', { class: 'feature-tip' }, feature.tip),
      el('div', { class: 'feature-example' },
        el('button', {
          class: 'r-play', type: 'button', title: 'Play example',
          onclick: () => speak(feature.example, { lang }),
        }, icon('volume', 13)),
        el('span', { class: 'feature-example-text' }, feature.example),
        el('small', {}, feature.example_en))),
    el('button', {
      class: 'btn btn-soft btn-sm practice-btn', type: 'button',
      title: 'Compose a piece that practices this structure',
      onclick: () => {
        ctx.composePrefill = `A short piece that naturally practices "${feature.name}" (${feature.tip}) - pick any fun scenario.`;
        ctx.navigate('compose');
      },
    }, icon('sparkles', 14), 'Practice'));
}
