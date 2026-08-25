// Dashboard: today's plan at a glance - deliberately sparse.
// One guidance line, three actions, two compact status cards, one daily focus.

import { el, icon, fmtInt, fmtDuration } from '../ui.js';
import {
  state, deckCounts, dueCards, newCards, streak, statDay,
  todayKey, newCardsIntroducedToday, accuracyOverDays, currentLanguage,
} from '../store.js';
import { api } from '../api.js';
import { ctx, languageProfile } from '../context.js';

const DAY_MS = 86400000;

function greeting() {
  const hour = new Date().getHours();
  const byLang = {
    fr: hour < 18 ? 'Bonjour' : 'Bonsoir',
    es: hour < 18 ? 'Hola' : 'Buenas noches',
    ru: 'Привет',
    zh: '你好',
  };
  return byLang[currentLanguage()] || 'Hello';
}

function actionCard({ iconName, title, sub, cta, view, accent, disabled }) {
  return el('button', {
    class: `action-card${accent ? ' accent' : ''}`,
    type: 'button',
    disabled: disabled || undefined,
    onclick: () => ctx.navigate(view),
  },
    el('div', { class: 'action-icon' }, icon(iconName, 22)),
    el('div', { class: 'action-text' },
      el('strong', {}, title),
      el('span', {}, sub)),
    el('div', { class: 'action-cta' }, cta, icon('arrowRight', 16)));
}

function weekBars(lang) {
  const days = state.stats[lang]?.days || {};
  const labels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
  const now = new Date();
  const monday = new Date(now);
  monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));
  let max = 4;
  const data = [];
  for (let i = 0; i < 7; i++) {
    const d = new Date(monday.getTime() + i * DAY_MS);
    const rec = days[todayKey(d.getTime())];
    const total = d > now ? null : (rec ? rec.reviews + rec.newCards : 0);
    if (total) max = Math.max(max, total);
    data.push({ total, isToday: todayKey(d.getTime()) === todayKey() });
  }
  return el('div', { class: 'weekbars' }, data.map((d, i) =>
    el('div', { class: 'weekbar-col' },
      el('div', { class: `weekbar-bar${d.isToday ? ' today' : ''}` },
        el('span', { style: { height: `${d.total === null ? 0 : Math.max(d.total ? 12 : 3, (d.total / max) * 100)}%` } })),
      el('small', {}, labels[i]))));
}

function pipelineRow(label, value, cls, total) {
  const pct = total ? Math.max(value ? 2 : 0, Math.round((value / total) * 100)) : 0;
  return el('div', { class: 'pipeline-row' },
    el('span', { class: 'pipeline-label' }, label),
    el('div', { class: 'pipeline-track' }, el('span', { class: cls, style: { width: `${pct}%` } })),
    el('span', { class: 'pipeline-value' }, fmtInt(value)));
}

export function render(container) {
  const lang = currentLanguage();
  const profile = languageProfile(lang);
  const counts = deckCounts(lang);
  const due = dueCards(lang).length;
  const newRemaining = Math.max(0, Math.min(state.settings.newPerDay - newCardsIntroducedToday(lang), newCards(lang).length));
  const today = statDay(lang);
  const currentStreak = streak(lang);
  const acc = accuracyOverDays(30, lang);
  const empty = counts.total === 0;

  container.replaceChildren(
    el('div', { class: 'view-inner dashboard' },
      el('header', { class: 'dash-hero' },
        el('div', {},
          el('h1', {}, `${greeting()} 👋`),
          el('p', { class: 'muted' },
            empty
              ? `Let's build your first ${profile?.display || ''} deck.`
              : due > 0
                ? `${due} review${due === 1 ? '' : 's'} waiting - clearing them is today's highest-value minute.`
                : newRemaining > 0
                  ? 'Reviews are clear. Time to learn something new.'
                  : 'All caught up - compose something fun, or train your ear.')),
        el('div', { class: 'hero-stats' },
          el('div', { class: 'hero-stat', title: 'Days in a row with practice' },
            icon('flame', 20), el('strong', {}, currentStreak), el('span', {}, currentStreak === 1 ? 'day' : 'days')),
          el('div', { class: 'hero-stat', title: 'Practice time today' },
            icon('clock', 20), el('strong', {}, fmtDuration(today.timeMs)), el('span', {}, 'today')))),

      el('section', { class: 'action-row', dataset: { tour: 'actions' } },
        actionCard({
          iconName: 'refresh', view: 'review', accent: due > 0,
          title: due > 0 ? `Review ${due} card${due === 1 ? '' : 's'}` : 'Nothing due',
          sub: due > 0 ? 'Spaced repetition keeps memories alive' : 'Your queue is clear',
          cta: 'Review', disabled: due === 0,
        }),
        actionCard({
          iconName: 'sparkles', view: 'learn', accent: due === 0 && newRemaining > 0,
          title: empty ? 'Start learning' : newRemaining > 0 ? `Learn ${newRemaining} new` : 'Daily new done',
          sub: empty ? 'Generate your first lesson' : newRemaining > 0 ? 'Fresh items, fully guided' : 'Generate more or raise the limit',
          cta: 'Learn',
        }),
        actionCard({
          iconName: 'penLine', view: 'compose',
          title: 'Compose',
          sub: 'Any scene you can describe, at your level',
          cta: 'Create',
        })),

      el('div', { class: 'dash-grid' },
        el('section', { class: 'card' },
          el('h3', {}, 'This week'),
          weekBars(lang),
          el('div', { class: 'dash-meta' },
            el('span', {}, `${fmtInt(today.reviews)} reps today`),
            acc.rate !== null
              ? el('span', {}, `${Math.round(acc.rate * 100)}% accuracy · 30d`)
              : el('span', { class: 'muted' }, 'No reviews yet'))),

        el('section', { class: 'card' },
          el('h3', {}, `Deck · ${profile?.display || lang}`),
          counts.total === 0
            ? el('p', { class: 'muted' }, 'No cards yet. Generate a lesson in Learn to begin.')
            : el('div', { class: 'pipeline' },
              pipelineRow('New', counts.new, 'p-new', counts.total),
              pipelineRow('Learning', counts.learning, 'p-learning', counts.total),
              pipelineRow('Young', counts.review, 'p-young', counts.total),
              pipelineRow('Mature', counts.mature, 'p-mature', counts.total)),
          el('div', { class: 'dash-meta' },
            el('span', {}, `${fmtInt(counts.total)} cards total`),
            el('button', { class: 'btn-link', onclick: () => ctx.navigate('progress') }, 'Full progress ', icon('arrowRight', 14)))),

        el('section', { class: 'card wide focus-card', id: 'focusCard' },
          el('div', { class: 'focus-loading' }, el('span', { class: 'spinner dark' }))))));

  fillFocus(lang);
}

export function cleanup() {}

// "Today's structure": one grammar feature at the learner's level, rotating
// daily - a small deterministic nudge that links the grammar map to practice.
async function fillFocus(lang) {
  const card = document.querySelector('#focusCard');
  if (!card) return;
  let profile = null;
  try { profile = await api.grammar(lang); } catch { /* offline */ }
  const features = profile?.roadmap?.[state.settings.level] || [];
  if (!features.length) { card.remove(); return; }
  const dayNumber = Math.floor(Date.now() / DAY_MS);
  const feature = features[dayNumber % features.length];

  card.replaceChildren(
    el('div', { class: 'focus-body' },
      el('div', { class: 'focus-text' },
        el('span', { class: 'focus-kicker' }, icon('zap', 14), `Today's structure · ${state.settings.level}`),
        el('strong', { class: 'focus-name' }, feature.name),
        el('span', { class: 'focus-tip' }, feature.tip),
        el('span', { class: 'focus-example' }, `${feature.example} - ${feature.example_en}`)),
      el('div', { class: 'focus-actions' },
        el('button', {
          class: 'btn btn-soft btn-sm',
          onclick: () => { ctx.grammarFocus = feature.id; ctx.navigate('grammar'); },
        }, icon('book', 14), 'Grammar map'),
        el('button', {
          class: 'btn btn-primary btn-sm',
          onclick: () => {
            ctx.composePrefill = `A short piece that naturally practices "${feature.name}" (${feature.tip}) - pick any fun scenario.`;
            ctx.navigate('compose');
          },
        }, icon('sparkles', 14), 'Practice it'))));
}
