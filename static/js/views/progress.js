// Progress view: honest, local analytics of the whole learning system -
// vocabulary growth, memory strength, review history, grammar coverage,
// study mix, consistency, and leeches.

import { el, icon, fmtInt, fmtDuration, fmtInterval, toast, confirmDialog } from '../ui.js';
import {
  state, cards, deckCounts, streak, accuracyOverDays, todayKey,
  currentLanguage, removeCard, grammarSeen,
} from '../store.js';
import { retrievability } from '../srs.js';
import { api } from '../api.js';
import { ctx } from '../context.js';
import { levelName, levelBlurb, levelLabel } from '../levels.js';
import {
  ability, abilityBand, abilityRecord, isEnabled as adaptiveOn,
  levelRecommendation, modeForecast,
} from '../adaptive.js';

const DAY_MS = 86400000;

export function render(container) {
  const lang = currentLanguage();
  const counts = deckCounts(lang);
  const days = state.stats[lang]?.days || {};
  const acc30 = accuracyOverDays(30, lang);
  const totalReviews = Object.values(days).reduce((a, d) => a + d.reviews, 0);
  const totalTime = Object.values(days).reduce((a, d) => a + d.timeMs, 0);

  container.replaceChildren(
    el('div', { class: 'view-inner' },
      el('div', { class: 'stat-tiles' },
        tile('layers', fmtInt(counts.total), 'cards in deck'),
        tile('flame', fmtInt(streak(lang)), 'day streak'),
        tile('refresh', fmtInt(totalReviews), 'total reviews'),
        tile('target', acc30.rate === null ? '-' : `${Math.round(acc30.rate * 100)}%`, 'accuracy · 30d'),
        tile('clock', fmtDuration(totalTime), 'total study time'),
        tile('star', fmtInt(counts.mature), 'mature memories')),

      el('div', { class: 'dash-grid' },
        el('section', { class: 'card' },
          el('h3', {}, 'Vocabulary growth · 60d'),
          el('p', { class: 'muted small' }, 'Cumulative items introduced into your deck.'),
          growthChart(days)),
        el('section', { class: 'card' },
          el('h3', {}, 'Reviews · last 30 days'),
          reviewChart(days)),
        el('section', { class: 'card' },
          el('h3', {}, 'Memory strength now'),
          el('p', { class: 'muted small' }, 'Estimated recall probability of every learned card at this moment (FSRS retrievability).'),
          retentionHistogram(lang)),
        el('section', { class: 'card' },
          el('h3', {}, 'Study mix · 30d'),
          el('p', { class: 'muted small' }, 'Balanced skills grow together - reviews build memory, listening and speaking build fluency.'),
          studyMix(days)),
        adaptiveCard(lang),
        el('section', { class: 'card wide grammar-coverage-card' },
          el('div', { class: 'row gap', style: { justifyContent: 'space-between', alignItems: 'baseline' } },
            el('h3', {}, 'Grammar coverage'),
            el('button', { class: 'btn-link', onclick: () => ctx.navigate('grammar') }, 'Open grammar map ', icon('arrowRight', 14))),
          el('p', { class: 'muted small' }, 'How many structures per level your generated lessons and compositions have surfaced so far.'),
          el('div', { class: 'coverage-rows', id: 'coverageRows' }, el('span', { class: 'spinner dark' }))),
        el('section', { class: 'card wide' },
          el('h3', {}, 'Consistency'),
          el('p', { class: 'muted small' }, 'Memories consolidate during sleep - daily contact beats weekend marathons.'),
          heatmap(lang)),
        el('section', { class: 'card wide' },
          el('h3', {}, 'Leeches - hardest cards'),
          el('p', { class: 'muted small' }, 'Most lapses. Rewrite these with a mnemonic, or remove them.'),
          leechList(lang, container)))));

  fillCoverage(lang);
}

export function cleanup() {}

/**
 * What adaptive testing has measured, and what it is doing with it.
 * Sits in Progress because it is a measurement of the learner, not a setting.
 */
function adaptiveCard(lang) {
  const record = abilityRecord(lang);
  const score = ability(lang);
  const band = abilityBand(score);
  const recommendation = levelRecommendation(lang, state.settings.level);
  const trend = record.recent.map((sample) => sample.score);

  return el('section', { class: 'card wide' },
    el('div', { class: 'row gap', style: { justifyContent: 'space-between', alignItems: 'baseline' } },
      el('h3', {}, 'Adaptive testing'),
      el('span', { class: 'muted small' },
        adaptiveOn() ? `${record.samples} answers measured` : 'Currently switched off in Settings')),
    el('p', { class: 'muted small' },
      'One ability estimate per language, moved by every graded answer. It decides which retrieval each review asks of you.'),
    record.samples === 0
      ? el('p', { class: 'muted' }, 'Nothing measured yet. Answer a few reviews and this fills in.')
      : el('div', { class: 'adaptive-progress' },
        el('div', { class: 'adaptive-progress-head' },
          el('div', {},
            el('strong', {}, band.label),
            el('span', { class: 'muted small' }, band.hint)),
          el('div', { class: 'adaptive-score' },
            el('strong', {}, String(Math.round(score * 100))),
            el('span', {}, 'ability'))),
        trend.length > 1 ? abilityTrend(trend) : null,
        el('div', { class: 'coverage-rows' },
          modeForecast(lang).map((row) => el('div', { class: `coverage-row${row.selected ? ' current' : ''}` },
            el('span', { class: 'muted small forecast-label' }, row.label),
            el('div', { class: 'coverage-track big' }, el('span', { style: { width: `${Math.round(row.predicted * 100)}%` } })),
            el('span', { class: 'coverage-count' }, `${Math.round(row.predicted * 100)}%`)))),
        recommendation.action === 'hold'
          ? el('p', { class: 'muted small' }, recommendation.reason)
          : el('div', { class: `level-suggestion ${recommendation.action}` },
            el('div', {},
              el('strong', {}, recommendation.action === 'up'
                ? `Ready for ${levelLabel(recommendation.suggested)}`
                : `Consider ${levelLabel(recommendation.suggested)}`),
              el('span', { class: 'muted small' }, recommendation.reason)),
            el('span', { class: 'muted small' }, 'Change it in Settings'))));
}

/** How the estimate has moved over the last stretch of answers. */
function abilityTrend(scores) {
  const W = 560, H = 70, P = 5;
  const stepX = (W - P * 2) / Math.max(scores.length - 1, 1);
  const coords = scores.map((value, i) => [P + i * stepX, H - P - value * (H - P * 2)]);
  const line = coords.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const wrap = el('div', { class: 'growth-chart ability-trend' });
  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Ability estimate over recent answers">
      <path d="${line}" class="growth-line" fill="none"></path>
    </svg>`;
  return wrap;
}

function tile(iconName, value, label) {
  return el('div', { class: 'stat-tile' },
    el('div', { class: 'stat-tile-icon' }, icon(iconName, 18)),
    el('strong', {}, value),
    el('span', {}, label));
}

// -- charts -------------------------------------------------------------------
function growthChart(days) {
  const points = [];
  let cumulative = 0;
  const backfill = [];
  for (let i = 59; i >= 0; i--) {
    const rec = days[todayKey(Date.now() - i * DAY_MS)];
    backfill.push(rec ? rec.newCards : 0);
  }
  for (const v of backfill) { cumulative += v; points.push(cumulative); }
  const max = Math.max(cumulative, 1);
  const W = 560, H = 120, P = 4;
  const stepX = (W - P * 2) / Math.max(points.length - 1, 1);
  const coords = points.map((v, i) => [P + i * stepX, H - P - (v / max) * (H - P * 2)]);
  const line = coords.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area = `${line} L${coords[coords.length - 1][0].toFixed(1)},${H - P} L${P},${H - P} Z`;
  const svg = el('div', { class: 'growth-chart' });
  svg.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Cumulative vocabulary growth">
      <path d="${area}" class="growth-area"></path>
      <path d="${line}" class="growth-line" fill="none"></path>
    </svg>`;
  return el('div', {}, svg,
    el('div', { class: 'dash-meta' },
      el('span', {}, `${fmtInt(cumulative)} items in 60 days`),
      el('span', { class: 'muted' }, cumulative ? `~${(cumulative / 60).toFixed(1)}/day` : 'Start in Learn')));
}

function reviewChart(days) {
  const bars = [];
  let max = 5;
  const data = [];
  for (let i = 29; i >= 0; i--) {
    const key = todayKey(Date.now() - i * DAY_MS);
    const rec = days[key];
    const value = rec ? rec.reviews + rec.newCards : 0;
    max = Math.max(max, value);
    data.push({ key, value });
  }
  for (const d of data) {
    bars.push(el('div', { class: 'chart-bar', title: `${d.key} · ${d.value}` },
      el('span', { style: { height: `${Math.max(d.value ? 6 : 2, (d.value / max) * 100)}%` } })));
  }
  return el('div', { class: 'chart-bars' }, bars);
}

function retentionHistogram(lang) {
  const now = Date.now();
  const learned = cards(lang).filter((c) => c.state === 'review' && !c.suspended);
  if (!learned.length) return el('p', { class: 'muted' }, 'No graduated cards yet.');
  const buckets = [0, 0, 0, 0, 0];
  for (const c of learned) {
    const elapsed = c.srs.lastReview ? (now - c.srs.lastReview) / DAY_MS : 0;
    const r = retrievability(elapsed, c.srs.S) * 100;
    if (r < 60) buckets[0] += 1;
    else if (r < 75) buckets[1] += 1;
    else if (r < 85) buckets[2] += 1;
    else if (r < 95) buckets[3] += 1;
    else buckets[4] += 1;
  }
  const labels = ['<60%', '60-75', '75-85', '85-95', '95%+'];
  const max = Math.max(...buckets, 1);
  return el('div', { class: 'hist' },
    buckets.map((b, i) => el('div', { class: 'hist-col' },
      el('div', { class: 'hist-bar' }, el('span', { style: { height: `${Math.max(b ? 8 : 2, (b / max) * 100)}%` } })),
      el('small', {}, labels[i]),
      el('small', { class: 'muted' }, String(b)))));
}

function studyMix(days) {
  let reviews = 0, listening = 0, speaking = 0;
  for (let i = 0; i < 30; i++) {
    const rec = days[todayKey(Date.now() - i * DAY_MS)];
    if (!rec) continue;
    listening += rec.listening || 0;
    speaking += rec.speaking || 0;
    reviews += Math.max(0, rec.reviews - (rec.listening || 0) - (rec.speaking || 0));
  }
  const total = Math.max(reviews + listening + speaking, 1);
  const rows = [
    ['Recall reviews', reviews, 'mix-review'],
    ['Listening', listening, 'mix-listen'],
    ['Speaking', speaking, 'mix-speak'],
  ];
  return el('div', { class: 'pipeline' },
    rows.map(([label, value, cls]) => el('div', { class: 'pipeline-row' },
      el('span', { class: 'pipeline-label wide' }, label),
      el('div', { class: 'pipeline-track' },
        el('span', { class: cls, style: { width: `${Math.max(value ? 3 : 0, Math.round((value / total) * 100))}%` } })),
      el('span', { class: 'pipeline-value' }, fmtInt(value)))));
}

function heatmap(lang) {
  const days = state.stats[lang]?.days || {};
  const weeks = 20;
  const today = new Date();
  const start = new Date(today);
  start.setDate(start.getDate() - (weeks * 7 - 1) - ((start.getDay() + 6) % 7));
  const values = [];
  for (let i = 0; i < weeks * 7; i++) {
    const d = new Date(start.getTime() + i * DAY_MS);
    if (d > today) break;
    const key = todayKey(d.getTime());
    const rec = days[key];
    values.push({ key, total: rec ? rec.reviews + rec.newCards : 0 });
  }
  const max = Math.max(4, ...values.map((v) => v.total));
  return el('div', { class: 'heatmap' },
    values.map((v) => el('div', {
      class: `heat-cell l${v.total === 0 ? 0 : Math.min(4, Math.ceil((v.total / max) * 4))}`,
      title: `${v.key} · ${v.total} ${v.total === 1 ? 'rep' : 'reps'}`,
    })));
}

// -- grammar coverage ---------------------------------------------------------
async function fillCoverage(lang) {
  const rowsEl = document.querySelector('#coverageRows');
  if (!rowsEl) return;
  let profile = null;
  try { profile = await api.grammar(lang); } catch { /* offline */ }
  if (!profile) {
    rowsEl.replaceChildren(el('p', { class: 'muted' }, 'Grammar map unavailable.'));
    return;
  }
  const seen = grammarSeen(lang);
  const userLevel = state.settings.level;
  rowsEl.replaceChildren(...Object.entries(profile.roadmap).map(([level, features]) => {
    const covered = features.filter((f) => seen[f.id]?.seen).length;
    const pct = Math.round((covered / features.length) * 100);
    return el('div', { class: `coverage-row${level === userLevel ? ' current' : ''}`, title: levelBlurb(level) },
      el('span', { class: `level-chip${level === userLevel ? ' active' : ''}` },
        el('strong', {}, level),
        el('span', { class: 'level-chip-name' }, levelName(level))),
      el('div', { class: 'coverage-track big' }, el('span', { style: { width: `${pct}%` } })),
      el('span', { class: 'coverage-count' }, `${covered} / ${features.length}`));
  }));
}

// -- leeches ------------------------------------------------------------------
function leechList(lang, container) {
  const leeches = cards(lang)
    .filter((c) => c.srs.lapses >= 2)
    .sort((a, b) => b.srs.lapses - a.srs.lapses)
    .slice(0, 8);
  if (!leeches.length) return el('p', { class: 'muted' }, 'No leeches - nothing is repeatedly failing.');
  return el('div', { class: 'leech-list' },
    leeches.map((c) => el('div', { class: 'leech-row' },
      el('div', { class: 'leech-text' },
        el('strong', {}, c.target),
        el('span', { class: 'muted small' }, `${c.english} · ${c.srs.lapses} lapses · next ${fmtInterval((c.srs.due - Date.now()) / DAY_MS)}`)),
      el('button', {
        class: 'btn-icon', title: 'Remove card',
        onclick: async () => {
          if (await confirmDialog(`Remove “${c.target}” from your deck?`, { danger: true, confirmLabel: 'Remove' })) {
            removeCard(c.id, lang);
            toast('Card removed', 'info');
            render(container);
          }
        },
      }, icon('trash', 16)))));
}
