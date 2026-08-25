// Topics: the curated content library, browsable without any API key.
//
// Four tiers, mirroring backend/curriculum/taxonomy.py:
//   Topics  ->  Domain  ->  Unit  ->  Groups of items
// The route carries the position (#/topics/food/restaurant), so the browser
// back button, bookmarks, and deep links from other views all work.
//
// Nothing here needs the LLM. When a key IS configured, each unit additionally
// offers "extend with AI", which generates fresh material inside that unit's
// scope while avoiding everything already curated or already in the deck.

import { el, icon, toast, fmtInt, progressSteps } from '../ui.js';
import { api, ApiError } from '../api.js';
import {
  state, addCards, currentLanguage, unitCoverage, deckIndex, normalizeTarget,
  recordGrammarFeatures, updateSettings,
} from '../store.js';
import { audioButton } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';

const treeCache = new Map();   // lang -> { domains, path, summary }
const unitCache = new Map();   // `${lang}:${unitId}` -> unit detail
let searchTimer = null;

const LEVELS = ['A1', 'A2', 'B1', 'B2'];

export function cleanup() {
  clearTimeout(searchTimer);
}

export function render(container, params = []) {
  const lang = currentLanguage();
  const [first, second] = params;

  if (first === 'search') {
    renderShell(container, lang, (host) => renderSearch(host, lang, second || ''));
    return;
  }
  if (first && second) {
    renderShell(container, lang, (host) => renderUnit(host, lang, first, second));
    return;
  }
  if (first) {
    renderShell(container, lang, (host) => renderDomain(host, lang, first));
    return;
  }
  renderShell(container, lang, (host) => renderOverview(host, lang));
}

// -- shell: loads the tree once, then hands off to the level renderers -------
function renderShell(container, lang, fill) {
  const host = el('div', { class: 'view-inner topics' },
    el('div', { class: 'topics-loading' }, el('span', { class: 'spinner dark' }), ' Loading the library…'));
  container.replaceChildren(host);
  loadTree(lang).then((tree) => {
    if (currentLanguage() !== lang) return;
    if (!tree) {
      host.replaceChildren(el('div', { class: 'card start-card' },
        el('h2', {}, 'Library unavailable'),
        el('p', { class: 'muted' }, 'Could not reach the local server. Is app.py still running?'),
        el('button', { class: 'btn btn-primary', onclick: () => render(container) }, 'Try again')));
      return;
    }
    fill(host, tree);
  });
}

async function loadTree(lang) {
  if (!treeCache.has(lang)) {
    try {
      treeCache.set(lang, await api.curriculum(lang));
    } catch {
      return null;
    }
  }
  return treeCache.get(lang);
}

async function loadUnit(lang, unitId) {
  const key = `${lang}:${unitId}`;
  if (!unitCache.has(key)) {
    unitCache.set(key, await api.curriculumUnit(lang, unitId));
  }
  return unitCache.get(key);
}

function tree(lang) { return treeCache.get(lang); }

// -- shared pieces ------------------------------------------------------------
function crumbs(...parts) {
  const nodes = [];
  parts.forEach((part, index) => {
    if (index) nodes.push(el('span', { class: 'crumb-sep' }, '/'));
    nodes.push(part.to
      ? el('button', { class: 'crumb-link', type: 'button', onclick: () => ctx.navigate(...part.to) }, part.label)
      : el('span', { class: 'crumb-current' }, part.label));
  });
  return el('nav', { class: 'crumbs', 'aria-label': 'Breadcrumb' }, nodes);
}

/** Donut showing how much of a unit or domain is already in the deck. */
function ring(done, total, size = 40) {
  const pct = total ? Math.min(1, done / total) : 0;
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const wrap = el('div', { class: `ring${pct >= 1 ? ' complete' : ''}`, title: `${done} of ${total} in your deck` });
  // A round line cap on a zero-length arc still paints a dot, so an empty ring
  // has to omit the value arc entirely rather than draw 0%.
  const arc = pct > 0
    ? `<circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-value" fill="none" stroke-width="4"
        stroke-dasharray="${(c * pct).toFixed(1)} ${c.toFixed(1)}"
        transform="rotate(-90 ${size / 2} ${size / 2})" stroke-linecap="round"></circle>`
    : '';
  wrap.innerHTML = `
    <svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" aria-hidden="true">
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-track" fill="none" stroke-width="4"></circle>
      ${arc}
    </svg>
    <span class="ring-label">${Math.round(pct * 100)}%</span>`;
  return wrap;
}

function levelChip(level, extraClass = '') {
  return el('span', { class: `level-chip ${extraClass}`.trim() }, level);
}

function searchBox(lang, initial = '') {
  const input = el('input', {
    class: 'input topics-search', type: 'search', value: initial,
    placeholder: 'Search every topic and word in the library…',
    'aria-label': 'Search the library',
    oninput: (e) => {
      const value = e.target.value.trim();
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        if (value.length >= 2) ctx.navigate('topics', 'search', value);
        else if (!value) ctx.navigate('topics');
      }, 320);
    },
  });
  return el('div', { class: 'topics-search-wrap' }, icon('eye', 16), input);
}

// -- level 1: overview --------------------------------------------------------
function renderOverview(host, lang) {
  const data = tree(lang);
  const profile = languageProfile(lang);
  const coverage = unitCoverage(lang);
  const levelFilter = state.settings.topicLevelFilter || 'all';

  const allUnits = data.domains.flatMap((d) => d.units);
  const available = allUnits.filter((u) => u.available);
  const totalItems = available.reduce((a, u) => a + u.itemCount, 0);
  const inDeck = available.reduce((a, u) => a + Math.min(coverage[u.id]?.inDeck || 0, u.itemCount), 0);

  if (!available.length) {
    host.replaceChildren(
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('layers', 26)),
        el('h2', {}, `No curated ${profile?.display || ''} units yet`),
        el('p', { class: 'muted' }, 'The taxonomy is shared across languages, but this one has no authored content in this build. Generate a lesson in Learn instead.'),
        el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('learn') }, icon('sparkles', 16), 'Go to Learn')));
    return;
  }

  const next = nextUnit(data, coverage);

  host.replaceChildren(
    el('header', { class: 'topics-hero' },
      el('div', { class: 'topics-hero-text' },
        el('h1', {}, 'Topic library'),
        el('p', { class: 'muted' },
          `${fmtInt(totalItems)} curated ${profile?.display || ''} items across ${available.length} topics in ${data.domains.length} areas of life. Everything here works without an AI key.`)),
      el('div', { class: 'topics-hero-progress' },
        ring(inDeck, totalItems, 62),
        el('div', { class: 'topics-hero-stat' },
          el('strong', {}, `${fmtInt(inDeck)} / ${fmtInt(totalItems)}`),
          el('span', {}, 'in your deck')))),

    searchBox(lang),

    next ? el('section', { class: 'card next-unit-card' },
      el('div', { class: 'next-unit-text' },
        el('span', { class: 'focus-kicker' }, icon('target', 14), 'Suggested next'),
        el('strong', {}, next.title),
        el('span', { class: 'muted' }, next.goal)),
      el('div', { class: 'row gap' },
        levelChip(next.level, 'big'),
        el('button', {
          class: 'btn btn-primary',
          onclick: () => ctx.navigate('topics', next.domain, next.id),
        }, 'Open topic', icon('arrowRight', 16)))) : null,

    el('div', { class: 'level-filter', role: 'group', 'aria-label': 'Filter by level' },
      el('span', { class: 'muted small' }, 'Level'),
      ['all', ...LEVELS].map((lv) => el('button', {
        class: `filter-chip${levelFilter === lv ? ' active' : ''}`, type: 'button',
        onclick: () => { updateSettings({ topicLevelFilter: lv }); renderOverview(host, lang); },
      }, lv === 'all' ? 'All' : lv))),

    el('div', { class: 'domain-grid' },
      data.domains.map((domain) => domainCard(domain, coverage, levelFilter))));
}

function domainCard(domain, coverage, levelFilter) {
  const units = domain.units.filter((u) => u.available && (levelFilter === 'all' || u.level === levelFilter));
  const total = units.reduce((a, u) => a + u.itemCount, 0);
  const done = units.reduce((a, u) => a + Math.min(coverage[u.id]?.inDeck || 0, u.itemCount), 0);
  if (!units.length) return null;
  return el('button', {
    class: `domain-card accent-${domain.accent}`, type: 'button',
    onclick: () => ctx.navigate('topics', domain.id),
  },
    el('div', { class: 'domain-card-head' },
      el('span', { class: 'domain-icon' }, icon(domain.icon, 20)),
      ring(done, total, 38)),
    el('strong', { class: 'domain-title' }, domain.title),
    el('span', { class: 'domain-blurb' }, domain.blurb),
    el('div', { class: 'domain-meta' },
      el('span', {}, `${units.length} topic${units.length === 1 ? '' : 's'}`),
      el('span', { class: 'dot-sep' }, '·'),
      el('span', {}, `${total} items`),
      el('span', { class: 'domain-levels' }, [...new Set(units.map((u) => u.level))].sort().map((lv) => levelChip(lv)))));
}

/** First unit on the recommended path that is not yet fully in the deck. */
function nextUnit(data, coverage) {
  const byId = new Map(data.domains.flatMap((d) => d.units).map((u) => [u.id, u]));
  for (const id of data.path) {
    const unit = byId.get(id);
    if (!unit?.available) continue;
    if ((coverage[id]?.inDeck || 0) < unit.itemCount) return unit;
  }
  return null;
}

// -- level 2: one domain ------------------------------------------------------
function renderDomain(host, lang, domainId) {
  const data = tree(lang);
  const domain = data.domains.find((d) => d.id === domainId);
  if (!domain) { ctx.navigate('topics'); return; }
  const coverage = unitCoverage(lang);
  const units = domain.units.filter((u) => u.available);

  host.replaceChildren(
    crumbs({ label: 'Topics', to: ['topics'] }, { label: domain.title }),
    el('header', { class: `domain-hero accent-${domain.accent}` },
      el('span', { class: 'domain-icon big' }, icon(domain.icon, 26)),
      el('div', {},
        el('h1', {}, domain.title),
        el('p', { class: 'muted' }, domain.blurb))),
    searchBox(lang),
    units.length
      ? el('div', { class: 'unit-list' }, units.map((unit) => unitRow(unit, coverage[unit.id], domain)))
      : el('p', { class: 'muted' }, 'No authored units in this area for this language yet.'));
}

function unitRow(unit, cover, domain) {
  const inDeck = Math.min(cover?.inDeck || 0, unit.itemCount);
  const learned = Math.min(cover?.learned || 0, unit.itemCount);
  const pct = unit.itemCount ? Math.round((inDeck / unit.itemCount) * 100) : 0;
  return el('button', {
    class: `unit-row${inDeck >= unit.itemCount ? ' complete' : ''}`, type: 'button',
    onclick: () => ctx.navigate('topics', domain.id, unit.id),
  },
    el('div', { class: 'unit-row-main' },
      el('div', { class: 'unit-row-title' },
        levelChip(unit.level),
        el('strong', {}, unit.title),
        inDeck >= unit.itemCount ? el('span', { class: 'unit-done' }, icon('check', 12), 'in deck') : null),
      el('span', { class: 'unit-goal' }, unit.goal),
      el('div', { class: 'unit-keywords' }, unit.keywords.map((k) => el('span', { class: 'kw' }, k)))),
    el('div', { class: 'unit-row-side' },
      el('div', { class: 'unit-bar', title: `${inDeck} of ${unit.itemCount} in deck, ${learned} studied` },
        el('span', { class: 'unit-bar-deck', style: { width: `${pct}%` } }),
        el('span', { class: 'unit-bar-learned', style: { width: `${unit.itemCount ? (learned / unit.itemCount) * 100 : 0}%` } })),
      el('span', { class: 'unit-count muted small' }, `${inDeck}/${unit.itemCount}`),
      icon('arrowRight', 16)));
}

// -- level 3: one unit --------------------------------------------------------
function renderUnit(host, lang, domainId, unitId) {
  const data = tree(lang);
  const domain = data.domains.find((d) => d.id === domainId);
  host.replaceChildren(el('div', { class: 'topics-loading' }, el('span', { class: 'spinner dark' }), ' Loading topic…'));

  loadUnit(lang, unitId).then((unit) => {
    if (currentLanguage() !== lang) return;
    paintUnit(host, lang, domain, unit);
  }).catch((err) => {
    host.replaceChildren(
      crumbs({ label: 'Topics', to: ['topics'] }, { label: domain?.title || 'Topic' }),
      el('div', { class: 'card start-card' },
        el('h2', {}, 'Topic unavailable'),
        el('p', { class: 'muted' }, err instanceof ApiError && err.status === 404
          ? 'This language has no curated content for that topic yet.'
          : `Could not load the topic: ${err.message}`),
        el('button', { class: 'btn btn-primary', onclick: () => ctx.navigate('topics', domainId) }, 'Back to the area')));
  });
}

function paintUnit(host, lang, domain, unit) {
  const profile = languageProfile(lang);
  const owned = deckIndex(lang);
  const selection = new Set();
  const rowsById = new Map();

  const allItems = unit.groups.flatMap((g) => g.items);
  const newItems = allItems.filter((item) => !owned.has(normalizeTarget(item.target)));
  newItems.forEach((item) => selection.add(normalizeTarget(item.target)));

  const countEl = el('strong', {}, '');
  const addBtn = el('button', { class: 'btn btn-primary btn-lg', onclick: () => addSelection() });
  const actionBar = el('div', { class: 'unit-actionbar' },
    el('div', { class: 'unit-actionbar-text' },
      countEl,
      el('span', { class: 'muted small' }, `${allItems.length - newItems.length} of ${allItems.length} already in your deck`)),
    el('div', { class: 'row gap wrap' },
      el('button', { class: 'btn btn-ghost btn-sm', onclick: () => setAll(true) }, 'Select all new'),
      el('button', { class: 'btn btn-ghost btn-sm', onclick: () => setAll(false) }, 'Clear'),
      addBtn));

  const syncBar = () => {
    const n = selection.size;
    countEl.textContent = n ? `${n} item${n === 1 ? '' : 's'} selected` : 'Nothing selected';
    addBtn.disabled = n === 0;
    addBtn.replaceChildren(icon('plus', 18), n ? `Add ${n} to deck` : 'Add to deck');
  };

  const setAll = (on) => {
    selection.clear();
    if (on) newItems.forEach((item) => selection.add(normalizeTarget(item.target)));
    rowsById.forEach((row, id) => {
      const box = row.querySelector('.item-check');
      if (box && !box.disabled) box.checked = selection.has(id);
    });
    syncBar();
  };

  const addSelection = () => {
    const chosen = allItems.filter((item) => selection.has(normalizeTarget(item.target)));
    if (!chosen.length) return;
    const report = addCards(chosen, unit.title, lang, { unit: unit.id, level: unit.level });
    recordGrammarFeatures(unit.grammar || [], lang);
    toast(
      report.added
        ? `${report.added} card${report.added === 1 ? '' : 's'} added from ${unit.title}`
        : 'Those items were already in your deck',
      report.added ? 'success' : 'info');
    // Repaint so the rows show their new "in deck" state.
    paintUnit(host, lang, domain, unit);
    if (report.added) {
      host.querySelector('.unit-after-add')?.replaceChildren(
        el('span', { class: 'muted small' }, `${report.added} new card${report.added === 1 ? '' : 's'} waiting`),
        el('button', { class: 'btn btn-primary btn-sm', onclick: () => ctx.navigate('learn') },
          icon('sparkles', 15), 'Learn them now'));
    }
  };

  const groupNodes = unit.groups.map((group) => el('section', { class: 'item-group' },
    el('h3', { class: 'item-group-title' }, group.title, el('span', { class: 'muted small' }, `${group.items.length} items`)),
    el('div', { class: 'item-rows' }, group.items.map((item) => {
      const id = normalizeTarget(item.target);
      const already = owned.has(id);
      const row = itemRow(item, { lang, profile, already, id, selection, syncBar });
      rowsById.set(id, row);
      return row;
    }))));

  host.replaceChildren(
    crumbs(
      { label: 'Topics', to: ['topics'] },
      { label: domain?.title || unit.domainTitle, to: ['topics', unit.domain] },
      { label: unit.title }),

    el('header', { class: 'unit-hero' },
      el('div', { class: 'unit-hero-main' },
        el('div', { class: 'row gap' }, levelChip(unit.level, 'big'), el('span', { class: 'muted small' }, unit.domainTitle)),
        el('h1', {}, unit.title),
        el('p', { class: 'unit-hero-goal' }, icon('target', 15), el('span', {}, unit.goal)),
        unit.grammar?.length ? el('div', { class: 'unit-grammar' },
          el('span', { class: 'muted small' }, 'Grammar exercised:'),
          unit.grammar.map((fid) => el('button', {
            class: 'kw kw-link', type: 'button', title: 'Open in the Grammar map',
            onclick: () => { ctx.grammarFocus = fid; ctx.navigate('grammar'); },
          }, fid.replace(/^[a-z]{2}-/, '').replace(/-/g, ' ')))) : null),
      el('div', { class: 'unit-hero-side' }, ring(allItems.length - newItems.length, allItems.length, 62))),

    actionBar,
    el('div', { class: 'unit-after-add row gap center' }),
    el('div', { class: 'item-groups' }, groupNodes),
    extendCard(lang, unit, host, domain));

  syncBar();
}

function itemRow(item, { lang, profile, already, id, selection, syncBar }) {
  const check = already
    ? el('span', { class: 'item-owned', title: 'Already in your deck' }, icon('check', 14))
    : el('input', {
      class: 'item-check', type: 'checkbox', checked: selection.has(id) || undefined,
      'aria-label': `Select ${item.target}`,
      onchange: (e) => {
        if (e.target.checked) selection.add(id); else selection.delete(id);
        syncBar();
      },
    });

  const example = item.example
    ? el('div', { class: 'item-example' },
      el('span', { class: 'item-example-target' }, item.example),
      item.example_en ? el('small', {}, item.example_en) : null)
    : null;

  return el('label', { class: `item-row${already ? ' owned' : ''}` },
    check,
    el('div', { class: 'item-main' },
      el('div', { class: 'item-head' },
        el('strong', { class: 'item-target', style: { fontFamily: profile?.fontStack || 'inherit' } }, item.target),
        item.pronunciation && state.settings.showPronunciation
          ? el('span', { class: 'item-ipa' }, item.pronunciation) : null,
        item.tags?.includes('trap') ? el('span', { class: 'tag-trap', title: 'Common trap for English speakers' }, 'trap') : null),
      el('div', { class: 'item-english' }, item.english),
      example,
      item.note ? el('div', { class: 'item-note' }, icon('lightbulb', 13), el('span', {}, item.note)) : null),
    el('div', { class: 'item-actions' }, audioButton(item.target, { lang, kind: 'ghost' })));
}

// -- optional AI extension of a unit -----------------------------------------
function extendCard(lang, unit, host, domain) {
  if ((ctx.config?.provider || 'offline') === 'offline') {
    return el('section', { class: 'card extend-card muted-card' },
      el('h3', {}, icon('sparkles', 16), ' Want more than the curated set?'),
      el('p', { class: 'muted small' },
        'Connect an OpenRouter key in Settings and this topic can generate unlimited extra items inside the same scope, skipping everything you already have.'));
  }
  const countSelect = el('select', { class: 'input compact' },
    [8, 12, 16, 20].map((n) => el('option', { value: n, selected: n === 12 || undefined }, `${n} items`)));
  const progressHost = el('div', { class: 'progress-host' });
  const btn = el('button', {
    class: 'btn btn-soft',
    onclick: async () => {
      btn.disabled = true;
      btn.replaceChildren(el('span', { class: 'spinner' }), 'Generating');
      const progress = progressSteps([
        'Sending the topic scope to the model',
        `Writing ${countSelect.value} fresh items at ${unit.level}`,
        'Checking structure and pronunciation',
        'Adding cards to your deck',
      ], [1200, 12000, 4000]);
      progressHost.replaceChildren(progress.root);
      try {
        const pack = await api.lesson({
          language: lang, unit: unit.id, level: unit.level,
          count: Number(countSelect.value),
          knownWords: unit.groups.flatMap((g) => g.items.map((i) => i.target)),
        });
        const report = addCards(pack.items, unit.title, lang, { unit: unit.id, level: unit.level });
        recordGrammarFeatures(pack.grammar_features || [], lang);
        progress.finish();
        toast(report.added
          ? `${report.added} extra card${report.added === 1 ? '' : 's'} added to ${unit.title}`
          : 'The model produced only items you already have', report.added ? 'success' : 'info');
        setTimeout(() => renderUnit(host, lang, domain?.id || unit.domain, unit.id), 500);
      } catch (err) {
        progress.fail(err instanceof ApiError && err.status === 503
          ? 'Generation is unavailable right now. The curated items above still work.'
          : `Generation failed: ${err.message}`);
        btn.disabled = false;
        btn.replaceChildren(icon('sparkles', 16), 'Extend this topic');
      }
    },
  }, icon('sparkles', 16), 'Extend this topic');

  return el('section', { class: 'card extend-card' },
    el('h3', {}, icon('sparkles', 16), ' Extend this topic'),
    el('p', { class: 'muted small' },
      'Generates new items inside this topic only, avoiding everything curated above and everything already in your deck.'),
    el('div', { class: 'row gap wrap' }, countSelect, btn),
    progressHost);
}

// -- search results -----------------------------------------------------------
function renderSearch(host, lang, query) {
  const data = tree(lang);
  const profile = languageProfile(lang);
  host.replaceChildren(
    crumbs({ label: 'Topics', to: ['topics'] }, { label: `Search: ${query}` }),
    searchBox(lang, query),
    el('div', { class: 'topics-loading' }, el('span', { class: 'spinner dark' }), ' Searching…'));

  api.curriculumSearch(lang, query).then((results) => {
    if (currentLanguage() !== lang) return;
    const owned = deckIndex(lang);
    const body = [];

    if (results.units.length) {
      body.push(el('section', { class: 'search-section' },
        el('h3', {}, 'Topics'),
        el('div', { class: 'search-unit-grid' }, results.units.map((u) => el('button', {
          class: 'search-unit', type: 'button',
          onclick: () => ctx.navigate('topics', u.domain, u.id),
        },
          levelChip(u.level),
          el('strong', {}, u.title),
          el('span', { class: 'muted small' }, `${u.domainTitle} · ${u.itemCount} items`))))));
    }

    if (results.items.length) {
      const chosen = results.items.filter((i) => !owned.has(normalizeTarget(i.target)));
      body.push(el('section', { class: 'search-section' },
        el('div', { class: 'row gap', style: { justifyContent: 'space-between', alignItems: 'baseline' } },
          el('h3', {}, `Words and phrases (${results.items.length})`),
          chosen.length ? el('button', {
            class: 'btn btn-soft btn-sm',
            onclick: () => {
              const report = addCards(chosen, `Search: ${query}`, lang);
              toast(`${report.added} card${report.added === 1 ? '' : 's'} added`, report.added ? 'success' : 'info');
              renderSearch(host, lang, query);
            },
          }, icon('plus', 14), `Add all ${chosen.length} new`) : null),
        el('div', { class: 'item-rows' }, results.items.map((item) => {
          const already = owned.has(normalizeTarget(item.target));
          return el('div', { class: `item-row search-row${already ? ' owned' : ''}` },
            el('div', { class: 'item-main' },
              el('div', { class: 'item-head' },
                el('strong', { class: 'item-target', style: { fontFamily: profile?.fontStack || 'inherit' } }, item.target),
                item.pronunciation ? el('span', { class: 'item-ipa' }, item.pronunciation) : null),
              el('div', { class: 'item-english' }, item.english),
              el('button', {
                class: 'item-source', type: 'button',
                onclick: () => ctx.navigate('topics', item.domain, item.unit),
              }, icon('layers', 12), `${item.domainTitle} · ${item.unitTitle}`)),
            el('div', { class: 'item-actions' },
              audioButton(item.target, { lang, kind: 'ghost' }),
              already
                ? el('span', { class: 'item-owned', title: 'Already in your deck' }, icon('check', 14))
                : el('button', {
                  class: 'btn-icon', title: 'Add to deck',
                  onclick: (e) => {
                    const report = addCards([item], item.unitTitle, lang, { unit: item.unit, level: item.level });
                    toast(report.added ? `“${item.target}” added` : 'Already in your deck', report.added ? 'success' : 'info');
                    e.currentTarget.replaceWith(el('span', { class: 'item-owned' }, icon('check', 14)));
                  },
                }, icon('plus', 16))));
        }))));
    }

    if (!body.length) {
      body.push(el('div', { class: 'card start-card' },
        el('h2', {}, 'No matches'),
        el('p', { class: 'muted' }, `Nothing in the curated ${profile?.display || ''} library matches “${query}”.`),
        el('div', { class: 'row gap center' },
          el('button', { class: 'btn btn-ghost', onclick: () => ctx.navigate('topics') }, 'Back to the library'),
          el('button', {
            class: 'btn btn-primary',
            onclick: () => { ctx.learnTopic = query; ctx.navigate('learn'); },
          }, icon('sparkles', 16), 'Generate a lesson on this'))));
    }

    host.replaceChildren(
      crumbs({ label: 'Topics', to: ['topics'] }, { label: `Search: ${query}` }),
      searchBox(lang, query),
      ...body);
    const input = host.querySelector('.topics-search');
    if (input) { input.focus(); input.setSelectionRange(query.length, query.length); }
  }).catch(() => {
    host.replaceChildren(
      crumbs({ label: 'Topics', to: ['topics'] }, { label: 'Search' }),
      el('p', { class: 'muted' }, 'Search is unavailable right now.'));
  });
  void data;
}
