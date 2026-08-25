// App bootstrap: chrome (sidebar + topbar), hash router, global shortcuts.

import { el, icon, toast } from './ui.js';
import { api } from './api.js';
import { state, on, dueCards, streak, currentLanguage, updateSettings } from './store.js';
import { ctx, languageProfile } from './context.js';
import { applyTheme } from './theme.js';
import { maybeStartTour, startTour } from './tour.js';
import { openSettings } from './views/settings.js';
import { stopAudio } from './audio.js';

import * as dashboard from './views/dashboard.js';
import * as topics from './views/topics.js';
import * as learn from './views/learn.js';
import * as review from './views/review.js';
import * as listen from './views/listen.js';
import * as speak from './views/speak.js';
import * as compose from './views/compose.js';
import * as grammar from './views/grammar.js';
import * as progress from './views/progress.js';

const VIEWS = {
  dashboard: { module: dashboard, label: 'Home', icon: 'home', title: 'Today' },
  topics: { module: topics, label: 'Topics', icon: 'layers', title: 'Topic library' },
  learn: { module: learn, label: 'Learn', icon: 'sparkles', title: 'Learn new items' },
  review: { module: review, label: 'Review', icon: 'refresh', title: 'Review queue' },
  listen: { module: listen, label: 'Listen', icon: 'headphones', title: 'Listening lab' },
  speak: { module: speak, label: 'Speak', icon: 'mic', title: 'Speaking lab' },
  compose: { module: compose, label: 'Compose', icon: 'penLine', title: 'Compose' },
  grammar: { module: grammar, label: 'Grammar', icon: 'book', title: 'Grammar map' },
  progress: { module: progress, label: 'Progress', icon: 'chart', title: 'Progress' },
};

// Old bookmarks/hashes from earlier versions
const ROUTE_ALIASES = { read: 'compose', stats: 'progress', library: 'topics' };

let currentView = null;
let mainEl = null;

/** Build the hash for a route. Extra segments become view parameters. */
function routeHash(name, params = []) {
  const clean = params.filter((p) => p !== null && p !== undefined && p !== '');
  return `#/${[name, ...clean.map(encodeURIComponent)].join('/')}`;
}

export function navigate(name, ...params) {
  const resolved = ROUTE_ALIASES[name] || name;
  const target = VIEWS[resolved] ? resolved : 'dashboard';
  const hash = routeHash(target, params);
  if (location.hash === hash) {
    renderView(target, params); // same hash: re-render explicitly, no hashchange
    return;
  }
  location.hash = hash; // hashchange handler re-enters
}

function resolveRoute() {
  const raw = location.hash.replace(/^#\/?/, '');
  const parts = raw.split('/').filter(Boolean).map(decodeURIComponent);
  const name = ROUTE_ALIASES[parts[0]] || parts[0];
  return { name: name in VIEWS ? name : 'dashboard', params: parts.slice(1) };
}

function renderView(name, params = []) {
  VIEWS[currentView]?.module.cleanup?.();
  stopAudio();
  currentView = name;
  document.querySelectorAll('.nav-item').forEach((b) => {
    b.classList.toggle('active', b.dataset.view === name);
  });
  mainEl.dataset.view = name;
  const titleEl = document.querySelector('#topbarTitle');
  if (titleEl) titleEl.textContent = VIEWS[name].title || VIEWS[name].label;
  try {
    VIEWS[name].module.render(mainEl, params);
  } catch (err) {
    console.error(`View "${name}" failed to render`, err);
    renderCrash(name, err);
  }
  refreshBadges();
  mainEl.focus({ preventScroll: true });
}

/** A view that throws must not leave a blank screen with no way forward. */
function renderCrash(name, err) {
  mainEl.replaceChildren(
    el('div', { class: 'view-inner narrow' },
      el('div', { class: 'card start-card' },
        el('div', { class: 'start-icon' }, icon('x', 26)),
        el('h2', {}, 'This view hit an error'),
        el('p', { class: 'muted' }, `${name}: ${err?.message || err}`),
        el('p', { class: 'muted small' }, 'Your data is safe in this browser. Try another view, or reload the page.'),
        el('div', { class: 'row gap center' },
          el('button', { class: 'btn btn-primary', onclick: () => navigate('dashboard') }, 'Go to Home'),
          el('button', { class: 'btn btn-ghost', onclick: () => location.reload() }, 'Reload')))));
}

function refreshBadges() {
  const due = dueCards(currentLanguage()).length;
  const badge = document.querySelector('#reviewBadge');
  if (badge) {
    badge.textContent = due > 99 ? '99+' : String(due);
    badge.classList.toggle('hidden', due === 0);
  }
  const streakEl = document.querySelector('#streakCount');
  if (streakEl) streakEl.textContent = String(streak(currentLanguage()));
  const langBtn = document.querySelector('#langSwitcher');
  if (langBtn) {
    const profile = languageProfile(currentLanguage());
    langBtn.querySelector('.lang-flag').textContent = profile?.flag || '🌐';
    langBtn.querySelector('.lang-name').textContent = profile?.display || currentLanguage();
  }
}

// -- chrome ------------------------------------------------------------------
function buildChrome(root) {
  const navButtons = Object.entries(VIEWS).map(([name, def]) =>
    el('button', {
      class: 'nav-item', dataset: { view: name }, type: 'button',
      onclick: () => navigate(name),
    },
      icon(def.icon, 19),
      el('span', {}, def.label),
      name === 'review' ? el('span', { class: 'nav-badge hidden', id: 'reviewBadge' }, '0') : null));

  const sidebar = el('aside', { class: 'sidebar' },
    el('div', { class: 'brand' },
      el('div', { class: 'brand-mark' }, icon('globe', 20)),
      el('div', { class: 'brand-text' },
        el('strong', {}, 'AI-Studio for Learning Languages'),
        el('small', {}, 'science-based tutor'))),
    el('nav', { class: 'sidebar-nav', dataset: { tour: 'nav' } }, navButtons),
    el('div', { class: 'sidebar-foot' },
      el('button', {
        class: 'nav-item', type: 'button', dataset: { tour: 'settings' },
        onclick: () => openSettings({ onChange: () => { refreshBadges(); } }),
      }, icon('settings', 19), el('span', {}, 'Settings')),
      el('button', {
        class: 'nav-item', type: 'button',
        onclick: () => startTour(true),
      }, icon('help', 19), el('span', {}, 'Walkthrough'))));

  const langBtn = el('button', {
    class: 'lang-switcher', id: 'langSwitcher', type: 'button', dataset: { tour: 'lang' },
    'aria-haspopup': 'true',
    onclick: openLanguageMenu,
  },
    el('span', { class: 'lang-flag' }, '🌐'),
    el('span', { class: 'lang-name' }, ''),
    icon('arrowRight', 14));

  const topbar = el('header', { class: 'topbar' },
    el('div', { class: 'topbar-title', id: 'topbarTitle' }),
    el('div', { class: 'topbar-actions' },
      el('div', { class: 'streak-pill', dataset: { tour: 'streak' }, title: 'Current streak' },
        icon('flame', 16), el('strong', { id: 'streakCount' }, '0')),
      langBtn));

  mainEl = el('main', { class: 'main-view', id: 'mainView', tabindex: '-1' });
  root.append(sidebar, el('div', { class: 'content-column' }, topbar, mainEl));
}

/** Switch target language. Selecting the current one is a no-op, not a reset. */
function selectLanguage(code, menu) {
  menu?.remove();
  if (code === currentLanguage()) return; // already here: never wipe the open view
  updateSettings({ language: code });
  refreshBadges();
  // Drop any view parameters: unit ids and the like belong to the old language.
  navigate(currentView || 'dashboard');
  const profile = languageProfile(code);
  toast(`Switched to ${profile?.display || code}`, 'success');
}

function openLanguageMenu(event) {
  const existing = document.querySelector('.lang-menu');
  if (existing) { existing.remove(); return; }
  const trigger = event.currentTarget;
  const rect = trigger.getBoundingClientRect();
  const menu = el('div', { class: 'lang-menu', role: 'menu' },
    (ctx.config?.languages || []).map((l) => {
      const active = l.code === currentLanguage();
      return el('button', {
        class: `lang-option${active ? ' active' : ''}`, type: 'button', role: 'menuitem',
        'aria-current': active ? 'true' : null,
        onclick: () => selectLanguage(l.code, menu),
      },
        el('span', { class: 'lang-option-flag' }, l.flag),
        el('span', { class: 'lang-option-text' },
          el('strong', {}, l.display),
          el('small', {}, l.nativeName)),
        active ? icon('check', 15) : null);
    }));
  Object.assign(menu.style, { top: `${rect.bottom + 8}px`, right: `${window.innerWidth - rect.right}px` });
  document.body.append(menu);
  const dismiss = (e) => {
    if (!menu.contains(e.target) && e.target !== trigger) {
      menu.remove();
      document.removeEventListener('mousedown', dismiss);
    }
  };
  setTimeout(() => document.addEventListener('mousedown', dismiss), 0);
}

// -- global keyboard nav (g + key) ------------------------------------------
const GOTO_KEYS = {
  d: 'dashboard', t: 'topics', l: 'learn', r: 'review',
  i: 'listen', s: 'speak', c: 'compose', g: 'grammar', p: 'progress',
};

let pendingG = false;
let pendingTimer = null;
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (pendingG) {
    pendingG = false;
    clearTimeout(pendingTimer);
    const target = GOTO_KEYS[e.key.toLowerCase()];
    if (target) { e.preventDefault(); navigate(target); }
    return;
  }
  if (e.key.toLowerCase() === 'g') {
    pendingG = true;
    clearTimeout(pendingTimer);
    pendingTimer = setTimeout(() => { pendingG = false; }, 1200);
  }
});

// -- boot --------------------------------------------------------------------
async function boot() {
  applyTheme();
  const root = document.querySelector('#app');
  ctx.navigate = navigate;
  ctx.refreshChrome = () => { refreshBadges(); };
  // Route parameters belong to the old language, so a repaint drops them.
  ctx.rerenderView = () => { renderView(currentView || 'dashboard'); };

  try {
    ctx.config = await api.config();
  } catch {
    ctx.config = {
      languages: [], levels: ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
      provider: 'offline', curriculum: {},
    };
    toast('Could not load server config - is app.py running?', 'error', 8000);
  }

  buildChrome(root);
  document.querySelector('#bootSplash')?.remove();

  on('deck', refreshBadges);
  on('stats', refreshBadges);
  on('imported', () => { refreshBadges(); navigate('dashboard'); });

  window.addEventListener('hashchange', () => {
    const route = resolveRoute();
    renderView(route.name, route.params);
  });
  const initial = resolveRoute();
  renderView(initial.name, initial.params);

  // Re-check due counts periodically (learning steps come due within minutes)
  setInterval(refreshBadges, 30000);

  maybeStartTour();
}

boot();
