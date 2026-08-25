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
import * as learn from './views/learn.js';
import * as review from './views/review.js';
import * as listen from './views/listen.js';
import * as speak from './views/speak.js';
import * as compose from './views/compose.js';
import * as grammar from './views/grammar.js';
import * as progress from './views/progress.js';

const VIEWS = {
  dashboard: { module: dashboard, label: 'Home', icon: 'home' },
  learn: { module: learn, label: 'Learn', icon: 'sparkles' },
  review: { module: review, label: 'Review', icon: 'refresh' },
  listen: { module: listen, label: 'Listen', icon: 'headphones' },
  speak: { module: speak, label: 'Speak', icon: 'mic' },
  compose: { module: compose, label: 'Compose', icon: 'penLine' },
  grammar: { module: grammar, label: 'Grammar', icon: 'book' },
  progress: { module: progress, label: 'Progress', icon: 'chart' },
};

// Old bookmarks/hashes from v2.0
const ROUTE_ALIASES = { read: 'compose', stats: 'progress' };

let currentView = null;
let mainEl = null;

function navigate(name) {
  name = ROUTE_ALIASES[name] || name;
  const target = VIEWS[name] ? name : 'dashboard';
  if (location.hash !== `#/${target}`) {
    location.hash = `#/${target}`;
    return; // hashchange handler re-enters
  }
  renderView(target);
}

function renderView(name) {
  VIEWS[currentView]?.module.cleanup?.();
  stopAudio();
  currentView = name;
  document.querySelectorAll('.nav-item').forEach((b) => {
    b.classList.toggle('active', b.dataset.view === name);
  });
  mainEl.dataset.view = name;
  VIEWS[name].module.render(mainEl);
  refreshBadges();
  mainEl.focus({ preventScroll: true });
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
        el('strong', {}, 'Glotta'),
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

function openLanguageMenu(event) {
  const existing = document.querySelector('.lang-menu');
  if (existing) { existing.remove(); return; }
  const rect = event.currentTarget.getBoundingClientRect();
  const menu = el('div', { class: 'lang-menu' },
    (ctx.config?.languages || []).map((l) => el('button', {
      class: `lang-option${l.code === currentLanguage() ? ' active' : ''}`, type: 'button',
      onclick: () => {
        updateSettings({ language: l.code });
        menu.remove();
        refreshBadges();
        renderView(currentView || 'dashboard');
        toast(`Switched to ${l.display}`, 'success');
      },
    }, el('span', {}, l.flag), el('span', {}, l.display), el('small', {}, l.nativeName))));
  Object.assign(menu.style, { top: `${rect.bottom + 8}px`, right: `${window.innerWidth - rect.right}px` });
  document.body.append(menu);
  const dismiss = (e) => {
    if (!menu.contains(e.target) && e.target !== event.currentTarget) {
      menu.remove();
      document.removeEventListener('mousedown', dismiss);
    }
  };
  setTimeout(() => document.addEventListener('mousedown', dismiss), 0);
}

// -- global keyboard nav (g + key) ------------------------------------------
let pendingG = false;
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.metaKey || e.ctrlKey) return;
  if (pendingG) {
    pendingG = false;
    const map = { d: 'dashboard', l: 'learn', r: 'review', c: 'compose', g: 'grammar', p: 'progress' };
    if (map[e.key.toLowerCase()]) { e.preventDefault(); navigate(map[e.key.toLowerCase()]); }
    return;
  }
  if (e.key.toLowerCase() === 'g') pendingG = true;
  setTimeout(() => { pendingG = false; }, 900);
});

// -- boot --------------------------------------------------------------------
async function boot() {
  applyTheme();
  const root = document.querySelector('#app');
  ctx.navigate = navigate;
  ctx.refreshChrome = () => { refreshBadges(); };

  try {
    ctx.config = await api.config();
  } catch {
    ctx.config = { languages: [], levels: ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], provider: 'offline', seedLanguages: [] };
    toast('Could not load server config - is app.py running?', 'error', 8000);
  }

  buildChrome(root);
  document.querySelector('#bootSplash')?.remove();

  on('deck', refreshBadges);
  on('stats', refreshBadges);
  on('imported', () => { refreshBadges(); renderView(currentView || 'dashboard'); });

  const resolveHash = () => {
    const raw = location.hash.replace('#/', '');
    const name = ROUTE_ALIASES[raw] || raw;
    return name in VIEWS ? name : 'dashboard';
  };
  window.addEventListener('hashchange', () => renderView(resolveHash()));
  renderView(resolveHash());

  // Re-check due counts periodically (learning steps come due within minutes)
  setInterval(refreshBadges, 30000);

  maybeStartTour();
}

boot();
