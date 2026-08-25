// Theme handling: light / dark / system, applied via data-theme on <html>.

import { state } from './store.js';

const media = window.matchMedia('(prefers-color-scheme: dark)');

export function applyTheme() {
  const setting = state.settings.theme || 'system';
  const dark = setting === 'dark' || (setting === 'system' && media.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
}

media.addEventListener('change', () => {
  if ((state.settings.theme || 'system') === 'system') applyTheme();
});
