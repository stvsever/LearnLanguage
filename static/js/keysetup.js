// In-app OpenRouter key onboarding.
//
// Shown wherever generation is requested without a configured key, and in
// Settings. Two steps, zero friction: open the key page, paste the key, and it
// saves automatically to the local .env (no restart, no manual save button).

import { el, icon, toast } from './ui.js';
import { api } from './api.js';
import { ctx } from './context.js';

const KEY_PATTERN = /^sk-or-\S{16,}$/;

/**
 * A card that walks the user through connecting an OpenRouter key.
 * Calls onConnected() after the key is verified and saved.
 */
export function keySetupCard({ onConnected, compact = false } = {}) {
  const status = el('div', { class: 'key-status' });
  let saving = false;

  const input = el('input', {
    class: 'input key-input',
    type: 'text',
    autocomplete: 'off', spellcheck: 'false',
    placeholder: 'sk-or-...   (paste your key here, it saves automatically)',
    oninput: async () => {
      const value = input.value.trim();
      if (!value || saving) return;
      if (!KEY_PATTERN.test(value)) {
        status.replaceChildren(el('span', { class: 'key-hint' },
          'Keys start with sk-or-. Keep typing or paste the full key.'));
        return;
      }
      saving = true;
      input.disabled = true;
      status.replaceChildren(el('span', { class: 'key-saving' },
        el('span', { class: 'spinner dark' }), ' Saving and connecting...'));
      try {
        const result = await api.setupKey(value);
        delete result.saved;
        ctx.config = result;
        status.replaceChildren(el('span', { class: 'key-ok' },
          icon('check', 14), ` Connected as ${result.keyMasked}. Generation is unlocked.`));
        toast('OpenRouter key saved. You are connected.', 'success');
        ctx.refreshChrome?.();
        setTimeout(() => onConnected?.(), 700);
      } catch (err) {
        saving = false;
        input.disabled = false;
        input.classList.add('shake');
        setTimeout(() => input.classList.remove('shake'), 500);
        status.replaceChildren(el('span', { class: 'key-error' },
          icon('x', 14), ` ${err.message}`));
      }
    },
  });

  return el('div', { class: `key-setup${compact ? ' compact' : ''}` },
    el('div', { class: 'key-head' },
      icon('zap', 16),
      el('strong', {}, 'Connect a free AI key to unlock generation')),
    el('ol', { class: 'key-steps' },
      el('li', {},
        'Create a key at ',
        el('a', { href: 'https://openrouter.ai/keys', target: '_blank', rel: 'noopener noreferrer' },
          'openrouter.ai/keys'),
        ' (sign-up is free, the default model costs fractions of a cent).'),
      el('li', {}, 'Paste it below. It is stored only in the local .env file on your machine.')),
    input,
    status);
}

/** Compact "replace key" input for Settings when a key is already set. */
export function keyReplaceRow({ onConnected } = {}) {
  const holder = el('div', { class: 'key-replace' });
  const show = () => holder.replaceChildren(keySetupCard({ onConnected, compact: true }));
  holder.replaceChildren(
    el('div', { class: 'set-row', style: { padding: '6px 0' } },
      el('div', { class: 'set-text' },
        el('strong', {}, 'OpenRouter key'),
        el('span', {}, `Connected as ${ctx.config?.keyMasked || 'unknown'}`)),
      el('button', { class: 'btn btn-ghost btn-sm', onclick: show }, 'Replace key')));
  return holder;
}
