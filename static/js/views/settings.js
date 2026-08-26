// Settings panel: every dial of the app, grouped and explained.

import { el, icon, toast, openModal, confirmDialog } from '../ui.js';
import {
  state, updateSettings, exportData, importData, resetProgress, resetAll,
  currentLanguage, deckCounts,
} from '../store.js';
import { ctx, languageProfile } from '../context.js';
import { startTour } from '../tour.js';
import { applyTheme } from '../theme.js';
import { keySetupCard, keyReplaceRow } from '../keysetup.js';
import { allLevels, levelBlurb, levelLabel } from '../levels.js';
import {
  ability, abilityBand, abilityRecord, isEnabled as adaptiveOn, modeForecast,
  recentAccuracy, levelRecommendation, resetAbility, targetSuccess,
} from '../adaptive.js';

export function openSettings({ onChange: onChangeExternal } = {}) {
  // Every control change flashes a "Saved" confirmation in the header:
  // settings persist automatically, and the UI should say so.
  let flashEl = null;
  let flashTimer = null;
  const flashSaved = () => {
    if (!flashEl) return;
    flashEl.classList.add('show');
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => flashEl.classList.remove('show'), 1400);
  };
  const onChange = () => { flashSaved(); onChangeExternal?.(); };

  const rerender = () => {
    body.replaceChildren(...sections());
    onChange?.();
  };

  function toggleRow(label, hint, key, { onToggle } = {}) {
    const isOn = Boolean(state.settings[key]);
    return el('div', { class: 'set-row' },
      el('div', { class: 'set-text' }, el('strong', {}, label), hint ? el('span', {}, hint) : null),
      el('button', {
        class: `switch${isOn ? ' on' : ''}`, role: 'switch', 'aria-checked': String(isOn), type: 'button',
        onclick: (e) => {
          updateSettings({ [key]: !state.settings[key] });
          e.currentTarget.classList.toggle('on');
          e.currentTarget.setAttribute('aria-checked', String(state.settings[key]));
          onChange?.();
          // Some switches change what the rest of the section should say.
          if (onToggle) setTimeout(onToggle, 0);
        },
      }, el('span', { class: 'knob' })));
  }

  /** Level, with the adjective that makes the code mean something. */
  function levelRow(repaint) {
    const current = state.settings.level;
    const hint = el('span', {}, levelBlurb(current));
    return el('div', { class: 'set-row' },
      el('div', { class: 'set-text' }, el('strong', {}, 'Your level'), hint),
      el('select', {
        class: 'input compact level-select',
        onchange: (e) => {
          updateSettings({ level: e.target.value });
          hint.textContent = levelBlurb(e.target.value);
          onChange?.();
          ctx.rerenderView();
          setTimeout(repaint, 0);
        },
      }, allLevels().map((lv) => el('option', {
        value: lv.code, title: lv.blurb, selected: lv.code === current || undefined,
      }, `${lv.code} · ${lv.name}`))));
  }

  /**
   * What adaptive testing is currently doing, in plain words: the band it reads
   * you at, how it got there, which retrievals that selects, and whether your
   * CEFR level still matches the evidence.
   */
  function adaptiveReadout(repaint) {
    const lang = currentLanguage();
    const record = abilityRecord(lang);
    const score = ability(lang);
    const band = abilityBand(score);
    const accuracy = recentAccuracy(lang);
    const recommendation = levelRecommendation(lang, state.settings.level);

    if (!adaptiveOn()) {
      return el('div', { class: 'adaptive-panel off' },
        el('p', { class: 'muted small' },
          'Exercises follow a fixed rotation. Your ability estimate keeps updating in the background, so switching this on later starts from real evidence.'),
        el('p', { class: 'muted small' },
          `Current reading: ${band.label.toLowerCase()}, from ${record.samples} answer${record.samples === 1 ? '' : 's'}.`));
    }

    const forecast = modeForecast(lang);
    return el('div', { class: 'adaptive-panel' },
      el('div', { class: 'adaptive-head' },
        el('div', {},
          el('strong', {}, band.label),
          el('span', { class: 'muted small' }, band.hint)),
        el('div', { class: 'adaptive-score' },
          el('strong', {}, `${Math.round(score * 100)}`),
          el('span', {}, 'ability'))),
      el('div', { class: 'adaptive-meter' },
        el('span', { style: { width: `${Math.round(score * 100)}%` } })),
      el('p', { class: 'muted small' },
        `${record.samples} answer${record.samples === 1 ? '' : 's'} measured`
        + (accuracy === null ? '' : ` · ${Math.round(accuracy * 100)}% recent accuracy`)
        + ` · aiming for ${Math.round(targetSuccess() * 100)}% success`),
      el('div', { class: 'forecast' },
        forecast.map((row) => el('div', { class: `forecast-row${row.selected ? ' selected' : ''}` },
          el('span', { class: 'forecast-label' }, row.label),
          el('div', { class: 'forecast-track' },
            el('span', { style: { width: `${Math.round(row.predicted * 100)}%` } })),
          el('span', { class: 'forecast-value' }, `${Math.round(row.predicted * 100)}%`)))),
      el('p', { class: 'muted small' },
        'Bars are the predicted success rate for each retrieval. The hardest one still above your target is the one you get.'),
      recommendationRow(recommendation, repaint),
      el('div', { class: 'row gap wrap' },
        el('button', {
          class: 'btn btn-ghost btn-sm',
          onclick: async () => {
            if (await confirmDialog('Reset the adaptive estimate for this language? Exercises start from the easy end again.', { confirmLabel: 'Reset estimate' })) {
              resetAbility(lang);
              toast('Adaptive estimate reset', 'info');
              repaint();
            }
          },
        }, icon('refresh', 15), 'Reset estimate')));
  }

  function recommendationRow(recommendation, repaint) {
    if (recommendation.action === 'hold') {
      return el('p', { class: 'muted small' }, icon('check', 13), ' ', recommendation.reason);
    }
    return el('div', { class: `level-suggestion ${recommendation.action}` },
      el('div', {},
        el('strong', {}, recommendation.action === 'up'
          ? `Ready for ${levelLabel(recommendation.suggested)}`
          : `Consider dropping to ${levelLabel(recommendation.suggested)}`),
        el('span', { class: 'muted small' }, recommendation.reason)),
      el('button', {
        class: 'btn btn-soft btn-sm',
        onclick: () => {
          updateSettings({ level: recommendation.suggested });
          toast(`Level set to ${levelLabel(recommendation.suggested)}`, 'success');
          onChange?.();
          ctx.rerenderView();
          repaint();
        },
      }, 'Apply'));
  }

  function selectRow(label, hint, key, options, { onSet } = {}) {
    return el('div', { class: 'set-row' },
      el('div', { class: 'set-text' }, el('strong', {}, label), hint ? el('span', {}, hint) : null),
      el('select', {
        class: 'input compact',
        onchange: (e) => {
          const value = e.target.value;
          updateSettings({ [key]: value });
          onSet?.(value);
          onChange?.();
        },
      }, options.map(([value, text]) => el('option', { value, selected: String(state.settings[key]) === String(value) || undefined }, text))));
  }

  function sliderRow(label, hint, key, { min, max, step = 1, format = (v) => String(v) }) {
    const valueEl = el('strong', { class: 'slider-value' }, format(state.settings[key]));
    return el('div', { class: 'set-row column' },
      el('div', { class: 'set-text' }, el('strong', {}, label), hint ? el('span', {}, hint) : null),
      el('div', { class: 'slider-line' },
        el('input', {
          type: 'range', min, max, step, value: state.settings[key], class: 'slider',
          oninput: (e) => { valueEl.textContent = format(Number(e.target.value)); },
          onchange: (e) => { updateSettings({ [key]: Number(e.target.value) }); onChange?.(); },
        }),
        valueEl));
  }

  function sections() {
    const lang = currentLanguage();
    const profile = languageProfile(lang);
    const provider = ctx.config?.provider || 'offline';

    return [
      section('globe', 'Language & voice', [
        selectRow('Target language', 'Each language keeps its own deck, stats, and topic library', 'language',
          (ctx.config?.languages || []).map((l) => [l.code, `${l.flag} ${l.display}`]),
          { onSet: () => { ctx.refreshChrome(); ctx.rerenderView(); rerender(); } }),
        levelRow(rerender),
        el('div', { class: 'set-row' },
          el('div', { class: 'set-text' }, el('strong', {}, 'Voice'), el('span', {}, `Neural voice used for ${profile?.display || ''} audio`)),
          el('select', {
            class: 'input compact',
            onchange: (e) => {
              const voices = { ...state.settings.voices, [lang]: e.target.value };
              updateSettings({ voices });
            },
          }, (profile?.voices || []).map((v) => el('option', {
            value: v.id,
            selected: (state.settings.voices[lang] || profile.defaultVoice) === v.id || undefined,
          }, v.label)))),
        selectRow('Speech speed', 'Slow it down while training your ear', 'ttsRate',
          [['slow', 'Slow'], ['study', 'Study'], ['natural', 'Natural'], ['fast', 'Fast']]),
        toggleRow('Autoplay audio', 'Hear each item as soon as it appears', 'autoplayAudio'),
        toggleRow('Feedback sounds', 'Short tones on correct / incorrect answers', 'soundEffects'),
      ]),

      section('zap', 'Testing & difficulty', [
        toggleRow('Adaptive testing',
          'Pick each exercise from your measured ability instead of a fixed rotation',
          'adaptiveTesting', { onToggle: rerender }),
        adaptiveReadout(rerender),
      ]),

      section('target', 'Daily plan & scheduler', [
        sliderRow('New cards per day', 'Sustainable beats heroic - each new card generates future reviews', 'newPerDay', { min: 2, max: 40 }),
        sliderRow('Max reviews per day', 'Safety cap for busy days', 'maxReviewsPerDay', { min: 20, max: 400, step: 10 }),
        sliderRow('Target retention', 'Higher = more frequent reviews. 90% is the sweet spot for effort vs. recall', 'targetRetention',
          { min: 0.8, max: 0.97, step: 0.01, format: (v) => `${Math.round(v * 100)}%` }),
      ]),

      section('sparkles', 'AI model', [
        provider === 'openrouter'
          ? keyReplaceRow({ onConnected: rerender })
          : keySetupCard({ onConnected: rerender }),
        el('p', { class: 'muted small' },
          provider === 'openrouter'
            ? 'Content generation runs through OpenRouter. Pick any model - DeepSeek V4 Flash is the fast, inexpensive default.'
            : provider === 'openai'
              ? 'Currently running on the OpenAI fallback. Connect an OpenRouter key above to unlock model selection.'
              : 'Generation is off until a key is connected. Everything else works offline.'),
        modelRow(),
      ]),

      section('keyboard', 'Typing & grading', [
        toggleRow('Strict accents', 'When off, é/e slips count as correct with a reminder', 'strictAccents'),
        toggleRow('Typo tolerance', 'One or two-letter slips grade as “almost” instead of wrong', 'typoTolerance'),
        toggleRow('Accent toolbar', 'On-screen keys for é è ç … under answer fields', 'accentToolbar'),
      ]),

      section('eye', 'Appearance', [
        selectRow('Theme', '', 'theme', [['system', 'System'], ['light', 'Light'], ['dark', 'Dark']],
          { onSet: () => applyTheme() }),
        toggleRow('Show pronunciation', 'IPA / pinyin hints on cards', 'showPronunciation'),
      ]),

      section('download', 'Your data', [
        el('p', { class: 'muted small' }, 'Everything lives in this browser (localStorage) - nothing is uploaded. Export regularly if you care about the progress.'),
        el('div', { class: 'row gap wrap' },
          el('button', { class: 'btn btn-soft btn-sm', onclick: downloadExport }, icon('download', 15), 'Export JSON'),
          importButton(rerender),
          el('button', {
            class: 'btn btn-ghost btn-sm', onclick: async () => {
              const counts = deckCounts(lang);
              if (await confirmDialog(`Reset ${profile?.display || lang} progress? This deletes ${counts.total} cards and all ${profile?.display || ''} stats.`, { danger: true, confirmLabel: 'Reset language' })) {
                resetProgress(lang); toast('Language progress reset', 'info'); rerender(); ctx.refreshChrome(); ctx.navigate('dashboard');
              }
            },
          }, icon('trash', 15), `Reset ${profile?.display || ''}`),
          el('button', {
            class: 'btn btn-ghost btn-sm danger', onclick: async () => {
              if (await confirmDialog('Delete ALL data for every language - decks, stats, stories, settings?', { danger: true, confirmLabel: 'Delete everything' })) {
                resetAll(); toast('All data deleted', 'info'); modal.close(); ctx.refreshChrome(); ctx.navigate('dashboard');
              }
            },
          }, icon('trash', 15), 'Reset all')),
      ]),

      section('help', 'About', [
        el('div', { class: 'about-grid' },
          aboutRow('Version', ctx.config?.version || '-'),
          aboutRow('LLM provider', provider === 'offline' ? 'Offline (seed content only)' : provider),
          aboutRow('Active model', state.settings.model || ctx.config?.model || '-'),
          aboutRow('TTS', 'Edge neural voices (free) + browser fallback')),
        provider === 'offline' ? el('div', { class: 'notice' }, icon('zap', 16),
          el('span', {}, 'Generation is currently off. Paste an OpenRouter key in the AI model section above; it saves and connects instantly, no restart needed.')) : null,
        el('div', { class: 'row gap wrap', style: { marginTop: '10px' } },
          el('button', { class: 'btn btn-soft btn-sm', onclick: () => { modal.close(); startTour(true); } }, icon('sparkles', 15), 'Replay walkthrough'),
          el('button', { class: 'btn btn-ghost btn-sm', onclick: showShortcuts }, icon('keyboard', 15), 'Keyboard shortcuts')),
      ]),
    ];
  }

  function modelRow() {
    const choices = ctx.config?.modelChoices || [];
    const current = state.settings.model;
    const isCustom = Boolean(current) && !choices.some((c) => c.id === current);
    const disabled = (ctx.config?.provider || 'offline') !== 'openrouter';

    const customInput = el('input', {
      class: 'input model-custom', type: 'text',
      placeholder: 'vendor/model-slug (from openrouter.ai/models)',
      value: isCustom ? current : '',
      onchange: (e) => {
        const slug = e.target.value.trim();
        updateSettings({ model: slug || null });
        onChange?.();
      },
    });
    if (!isCustom) customInput.classList.add('hidden');

    const select = el('select', {
      class: 'input compact',
      disabled: disabled || undefined,
      onchange: (e) => {
        const value = e.target.value;
        if (value === '__custom') {
          customInput.classList.remove('hidden');
          customInput.focus();
          return;
        }
        customInput.classList.add('hidden');
        updateSettings({ model: value || null });
        onChange?.();
      },
    },
      el('option', { value: '', selected: !current || undefined }, 'Server default'),
      choices.map((c) => el('option', { value: c.id, selected: current === c.id || undefined }, c.label)),
      el('option', { value: '__custom', selected: isCustom || undefined }, 'Custom slug…'));

    return el('div', { class: 'set-row column' },
      el('div', { class: 'set-row', style: { padding: '0' } },
        el('div', { class: 'set-text' },
          el('strong', {}, 'Model'),
          el('span', {}, disabled ? 'Requires an OpenRouter key' : 'Applies to lessons, compositions, and glosses')),
        select),
      customInput);
  }

  const body = el('div', { class: 'settings-body' }, sections());
  const modal = openModal({ title: 'Settings', content: body, wide: true });
  flashEl = el('span', { class: 'saved-flash' }, icon('check', 13), 'Saved');
  modal.panel.querySelector('.modal-head h2')?.after(flashEl);
  return modal;
}

function section(iconName, title, children) {
  return el('section', { class: 'set-section' },
    el('h3', {}, icon(iconName, 16), title),
    ...children);
}

function aboutRow(label, value) {
  return el('div', { class: 'about-row' }, el('span', {}, label), el('strong', {}, value));
}

function downloadExport() {
  const blob = new Blob([exportData()], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: `learnlanguage-export-${new Date().toISOString().slice(0, 10)}.json` });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  toast('Export downloaded', 'success');
}

function importButton(rerender) {
  const input = el('input', {
    type: 'file', accept: 'application/json', style: { display: 'none' },
    onchange: async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        importData(await file.text());
        toast('Data imported', 'success');
        rerender();
        ctx.refreshChrome();
        ctx.navigate('dashboard');
      } catch (err) {
        toast(`Import failed: ${err.message}`, 'error');
      }
    },
  });
  return el('span', {},
    input,
    el('button', { class: 'btn btn-soft btn-sm', onclick: () => input.click() }, icon('upload', 15), 'Import JSON'));
}

function showShortcuts() {
  openModal({
    title: 'Keyboard shortcuts',
    content: el('div', { class: 'shortcut-grid' },
      shortcut('1 - 4', 'Answer MCQs / grade a review (Again · Hard · Good · Easy)'),
      shortcut('Enter', 'Submit answer / continue with the suggested grade'),
      shortcut('Space', 'Continue (presentation cards)'),
      shortcut('R', 'Replay audio'),
      shortcut('G then D/L/R/S', 'Navigate: Dashboard · Learn · Review · Stats'),
      shortcut('Esc', 'Close dialogs')),
  });
}

function shortcut(keys, text) {
  return el('div', { class: 'shortcut-row' }, el('kbd', {}, keys), el('span', {}, text));
}
