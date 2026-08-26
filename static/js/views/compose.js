// Compose view: build a graded text and then work through it.
//
// You describe a scene and choose exactly as much as you want to choose. Leave
// the format on Auto and one LLM call classifies the best shape (dialogue /
// monologue / story / article) and writes it. Pin the format, the register, the
// number of speakers, the grammar structures, or the vocabulary that has to
// appear, and the model is held to all of it.
//
// The result renders as comprehensible input: per-segment audio (dialogues get
// a distinct neural voice per speaker), tap-any-word glosses, aligned
// translations, grammar spotlights, and comprehension questions.

import { el, icon, toast, sample, progressSteps, confirmDialog } from '../ui.js';
import { keySetupCard } from '../keysetup.js';
import {
  state, compositions, saveComposition, removeComposition, addCards,
  currentLanguage, recordTime, dueCards, cards,
} from '../store.js';
import { api, ApiError } from '../api.js';
import { speak, stopAudio, feedbackTone } from '../audio.js';
import { normalize } from '../grading.js';
import { choiceGrid } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';
import { allLevels, levelBlurb, levelLabel } from '../levels.js';

const FORMAT_META = {
  dialogue: { label: 'Dialogue', icon: 'mic', hint: 'A spoken exchange between named speakers' },
  monologue: { label: 'Monologue', icon: 'volume', hint: 'One voice: a speech, voicemail, vlog, or inner thought' },
  story: { label: 'Story', icon: 'book', hint: 'Narrated fiction with a beginning and an end' },
  article: { label: 'Article', icon: 'penLine', hint: 'Expository prose: news, explainer, review' },
};

const REGISTERS = [
  ['casual', 'Casual', 'Informal address, contractions, everyday idiom'],
  ['neutral', 'Neutral', 'Standard everyday register'],
  ['formal', 'Formal', 'Polite address, full forms, no slang'],
];

const EXAMPLE_PROMPTS = [
  'Haggling at a flea market over an old accordion',
  'A taxi driver leaving a rambling voicemail for his brother',
  'A lighthouse keeper who receives a mysterious package',
  'Why this city banned cars from its centre',
  'Two friends arguing about whether to adopt a second cat',
  'A chef explaining her signature dish to a nervous apprentice',
  'A weather forecast that slowly turns apocalyptic but stays polite',
  'A grandmother teaching her grandson to bargain at the market',
  'A night receptionist dealing with a guest who lost their passport',
  'An overheard argument about who forgot to water the plants',
];

// Survives re-renders so a failed generation never loses the learner's setup.
const form = {
  prompt: '',
  format: 'auto',
  register: 'neutral',
  length: 'medium',
  level: null,
  speakers: 2,
  focus: new Set(),
  vocabSource: 'none',
  examples: sample(EXAMPLE_PROMPTS, 3),
};

const grammarNameCache = new Map(); // lang -> {featureId: {name, level}}
let openPopover = null;
let readingStart = null;

export function render(container) {
  cleanup();
  const lang = currentLanguage();
  const saved = compositions(lang);

  if (ctx.composePrefill) {
    form.prompt = ctx.composePrefill;
    ctx.composePrefill = null;
  }
  if (ctx.composeFocus) {
    form.focus = new Set([ctx.composeFocus]);
    ctx.composeFocus = null;
  }
  if (!form.level) form.level = state.settings.level;

  container.replaceChildren(
    el('div', { class: 'view-inner' },
      el('div', { class: 'compose-hero card', id: 'composePanel' },
        el('span', { class: 'phase-tag' }, 'Compose'),
        el('h1', { class: 'compose-title' }, 'What should we create?'),
        el('p', { class: 'muted' },
          'Describe any scene, speech, story, or subject. Leave everything on Auto and the model decides the best shape, or take control of the format, register, speakers, grammar, and vocabulary.'),
        composeForm(container)),
      saved.length ? el('div', { class: 'card library-card' },
        el('div', { class: 'library-head' },
          el('h3', {}, 'Library'),
          el('span', { class: 'muted small' }, `${saved.length} saved piece${saved.length === 1 ? '' : 's'}`)),
        el('div', { class: 'library-grid' },
          saved.map((pack) => libraryItem(container, pack, lang)))) : null));
}

export function cleanup() {
  stopAudio();
  closePopover();
  if (readingStart) {
    recordTime(Date.now() - readingStart);
    readingStart = null;
  }
}

function libraryItem(container, pack, lang) {
  const meta = FORMAT_META[pack.format] || FORMAT_META.story;
  const segments = pack.segments?.length || 0;
  return el('div', { class: 'library-item' },
    el('button', {
      class: 'library-item-open', type: 'button',
      onclick: () => renderComposition(container, pack),
    },
      el('span', { class: `format-badge f-${pack.format}` }, icon(meta.icon, 13), meta.label),
      el('strong', {}, pack.title),
      el('span', { class: 'muted small' }, pack.scene || `${segments} segments · ${pack.level || ''}`)),
    el('button', {
      class: 'btn-icon library-item-delete', title: 'Delete from library',
      onclick: async (e) => {
        e.stopPropagation();
        if (await confirmDialog(`Delete “${pack.title}” from your library?`, { danger: true, confirmLabel: 'Delete' })) {
          removeComposition(pack.id, lang);
          render(container);
        }
      },
    }, icon('trash', 15)));
}

// -- the form -----------------------------------------------------------------
function composeForm(container) {
  const lang = currentLanguage();
  const offline = (ctx.config?.provider || 'offline') === 'offline';

  const promptInput = el('textarea', {
    class: 'input compose-input', rows: 3, maxlength: 480, id: 'composePrompt',
    placeholder: 'e.g. two neighbours argue about a tree that drops leaves on the wrong side of the fence…',
    oninput: (e) => { form.prompt = e.target.value; },
  });
  promptInput.value = form.prompt;

  const exampleRow = el('div', { class: 'prompt-examples' });
  const paintExamples = () => exampleRow.replaceChildren(
    ...form.examples.map((example) => el('button', {
      class: 'prompt-chip', type: 'button', title: 'Use this idea',
      onclick: () => { form.prompt = example; promptInput.value = example; promptInput.focus(); },
    }, example)),
    el('button', {
      class: 'prompt-chip shuffle', type: 'button', title: 'Other ideas',
      onclick: () => { form.examples = sample(EXAMPLE_PROMPTS, 3); paintExamples(); },
    }, icon('shuffle', 13)));
  paintExamples();

  // -- format, the control that used to be missing entirely ------------------
  const speakerRow = el('label', { class: 'field grow speaker-field' },
    el('span', {}, 'Speakers'),
    el('select', {
      class: 'input',
      onchange: (e) => { form.speakers = Number(e.target.value); },
    }, [2, 3, 4].map((n) => el('option', { value: n, selected: n === form.speakers || undefined }, `${n} people`))));
  const syncSpeakerRow = () => speakerRow.classList.toggle('hidden', form.format !== 'dialogue');

  const formatRow = el('div', { class: 'format-picker', role: 'radiogroup', 'aria-label': 'Format' },
    [['auto', { label: 'Auto', icon: 'zap', hint: 'Let the model choose the best shape' }], ...Object.entries(FORMAT_META)]
      .map(([value, meta]) => el('button', {
        class: `format-option${form.format === value ? ' active' : ''}`, type: 'button',
        role: 'radio', 'aria-checked': String(form.format === value), title: meta.hint,
        onclick: (e) => {
          form.format = value;
          formatRow.querySelectorAll('.format-option').forEach((b) => {
            b.classList.toggle('active', b === e.currentTarget);
            b.setAttribute('aria-checked', String(b === e.currentTarget));
          });
          syncSpeakerRow();
        },
      }, icon(meta.icon, 16), el('span', {}, meta.label))));
  syncSpeakerRow();

  const levelHint = el('small', { class: 'muted' }, levelBlurb(form.level));
  const levelSelect = el('select', {
    class: 'input',
    onchange: (e) => {
      form.level = e.target.value;
      levelHint.textContent = levelBlurb(form.level);
      refreshFocusPicker();
    },
  }, allLevels().map((lv) => el('option', {
    value: lv.code, title: lv.blurb, selected: lv.code === form.level || undefined,
  }, `${lv.code} · ${lv.name}`)));

  const lengthSelect = el('select', {
    class: 'input',
    onchange: (e) => { form.length = e.target.value; },
  }, [['short', 'Short · ~8 segments'], ['medium', 'Medium · ~13'], ['long', 'Long · ~20']]
    .map(([v, label]) => el('option', { value: v, selected: v === form.length || undefined }, label)));

  const registerSelect = el('select', {
    class: 'input',
    onchange: (e) => { form.register = e.target.value; },
  }, REGISTERS.map(([v, label, hint]) => el('option', { value: v, title: hint, selected: v === form.register || undefined }, label)));

  // -- grammar focus ---------------------------------------------------------
  const focusHost = el('div', { class: 'focus-picker' });
  const refreshFocusPicker = () => {
    grammarNames(lang).then((names) => {
      const atLevel = Object.entries(names).filter(([, meta]) => meta.level === form.level);
      if (!atLevel.length) { focusHost.replaceChildren(el('span', { class: 'muted small' }, 'No structures listed for this level.')); return; }
      focusHost.replaceChildren(...atLevel.map(([fid, meta]) => el('button', {
        class: `focus-chip${form.focus.has(fid) ? ' active' : ''}`, type: 'button',
        title: meta.tip || '',
        onclick: (e) => {
          if (form.focus.has(fid)) form.focus.delete(fid);
          else if (form.focus.size < 4) form.focus.add(fid);
          else { toast('Four structures is the practical maximum for one text', 'info'); return; }
          e.currentTarget.classList.toggle('active', form.focus.has(fid));
        },
      }, meta.name)));
    });
  };
  refreshFocusPicker();

  // -- vocabulary seeding ----------------------------------------------------
  const vocabSelect = el('select', {
    class: 'input',
    onchange: (e) => { form.vocabSource = e.target.value; },
  },
    el('option', { value: 'none', selected: form.vocabSource === 'none' || undefined }, 'Whatever fits the scene'),
    el('option', { value: 'due', selected: form.vocabSource === 'due' || undefined }, 'Words due for review'),
    el('option', { value: 'recent', selected: form.vocabSource === 'recent' || undefined }, 'My most recent cards'));

  const advanced = el('details', { class: 'compose-advanced' },
    el('summary', {}, icon('settings', 15), 'Fine control'),
    el('div', { class: 'compose-advanced-body' },
      el('div', { class: 'row gap wrap' },
        el('label', { class: 'field grow' }, el('span', {}, 'Register'), registerSelect),
        speakerRow),
      el('div', { class: 'field' },
        el('span', {}, 'Grammar to exercise'),
        el('small', { class: 'muted' }, `Pick up to four structures from ${levelLabel(form.level)}. They must appear in the text and in the spotlights.`),
        focusHost),
      el('label', { class: 'field' },
        el('span', {}, 'Vocabulary to weave in'),
        el('small', { class: 'muted' }, 'Recycling your own cards inside a story is the strongest kind of review.'),
        vocabSelect)));

  const progressHost = el('div', { class: 'progress-host' });
  const resultHost = el('div', {});

  const composeBtn = el('button', {
    class: 'btn btn-primary btn-lg', id: 'composeBtn', disabled: offline || undefined,
    onclick: () => runCompose(container, { composeBtn, progressHost, resultHost, lang }),
  }, icon('sparkles', 18), 'Compose');

  return el('div', {},
    offline ? keySetupCard({ onConnected: () => render(container) }) : null,
    promptInput,
    exampleRow,
    el('div', { class: 'field' }, el('span', {}, 'Format'), formatRow),
    el('div', { class: 'row gap wrap compose-controls' },
      el('label', { class: 'field grow' }, el('span', {}, 'Level'), levelSelect, levelHint),
      el('label', { class: 'field grow' }, el('span', {}, 'Length'), lengthSelect)),
    advanced,
    el('div', { class: 'compose-actions row gap wrap' }, composeBtn),
    progressHost,
    resultHost);
}

function vocabularyList(lang) {
  if (form.vocabSource === 'due') return dueCards(lang).slice(0, 12).map((c) => c.target);
  if (form.vocabSource === 'recent') {
    return [...cards(lang)].sort((a, b) => b.addedAt - a.addedAt).slice(0, 12).map((c) => c.target);
  }
  return null;
}

async function runCompose(container, { composeBtn, progressHost, resultHost, lang }) {
  resultHost.replaceChildren();
  composeBtn.disabled = true;
  composeBtn.replaceChildren(el('span', { class: 'spinner' }), 'Composing');
  const restore = () => {
    composeBtn.disabled = false;
    composeBtn.replaceChildren(icon('sparkles', 18), 'Compose');
  };

  const model = (state.settings.model || ctx.config?.model || 'the model').split('/').pop();
  const vocabulary = vocabularyList(lang);
  const progress = progressSteps([
    `Sending your request to ${model}`,
    form.format === 'auto'
      ? 'Choosing the best format: dialogue, monologue, story, or article'
      : `Shaping it as a ${FORMAT_META[form.format].label.toLowerCase()}`,
    `Writing at ${levelLabel(form.level)} in a ${form.register} register`,
    'Validating structure, glossary, spotlights, and questions',
  ], [1500, 5000, 16000]);
  progressHost.replaceChildren(progress.root);

  let pack;
  try {
    pack = await api.compose({
      prompt: form.prompt.trim(),
      language: lang,
      level: form.level,
      length: form.length,
      format: form.format === 'auto' ? null : form.format,
      register: form.register,
      speakers: form.format === 'dialogue' ? form.speakers : null,
      focus: [...form.focus],
      vocabulary,
    });
  } catch (err) {
    progress.fail(err instanceof ApiError && err.status === 503
      ? 'No AI key configured yet. Connect one above.'
      : `Composition failed: ${err.message}`);
    resultHost.replaceChildren(
      el('div', { class: 'result-card error' },
        el('div', { class: 'result-head' }, icon('x', 18), el('strong', {}, 'That did not work')),
        el('p', { class: 'muted small' }, err?.message || String(err)),
        el('p', { class: 'muted small' }, 'Your prompt and settings are kept. Shorter pieces and simpler levels succeed more often.'),
        el('button', { class: 'btn btn-soft btn-sm', onclick: () => composeBtn.click() }, icon('refresh', 15), 'Try again')));
    restore();
    return;
  }

  try {
    const stored = saveComposition(pack, lang);
    progress.finish();
    if (pack.notice) toast(pack.notice, 'info', 6000);
    toast(`Composed a ${stored.format}: “${stored.title}”`, 'success');
    setTimeout(() => renderComposition(container, stored), 400);
  } catch (err) {
    console.error('Saving composition failed', err);
    progress.fail(`The text arrived but could not be saved: ${err.message}`);
    resultHost.replaceChildren(
      el('div', { class: 'result-card warn' },
        el('div', { class: 'result-head' }, icon('lightbulb', 18), el('strong', {}, 'Saved copy failed, showing it anyway')),
        el('button', { class: 'btn btn-primary btn-sm', onclick: () => renderComposition(container, pack) }, 'Open the text')));
    restore();
  }
}

// -- voices for dialogue speakers -------------------------------------------
function speakerVoices(pack, lang) {
  const profile = languageProfile(lang);
  const map = new Map();
  if (pack.format !== 'dialogue' || !profile) return map;
  const voices = profile.voices || [];
  const females = voices.filter((v) => v.gender === 'f');
  const males = voices.filter((v) => v.gender === 'm');
  // Alternate genders so adjacent speakers sound clearly distinct.
  const rotation = [];
  for (let i = 0; i < Math.max(females.length, males.length); i++) {
    if (females[i]) rotation.push(females[i]);
    if (males[i]) rotation.push(males[i]);
  }
  const names = pack.participants?.length
    ? pack.participants
    : [...new Set(pack.segments.map((s) => (s.speaker || '').trim()).filter(Boolean))];
  names.forEach((name, i) => {
    map.set(name, rotation[i % Math.max(rotation.length, 1)]?.id || profile.defaultVoice);
  });
  return map;
}

// -- composition renderer -----------------------------------------------------
async function grammarNames(lang) {
  if (!grammarNameCache.has(lang)) {
    try {
      const profile = await api.grammar(lang);
      const names = {};
      for (const [level, features] of Object.entries(profile.roadmap || {})) {
        for (const f of features) names[f.id] = { name: f.name, level, tip: f.tip };
      }
      grammarNameCache.set(lang, names);
    } catch {
      grammarNameCache.set(lang, {});
    }
  }
  return grammarNameCache.get(lang);
}

export async function renderComposition(container, pack) {
  cleanup();
  readingStart = Date.now();
  // Read the pack's own language: a piece opened from the library must sound
  // like what it is, even if the UI language moved on.
  const lang = pack.language || currentLanguage();
  const profile = languageProfile(lang);
  const meta = FORMAT_META[pack.format] || FORMAT_META.story;
  const glossaryMap = new Map((pack.glossary || []).map((g) => [normalize(g.word, { keepDiacritics: false }), g.gloss]));
  const voices = speakerVoices(pack, lang);
  const segments = pack.segments || [];

  const segmentNodes = segments.map((segment) =>
    segmentBlock(segment, glossaryMap, lang, voices, pack.format));

  let playing = false;
  const playBtn = el('button', {
    class: 'btn btn-soft',
    onclick: async () => {
      if (playing) { playing = false; stopAudio(); playBtn.replaceChildren(icon('play', 16), 'Play'); return; }
      playing = true;
      playBtn.replaceChildren(icon('pause', 16), 'Stop');
      for (let i = 0; i < segments.length && playing; i++) {
        segmentNodes.forEach((n) => n.classList.remove('reading'));
        segmentNodes[i].classList.add('reading');
        segmentNodes[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        await speak(segments[i].text, { lang, voice: voices.get(segments[i].speaker) || null });
      }
      segmentNodes.forEach((n) => n.classList.remove('reading'));
      playing = false;
      playBtn.replaceChildren(icon('play', 16), 'Play');
    },
  }, icon('play', 16), 'Play');

  const showAllBtn = el('button', {
    class: 'btn btn-ghost', title: 'Show or hide every translation',
    onclick: () => {
      const hidden = container.querySelectorAll('.r-translation.hidden').length;
      container.querySelectorAll('.r-translation').forEach((n) => n.classList.toggle('hidden', hidden === 0));
      showAllBtn.replaceChildren(icon('eye', 16), hidden ? 'Hide English' : 'Show English');
    },
  }, icon('eye', 16), 'Show English');

  const deleteBtn = el('button', {
    class: 'btn-icon', title: 'Delete from library',
    onclick: async () => {
      if (await confirmDialog(`Delete “${pack.title}” from your library?`, { danger: true, confirmLabel: 'Delete' })) {
        removeComposition(pack.id, lang);
        render(container);
      }
    },
  }, icon('trash', 16));

  container.replaceChildren(
    el('div', { class: 'view-inner reader-inner' },
      el('div', { class: 'reader-head' },
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => render(container) }, icon('arrowLeft', 16), 'Compose'),
        el('div', { class: 'row gap wrap' }, playBtn, showAllBtn, deleteBtn)),
      el('article', { class: 'card reader-card', style: { fontFamily: profile?.fontStack || 'inherit' } },
        el('div', { class: 'reader-meta' },
          el('span', { class: `format-badge f-${pack.format}` }, icon(meta.icon, 13), meta.label),
          pack.level ? el('span', { class: 'level-chip' }, pack.level) : null,
          pack.participants?.length ? el('span', { class: 'muted small' }, pack.participants.join(', ')) : null),
        el('h1', { class: 'reader-title' }, pack.title),
        pack.scene ? el('p', { class: 'reader-scene' }, pack.scene) : null,
        el('p', { class: 'muted small' }, 'Tap a word for its meaning · tap the margin marker for audio and translation'),
        el('div', { class: `reader-body ${pack.format === 'dialogue' ? 'is-dialogue' : ''}` }, segmentNodes)),
      pack.grammar_spotlights?.length ? spotlightsSection(pack, lang) : null,
      pack.questions?.length ? questionsSection(pack) : null));
}

function segmentBlock(segment, glossaryMap, lang, voices, format) {
  const words = segment.text.split(/(\s+)/).map((chunk) => {
    if (/^\s+$/.test(chunk)) return chunk;
    return el('span', {
      class: 'r-word',
      onclick: (e) => { e.stopPropagation(); showGloss(e.currentTarget, chunk, segment.text, glossaryMap, lang); },
    }, chunk);
  });

  const translationEl = el('div', { class: 'r-translation hidden' }, segment.text_en);
  const voice = voices.get(segment.speaker) || null;
  const playControl = el('button', {
    class: 'r-play', type: 'button', title: 'Play and toggle translation',
    onclick: () => {
      speak(segment.text, { lang, voice });
      translationEl.classList.toggle('hidden');
    },
  }, format === 'dialogue' ? icon('volume', 13) : '¶');

  if (format === 'dialogue' && segment.speaker) {
    return el('div', { class: 'r-turn' },
      el('div', { class: 'turn-head' },
        el('span', { class: `speaker-chip s-${hashHue(segment.speaker)}` }, segment.speaker),
        playControl),
      el('div', { class: 'turn-body' },
        el('span', { class: 'r-text' }, words),
        translationEl));
  }
  return el('div', { class: 'r-sentence' },
    playControl,
    el('span', { class: 'r-text' }, words),
    translationEl);
}

function hashHue(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) % 6;
  return h;
}

// -- grammar spotlights -------------------------------------------------------
function spotlightsSection(pack, lang) {
  const section = el('section', { class: 'card spotlights-card' },
    el('h3', {}, icon('zap', 16), ' Grammar inside'),
    el('p', { class: 'muted small' }, 'Structures from your level, used live in this text. Tap one to see it in the Grammar map.'));
  const list = el('div', { class: 'spotlight-list' });
  section.append(list);

  grammarNames(lang).then((names) => {
    list.replaceChildren(...pack.grammar_spotlights.map((spot) => {
      const meta = names[spot.feature];
      return el('button', {
        class: 'spotlight', type: 'button',
        onclick: () => { ctx.grammarFocus = spot.feature; ctx.navigate('grammar'); },
      },
        el('div', { class: 'spotlight-head' },
          el('strong', {}, meta?.name || spot.feature),
          meta?.level ? el('span', { class: 'level-chip' }, meta.level) : null),
        el('span', { class: 'spotlight-excerpt' }, `“${spot.excerpt}”`),
        el('span', { class: 'muted small' }, spot.explanation));
    }));
  });
  return section;
}

// -- gloss popover ------------------------------------------------------------
function closePopover() {
  openPopover?.remove();
  openPopover = null;
}

async function showGloss(anchor, rawWord, context, glossaryMap, lang) {
  closePopover();
  const word = rawWord.replace(/^[«"'(\[]+|[»"'.,;:!?)\]]+$/g, '');
  if (!word) return;

  const pop = el('div', { class: 'gloss-pop' },
    el('div', { class: 'gloss-word' }, word),
    el('div', { class: 'gloss-body' }, el('span', { class: 'spinner dark' })));
  document.body.append(pop);
  positionPopover(pop, anchor);
  openPopover = pop;
  const dismiss = (e) => {
    if (!pop.contains(e.target)) { closePopover(); document.removeEventListener('mousedown', dismiss); }
  };
  document.addEventListener('mousedown', dismiss);

  speak(word, { lang });

  const local = glossaryMap.get(normalize(word, { keepDiacritics: false }));
  let data = local ? { gloss: local, note: '', lemma: '', pronunciation: '' } : null;
  if (!data) {
    try {
      data = await api.gloss({ text: word, context, language: lang });
    } catch {
      data = { gloss: 'No gloss available (connect an API key for tap-to-define)', note: '', lemma: '', pronunciation: '' };
    }
  }
  if (openPopover !== pop) return;

  pop.querySelector('.gloss-body').replaceChildren(el('div', {},
    el('div', { class: 'gloss-meaning' }, data.gloss),
    data.lemma && data.lemma !== word ? el('div', { class: 'gloss-meta' }, `→ ${data.lemma}`) : null,
    data.pronunciation ? el('div', { class: 'gloss-meta ipa' }, data.pronunciation) : null,
    data.note ? el('div', { class: 'gloss-note' }, data.note) : null,
    el('button', {
      class: 'btn btn-soft btn-sm', style: { marginTop: '8px' },
      onclick: () => {
        const report = addCards([{
          target: data.lemma || word, english: data.gloss,
          pronunciation: data.pronunciation || '', example: context, example_en: '', note: data.note || '',
        }], 'From composing', lang);
        toast(report.added ? `“${data.lemma || word}” added to your deck` : 'Already in your deck',
          report.added ? 'success' : 'info');
        closePopover();
      },
    }, icon('plus', 14), 'Add to deck')));
  positionPopover(pop, anchor);
}

function positionPopover(pop, anchor) {
  const rect = anchor.getBoundingClientRect();
  const popRect = pop.getBoundingClientRect();
  let left = rect.left + rect.width / 2 - popRect.width / 2;
  left = Math.max(12, Math.min(left, window.innerWidth - popRect.width - 12));
  let top = rect.bottom + 8;
  if (top + popRect.height > window.innerHeight - 12) top = rect.top - popRect.height - 8;
  pop.style.left = `${left}px`;
  pop.style.top = `${top + window.scrollY}px`;
}

// -- comprehension questions --------------------------------------------------
function questionsSection(pack) {
  let score = 0, answeredCount = 0;
  const scoreLine = el('p', { class: 'muted small' }, 'Answer in the language you are learning. Inference counts, rereading is allowed.');

  const blocks = pack.questions.map((q, qi) => {
    const explanation = el('div', { class: 'q-explanation hidden' }, q.explanation || '');
    const grid = choiceGrid({
      choices: q.choices,
      correctIndex: q.correct_choice,
      onPick: (correct) => {
        feedbackTone(correct ? 'correct' : 'wrong');
        answeredCount += 1;
        if (correct) score += 1;
        explanation.classList.remove('hidden');
        if (answeredCount === pack.questions.length) {
          scoreLine.textContent = `Comprehension: ${score} / ${pack.questions.length}${score === pack.questions.length ? ' - flawless.' : ''}`;
          scoreLine.classList.add('q-score');
        }
      },
    });
    return el('div', { class: 'q-block' },
      el('h4', {}, `${qi + 1}. ${q.question}`),
      grid,
      explanation);
  });

  return el('section', { class: 'card questions-card' },
    el('h3', {}, 'Did you understand?'),
    scoreLine,
    blocks);
}
