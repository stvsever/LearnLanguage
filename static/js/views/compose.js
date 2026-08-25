// Compose view: describe anything - a scene, a rant, a news piece, a debate -
// and one LLM call classifies the best format (dialogue / monologue / story /
// article) AND writes it at your level, grammar-aware.
//
// The result renders as comprehensible input: per-segment audio (dialogues get
// a distinct neural voice per speaker), tap-any-word glosses, aligned
// translations, grammar spotlights, and comprehension questions.

import { el, icon, toast, sample } from '../ui.js';
import {
  state, compositions, saveComposition, removeComposition, addCards,
  currentLanguage, recordTime,
} from '../store.js';
import { api, ApiError } from '../api.js';
import { speak, stopAudio, feedbackTone } from '../audio.js';
import { normalize } from '../grading.js';
import { choiceGrid } from '../exercises.js';
import { ctx, languageProfile } from '../context.js';
import { confirmDialog } from '../ui.js';

const FORMAT_META = {
  dialogue: { label: 'Dialogue', icon: 'mic' },
  monologue: { label: 'Monologue', icon: 'volume' },
  story: { label: 'Story', icon: 'book' },
  article: { label: 'Article', icon: 'penLine' },
};

const EXAMPLE_PROMPTS = [
  'A dialogue: haggling at a flea market over an old accordion',
  'A monologue: a taxi driver leaving a rambling voicemail for his brother',
  'A story about a lighthouse keeper who receives a mysterious package',
  'A short article on why this city banned cars from its center',
  'Two friends arguing about whether to adopt a second cat',
  'A chef explaining her signature dish to a nervous apprentice',
  'A weather forecast that slowly turns apocalyptic (but stays polite)',
  'A grandmother teaching her grandson to bargain at the market',
];

const grammarNameCache = new Map(); // lang -> {featureId: name}
let openPopover = null;
let readingStart = null;

export function render(container) {
  cleanup();
  const lang = currentLanguage();
  const saved = compositions(lang);
  const prefill = ctx.composePrefill || '';
  ctx.composePrefill = null;

  container.replaceChildren(
    el('div', { class: 'view-inner' },
      el('div', { class: 'compose-hero card', id: 'composePanel' },
        el('span', { class: 'phase-tag' }, 'Compose'),
        el('h1', { class: 'compose-title' }, 'What should we create?'),
        el('p', { class: 'muted' },
          'Describe any scene, speech, story, or subject. The model picks the best format - dialogue, monologue, story, or article - and writes it at your level with your target grammar woven in.'),
        composeForm(container, prefill)),
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
  return el('button', {
    class: 'library-item', type: 'button',
    onclick: () => renderComposition(container, pack),
  },
    el('span', { class: `format-badge f-${pack.format}` }, icon(meta.icon, 13), meta.label),
    el('strong', {}, pack.title),
    el('span', { class: 'muted small' }, pack.scene || `${pack.segments.length} segments · ${pack.level}`));
}

function composeForm(container, prefill) {
  const lang = currentLanguage();
  const offline = ctx.config?.provider === 'offline';
  const hasSeed = ctx.config?.seedLanguages?.includes(lang);

  const promptInput = el('textarea', {
    class: 'input compose-input', rows: 2, maxlength: 480, id: 'composePrompt',
    placeholder: 'e.g. two neighbours argue about a tree that drops leaves on the wrong side of the fence…',
  });
  if (prefill) promptInput.value = prefill;

  const exampleRow = el('div', { class: 'prompt-examples' },
    sample(EXAMPLE_PROMPTS, 3).map((example) => el('button', {
      class: 'prompt-chip', type: 'button', title: 'Use this idea',
      onclick: () => { promptInput.value = example; promptInput.focus(); },
    }, example)));

  const levelSelect = el('select', { class: 'input' },
    (ctx.config?.levels || []).map((lv) => el('option', { value: lv, selected: lv === state.settings.level || undefined }, lv)));
  const lengthSelect = el('select', { class: 'input' },
    [['short', 'Short · ~8 segments'], ['medium', 'Medium · ~13'], ['long', 'Long · ~20']]
      .map(([v, label]) => el('option', { value: v, selected: v === 'medium' || undefined }, label)));

  const composeBtn = el('button', {
    class: 'btn btn-primary btn-lg', id: 'composeBtn',
    onclick: async () => {
      composeBtn.disabled = true;
      composeBtn.replaceChildren(el('span', { class: 'spinner' }), 'Composing…');
      try {
        const pack = await api.compose({
          prompt: promptInput.value.trim(),
          language: lang,
          level: levelSelect.value,
          length: lengthSelect.value,
        });
        saveComposition(pack, lang);
        if (pack.notice) toast(pack.notice, 'info', 6000);
        renderComposition(container, pack);
      } catch (err) {
        toast(err instanceof ApiError && err.status === 503
          ? 'No LLM configured - add OPENROUTER_API_KEY to .env to compose.'
          : `Composition failed: ${err.message}`, 'error', 6000);
        composeBtn.disabled = false;
        composeBtn.replaceChildren(icon('sparkles', 18), 'Compose');
      }
    },
  }, icon('sparkles', 18), 'Compose');

  const seedBtn = hasSeed && offline ? el('button', {
    class: 'btn btn-soft',
    onclick: async () => {
      const pack = await api.compose({ language: lang, prompt: '' });
      saveComposition(pack, lang);
      renderComposition(container, pack);
    },
  }, icon('book', 16), 'Open starter piece') : null;

  return el('div', {},
    offline && !hasSeed ? el('div', { class: 'notice' }, icon('zap', 16),
      el('span', {}, 'Add OPENROUTER_API_KEY to .env to unlock composing.')) : null,
    promptInput,
    exampleRow,
    el('div', { class: 'row gap wrap compose-controls' },
      el('label', { class: 'field grow' }, el('span', {}, 'Level'), levelSelect),
      el('label', { class: 'field grow' }, el('span', {}, 'Length'), lengthSelect),
      el('div', { class: 'compose-actions' }, composeBtn, seedBtn)));
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
  (pack.participants || []).forEach((name, i) => {
    map.set(name, rotation[i % rotation.length]?.id || profile.defaultVoice);
  });
  return map;
}

// -- composition renderer -----------------------------------------------------
async function grammarNames(lang) {
  if (!grammarNameCache.has(lang)) {
    try {
      const profile = await api.grammar(lang);
      const names = {};
      for (const features of Object.values(profile.roadmap || {})) {
        for (const f of features) names[f.id] = { name: f.name, level: featureLevel(profile, f.id) };
      }
      grammarNameCache.set(lang, names);
    } catch {
      grammarNameCache.set(lang, {});
    }
  }
  return grammarNameCache.get(lang);
}

function featureLevel(profile, fid) {
  for (const [level, features] of Object.entries(profile.roadmap || {})) {
    if (features.some((f) => f.id === fid)) return level;
  }
  return '';
}

export async function renderComposition(container, pack) {
  cleanup();
  readingStart = Date.now();
  const lang = currentLanguage();
  const profile = languageProfile(lang);
  const meta = FORMAT_META[pack.format] || FORMAT_META.story;
  const glossaryMap = new Map((pack.glossary || []).map((g) => [normalize(g.word, { keepDiacritics: false }), g.gloss]));
  const voices = speakerVoices(pack, lang);

  const segmentNodes = pack.segments.map((segment) =>
    segmentBlock(segment, glossaryMap, lang, voices, pack.format));

  let playing = false;
  const playBtn = el('button', {
    class: 'btn btn-soft',
    onclick: async () => {
      if (playing) { playing = false; stopAudio(); playBtn.replaceChildren(icon('play', 16), 'Play'); return; }
      playing = true;
      playBtn.replaceChildren(icon('pause', 16), 'Stop');
      for (let i = 0; i < pack.segments.length && playing; i++) {
        segmentNodes.forEach((n) => n.classList.remove('reading'));
        segmentNodes[i].classList.add('reading');
        segmentNodes[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        const segment = pack.segments[i];
        await speak(segment.text, { lang, voice: voices.get(segment.speaker) || null });
      }
      segmentNodes.forEach((n) => n.classList.remove('reading'));
      playing = false;
      playBtn.replaceChildren(icon('play', 16), 'Play');
    },
  }, icon('play', 16), 'Play');

  const deleteBtn = el('button', {
    class: 'btn-icon', title: 'Delete from library',
    onclick: async () => {
      if (await confirmDialog(`Delete “${pack.title}” from your library?`, { danger: true, confirmLabel: 'Delete' })) {
        removeComposition(pack.title, lang);
        render(container);
      }
    },
  }, icon('trash', 16));

  container.replaceChildren(
    el('div', { class: 'view-inner reader-inner' },
      el('div', { class: 'reader-head' },
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => render(container) }, icon('arrowLeft', 16), 'Compose'),
        el('div', { class: 'row gap' }, playBtn, deleteBtn)),
      el('article', { class: 'card reader-card', style: { fontFamily: profile?.fontStack || 'inherit' } },
        el('div', { class: 'reader-meta' },
          el('span', { class: `format-badge f-${pack.format}` }, icon(meta.icon, 13), meta.label),
          el('span', { class: 'level-chip' }, pack.level || '')),
        el('h1', { class: 'reader-title' }, pack.title),
        pack.scene ? el('p', { class: 'reader-scene' }, pack.scene) : null,
        el('p', { class: 'muted small' }, 'Tap a word for its meaning · tap the margin marker for audio + translation'),
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
    class: 'r-play', type: 'button', title: 'Play + toggle translation',
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
      data = { gloss: 'No gloss available (configure an API key for tap-to-define)', note: '', lemma: '', pronunciation: '' };
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
        const added = addCards([{
          target: data.lemma || word, english: data.gloss,
          pronunciation: data.pronunciation || '', example: context, example_en: '', note: data.note || '',
        }], 'From composing', lang);
        toast(added ? `“${data.lemma || word}” added to your deck` : 'Already in your deck', added ? 'success' : 'info');
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
  const scoreLine = el('p', { class: 'muted small' }, 'Answer in the language you\'re learning - inference counts, rereading is allowed.');

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
