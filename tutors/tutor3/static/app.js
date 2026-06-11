const state = {
  config: null,
  topics: [],
  section: "study",
  generateMode: "vocabulary",
  source: "custom",
  language: "es",
  voice: "",
  difficulty: "intermediate",
  vocabulary: [],
  scenario: null,
  currentIndex: 0,
  scenarioIndex: 0,
  sessionKind: "starter",
  sessionsByLanguage: new Map(),
  testMode: "orthographic",
  activeTest: null,
  testAttempts: [],
  audio: new Audio(),
  audioCache: new Map(),
  isPlayingAll: false,
};

const els = {};
const testLabels = {
  orthographic: "Orthographic recall",
  dictation: "Dictation",
  phonemic: "Sound discrimination",
  scenario: "Scenario comprehension",
};

const starterSets = {
  es: [
    { english: "Where is the station?", target: "¿Dónde está la estación?", pronunciation: "DON-de es-TA la es-ta-SYON", note: "Useful direction question.", tags: ["starter", "travel"] },
    { english: "I would like a ticket.", target: "Quisiera un billete.", pronunciation: "kee-SYE-ra un bee-YE-te", note: "Polite request form.", tags: ["starter", "travel"] },
    { english: "How much does it cost?", target: "¿Cuánto cuesta?", pronunciation: "KWAN-to KWES-ta", note: "Price question.", tags: ["starter", "shopping"] },
    { english: "Can you repeat that?", target: "¿Puede repetirlo?", pronunciation: "PWE-de re-pe-TEER-lo", note: "Repair strategy.", tags: ["starter", "conversation"] },
    { english: "I do not understand yet.", target: "Todavía no entiendo.", pronunciation: "to-da-VEE-a no en-TYEN-do", note: "Honest comprehension signal.", tags: ["starter", "conversation"] },
  ],
  ru: [
    { english: "Where is the station?", target: "Где находится вокзал?", pronunciation: "gdye na-KHO-deet-sya vak-ZAL", note: "Cyrillic direction question.", tags: ["starter", "travel"] },
    { english: "I would like a ticket.", target: "Я хотел бы билет.", pronunciation: "ya kha-TYEL by bee-LYET", note: "Male speaker form. Female speakers often use хотела.", tags: ["starter", "travel"] },
    { english: "How much does it cost?", target: "Сколько это стоит?", pronunciation: "SKOL-ka EH-ta STO-it", note: "Price question.", tags: ["starter", "shopping"] },
    { english: "Can you repeat that?", target: "Можете повторить?", pronunciation: "MO-zhe-tye pov-ta-REET", note: "Polite repair strategy.", tags: ["starter", "conversation"] },
    { english: "I do not understand yet.", target: "Я пока не понимаю.", pronunciation: "ya pa-KA ne pa-nee-MA-yu", note: "Useful learner phrase.", tags: ["starter", "conversation"] },
  ],
  fr: [
    { english: "Where is the station?", target: "Où est la gare ?", pronunciation: "oo eh la gar", note: "Short direction question.", tags: ["starter", "travel"] },
    { english: "I would like a ticket.", target: "Je voudrais un billet.", pronunciation: "zhuh voo-DREH un bee-YEH", note: "Polite request form.", tags: ["starter", "travel"] },
    { english: "How much does it cost?", target: "Ça coûte combien ?", pronunciation: "sa koot kom-BYEN", note: "Price question.", tags: ["starter", "shopping"] },
    { english: "Can you repeat that?", target: "Vous pouvez répéter ?", pronunciation: "voo poo-VAY ray-pay-TAY", note: "Repair strategy.", tags: ["starter", "conversation"] },
    { english: "I do not understand yet.", target: "Je ne comprends pas encore.", pronunciation: "zhuh nuh kom-PRAHN pa zahn-KOR", note: "Useful learner phrase.", tags: ["starter", "conversation"] },
  ],
  zh: [
    { english: "Where is the station?", target: "车站在哪里？", pronunciation: "Chēzhàn zài nǎlǐ?", note: "Simplified Chinese direction question.", tags: ["starter", "travel"] },
    { english: "I would like a ticket.", target: "我想买一张票。", pronunciation: "Wǒ xiǎng mǎi yì zhāng piào.", note: "Ticket request.", tags: ["starter", "travel"] },
    { english: "How much does it cost?", target: "这个多少钱？", pronunciation: "Zhège duōshǎo qián?", note: "Price question.", tags: ["starter", "shopping"] },
    { english: "Can you repeat that?", target: "你可以再说一遍吗？", pronunciation: "Nǐ kěyǐ zài shuō yí biàn ma?", note: "Repair strategy.", tags: ["starter", "conversation"] },
    { english: "I do not understand yet.", target: "我还不明白。", pronunciation: "Wǒ hái bù míngbai.", note: "Useful learner phrase.", tags: ["starter", "conversation"] },
  ],
};

function $(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "languageStrip",
    "modelStatus",
    "saveSessionBtn",
    "activeLanguageLabel",
    "vocabularyModeBtn",
    "scenarioModeBtn",
    "sourceSelect",
    "topicField",
    "topicSelect",
    "conceptInput",
    "difficultySelect",
    "itemCountInput",
    "voiceSelect",
    "rateSelect",
    "englishToggle",
    "maskToggle",
    "autoQueueToggle",
    "generateBtn",
    "generationStepCount",
    "generationSteps",
    "generationProgress",
    "statusLine",
    "sessionCounter",
    "sessionProgress",
    "accuracyLabel",
    "queueCount",
    "testCount",
    "vocabListTitle",
    "listSortSelect",
    "vocabList",
    "focusCard",
    "testStage",
    "insightSummary",
    "insightRecommendation",
    "orthographyScore",
    "phonemicScore",
    "weakList",
    "playAllBtn",
    "focusModeBtn",
  ].forEach((id) => {
    els[id] = $(id);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function cleanAnswer(value) {
  return String(value ?? "")
    .normalize("NFC")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function looseAnswer(value) {
  return cleanAnswer(value).replace(/[¿?¡!.,;:()[\]{}"']/g, "");
}

function shuffle(items) {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const other = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[other]] = [copy[other], copy[index]];
  }
  return copy;
}

function language() {
  return state.config.languages.find((item) => item.code === state.language) ?? state.config.languages[0];
}

function selectedVoice() {
  return state.voice || language().defaultVoice;
}

function currentItem() {
  if (!state.vocabulary.length) return null;
  state.currentIndex = Math.max(0, Math.min(state.currentIndex, state.vocabulary.length - 1));
  return state.vocabulary[state.currentIndex];
}

function setBusy(isBusy) {
  document.body.classList.toggle("loading", isBusy);
  els.generateBtn.disabled = isBusy;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Request failed");
  }
  return payload;
}

function renderConfig() {
  els.modelStatus.textContent = state.config.hasOpenAIKey ? `Model ${state.config.model}` : "Demo mode";
  els.languageStrip.innerHTML = state.config.languages
    .map((item) => {
      const active = item.code === state.language ? "active" : "";
      const switcherLabel = item.code === "zh" ? "Mandarin" : item.display;
      return `
        <button class="language-button ${active}" data-language="${item.code}" type="button">
          <span class="flag" aria-hidden="true">${escapeHtml(item.flag || item.code.toUpperCase())}</span>
          <span><strong>${escapeHtml(switcherLabel)}</strong><small>${escapeHtml(item.nativeName)}</small></span>
        </button>
      `;
    })
    .join("");

  els.difficultySelect.innerHTML = state.config.difficulties
    .map((item) => `<option value="${item.id}">${escapeHtml(item.label)} (${escapeHtml(item.cefr)})</option>`)
    .join("");
  els.difficultySelect.value = state.difficulty;
  renderVoiceOptions();
}

function renderVoiceOptions() {
  const lang = language();
  const previousVoice = state.voice || lang.defaultVoice;
  const validVoices = lang.voices.map((voice) => voice.id);
  document.documentElement.style.setProperty("--target-font", lang.fontStack);
  els.activeLanguageLabel.textContent = `${lang.flag || ""} ${lang.display}`.trim();
  els.voiceSelect.innerHTML = lang.voices
    .map((voice) => `<option value="${voice.id}">${escapeHtml(voice.label)}</option>`)
    .join("");
  state.voice = validVoices.includes(previousVoice) ? previousVoice : lang.defaultVoice;
  els.voiceSelect.value = state.voice;
}

function seedStarterSession(message = "") {
  state.vocabulary = normalizeVocabulary(starterSets[state.language] || starterSets.es);
  state.currentIndex = 0;
  state.scenario = null;
  state.scenarioIndex = 0;
  state.testMode = "orthographic";
  state.activeTest = null;
  state.testAttempts = [];
  state.sessionKind = "starter";
  if (message) els.statusLine.textContent = message;
}

function snapshotCurrentSession() {
  state.sessionsByLanguage.set(state.language, {
    vocabulary: state.vocabulary,
    scenario: state.scenario,
    currentIndex: state.currentIndex,
    scenarioIndex: state.scenarioIndex,
    sessionKind: state.sessionKind,
    testMode: state.testMode,
    activeTest: state.activeTest,
    testAttempts: state.testAttempts,
  });
}

function restoreLanguageSession() {
  const saved = state.sessionsByLanguage.get(state.language);
  if (saved) {
    state.vocabulary = saved.vocabulary || [];
    state.scenario = saved.scenario || null;
    state.currentIndex = saved.currentIndex || 0;
    state.scenarioIndex = saved.scenarioIndex || 0;
    state.sessionKind = saved.sessionKind || "generated";
    state.testMode = saved.testMode || "orthographic";
    state.activeTest = saved.activeTest || null;
    state.testAttempts = saved.testAttempts || [];
    els.statusLine.textContent = `Restored your ${language().display} session.`;
    return;
  }
  seedStarterSession(`${language().display} starter set loaded. Generate a custom session when ready.`);
}

function renderTopics() {
  els.topicSelect.innerHTML = state.topics
    .map((topic) => `<option value="${topic.id}">${escapeHtml(topic.nameEn)} (${topic.count})</option>`)
    .join("");
}

function setSection(section) {
  state.section = section;
  document.querySelectorAll("[data-section]").forEach((button) => {
    button.classList.toggle("active", button.dataset.section === section);
  });
  document.querySelectorAll(".section-view").forEach((view) => {
    view.classList.toggle("active", view.id === `${section}View`);
  });
  if (section === "tests") renderTestStage();
  if (section === "insights") renderInsights();
}

function setGenerateMode(mode) {
  state.generateMode = mode;
  els.vocabularyModeBtn.classList.toggle("active", mode === "vocabulary");
  els.scenarioModeBtn.classList.toggle("active", mode === "scenario");
}

function updateSourceVisibility() {
  state.source = els.sourceSelect.value;
  els.topicField.style.display = state.source === "topic" ? "grid" : "none";
}

function generationTemplate() {
  const mode = state.generateMode === "scenario" ? "scenario" : "vocabulary";
  return [
    `Reading ${mode} settings`,
    "Preparing language and voice",
    state.source === "topic" && mode === "vocabulary" ? "Loading topic vocabulary" : "Calling AI generator",
    "Building study and test queues",
    "Finalizing session",
  ];
}

function setGenerationStep(index, message = "") {
  const steps = generationTemplate();
  const safeIndex = Math.max(0, Math.min(index, steps.length));
  els.generationStepCount.textContent = `${safeIndex} / ${steps.length}`;
  els.generationProgress.style.width = `${Math.round((safeIndex / steps.length) * 100)}%`;
  els.generationSteps.innerHTML = steps
    .map((step, stepIndex) => {
      const done = stepIndex < safeIndex;
      const active = stepIndex === safeIndex && safeIndex < steps.length;
      return `
        <li class="${done ? "done" : ""} ${active ? "active" : ""}">
          <span class="step-dot"></span>
          <span>${escapeHtml(step)}</span>
          <span class="step-check">${done ? "✓" : ""}</span>
        </li>
      `;
    })
    .join("");
  if (message) els.statusLine.textContent = message;
}

function normalizeVocabulary(items) {
  return (items || []).map((item, index) => ({
    english: item.english || "",
    target: item.target || "",
    pronunciation: item.pronunciation || "",
    note: item.note || "",
    tags: item.tags || [],
    index,
    confidence: item.confidence || 0,
    due: item.due || "New",
    wrong: item.wrong || 0,
    correct: item.correct || 0,
  }));
}

async function generateSession() {
  setBusy(true);
  setGenerationStep(0, "Reading your settings.");
  try {
    state.difficulty = els.difficultySelect.value;
    await delay(150);
    setGenerationStep(1, `Preparing ${language().display} with ${els.voiceSelect.selectedOptions[0]?.textContent || "default voice"}.`);
    await delay(150);
    setGenerationStep(2, state.source === "topic" ? "Loading and translating the source topic." : "Sending the prompt to the AI generator.");

    if (state.generateMode === "scenario") {
      const scenario = await api("/api/scenario/generate", {
        method: "POST",
        body: {
          topic: els.conceptInput.value.trim(),
          language: state.language,
          difficulty: state.difficulty,
        },
      });
      state.scenario = scenario;
      state.scenarioIndex = 0;
      state.vocabulary = [];
      state.currentIndex = 0;
      state.sessionKind = "generated";
      state.testAttempts = [];
      setGenerationStep(3, "Scenario questions are ready for the Tests section.");
      await delay(150);
      setGenerationStep(5, `Scenario loaded from ${scenario.source}.`);
      setSection("tests");
      state.testMode = "scenario";
      syncTestModeButtons();
      createActiveTest();
    } else {
      const body = {
        language: state.language,
        count: Number(els.itemCountInput.value || 20),
        difficulty: state.difficulty,
      };
      const payload = state.source === "topic"
        ? await api("/api/vocabulary/topic", { method: "POST", body: { ...body, topicId: els.topicSelect.value } })
        : await api("/api/vocabulary/generate", { method: "POST", body: { ...body, concept: els.conceptInput.value.trim() } });
      state.vocabulary = normalizeVocabulary(payload.items);
      state.currentIndex = 0;
      state.scenario = null;
      state.scenarioIndex = 0;
      state.sessionKind = "generated";
      state.testAttempts = [];
      setGenerationStep(3, "Creating audio rows, recall prompts, spelling tests, and sound tests.");
      await delay(150);
      setGenerationStep(5, `${state.vocabulary.length} vocabulary items loaded from ${payload.source}.`);
      setSection("study");
      drawIdleWaveform();
    }
    snapshotCurrentSession();
    renderAll();
  } catch (error) {
    els.statusLine.textContent = `Generation failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sortedVocabulary() {
  const items = [...state.vocabulary];
  const sort = els.listSortSelect.value;
  if (sort === "hard") {
    return items.sort((a, b) => (b.wrong + (5 - b.confidence)) - (a.wrong + (5 - a.confidence)));
  }
  if (sort === "due") {
    return items.sort((a, b) => Number(a.confidence > 0) - Number(b.confidence > 0));
  }
  return items;
}

function renderVocabList() {
  els.vocabListTitle.textContent = state.vocabulary.length ? `${state.vocabulary.length} items ready` : "No session loaded";
  if (!state.vocabulary.length) {
    els.vocabList.innerHTML = `
      <div class="empty-state">
        <div>
          <h2>No vocabulary yet</h2>
          <p>Generate a custom session or load a source topic. The list will give every item its own play button.</p>
        </div>
      </div>
    `;
    return;
  }

  els.vocabList.innerHTML = sortedVocabulary()
    .map((item) => {
      const active = item.index === state.currentIndex ? "active" : "";
      const dots = Array.from({ length: 3 }, (_, index) => `<span class="${index < item.confidence ? "on" : ""}"></span>`).join("");
      const due = item.confidence >= 4 ? "Known" : item.confidence > 0 ? "Review" : "New";
      return `
        <div class="vocab-row ${active}" data-row-index="${item.index}">
          <button class="row-play" data-play-index="${item.index}" type="button" aria-label="Play ${escapeHtml(item.target)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7-11-7Z"></path></svg>
          </button>
          <button class="row-copy" data-select-index="${item.index}" type="button">
            <strong>${escapeHtml(item.target)}</strong>
            <small>${escapeHtml(item.english)}</small>
          </button>
          <button class="row-repeat" data-repeat-index="${item.index}" type="button" aria-label="Repeat ${escapeHtml(item.target)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a7 7 0 0 1 12-5l2 2"></path><path d="M18 4v5h-5"></path><path d="M20 12a7 7 0 0 1-12 5l-2-2"></path><path d="M6 20v-5h5"></path></svg>
          </button>
          <div>
            <div class="row-meter">${dots}</div>
            <div class="due-label">${escapeHtml(due)}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderFocusCard() {
  const item = currentItem();
  if (!item) {
    els.focusCard.innerHTML = `
      <div class="empty-state">
        <div>
          <h2>Generate a study session</h2>
          <p>Study is intentionally quiet. Tests, insights, and review are separate so the card can stay focused.</p>
        </div>
      </div>
    `;
    drawIdleWaveform();
    return;
  }

  const showEnglish = els.englishToggle.checked;
  const masked = els.maskToggle.checked ? "masked" : "";
  els.focusCard.innerHTML = `
    <div class="focus-top">
      <div>
        <span class="field-label">Study card</span>
        <strong>${state.currentIndex + 1} / ${state.vocabulary.length}</strong>
      </div>
      <div class="focus-actions">
        <button class="nav-button" id="prevItemBtn" type="button">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6"></path></svg>
          Previous
        </button>
        <button class="nav-button" id="nextItemBtn" type="button">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18l6-6-6-6"></path></svg>
          Next
        </button>
      </div>
    </div>
    <div class="focus-body">
      <div class="target-text ${masked}">${escapeHtml(item.target)}</div>
      ${showEnglish ? `<div class="english-support">${escapeHtml(item.english)}</div>` : `<button class="command-button" id="revealEnglishBtn" type="button">Reveal English</button>`}
      ${item.note ? `<div class="note-line">${escapeHtml(item.note)}</div>` : ""}
    </div>
    <div class="audio-panel">
      <button class="main-play" id="playCurrentBtn" type="button" aria-label="Play current item">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7-11-7Z"></path></svg>
      </button>
      <canvas id="waveCanvas" width="720" height="76" aria-label="Audio waveform"></canvas>
      <span id="audioTime">0:00</span>
    </div>
    <div class="focus-controls">
      <button id="shadowBtn" type="button">Shadow aloud</button>
      <button id="recallBtn" type="button">Recall test</button>
      <button id="addReviewBtn" type="button">Add to review</button>
    </div>
  `;
  drawIdleWaveform();
}

function renderStats() {
  const total = state.vocabulary.length || state.scenario?.items?.length || 0;
  const index = state.section === "tests" && state.testMode === "scenario" ? state.scenarioIndex : state.currentIndex;
  els.sessionCounter.textContent = `${total ? index + 1 : 0} / ${total} items`;
  els.sessionProgress.style.width = total ? `${Math.round(((index + 1) / total) * 100)}%` : "0";
  const correct = state.testAttempts.filter((attempt) => attempt.correct).length;
  const attempts = state.testAttempts.length;
  els.accuracyLabel.textContent = attempts ? `${Math.round((correct / attempts) * 100)}%` : "0%";
  const queue = state.vocabulary.filter((item) => item.confidence < 4 || item.wrong > 0).length;
  els.queueCount.textContent = String(queue);
  els.testCount.textContent = String(attempts);
}

function renderAll() {
  renderConfig();
  renderVocabList();
  renderFocusCard();
  renderStats();
  renderTestStage();
  renderInsights();
}

async function synthesize(text) {
  const key = JSON.stringify([text, state.language, selectedVoice(), els.rateSelect.value]);
  if (state.audioCache.has(key)) return state.audioCache.get(key);
  const payload = await api("/api/tts", {
    method: "POST",
    body: {
      text,
      language: state.language,
      voice: selectedVoice(),
      rate: els.rateSelect.value,
    },
  });
  state.audioCache.set(key, payload);
  return payload;
}

async function playText(text) {
  if (!text) return null;
  const timeNode = $("audioTime");
  if (timeNode) timeNode.textContent = "loading";
  const payload = await synthesize(text);
  await renderWaveformFromUrl(payload.url);
  state.audio.src = payload.url;
  try {
    await state.audio.play();
    if (timeNode) timeNode.textContent = payload.durationSeconds ? `0:00 / ${formatTime(payload.durationSeconds)}` : "playing";
  } catch {
    if (timeNode) timeNode.textContent = payload.durationSeconds ? `ready / ${formatTime(payload.durationSeconds)}` : "ready";
    els.statusLine.textContent = "Audio is ready. Click play again if the browser blocked immediate playback.";
  }
  return payload;
}

async function playVocabulary(index) {
  if (!state.vocabulary[index]) return;
  state.currentIndex = index;
  renderVocabList();
  renderFocusCard();
  await playText(state.vocabulary[index].target);
}

async function playAll() {
  if (!state.vocabulary.length || state.isPlayingAll) return;
  state.isPlayingAll = true;
  els.statusLine.textContent = "Playing the vocabulary list from the current item.";
  for (let index = state.currentIndex; index < state.vocabulary.length; index += 1) {
    if (!state.isPlayingAll) break;
    state.currentIndex = index;
    renderVocabList();
    renderFocusCard();
    await playText(state.vocabulary[index].target);
    await waitForAudioEnd();
  }
  state.isPlayingAll = false;
}

function waitForAudioEnd() {
  return new Promise((resolve) => {
    if (!state.audio.src || state.audio.paused) {
      resolve();
      return;
    }
    state.audio.onended = () => resolve();
    setTimeout(resolve, 9000);
  });
}

function formatTime(seconds) {
  const safe = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

function drawIdleWaveform() {
  const canvas = $("waveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#c3ced3";
  ctx.lineWidth = 2;
  const mid = canvas.height / 2;
  for (let index = 0; index < 80; index += 1) {
    const x = 10 + index * ((canvas.width - 20) / 80);
    const h = 6 + (index % 9) * 2;
    ctx.beginPath();
    ctx.moveTo(x, mid - h);
    ctx.lineTo(x, mid + h);
    ctx.stroke();
  }
}

async function renderWaveformFromUrl(url) {
  const canvas = $("waveCanvas");
  if (!canvas) return;
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    const buffer = await fetch(url).then((response) => response.arrayBuffer());
    const audioBuffer = await context.decodeAudioData(buffer.slice(0));
    const data = audioBuffer.getChannelData(0);
    const bars = 120;
    const block = Math.floor(data.length / bars);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    const mid = canvas.height / 2;
    for (let index = 0; index < bars; index += 1) {
      let sum = 0;
      for (let sample = 0; sample < block; sample += 1) {
        sum += Math.abs(data[index * block + sample] || 0);
      }
      const amp = Math.min(1, (sum / block) * 6);
      const height = Math.max(4, amp * (canvas.height * 0.44));
      const x = 8 + index * ((canvas.width - 16) / bars);
      ctx.strokeStyle = index < bars * 0.42 ? "#007b78" : "#b9c4c9";
      ctx.beginPath();
      ctx.moveTo(x, mid - height);
      ctx.lineTo(x, mid + height);
      ctx.stroke();
    }
    await context.close();
  } catch {
    drawIdleWaveform();
  }
}

function createActiveTest() {
  if (state.testMode === "scenario" && state.scenario?.items?.length) {
    const item = state.scenario.items[state.scenarioIndex] || state.scenario.items[0];
    state.activeTest = {
      type: "scenario",
      prompt: item.question,
      answer: item.choices[item.correct_choice],
      choices: item.choices,
      correctIndex: item.correct_choice,
      explanation: item.reasoning_note_en || item.reasoning_note,
    };
    return;
  }

  const item = currentItem();
  if (!item) {
    state.activeTest = null;
    return;
  }

  if (state.testMode === "phonemic") {
    const pool = shuffle(state.vocabulary.filter((candidate) => candidate.index !== item.index)).slice(0, 3);
    state.activeTest = {
      type: "phonemic",
      prompt: "Listen and choose the written form that matches the sound.",
      answer: item.target,
      choices: shuffle([item, ...pool]).map((choice) => choice.target),
      item,
    };
    return;
  }

  if (state.testMode === "dictation") {
    state.activeTest = {
      type: "dictation",
      prompt: "Listen, then type the target text exactly.",
      answer: item.target,
      item,
    };
    return;
  }

  state.activeTest = {
    type: "orthographic",
    prompt: `Write this in ${language().display}: ${item.english}`,
    answer: item.target,
    item,
  };
}

function renderTestStage() {
  if (!els.testStage || state.section !== "tests") return;
  if (!state.activeTest) createActiveTest();
  const test = state.activeTest;
  if (!test) {
    els.testStage.innerHTML = `
      <div class="empty-state">
        <div>
          <h2>No test material yet</h2>
          <p>Generate vocabulary for spelling, dictation, and sound tests. Generate a scenario for comprehension tests.</p>
        </div>
      </div>
    `;
    return;
  }

  if (test.type === "phonemic" || test.type === "scenario") {
    els.testStage.innerHTML = `
      <div class="test-card">
        <p class="test-prompt">${escapeHtml(testLabels[test.type] || "Test")}</p>
        <h2>${escapeHtml(test.prompt)}</h2>
        ${test.type === "phonemic" ? `<button class="main-play" id="testPlayBtn" type="button" aria-label="Play test audio"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7-11-7Z"></path></svg></button>` : ""}
        <div class="choice-grid">
          ${test.choices.map((choice, index) => `<button class="choice-button" data-test-choice="${index}" type="button">${String.fromCharCode(65 + index)}. ${escapeHtml(choice)}</button>`).join("")}
        </div>
        <div class="test-feedback" id="testFeedback"></div>
        <div class="test-actions">
          <button id="nextTestBtn" type="button">Next prompt</button>
          <button id="studyCurrentBtn" type="button">Study current</button>
          <button id="saveTestBtn" type="button">Save session</button>
        </div>
      </div>
    `;
    return;
  }

  els.testStage.innerHTML = `
    <div class="test-card">
      <p class="test-prompt">${escapeHtml(testLabels[test.type])}</p>
      <h2>${escapeHtml(test.prompt)}</h2>
      ${test.type === "dictation" ? `<button class="main-play" id="testPlayBtn" type="button" aria-label="Play dictation audio"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7-11-7Z"></path></svg></button>` : ""}
      <div class="test-input-row">
        <input id="testAnswerInput" type="text" autocomplete="off" placeholder="Type your answer" />
        <button class="primary-button" id="checkTestBtn" type="button">Check</button>
      </div>
      <div class="test-feedback" id="testFeedback"></div>
      <div class="test-actions">
        <button id="nextTestBtn" type="button">Next prompt</button>
        <button id="showAnswerBtn" type="button">Show answer</button>
        <button id="saveTestBtn" type="button">Save session</button>
      </div>
    </div>
  `;
}

function checkTypedTest() {
  const test = state.activeTest;
  const input = $("testAnswerInput");
  if (!test || !input) return;
  const expected = cleanAnswer(test.answer);
  const actual = cleanAnswer(input.value);
  const exact = actual === expected;
  const near = !exact && looseAnswer(input.value) === looseAnswer(test.answer);
  recordAttempt(test, exact, near ? "Near miss. Check punctuation, accents, Cyrillic, or characters." : "");
}

function chooseTest(index) {
  const test = state.activeTest;
  if (!test) return;
  const correct = test.type === "scenario" ? index === test.correctIndex : test.choices[index] === test.answer;
  document.querySelectorAll("[data-test-choice]").forEach((button) => {
    const buttonIndex = Number(button.dataset.testChoice);
    const isCorrect = test.type === "scenario" ? buttonIndex === test.correctIndex : test.choices[buttonIndex] === test.answer;
    button.classList.toggle("correct", isCorrect);
    button.classList.toggle("wrong", buttonIndex === index && !isCorrect);
  });
  recordAttempt(test, correct);
}

function recordAttempt(test, correct, note = "") {
  const item = test.item || currentItem();
  if (item) {
    if (correct) {
      item.correct += 1;
      item.confidence = Math.min(3, item.confidence + 1);
    } else {
      item.wrong += 1;
      item.confidence = Math.max(0, item.confidence - 1);
    }
  }
  state.testAttempts.push({
    type: test.type,
    language: state.language,
    correct,
    answer: test.answer,
    timestamp: Date.now(),
  });
  const feedback = $("testFeedback");
  if (feedback) {
    feedback.className = `test-feedback ${correct ? "good" : "bad"}`;
    feedback.textContent = correct ? "Correct. Move to the next prompt." : note || `Answer: ${test.answer}`;
  }
  renderStats();
  renderVocabList();
  renderInsights();
}

function nextTestPrompt() {
  if (state.testMode === "scenario" && state.scenario?.items?.length) {
    state.scenarioIndex = (state.scenarioIndex + 1) % state.scenario.items.length;
  } else if (state.vocabulary.length) {
    state.currentIndex = (state.currentIndex + 1) % state.vocabulary.length;
  }
  createActiveTest();
  renderTestStage();
  renderStats();
}

function syncTestModeButtons() {
  document.querySelectorAll("[data-test-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.testMode === state.testMode);
  });
}

function renderInsights() {
  if (!els.insightSummary) return;
  const attempts = state.testAttempts;
  const correct = attempts.filter((attempt) => attempt.correct).length;
  const orth = attempts.filter((attempt) => ["orthographic", "dictation"].includes(attempt.type));
  const phon = attempts.filter((attempt) => attempt.type === "phonemic");
  const orthCorrect = orth.filter((attempt) => attempt.correct).length;
  const phonCorrect = phon.filter((attempt) => attempt.correct).length;
  const accuracy = attempts.length ? Math.round((correct / attempts.length) * 100) : 0;
  els.insightSummary.textContent = attempts.length ? `${accuracy}% overall accuracy` : "No tests yet";
  els.orthographyScore.textContent = orth.length ? `${Math.round((orthCorrect / orth.length) * 100)}%` : "0%";
  els.phonemicScore.textContent = phon.length ? `${Math.round((phonCorrect / phon.length) * 100)}%` : "0%";
  if (accuracy >= 85 && attempts.length >= 5) {
    els.insightRecommendation.textContent = "Raise the difficulty or reduce English support. You are ready for more retrieval pressure.";
  } else if (attempts.length) {
    els.insightRecommendation.textContent = "Prioritize missed items with dictation and sound discrimination before adding new material.";
  } else {
    els.insightRecommendation.textContent = "Complete one spelling test and one sound test to unlock a useful evaluation.";
  }

  const weak = [...state.vocabulary]
    .filter((item) => item.wrong > 0 || item.confidence < 2)
    .sort((a, b) => (b.wrong - a.wrong) || (a.confidence - b.confidence))
    .slice(0, 8);
  els.weakList.innerHTML = weak.length
    ? weak.map((item) => `<div class="weak-item"><strong>${escapeHtml(item.target)}</strong><span>${escapeHtml(item.english)}</span><span>${item.wrong} misses, confidence ${item.confidence}/3</span></div>`).join("")
    : `<div class="weak-item"><strong>No weak items yet</strong><span>Missed tests will appear here.</span></div>`;
}

async function saveSession() {
  els.statusLine.textContent = "Saving session locally.";
  const payload = await api("/api/session/save", {
    method: "POST",
    body: {
      language: state.language,
      difficulty: state.difficulty,
      vocabulary: state.vocabulary,
      scenario: state.scenario,
      testAttempts: state.testAttempts,
    },
  });
  els.statusLine.textContent = `Session saved as ${payload.file}.`;
}

function bindEvents() {
  els.languageStrip.addEventListener("click", (event) => {
    const button = event.target.closest("[data-language]");
    if (!button) return;
    if (button.dataset.language === state.language) return;
    snapshotCurrentSession();
    state.language = button.dataset.language;
    state.voice = "";
    state.audioCache.clear();
    state.audio.pause();
    restoreLanguageSession();
    syncTestModeButtons();
    renderAll();
  });

  els.vocabularyModeBtn.addEventListener("click", () => setGenerateMode("vocabulary"));
  els.scenarioModeBtn.addEventListener("click", () => setGenerateMode("scenario"));
  els.sourceSelect.addEventListener("change", updateSourceVisibility);
  els.voiceSelect.addEventListener("change", () => {
    state.voice = els.voiceSelect.value;
    state.audioCache.clear();
  });
  els.rateSelect.addEventListener("change", () => state.audioCache.clear());
  els.englishToggle.addEventListener("change", renderFocusCard);
  els.maskToggle.addEventListener("change", renderFocusCard);
  els.listSortSelect.addEventListener("change", renderVocabList);
  els.generateBtn.addEventListener("click", generateSession);
  els.saveSessionBtn.addEventListener("click", saveSession);
  els.playAllBtn.addEventListener("click", playAll);
  els.focusModeBtn.addEventListener("click", () => {
    document.documentElement.requestFullscreen?.();
    els.statusLine.textContent = "Focus mode started. Use left and right keys to navigate.";
  });

  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => setSection(button.dataset.section));
  });

  document.querySelectorAll("[data-start-test]").forEach((button) => {
    button.addEventListener("click", () => {
      state.testMode = button.dataset.startTest;
      syncTestModeButtons();
      createActiveTest();
      setSection("tests");
    });
  });

  document.querySelectorAll("[data-test-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.testMode = button.dataset.testMode;
      syncTestModeButtons();
      createActiveTest();
      renderTestStage();
    });
  });

  els.vocabList.addEventListener("click", async (event) => {
    const play = event.target.closest("[data-play-index]");
    const repeat = event.target.closest("[data-repeat-index]");
    const select = event.target.closest("[data-select-index]");
    if (play) {
      await playVocabulary(Number(play.dataset.playIndex));
      return;
    }
    if (repeat) {
      await playVocabulary(Number(repeat.dataset.repeatIndex));
      return;
    }
    if (select) {
      state.currentIndex = Number(select.dataset.selectIndex);
      renderVocabList();
      renderFocusCard();
    }
  });

  els.focusCard.addEventListener("click", async (event) => {
    if (event.target.closest("#prevItemBtn")) {
      state.currentIndex = (state.currentIndex - 1 + state.vocabulary.length) % state.vocabulary.length;
      renderVocabList();
      renderFocusCard();
    }
    if (event.target.closest("#nextItemBtn")) {
      state.currentIndex = (state.currentIndex + 1) % state.vocabulary.length;
      renderVocabList();
      renderFocusCard();
    }
    if (event.target.closest("#playCurrentBtn")) await playText(currentItem()?.target);
    if (event.target.closest("#revealEnglishBtn")) {
      els.englishToggle.checked = true;
      renderFocusCard();
    }
    if (event.target.closest("#shadowBtn")) {
      await playText(currentItem()?.target);
      els.statusLine.textContent = "Shadow aloud, then test the item from memory.";
    }
    if (event.target.closest("#recallBtn")) {
      state.testMode = "orthographic";
      syncTestModeButtons();
      createActiveTest();
      setSection("tests");
    }
    if (event.target.closest("#addReviewBtn")) {
      const item = currentItem();
      if (item) item.confidence = 0;
      renderVocabList();
      renderStats();
      els.statusLine.textContent = "Current item added to the review queue.";
    }
  });

  els.testStage.addEventListener("click", async (event) => {
    const choice = event.target.closest("[data-test-choice]");
    if (choice) chooseTest(Number(choice.dataset.testChoice));
    if (event.target.closest("#checkTestBtn")) checkTypedTest();
    if (event.target.closest("#testPlayBtn")) await playText(state.activeTest?.answer);
    if (event.target.closest("#nextTestBtn")) nextTestPrompt();
    if (event.target.closest("#showAnswerBtn")) {
      const feedback = $("testFeedback");
      if (feedback && state.activeTest) feedback.textContent = `Answer: ${state.activeTest.answer}`;
    }
    if (event.target.closest("#studyCurrentBtn")) setSection("study");
    if (event.target.closest("#saveTestBtn")) saveSession();
  });

  document.addEventListener("keydown", (event) => {
    if (!state.vocabulary.length) return;
    if (event.key === "ArrowLeft") {
      state.currentIndex = (state.currentIndex - 1 + state.vocabulary.length) % state.vocabulary.length;
      renderVocabList();
      renderFocusCard();
    }
    if (event.key === "ArrowRight") {
      state.currentIndex = (state.currentIndex + 1) % state.vocabulary.length;
      renderVocabList();
      renderFocusCard();
    }
  });

  state.audio.addEventListener("timeupdate", () => {
    const node = $("audioTime");
    if (!node) return;
    const current = formatTime(state.audio.currentTime);
    const duration = Number.isFinite(state.audio.duration) ? ` / ${formatTime(state.audio.duration)}` : "";
    node.textContent = `${current}${duration}`;
  });
}

async function init() {
  cacheElements();
  setGenerationStep(0);
  drawIdleWaveform();
  const [config, topicsPayload] = await Promise.all([api("/api/config"), api("/api/topics")]);
  state.config = config;
  state.topics = topicsPayload.topics || [];
  renderConfig();
  renderTopics();
  updateSourceVisibility();
  seedStarterSession("Starter set loaded. Generate custom vocabulary or a scenario when ready.");
  snapshotCurrentSession();
  bindEvents();
  renderAll();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="empty-state"><h2>LearnLanguage could not start.</h2><p>${escapeHtml(error.message)}</p></main>`;
});
