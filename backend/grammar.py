"""Grammar-theoretical architecture per language.

Each language carries a structured model of *what there is to learn* and *in
what order*, grounded in its typology:

- ``pillars``   - the load-bearing systems of the language (the mental model).
- ``roadmap``   - CEFR-staged features (A1-C2), each with a learner-facing tip
                  and a minimal example. These drive three things:
                  1. generation prompts (lessons/compositions weave in features
                     at the learner's level),
                  2. the in-app Grammar view (the reference itself),
                  3. progress tracking (features surfaced in generated content
                     are recorded as "encountered" locally).
- ``challenges``- transfer problems for an English/Dutch-speaking learner.
- ``phonology`` - the sound-system traps that shape listening and speaking.

This is deliberately data, not prose: one source of truth for prompts, UI,
and progress.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .languages import normalize_language_code

CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


def _f(fid: str, name: str, tip: str, example: str, example_en: str) -> dict:
    return {"id": fid, "name": name, "tip": tip, "example": example, "example_en": example_en}


GRAMMAR: Dict[str, dict] = {
    # ------------------------------------------------------------------ FRENCH
    "fr": {
        "overview": (
            "A Romance language with fusional morphology: meaning is packed into word endings "
            "and agreement. The verb is the engine of the sentence; gender and agreement ripple "
            "through everything; spelling and sound diverge constantly."
        ),
        "typology": [
            "Word order: subject-verb-object, with object pronouns jumping before the verb.",
            "Two grammatical genders; determiners, adjectives, and some participles agree.",
            "Rich verb morphology: person, number, tense, mood fused into endings.",
            "Spoken and written French are two registers of one system - much written morphology is silent.",
        ],
        "pillars": [
            {"id": "gender-agreement", "title": "Gender & agreement", "summary": "Every noun is le or la. Determiners, adjectives, and past participles copy gender and number - agreement is the glue of the sentence."},
            {"id": "verb-engine", "title": "The verb engine", "summary": "Three conjugation families plus a core of irregulars (être, avoir, aller, faire). Tense and mood live in the ending: présent, passé composé/imparfait, futur, conditionnel, subjonctif."},
            {"id": "pronoun-machinery", "title": "Pronoun machinery", "summary": "Object pronouns (me, te, le, la, lui, y, en) are clitics: they stand before the verb in a fixed order. Mastering them is mastering French rhythm."},
            {"id": "negation-questions", "title": "Negation & questions", "summary": "Negation is a frame: ne … pas around the verb (ne often dropped in speech). Questions come in three registers: intonation, est-ce que, inversion."},
            {"id": "sound-spelling", "title": "Elision & liaison", "summary": "Words fuse: le + eau → l'eau; final consonants awaken before vowels (les‿amis). The writing system remembers history the mouth has abandoned."},
            {"id": "register", "title": "Register", "summary": "tu vs vous, on vs nous, dropped ne - French signals social distance grammatically. Spoken French is not sloppy written French; it is its own system."},
        ],
        "roadmap": {
            "A1": [
                _f("fr-present-er", "Present tense of -er verbs", "One stem + silent endings: je parle, tu parles, ils parlent all sound identical.", "Je parle un peu français.", "I speak a little French."),
                _f("fr-articles-gender", "Articles & gender", "Learn every noun with its article - un/une is part of the word.", "un café, une bière, l'eau", "a coffee, a beer, the water"),
                _f("fr-etre-avoir", "être & avoir", "The two power verbs: identity/state vs possession - and later, the keys to the past tense.", "J'ai froid, mais je suis content.", "I'm cold, but I'm happy."),
                _f("fr-negation-nepas", "Negation ne … pas", "A frame around the verb; spoken French often drops the ne.", "Je ne comprends pas. / Je comprends pas.", "I don't understand."),
                _f("fr-questions-basic", "Three ways to ask", "Intonation (Tu viens ?), est-ce que, and inversion (Viens-tu ?) - register rises left to right.", "Est-ce que tu viens ?", "Are you coming?"),
            ],
            "A2": [
                _f("fr-passe-compose", "Passé composé", "avoir/être + past participle for completed events; être-verbs (aller, venir, rester…) agree with the subject.", "Elle est allée à Lyon.", "She went to Lyon."),
                _f("fr-futur-proche", "Futur proche", "aller + infinitive - the everyday future.", "On va manger à midi.", "We're going to eat at noon."),
                _f("fr-object-pronouns", "Object pronouns", "le/la/les and lui/leur move before the verb.", "Je le vois demain.", "I'm seeing him tomorrow."),
                _f("fr-comparatives", "Comparison", "plus/moins/aussi … que; bon → meilleur, bien → mieux.", "C'est mieux que hier.", "It's better than yesterday."),
                _f("fr-imperatif", "Imperative", "Drop the subject; -er verbs lose the -s: Parle ! Pronouns attach after: Dis-le-moi.", "Écoute-moi !", "Listen to me!"),
            ],
            "B1": [
                _f("fr-imparfait-vs-pc", "Imparfait vs passé composé", "Backdrop vs event: the imparfait paints the scene, the passé composé moves the plot.", "Il pleuvait quand elle est arrivée.", "It was raining when she arrived."),
                _f("fr-futur-simple", "Futur simple", "Infinitive + -ai, -as, -a…: the written and formal future.", "Je te dirai tout demain.", "I'll tell you everything tomorrow."),
                _f("fr-conditionnel", "Conditionnel", "Politeness and hypothesis: je voudrais, on pourrait.", "À ta place, je partirais tôt.", "In your place, I would leave early."),
                _f("fr-subjonctif-intro", "Subjonctif after il faut que", "First contact with the mood of necessity and desire.", "Il faut que tu viennes.", "You have to come."),
                _f("fr-relatifs", "Relative pronouns qui/que/dont", "qui = subject, que = object, dont = of which/whose.", "Le livre dont je t'ai parlé.", "The book I told you about."),
                _f("fr-y-en", "The pronouns y & en", "y replaces à + place/thing, en replaces de + quantity.", "Des pommes ? J'en ai trois.", "Apples? I have three (of them)."),
            ],
            "B2": [
                _f("fr-subjonctif-full", "Subjonctif across triggers", "Emotion, doubt, will, and conjunctions (bien que, pour que) all open the subjunctive door.", "Je doute qu'il soit là.", "I doubt he's there."),
                _f("fr-plus-que-parfait", "Plus-que-parfait", "The past of the past: avait/était + participle.", "Elle avait déjà mangé quand je suis rentré.", "She had already eaten when I got home."),
                _f("fr-si-clauses", "Si-clauses", "Three ladders: si + présent → futur; si + imparfait → conditionnel; si + PQP → conditionnel passé.", "Si j'avais su, je serais venu.", "If I had known, I would have come."),
                _f("fr-passive-on", "Passive & on", "French avoids the passive: on dit que… does the work of 'it is said that…'.", "On m'a volé mon vélo.", "My bike was stolen."),
                _f("fr-gerondif", "Gérondif", "en + -ant: doing two things at once, or how something is done.", "Elle lit en marchant.", "She reads while walking."),
                _f("fr-pronoun-order", "Stacked pronouns", "me/te/nous/vous before le/la/les before lui/leur before y before en.", "Je le lui ai donné.", "I gave it to him."),
            ],
            "C1": [
                _f("fr-passe-simple", "Passé simple (recognition)", "The literary past - you read it far more than you say it.", "Il ouvrit la porte et sortit.", "He opened the door and left."),
                _f("fr-participle-agreement", "Fine participle agreement", "With avoir, the participle agrees with a preceding direct object.", "Les lettres qu'il a écrites.", "The letters he wrote."),
                _f("fr-nominalisation", "Nominalisation & register", "Formal French turns verbs into nouns: partir → le départ.", "La fermeture de l'usine a surpris tout le monde.", "The closing of the factory surprised everyone."),
                _f("fr-connecteurs", "Advanced connectors", "quoique, en dépit de, dès lors que - the joints of argued prose.", "Quoiqu'il soit tard, restons.", "Although it's late, let's stay."),
            ],
            "C2": [
                _f("fr-litteraire", "Literary moods", "Recognize the subjonctif imparfait (qu'il fût) and other museum pieces of written style.", "Il craignait qu'elle ne vînt.", "He feared she might come."),
                _f("fr-dislocation", "Dislocation & emphasis", "Spoken French fronts and doubles: Moi, le café, j'adore ça.", "Ce film, je l'ai vu trois fois.", "That film - I've seen it three times."),
                _f("fr-regimes", "Prepositional regimes", "Verbs govern idiosyncratic prepositions: se fier à, se méfier de, tenir à.", "Je tiens à vous remercier.", "I am keen to thank you."),
                _f("fr-discourse", "Discourse particles", "enfin, quand même, du coup, bref - the particles that make speech native.", "Du coup, on fait quoi ?", "So... what do we do then?"),
            ],
        },
        "challenges": [
            "Gender is unpredictable - learn nouns with their article, never bare.",
            "Much verb morphology is silent: parle/parles/parlent sound identical.",
            "Passé composé vs imparfait is aspect, not translation of English tenses.",
            "Object pronouns move before the verb, against English instinct.",
            "Liaison and elision mean word boundaries you hear are not the ones you read.",
        ],
        "phonology": [
            "Nasal vowels /ɑ̃/ /ɔ̃/ /ɛ̃/ (banc, bon, bain) carry meaning - vent vs vend vs vin.",
            "Final consonants are usually silent, but wake up in liaison: les‿enfants.",
            "/y/ (tu) vs /u/ (tout) is a new vowel for English/Dutch ears.",
            "Stress always falls on the final syllable of the group - no lexical stress.",
            "The French r /ʁ/ is uvular, closer to Dutch g than English r.",
        ],
    },

    # ----------------------------------------------------------------- SPANISH
    "es": {
        "overview": (
            "A Romance language with transparent spelling and a verb system that carries the "
            "whole sentence: subject pronouns usually vanish because the ending says who acts. "
            "Two past aspects and a fully alive subjunctive organize how reality is reported."
        ),
        "typology": [
            "Pro-drop: hablo means 'I speak' - subject pronouns appear only for contrast.",
            "Flexible SVO; objects marked by the 'personal a' when human.",
            "Two genders; adjectives follow and agree with their noun.",
            "Nearly phonetic orthography - what you see is what you say.",
        ],
        "pillars": [
            {"id": "verb-person", "title": "The verb says who", "summary": "Endings encode person and number, so subjects drop. Three families (-ar/-er/-ir) plus strong irregulars (ser, ir, tener, hacer)."},
            {"id": "ser-estar", "title": "Two verbs 'to be'", "summary": "ser for essence and identity, estar for state and location. Es aburrido (he's boring) vs está aburrido (he's bored)."},
            {"id": "past-aspect", "title": "Two pasts", "summary": "Indefinido moves the story forward; imperfecto paints the background. The contrast is aspect, not time."},
            {"id": "subjunctive", "title": "The living subjunctive", "summary": "Wishes, doubts, emotions, and unreal worlds switch the verb into subjunctive - everyday Spanish, not bookish Spanish."},
            {"id": "clitics-a", "title": "Clitics & personal a", "summary": "Object pronouns attach before verbs or onto infinitives (dímelo); human direct objects take a: Veo a María."},
            {"id": "variation", "title": "A pluricentric language", "summary": "tú/usted/vos, vosotros vs ustedes, seseo - Spanish is one system with many voices."},
        ],
        "roadmap": {
            "A1": [
                _f("es-present-regular", "Present tense, three families", "-ar/-er/-ir endings say who acts; drop the pronoun.", "Hablo español y aprendo más.", "I speak Spanish and I'm learning more."),
                _f("es-gender-articles", "Gender & articles", "el/la, un/una; most -o nouns masculine, -a feminine - with famous traitors (el día, la mano).", "el mapa, la foto", "the map, the photo"),
                _f("es-ser-estar-basic", "ser vs estar, first pass", "ser = what it is; estar = how/where it is.", "Soy belga. Estoy en Madrid.", "I'm Belgian. I'm in Madrid."),
                _f("es-gustar", "The gustar pattern", "Things please you: me gusta el café - the 'subject' comes last.", "Me gustan los idiomas.", "I like languages."),
                _f("es-questions-negation", "Questions & negation", "¿…? wraps questions; no before the verb; double negatives are correct.", "No veo nada.", "I don't see anything."),
            ],
            "A2": [
                _f("es-indefinido", "Pretérito indefinido", "Completed events: hablé, comí, viví - plus the irregular spine (fui, tuve, hice).", "Ayer fui al mercado.", "Yesterday I went to the market."),
                _f("es-imperfecto", "Pretérito imperfecto", "Habits and scenery in the past: hablaba, comía.", "De niño jugaba en la calle.", "As a child I used to play in the street."),
                _f("es-ir-a", "ir a + infinitive", "The everyday future.", "Voy a llamarte esta noche.", "I'm going to call you tonight."),
                _f("es-reflexives", "Reflexive daily verbs", "levantarse, ducharse - the pronoun is part of the routine.", "Me levanto a las siete.", "I get up at seven."),
                _f("es-clitics-intro", "Object pronouns", "lo/la/los/las and le/les before the verb or glued to the infinitive.", "¿El libro? Lo leí ayer.", "The book? I read it yesterday."),
            ],
            "B1": [
                _f("es-aspect-contrast", "Indefinido vs imperfecto", "Event vs backdrop in one sentence - the heart of Spanish narration.", "Llovía cuando llegué.", "It was raining when I arrived."),
                _f("es-subj-present", "Present subjunctive, first triggers", "querer que, esperar que, para que flip the verb: hable, coma, viva.", "Quiero que vengas.", "I want you to come."),
                _f("es-commands", "Commands", "Affirmative tú (¡habla!) vs negative (¡no hables!) - the negative borrows the subjunctive.", "Dímelo, pero no se lo digas a él.", "Tell me, but don't tell him."),
                _f("es-por-para", "por vs para", "por = cause, route, exchange; para = goal, destination, deadline.", "Gracias por venir; esto es para ti.", "Thanks for coming; this is for you."),
                _f("es-se-impersonal", "Impersonal & passive se", "se habla español, se venden casas - Spanish's favorite passive.", "Aquí se come bien.", "One eats well here."),
            ],
            "B2": [
                _f("es-subj-full", "Subjunctive across the board", "Doubt, emotion, denial, indefinite antecedents: busco a alguien que sepa ruso.", "No creo que sea verdad.", "I don't think it's true."),
                _f("es-subj-past", "Past subjunctive & si-clauses", "si tuviera tiempo, iría - the unreal past in -ra.", "Si pudiera, viajaría más.", "If I could, I'd travel more."),
                _f("es-conjecture", "Future & conditional of conjecture", "¿Qué hora será? = I wonder what time it is.", "Serán las diez.", "It must be around ten."),
                _f("es-clitic-stacking", "Clitic stacking", "le + lo → se lo: Se lo di ayer.", "¿La carta? Se la mandé.", "The letter? I sent it to her."),
                _f("es-passive-ser", "ser-passive vs se-passive", "Written Spanish tolerates fue construido; speech prefers se construyó.", "El puente fue construido en 1920.", "The bridge was built in 1920."),
            ],
            "C1": [
                _f("es-subj-nuance", "Indicative/subjunctive minimal pairs", "aunque llueve (it is raining) vs aunque llueva (even if it rains).", "Aunque sea tarde, iré.", "Even if it's late, I'll go."),
                _f("es-discourse", "Discourse markers", "o sea, en cambio, por lo tanto, sin embargo - the hinges of argument.", "No vino; sin embargo, avisó.", "He didn't come; however, he let us know."),
                _f("es-voseo", "Regional systems", "vos tenés, ustedes as universal plural - reading the map of Spanish.", "¿Vos qué pensás?", "What do you think? (Río de la Plata)"),
                _f("es-relative-advanced", "Advanced relatives", "cuyo, el cual, lo que - precision joinery for formal prose.", "La casa cuyas ventanas ves…", "The house whose windows you see..."),
            ],
            "C2": [
                _f("es-estilo", "Word order as style", "Fronting for focus: Eso no lo sabía yo.", "A María la vi ayer.", "María - I saw her yesterday (topicalized)."),
                _f("es-regimes", "Prepositional regimes", "soñar con, contar con, empeñarse en - verb+preposition idiom.", "Cuento contigo.", "I'm counting on you."),
                _f("es-literary", "Literary tenses", "The -se subjunctive (hablase) and archaic futures of formal prose.", "Si él lo supiese…", "If he but knew it..."),
                _f("es-idiom", "Idiom & register range", "From refranes to bureaucratic Spanish - matching voice to situation.", "A quien madruga, Dios le ayuda.", "The early bird catches the worm."),
            ],
        },
        "challenges": [
            "ser vs estar has no English equivalent - sort by essence vs state, then learn the exceptions.",
            "Two past aspects: choosing indefinido vs imperfecto is the classic plateau.",
            "The subjunctive is everyday grammar, not an advanced ornament.",
            "gustar-type verbs invert your instinct: the thing liked is the subject.",
            "The personal a has no translation and is easy to drop by accident.",
        ],
        "phonology": [
            "Five pure vowels, never reduced - resist the English schwa.",
            "r vs rr contrast: pero (but) vs perro (dog); the trill is learnable with practice.",
            "b and v are the same sound; j/g before e,i is a rough /x/.",
            "Stress is rule-based; the written accent marks the exceptions (está, sábado).",
            "In much of Spain, z/ce/ci = /θ/; in Latin America it merges with s (seseo).",
        ],
    },

    # ----------------------------------------------------------------- RUSSIAN
    "ru": {
        "overview": (
            "A Slavic language where endings do the work word order does in English: six cases "
            "mark who does what to whom, freeing word order for emphasis. Verbs come in aspect "
            "pairs - every action is framed as ongoing or completed. No articles, no present-tense "
            "'to be': Russian trims what context can carry."
        ),
        "typology": [
            "Highly fusional: one ending can encode case + number + gender at once.",
            "Free word order driven by information structure (old info first, news last).",
            "Aspect (imperfective/perfective) is as central as tense.",
            "Cyrillic script; spelling is largely regular once stress is known.",
        ],
        "pillars": [
            {"id": "case-system", "title": "Six cases", "summary": "Nominative, accusative, genitive, dative, instrumental, prepositional - nouns, adjectives, and pronouns all decline. Cases are the sentence's wiring."},
            {"id": "aspect", "title": "Aspect pairs", "summary": "читать/прочитать: almost every verb is a pair. Imperfective = process/habit; perfective = completed result. Choosing aspect is choosing meaning."},
            {"id": "motion-verbs", "title": "Verbs of motion", "summary": "идти vs ходить (one-way vs habitual, on foot), ехать vs ездить (by vehicle) - then a prefix system (при-, у-, вы-…) that maps trajectories."},
            {"id": "zero-forms", "title": "What Russian omits", "summary": "No articles; no present-tense 'to be' (Я студент); possession via у меня есть. Absence is grammatical."},
            {"id": "sound-system", "title": "Hard & soft", "summary": "Nearly every consonant has a palatalized twin (мат/мать differ by softness alone). Stress is mobile and unmarked, and unstressed о reduces to /a/."},
            {"id": "info-structure", "title": "Word order as emphasis", "summary": "Я тебе книгу дал vs Книгу я тебе дал - same facts, different spotlight. The end of the sentence carries the news."},
        ],
        "roadmap": {
            "A1": [
                _f("ru-cyrillic", "Cyrillic fluency", "33 letters; the traps are the look-alikes: в=/v/, н=/n/, р=/r/, с=/s/.", "Вот мой дом.", "Here is my house."),
                _f("ru-gender", "Three genders by ending", "-consonant = masculine, -а/-я = feminine, -о/-е = neuter (mostly).", "стол, книга, окно", "table, book, window"),
                _f("ru-present-conj", "Present tense, two conjugations", "-е- type (читаю, читаешь) vs -и- type (говорю, говоришь).", "Я читаю, а ты говоришь.", "I read and you talk."),
                _f("ru-prepositional", "Prepositional of place", "в/на + -е for location: в Москве, на работе.", "Я живу в Бельгии.", "I live in Belgium."),
                _f("ru-u-menya", "Possession: у меня есть", "'By me there is' - Russian's have.", "У меня есть вопрос.", "I have a question."),
            ],
            "A2": [
                _f("ru-accusative", "Accusative & animacy", "Direct objects; animate masculines borrow the genitive: вижу брата.", "Я вижу маму и дом.", "I see mom and the house."),
                _f("ru-genitive", "Genitive basics", "Possession, absence (нет времени), and after numbers/quantities.", "У меня нет денег.", "I have no money."),
                _f("ru-past", "Past tense by gender", "One л-form that agrees: он читал, она читала, они читали.", "Она уже ушла.", "She has already left."),
                _f("ru-dative-exp", "Dative experiencer", "Feelings happen to you: мне нравится, мне холодно, мне 30 лет.", "Мне нравится этот город.", "I like this city."),
                _f("ru-future-budu", "Future with буду", "буду + imperfective infinitive - first future, before aspect nuance.", "Завтра я буду работать.", "Tomorrow I will work / be working."),
            ],
            "B1": [
                _f("ru-aspect-core", "Aspect pairs in earnest", "делал (was doing/did habitually) vs сделал (got it done) - learn verbs in pairs.", "Я писал письмо и наконец написал его.", "I was writing the letter and finally finished it."),
                _f("ru-instrumental", "Instrumental", "Tool, accompaniment (с сестрой), professions (работаю врачом).", "Пишу карандашом.", "I write with a pencil."),
                _f("ru-motion-basic", "Motion: идти/ходить, ехать/ездить", "One direction now vs there-and-back/habit.", "Сейчас я иду в парк; я хожу туда каждый день.", "I'm walking to the park now; I go there every day."),
                _f("ru-imperative", "Imperative & aspect", "Пиши! (keep at it) vs Напиши! (get it written) - aspect changes the command.", "Садитесь, пожалуйста!", "Please sit down."),
                _f("ru-dative-full", "Dative in full", "Indirect objects, к + D, по + D; помогать and нужен govern dative.", "Мне нужно помочь брату.", "I need to help my brother."),
            ],
            "B2": [
                _f("ru-motion-prefixed", "Prefixed motion verbs", "при- arrive, у- leave, вы- exit, за- drop by: prefixes draw the trajectory.", "Он пришёл, но скоро ушёл.", "He arrived but soon left."),
                _f("ru-conditional", "Conditional with бы", "past form + бы: если бы я знал… - unreality in two particles.", "Если бы я знал, я бы пришёл.", "If I had known, I would have come."),
                _f("ru-participles-rec", "Participles (recognition)", "читающий, прочитанный - bookish verb-adjectives you must read fluently.", "человек, читающий газету", "a person reading a newspaper"),
                _f("ru-numeral-case", "Numbers govern case", "2-4 take genitive singular, 5+ genitive plural: два часа, пять часов.", "три больших окна", "three big windows"),
                _f("ru-verbal-adverbs", "Verbal adverbs (recognition)", "читая, прочитав - 'while reading / having read' in one word.", "Прочитав письмо, она улыбнулась.", "Having read the letter, she smiled."),
            ],
            "C1": [
                _f("ru-participles-active", "Participles in production", "Deploy participial clauses to compress relative clauses in writing.", "документы, подписанные директором", "documents signed by the director"),
                _f("ru-aspect-nuance", "Aspect in infinitive & negation", "не надо звонить vs не надо позвонить - negation prefers imperfective.", "Не забудь позвонить!", "Don't forget to call!"),
                _f("ru-particles", "Particles же, ведь, ли", "The flavor particles: же insists, ведь appeals to shared knowledge, ли asks formally.", "Ты же обещал!", "But you promised!"),
                _f("ru-subordination", "Complex subordination", "то, что…; тот, кто…; чтобы + past for purpose/wish.", "Я хочу, чтобы ты остался.", "I want you to stay."),
            ],
            "C2": [
                _f("ru-word-order-style", "Word order as stylistics", "Inversion and end-focus as literary devices; poetry's freedom.", "Тихо падал снег.", "Softly fell the snow."),
                _f("ru-bookish", "Bookish syntax", "Chained participial and gerundial constructions of formal prose.", "Учитывая вышесказанное…", "Considering the above..."),
                _f("ru-case-idiom", "Idiomatic case government", "ждать письма (G) vs ждать маму (A); бояться темноты - government as idiom.", "Он боится высоты.", "He is afraid of heights."),
                _f("ru-register", "Register range", "From canceled-formality разговорный to bureaucratic канцелярит - and what to avoid.", "Доводим до вашего сведения…", "We hereby inform you... (officialese)"),
            ],
        },
        "challenges": [
            "Cases replace word-order logic - trust endings, not position.",
            "Aspect has no English mirror: learn verbs as pairs from day one.",
            "Mobile stress changes vowel sounds and is not written (замо́к vs за́мок).",
            "Motion verbs multiply 'to go' into a small system.",
            "No articles: definiteness moves into word order and context.",
        ],
        "phonology": [
            "Unstressed о sounds like /a/: молоко = /məlɐko/.",
            "Hard vs soft consonants distinguish words: брат (brother) vs брать (to take).",
            "ы is a back /i/ with no English/Dutch equivalent - practice был vs бил.",
            "Final consonants devoice: год sounds like 'got'.",
            "ш vs щ, ц vs ч - four sounds English folds into two.",
        ],
    },

    # ----------------------------------------------------------------- MANDARIN
    "zh": {
        "overview": (
            "An analytic language: words never change form. Grammar lives in word order, "
            "particles, and context; tones are part of every word's identity. Learning Mandarin "
            "is learning a new axis (tone), a new script (characters), and a new habit: marking "
            "aspect and result instead of tense."
        ),
        "typology": [
            "No conjugation, no plurals, no gender, no case - zero inflection.",
            "SVO base order with topic-prominence: the topic comes first, comment follows.",
            "All modifiers precede what they modify (big-red-that book → 那本大红书).",
            "Four lexical tones plus neutral; characters combine semantic + phonetic parts.",
        ],
        "pillars": [
            {"id": "tones", "title": "Tones", "summary": "mā má mǎ mà are four different words. Tone is not intonation on top of the word - it is the word. Sandhi: 3rd+3rd → 2nd+3rd; 不 and 一 shift by context."},
            {"id": "word-order", "title": "Order & topic", "summary": "Time before place before verb; modifiers always in front; topics fronted: 北京我去过 'Beijing, I've been'."},
            {"id": "aspect-particles", "title": "Aspect, not tense", "summary": "了 (completed/changed), 过 (experienced), 着 (ongoing state) mark how an event relates to time - the when comes from time words."},
            {"id": "measure-words", "title": "Measure words", "summary": "Numbers never touch nouns directly: 三本书 'three volume book'. 个 is the default; good Mandarin picks the right classifier."},
            {"id": "complements", "title": "Complement system", "summary": "Verbs bolt on results and directions: 听懂 hear-understand, 跑出去 run-exit-go. Potential infix: 听得懂 / 听不懂 can/can't understand."},
            {"id": "characters", "title": "Characters", "summary": "Most characters = semantic radical + phonetic hint (妈 = 女 woman + 马 mǎ). Learn components, not strokes, and characters become a network."},
        ],
        "roadmap": {
            "A1": [
                _f("zh-pinyin-tones", "Pinyin & the four tones", "Master the tone pairs early; 买 mǎi (buy) vs 卖 mài (sell) is rent money.", "妈妈骂马。", "Mom scolds the horse. (tone drill)"),
                _f("zh-svo", "Basic SVO with 是/有", "我是学生, 我有时间 - no conjugation, ever.", "我是比利时人。", "I am Belgian."),
                _f("zh-questions", "Questions with 吗 / 呢", "Statement + 吗 = yes/no question; 呢 bounces the question back.", "你好吗？我很好，你呢？", "How are you? I'm fine, and you?"),
                _f("zh-measure-ge", "Numbers + 个", "三个人, 两个问题 - the universal classifier first.", "我有两个哥哥。", "I have two older brothers."),
                _f("zh-negation", "不 vs 没", "不 negates present/future/habits; 没(有) negates completed events and having.", "我不喝咖啡；昨天我没喝。", "I don't drink coffee; yesterday I didn't."),
            ],
            "A2": [
                _f("zh-le-completed", "了 for completion/change", "V + 了 = done; sentence-final 了 = new situation. Don't translate as past tense.", "我吃了饭。/ 下雨了。", "I ate. / It's (started) raining."),
                _f("zh-guo", "过 for experience", "Ever done it in your life: 我去过中国.", "你吃过火锅吗？", "Have you ever had hotpot?"),
                _f("zh-bi", "Comparison with 比", "A 比 B + adjective - no 'more' needed: 他比我高.", "今天比昨天冷。", "Today is colder than yesterday."),
                _f("zh-auxiliaries", "会 / 能 / 可以", "Learned skill vs capability vs permission - three flavors of 'can'.", "我会游泳，但今天不能游。", "I can swim, but today I can't."),
                _f("zh-zai-location", "在 + place before verb", "Location comes before the action: 我在家工作.", "他在北京学习。", "He studies in Beijing."),
            ],
            "B1": [
                _f("zh-result-complements", "Result complements", "V + 完/好/到/错: 听懂 hear-and-understand, 找到 find-successfully.", "我看完了这本书。", "I finished reading this book."),
                _f("zh-direction-complements", "Direction complements", "上来/下去/出来… trajectories glued to verbs: 走进来 walk in (toward me).", "他跑出去了。", "He ran out."),
                _f("zh-ba", "The 把 construction", "Fronts the object to focus on what happened to it: 把门关上.", "请把手机关掉。", "Please turn off your phone."),
                _f("zh-bei", "Passive with 被", "被 marks the done-to: 他被老师批评了 - often for mishaps.", "我的伞被人拿走了。", "My umbrella was taken."),
                _f("zh-shi-de", "是…的 focus", "Spotlights time/place/manner of a known event: 我是坐飞机来的.", "你是什么时候到的？", "When was it that you arrived?"),
            ],
            "B2": [
                _f("zh-potential", "Potential complements", "得/不 inside the complement: 听得懂 / 听不懂, 买得起 / 买不起.", "这么多菜我吃不完。", "I can't finish this much food."),
                _f("zh-le-nuance", "了 in two positions", "Verb-了 vs sentence-了 vs both: 我学了三年了 - still learning.", "我在这儿住了两年了。", "I've been living here for two years (and still am)."),
                _f("zh-conjunction-pairs", "Paired conjunctions", "虽然…但是, 因为…所以, 不但…而且 - both halves appear.", "虽然贵，但是值得。", "Although expensive, it's worth it."),
                _f("zh-yue", "越…越 and 一…就", "The more...the more; as soon as: 越学越有意思.", "我一到家就给你打电话。", "I'll call you as soon as I get home."),
                _f("zh-duration-order", "Duration & frequency order", "V + time-amount: 学了两个小时 - duration follows the verb.", "我等了你半个小时。", "I waited for you half an hour."),
            ],
            "C1": [
                _f("zh-written-register", "书面语: written register", "之, 于, 而, 以 - classical function words that formal prose still runs on.", "总而言之，此事关系重大。", "In short, this matter is of great importance."),
                _f("zh-chengyu", "成语 in action", "Four-character idioms as compressed stories: 入乡随俗 'enter village, follow customs'.", "我们应该入乡随俗。", "We should do as the locals do."),
                _f("zh-topicalization", "Topic chains", "Drop what's understood; let topics govern several clauses - Mandarin's paragraph logic.", "那家饭馆，菜好吃，价钱也不贵。", "That restaurant - food's great, prices aren't high either."),
                _f("zh-discourse", "Discourse connectors", "其实, 反正, 毕竟, 难道 - stance in a single word.", "难道你不知道吗？", "Don't tell me you didn't know?"),
            ],
            "C2": [
                _f("zh-classical-echo", "Classical echoes", "Reading 文言 flavor in headlines and formal writing: 无需, 即可, 者.", "凭票即可入场。", "Admission upon ticket presentation."),
                _f("zh-parallelism", "Parallel prose & rhythm", "Four-and-four rhythms, antithesis - what makes writing 地道 (authentic).", "山高水长，来日方长。", "Mountains high, rivers long - there is yet time ahead."),
                _f("zh-xiehouyu", "歇后语 & humor", "Two-part allegorical sayings: 泥菩萨过江——自身难保.", "这真是泥菩萨过江。", "A clay Buddha crossing the river - can't even save itself."),
                _f("zh-register-range", "Full register range", "From 网络用语 internet slang to bureaucratic 公文 - tuning voice to context.", "此致敬礼。", "Respectfully yours (formal letter close)."),
            ],
        },
        "challenges": [
            "Tones are lexical: a wrong tone is a wrong word, not an accent.",
            "了 is aspect and change-of-state, never a simple past-tense marker.",
            "Measure words are obligatory and noun-specific.",
            "Characters decouple sound from writing - build the radical network early.",
            "把 and 被 reorganize the sentence in ways English never does.",
        ],
        "phonology": [
            "Third tone is low-dipping in isolation but usually just low in speech.",
            "Tone sandhi: 3rd+3rd → 2nd+3rd (你好 = ní hǎo); 不 and 一 change tone by context.",
            "zh/ch/sh (retroflex) vs j/q/x (palatal) - two rows English merges.",
            "ü after j/q/x is written u but still sounds /y/ (去 qù).",
            "-n vs -ng finals distinguish words: 反 fǎn vs 房 fáng.",
        ],
    },
}


def grammar_profile(language_code: str) -> dict:
    """Public payload for the Grammar view."""
    lang = normalize_language_code(language_code)
    data = GRAMMAR[lang]
    return {"language": lang, **data}


def feature_index(language_code: str) -> Dict[str, dict]:
    """feature id -> {name, level} for validation and progress tracking."""
    lang = normalize_language_code(language_code)
    index: Dict[str, dict] = {}
    for level, features in GRAMMAR[lang]["roadmap"].items():
        for feature in features:
            index[feature["id"]] = {"name": feature["name"], "level": level}
    return index


def prompt_brief(language_code: str, level: str) -> str:
    """Compact grammar brief injected into generation prompts.

    Tells the model which structures the learner already owns (at/below level)
    and which are the current targets (at level), so generated text sits at i+1.
    """
    lang = normalize_language_code(language_code)
    data = GRAMMAR[lang]
    level = level if level in CEFR_ORDER else "A2"
    position = CEFR_ORDER.index(level)
    known: List[str] = []
    for lv in CEFR_ORDER[:position]:
        known += [f["name"] for f in data["roadmap"][lv]]
    targets = [f"{f['id']}: {f['name']} - {f['tip']}" for f in data["roadmap"][level]]
    lines = [
        f"Language typology: {data['overview']}",
        "Structures the learner already knows (use freely): " + (", ".join(known) if known else "only the most basic patterns"),
        "TARGET structures for this level (weave several in naturally):",
        *[f"  - {t}" for t in targets],
        "Do not use structures from higher levels except as unavoidable fixed chunks.",
    ]
    return "\n".join(lines)
