import os
import logging
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, scrolledtext, colorchooser
import threading
import random
import json
from datetime import datetime
from pathlib import Path


from class_tutor import (
    AUDIO_DIR,
    DATA_DIR,
    SPANISH_AUDIO_DIR,
    TEST_RESULTS_DIR,
    VOCAB_SOURCE_PATH,
    LANGUAGE_PROFILES,
    Tutor,
    BilingualDict,
    get_language_profile,
    normalize_language_code,
)
from class_scroll import ScrollableFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("PGB_Model_Log.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = SPANISH_AUDIO_DIR
VOCAB_JSON = VOCAB_SOURCE_PATH

class TutorGUI:
    """
    A graphical user interface for the Language Tutor using Tkinter.
    """

    def __init__(self, root):
        self.tutor = Tutor()
        self.root = root
        self.root.title("Language Tutor")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        # make all content scrollable
        container = ScrollableFrame(self.root)
        container.pack(fill="both", expand=True)
        self.content = container.scrollable_frame

        # keep your existing attributes
        self.current_bg_color = "#f0f0f0"
        self.current_fg_color = "#000000"
        self.current_font_size = 14
        self.min_font_size = 12
        self.max_font_size = 30
        self.last_bilingual_content = None
        self.current_language_code = "es"
        self.current_voice_name = get_language_profile("es").voices[get_language_profile("es").default_voice_label]

        # build UI into self.content instead of self.root
        self.create_menu()               # menu bar stays on root
        self.create_widgets(self.content)
        self.load_vocabulary()
        self.create_vocab_selector(self.content)

        self.last_bilingual_content = None  # Store most recent bilingual content for testing

        self.audio_files_en = {}
        self.audio_files_target = {}

    def create_menu(self):
        """
        Creates the menu bar with options for color configuration.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        appearance_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Appearance", menu=appearance_menu)
        appearance_menu.add_command(label="Change Colors", command=self.change_colors)

    def change_colors(self):
        """
        Opens color chooser dialogs to allow the user to select background and text colors.
        Applies the selected colors to the GUI.
        """
        # Choose Background Color
        bg_color = colorchooser.askcolor(title="Choose Background Color", initialcolor=self.current_bg_color)
        if bg_color[1]:
            self.current_bg_color = bg_color[1]
            self.apply_colors()

        # Choose Text Color
        fg_color = colorchooser.askcolor(title="Choose Text Color", initialcolor=self.current_fg_color)
        if fg_color[1]:
            self.current_fg_color = fg_color[1]
            self.apply_colors()

    def apply_colors(self):
        """
        Applies the selected background and text colors to all relevant widgets.
        """
        # Apply to root window
        self.root.configure(bg=self.current_bg_color)

        # Apply to all frames and widgets
        for widget in self.root.winfo_children():
            self.apply_color_recursive(widget)

    def apply_color_recursive(self, widget):
        """
        Recursively applies colors to widgets.
        """
        try:
            if isinstance(widget, (ttk.LabelFrame, ttk.Frame)):
                widget.configure(style="Custom.TLabelframe")
                widget.configure(borderwidth=2, relief="groove")
            elif isinstance(widget, (ttk.Label, ttk.Button, ttk.Radiobutton)):
                widget.configure(style="Custom.TLabel")
            elif isinstance(widget, ttk.Treeview):
                widget.configure(style="Custom.Treeview")
            elif isinstance(widget, tk.Toplevel):
                widget.configure(bg=self.current_bg_color)
                for child in widget.winfo_children():
                    self.apply_color_recursive(child)
            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.configure(bg=self.current_bg_color, fg=self.current_fg_color)
        except tk.TclError:
            pass  # Some widgets may not support certain configurations

        # Recurse into child widgets
        for child in widget.winfo_children():
            self.apply_color_recursive(child)

        # Define custom styles
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.TLabel", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.Treeview", background=self.current_bg_color, foreground=self.current_fg_color,
                        fieldbackground=self.current_bg_color)
        style.map("Custom.Treeview", background=[('selected', '#ececec')], foreground=[('selected', '#000000')])

    def create_widgets(self, parent):
        # Style Configuration
        style = ttk.Style()
        style.configure("TFrame", background=self.current_bg_color)
        style.configure("TButton", font=("Helvetica", self.current_font_size))
        style.configure("TLabel", font=("Helvetica", self.current_font_size),
                        background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Header.TLabel", font=("Helvetica", self.current_font_size + 2, "bold"),
                        background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Treeview.Heading", font=("Helvetica", self.current_font_size, "bold"),
                        background="#d3d3d3", foreground="black")
        style.configure("Treeview", font=("Helvetica", self.current_font_size),
                        rowheight=self.current_font_size + 10)

        style.configure("Custom.TLabelframe", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.TLabel", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.Treeview", background=self.current_bg_color, foreground=self.current_fg_color,
                        fieldbackground=self.current_bg_color)
        style.map("Custom.Treeview", background=[('selected', '#ececec')], foreground=[('selected', '#000000')])

        # Concept Input
        input_frame = ttk.LabelFrame(parent, text="Explore Vocabulary Across Languages",
                                     padding=(20, 10), style="Custom.TLabelframe")
        input_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(input_frame, text="Enter instruction or concept of choice:", style="Header.TLabel") \
            .grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.concept_entry = ttk.Entry(input_frame, width=50,
                                       font=("Helvetica", self.current_font_size))
        self.concept_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        ttk.Label(input_frame, text="Number of items to generate:", style="Header.TLabel") \
            .grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.num_items_entry = ttk.Entry(input_frame, width=10,
                                         font=("Helvetica", self.current_font_size))
        self.num_items_entry.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        self.num_items_entry.insert(0, "10")

        ttk.Label(input_frame, text="Select difficulty level:", style="Header.TLabel") \
            .grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.difficulty_var = tk.StringVar(value='intermediate')
        difficulty_frame = ttk.Frame(input_frame, style="Custom.TLabelframe")
        difficulty_frame.grid(row=2, column=1, padx=5, pady=10, sticky="w")
        for level in ['Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert']:
            ttk.Radiobutton(difficulty_frame, text=level, variable=self.difficulty_var,
                            value=level.lower(), style="Custom.TRadiobutton") \
                .pack(side="left", padx=5)

        ttk.Label(input_frame, text="Choose target language:", style="Header.TLabel") \
            .grid(row=3, column=0, padx=5, pady=10, sticky="w")
        self.language_var = tk.StringVar(value='es')
        self.language_combo = ttk.Combobox(
            input_frame,
            values=[f"{profile.display} ({code})" for code, profile in LANGUAGE_PROFILES.items()],
            state="readonly",
            width=28,
        )
        self.language_combo.set("Spanish (es)")
        self.language_combo.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        self.language_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        ttk.Label(input_frame, text="Voice:", style="Header.TLabel") \
            .grid(row=4, column=0, padx=5, pady=10, sticky="w")
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(input_frame, textvariable=self.voice_var, state="readonly", width=28)
        self.voice_combo.grid(row=4, column=1, padx=5, pady=10, sticky="w")
        self.update_voice_options("es")

        self.learn_button = tk.Button(
            input_frame, text="Learn Concept",
            bg="#007BFF", fg="black",
            activebackground="#0056b3", activeforeground="white",
            font=("Helvetica", self.current_font_size),
            padx=10, pady=5, command=self.learn_concept
        )
        self.learn_button.grid(row=5, column=1, padx=5, pady=20, sticky="e")

        # Font Controls
        font_frame = ttk.LabelFrame(parent, text="Font Controls",
                                    padding=(20, 10), style="Custom.TLabelframe")
        font_frame.pack(fill="x", padx=20, pady=10)
        self.decrease_font_button = ttk.Button(font_frame, text="--smaller--",
                                               command=self.decrease_font_size, style="Custom.TButton")
        self.decrease_font_button.pack(side="left", padx=5, pady=5)
        self.font_size_label = ttk.Label(font_frame,
                                         text=f"Font Size: {self.current_font_size}",
                                         style="Header.TLabel")
        self.font_size_label.pack(side="left", padx=10)
        self.increase_font_button = ttk.Button(font_frame, text="++bigger++",
                                               command=self.increase_font_size, style="Custom.TButton")
        self.increase_font_button.pack(side="left", padx=5, pady=5)

        # Bilingual Translations
        display_frame = ttk.LabelFrame(parent, text="Bilingual Translations",
                                       padding=(20, 10), style="Custom.TLabelframe")
        display_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("English", "Target Language", "Play")
        self.translations_tree = ttk.Treeview(display_frame, columns=columns,
                                              show='headings', style="Custom.Treeview")
        for col in columns:
            self.translations_tree.heading(col, text=col)
            if col == "Play":
                # fixed-width, no stretch
                self.translations_tree.column(col, width=150, anchor="center", stretch=False)
            else:
                # auto-width, allow stretch
                self.translations_tree.column(col, anchor="center", stretch=True)

        self.translations_tree.pack(fill="both", expand=True, side="left", padx=(0, 10), pady=10)
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL,
                                  command=self.translations_tree.yview)
        scrollbar.pack(side="right", fill="y", pady=10)
        self.translations_tree.configure(yscroll=scrollbar.set)

        # Audio Controls
        audio_frame = ttk.LabelFrame(parent, text="Audio Controls",
                                     padding=(20, 10), style="Custom.TLabelframe")
        audio_frame.pack(fill="x", padx=20, pady=10)
        self.play_all_button = ttk.Button(audio_frame, text="Play All Audio",
                                          command=self.play_all_audio,
                                          state='disabled', style="Custom.TButton")
        self.play_all_button.pack(side="left", padx=5, pady=5)

        # Testing Mode
        test_frame = ttk.LabelFrame(parent, text="Testing Mode",
                                    padding=(20, 10), style="Custom.TLabelframe")
        test_frame.pack(fill="x", padx=20, pady=10)
        self.start_verbal_test_button = ttk.Button(test_frame,
                                                   text="Start Test: type 1 - orthographic",
                                                   command=self.start_test_verbal,
                                                   style="Custom.TButton")
        self.start_verbal_test_button.pack(side="left", padx=5, pady=5)
        self.start_audio_test_button = ttk.Button(test_frame,
                                                  text="Start Test: type 2 - phonologic",
                                                  command=self.start_test_audio,
                                                  style="Custom.TButton")
        self.start_audio_test_button.pack(side="left", padx=5, pady=5)

        # Logs
        log_frame = ttk.LabelFrame(parent, text="Logs",
                                   padding=(20, 10), style="Custom.TLabelframe")
        log_frame.pack(fill="x", padx=20, pady=10)

    def selected_language_code(self):
        raw = self.language_combo.get() if hasattr(self, "language_combo") else self.language_var.get()
        if "(" in raw and ")" in raw:
            raw = raw.split("(")[-1].split(")")[0]
        code = normalize_language_code(raw)
        self.language_var.set(code)
        self.current_language_code = code
        return code

    def update_voice_options(self, language_code):
        profile = get_language_profile(language_code)
        voices = list(profile.voices.keys())
        self.voice_combo.configure(values=voices)
        if self.voice_var.get() not in voices:
            self.voice_var.set(profile.default_voice_label)
        self.current_voice_name = profile.voices[self.voice_var.get()]

    def on_language_change(self, _event=None):
        code = self.selected_language_code()
        self.update_voice_options(code)
        profile = get_language_profile(code)
        self.translations_tree.heading("Target Language", text=profile.display)
        self.update_font_family_for_language(code)

    def selected_voice_name(self):
        profile = get_language_profile(self.selected_language_code())
        return profile.voices.get(self.voice_var.get(), profile.voices[profile.default_voice_label])

    def update_font_family_for_language(self, language_code):
        profile = get_language_profile(language_code)
        try:
            available = set(tkfont.families(self.root))
        except Exception:
            available = set()
        family = "Helvetica"
        for candidate in profile.preferred_fonts:
            if not available or candidate in available:
                family = candidate
                break
        try:
            style = ttk.Style()
            style.configure("Treeview", font=(family, self.current_font_size), rowheight=self.current_font_size + 10)
            style.configure("Treeview.Heading", font=(family, self.current_font_size, "bold"))
        except Exception as e:
            logger.warning("Could not update language font family: %s", e)

    def increase_font_size(self):
        """
        Increases the font size of the Treeview and updates the row height.
        """
        if self.current_font_size < self.max_font_size:
            self.current_font_size += 1
            self.update_font_size()
        else:
            messagebox.showinfo("Font Size", f"Maximum font size of {self.max_font_size} reached.")

    def decrease_font_size(self):
        """
        Decreases the font size of the Treeview and updates the row height.
        """
        if self.current_font_size > self.min_font_size:
            self.current_font_size -= 1
            self.update_font_size()
        else:
            messagebox.showinfo("Font Size", f"Minimum font size of {self.min_font_size} reached.")

    def update_font_size(self):
        """
        Updates the font size of the Treeview and adjusts the row height accordingly.
        """
        try:
            style = ttk.Style()
            style.configure("TButton", font=("Helvetica", self.current_font_size))
            style.configure("TLabel", font=("Helvetica", self.current_font_size))
            style.configure("Header.TLabel", font=("Helvetica", self.current_font_size + 2, "bold"))
            style.configure("Treeview.Heading", font=("Helvetica", self.current_font_size, "bold"))
            style.configure("Treeview", font=("Helvetica", self.current_font_size),
                            rowheight=self.current_font_size + 10)
            self.font_size_label.config(text=f"Font Size: {self.current_font_size}")
            logger.info(f"Font size updated to {self.current_font_size}.")
        except Exception as e:
            logger.error(f"Error updating font size: {e}")

    def learn_concept(self):
        """
        Called when the user clicks “Learn Concept.”
        Records that this was an LLM‐generated run (stores prompt),
        and clears any previously selected topic.
        """
        concept = self.concept_entry.get().strip()
        language = self.selected_language_code()
        voice_name = self.selected_voice_name()
        num_items_str = self.num_items_entry.get().strip()
        difficulty_level = self.difficulty_var.get().strip()

        if not concept:
            messagebox.showwarning("Input Error", "Please enter a concept to learn about.")
            return

        # record that we're using an LLM prompt, clear any topic
        self.last_concept_prompt = concept
        self.current_topic = None

        if num_items_str:
            if not num_items_str.isdigit() or int(num_items_str) <= 0:
                messagebox.showwarning("Input Error", "Please enter a valid positive integer for the number of items.")
                return
            num_items = int(num_items_str)
        else:
            num_items = 10
            logger.info(f"Number of items not provided. Defaulting to {num_items}.")
            messagebox.showinfo("Default Number of Items", "Defaulting to 10.")

        self.learn_button.config(state='disabled')
        self.play_all_button.config(state='disabled')
        self.translations_tree.delete(*self.translations_tree.get_children())

        threading.Thread(
            target=self.process_learning,
            args=(concept, language, voice_name, num_items, difficulty_level),
            daemon=True
        ).start()

    def process_learning(self, concept, language, voice_name, num_items, difficulty_level):
        try:
            logger.info(
                "Starting concept generation for %r (%s, %d items, %s).",
                concept,
                language,
                num_items,
                difficulty_level,
            )
            bilingual_content = self.tutor.request_concept(concept, num_items, language, difficulty_level)
            if not bilingual_content.translated_words:
                self.root.after(0, lambda: self.display_message("Failed to retrieve content."))
                return

            logger.info("LLM returned %d item(s).", len(bilingual_content.translated_words))

            # Save content for testing later
            self.last_bilingual_content = bilingual_content

            # Determine target language display name
            target_lang_display = self.get_language_display(language)

            def update_table():
                self.translations_tree.heading("Target Language", text=target_lang_display)
                for eng, target in zip(bilingual_content.untranslated_words, bilingual_content.translated_words):
                    self.translations_tree.insert('', tk.END, values=(eng, target, "▶"))
                self.translations_tree.bind('<ButtonRelease-1>', self.on_tree_click)
                self.play_all_button.config(state='normal')

            self.root.after(0, update_table)

            # Prepare audio for all words
            self.prepare_audio_files(bilingual_content, language, voice_name)

        except Exception as e:
            logger.error(f"Error in process_learning: {e}")
            self.root.after(0, lambda: self.display_message("An error occurred. Check logs for details."))
        finally:
            self.root.after(0, lambda: self.learn_button.config(state='normal'))

    def get_language_display(self, language_code):
        return get_language_profile(language_code).display

    def prepare_audio_files(self, bilingual_content, language, voice_name=None):
        """
        Pre-generates audio files for target language words based on the selected language.
        Stores the file paths in a dictionary for easy access during playback.
        """
        self.audio_files_en = {}  # English audio not generated intentionally.
        self.audio_files_target = self.tutor.text_to_speech_batch(
            bilingual_content.translated_words,
            language,
            voice_name,
        )

    def on_tree_click(self, event):
        """
        Handles click events on the Treeview to play audio for individual words.
        """
        item = self.translations_tree.identify_row(event.y)
        column = self.translations_tree.identify_column(event.x)
        if not item:
            return

        values = self.translations_tree.item(item, 'values')
        if column == '#3':  # 'Play' column
            target_word = values[1]
            audio_file = self.audio_files_target.get(target_word)
            if audio_file and os.path.exists(audio_file):
                threading.Thread(target=self.play_audio_thread, args=(audio_file,), daemon=True).start()
            else:
                messagebox.showerror("Audio Error", f"No audio file found for '{target_word}'.")

    def play_audio_thread(self, audio_file):
        """
        Plays the specified audio file in a separate thread.
        """
        try:
            self.tutor.play_audio(audio_file)
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            messagebox.showerror("Playback Error", "Failed to play audio. Check logs for details.")

    def play_all_audio(self):
        """
        Plays all target language audio files sequentially.
        """
        threading.Thread(target=self.play_all_audio_thread, daemon=True).start()

    def play_all_audio_thread(self):
        """
        Plays all audio files in a separate thread in the desired sequence.
        """
        try:
            items = self.translations_tree.get_children()
            for item in items:
                values = self.translations_tree.item(item, 'values')
                target_word = values[1]
                audio_target = self.audio_files_target.get(target_word)
                if audio_target and os.path.exists(audio_target):
                    self.tutor.play_audio(audio_target)

            messagebox.showinfo("Audio Playback", "Finished playing all audio.")
        except Exception as e:
            logger.error(f"Error during all audio playback: {e}")
            messagebox.showerror("Playback Error", "Failed to play all audio. Check logs for details.")

    def display_message(self, message):
        messagebox.showinfo("Information", message)

    # ====== Testing Component Methods ======

    def start_test_verbal(self):
        """
        Initiates verbal test mode. Opens a new window for testing with test type set to 'verbal'.
        """
        self.start_test_common("verbal")

    def start_test_audio(self):
        """
        Initiates audio test mode. Opens a new window for testing with test type set to 'audio'.
        """
        self.start_test_common("audio")

    def start_test_common(self, test_mode):
        """
        Initiates test mode with the specified test_mode ("verbal" or "audio").
        Checks that bilingual content exists, then opens a new window for testing.
        """
        if not self.last_bilingual_content or not self.last_bilingual_content.untranslated_words:
            messagebox.showwarning("Test Error", "No content available for testing. Please learn a concept first.")
            return

        # Prepare test questions as a list of tuples (english, target)
        self.test_questions = list(zip(self.last_bilingual_content.untranslated_words,
                                       self.last_bilingual_content.translated_words))
        random.shuffle(self.test_questions)
        self.score = 0
        self.total_questions = len(self.test_questions)
        self.question_count = 0
        self.incorrect_items = []

        # Set the test mode for use in the testing interface
        self.test_mode = test_mode  # "verbal" or "audio"

        # Open test window
        self.test_window = tk.Toplevel(self.root)
        self.test_window.title("Test Mode")
        self.test_window.geometry("600x500")
        self.current_question = None
        self.create_test_widgets()
        self.show_next_question()

    def create_test_widgets(self):
        """
        Creates widgets for the test window including settings for the max number of options.
        The question sentence is displayed in italic.
        """
        self.test_frame = ttk.Frame(self.test_window, padding=20)
        self.test_frame.pack(fill="both", expand=True)

        # ----- Test Settings -----
        settings_frame = ttk.Frame(self.test_frame)
        settings_frame.pack(fill="x", pady=10)

        ttk.Label(settings_frame, text="Max Options:").pack(side="left", padx=5)
        self.max_display_entry = ttk.Entry(settings_frame, width=5)
        self.max_display_entry.insert(0, "5")
        self.max_display_entry.pack(side="left", padx=5)

        # ----- Question Label (in italic) -----
        self.question_label = ttk.Label(self.test_frame, text="", font=("Helvetica", 16, "italic"), wraplength=550)
        self.question_label.pack(pady=10)

        # If audio test mode, add a "Play Audio" button (it will be used per question)
        self.play_audio_button = ttk.Button(self.test_frame, text="Play Audio", command=self.play_current_audio)
        # Initially hidden; will be shown for audio test mode.
        if self.test_mode == "audio":
            self.play_audio_button.pack(pady=5)
        else:
            self.play_audio_button.pack_forget()

        # ----- Options Frame -----
        self.options_frame = ttk.Frame(self.test_frame)
        self.options_frame.pack(pady=10, fill="both", expand=True)

        # Variable to hold user's answer
        self.selected_option = tk.StringVar()

        # List to store radio button widgets for each question
        self.radio_buttons = []

        # ----- Submit Button -----
        self.submit_button = ttk.Button(self.test_frame, text="Submit Answer", command=self.check_answer)
        self.submit_button.pack(pady=10)

        # ----- Final Feedback (hidden until end) -----
        self.final_feedback_label = ttk.Label(self.test_frame, text="", font=("Helvetica", 14))
        self.final_feedback_label.pack(pady=10)

    def play_current_audio(self):
        """
        Plays the audio associated with the current question (used in audio test mode).
        """
        if self.current_question and self.test_mode == "audio":
            target_text = self.current_question[1]
            audio_file = self.audio_files_target.get(target_text)
            if audio_file and os.path.exists(audio_file):
                threading.Thread(target=self.play_audio_thread, args=(audio_file,), daemon=True).start()
            else:
                messagebox.showerror("Audio Error", "No audio available for this question.")

    def show_next_question(self):
        """
        Displays the next question in the test window.
        Saves results when the test is complete.
        """
        # Clear previous radio buttons
        for rb in self.radio_buttons:
            rb.destroy()
        self.radio_buttons.clear()
        self.selected_option.set("")
        self.final_feedback_label.config(text="")

        # Increment question count
        self.question_count += 1

        # Hide or show the play audio button based on test mode.
        if self.test_mode == "verbal":
            self.play_audio_button.pack_forget()
        else:
            self.play_audio_button.pack(pady=5)

        if not self.test_questions:
            # Test finished: hide submit button and options; display final feedback.
            percentage = (self.score / self.total_questions) * 100 if self.total_questions > 0 else 0
            feedback = (
                "Excellent job!" if percentage >= 80
                else "Good effort, keep practicing!" if percentage >= 50
                else "Needs more practice."
            )
            final_text = f"Final Score: {self.score}/{self.total_questions}\nFeedback: {feedback}"
            if self.incorrect_items:
                final_text += "\n\nIncorrect Items:\n"
                for order, question, correct, _ in sorted(self.incorrect_items, key=lambda x: x[0]):
                    final_text += f"{order}. {question} --> {correct}\n"

            self.question_label.config(text="Test Completed!")
            self.submit_button.pack_forget()
            self.options_frame.pack_forget()
            self.play_audio_button.pack_forget()
            self.final_feedback_label.config(text=final_text)

            # Save results once test is complete
            self.save_test_results(self.test_mode)
            return

        # Get the next question from the list.
        self.current_question = self.test_questions.pop(0)

        # Determine max options from entry (default to 4 if invalid)
        try:
            max_display = int(self.max_display_entry.get())
            if max_display < 2:
                max_display = 4
        except:
            max_display = 4

        if self.test_mode == "verbal":
            question_text = f'What is the translation for:\n\n---- "{self.current_question[0]}" ----'
            correct_answer = self.current_question[1]
            all_options = set(self.last_bilingual_content.translated_words)
        else:
            question_text = "Listen to the audio and select the correct English sentence."
            correct_answer = self.current_question[0]
            all_options = set(self.last_bilingual_content.untranslated_words)

        self.question_label.config(text=question_text)

        # Prepare options: include correct answer plus distractors.
        all_options.discard(correct_answer)
        distractors = random.sample(sorted(all_options), min(max_display - 1, len(all_options)))
        options = distractors + [correct_answer]
        random.shuffle(options)

        # Create radio buttons for options
        for opt in options:
            rb = ttk.Radiobutton(self.options_frame, text=opt, variable=self.selected_option, value=opt)
            rb.pack(fill="x", padx=5, pady=5, anchor="w")
            self.radio_buttons.append(rb)

    def check_answer(self):
        """
        Records the user's answer for the current question and moves to the next question.
        Provides partial feedback immediately if the answer is incorrect.
        """
        if not self.selected_option.get():
            messagebox.showwarning("Select an answer", "Please select an answer.")
            return

        if self.test_mode == "verbal":
            correct_answer = self.current_question[1]
        else:
            correct_answer = self.current_question[0]

        if self.selected_option.get() != correct_answer:
            # Record the incorrect item: question order, question text, correct answer, and user's answer.
            self.incorrect_items.append(
                (self.question_count, self.current_question[0], correct_answer, self.selected_option.get()))
            # Display partial feedback for incorrect answer.
            self.final_feedback_label.config(text=f"Incorrect! Correct answer: {correct_answer}")
            self.submit_button.config(state="disabled")
            # Proceed to the next question after a short delay.
            self.test_window.after(3000, self.proceed_to_next_question)
        else:
            self.score += 1
            self.show_next_question()

    def proceed_to_next_question(self):
        """
        Clears the partial feedback and re-enables the submit button then moves to the next question.
        """
        self.final_feedback_label.config(text="")
        self.submit_button.config(state="normal")
        self.show_next_question()

    def load_vocabulary(self):
        """Load source Spanish vocabulary and build display names."""
        with open(VOCAB_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.topics_data = {}  # topic_id -> metadata + entries
        self.display_to_key = {}  # display -> topic_id

        for topic in data.get("topics", []):
            eng, spa = topic["name"]
            topic_id = self.slugify(eng)
            display = f"{eng} / {spa}"
            self.display_to_key[display] = topic_id
            self.topics_data[topic_id] = {
                "name_en": eng,
                "name_es": spa,
                "entries": topic["entries"],
            }

    def slugify(self, value):
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")

    def create_vocab_selector(self, parent):
        frame = ttk.LabelFrame(parent, text="Choose Vocabulary Topic",
                               padding=10, style="Custom.TLabelframe")
        frame.pack(fill="x", padx=20, pady=5)

        # sort the display names alphabetically
        sorted_displays = sorted(self.display_to_key.keys())

        self.topic_combo = ttk.Combobox(
            frame,
            values=sorted_displays,
            state="readonly"
        )
        self.topic_combo.set("Select a topic…")
        self.topic_combo.pack(fill="x", padx=5, pady=5)
        self.topic_combo.bind("<<ComboboxSelected>>", self.on_topic_select)

    def on_topic_select(self, _evt):
        display = self.topic_combo.get()
        topic_id = self.display_to_key[display]
        topic = self.topics_data.get(topic_id, {})
        language = self.selected_language_code()
        voice_name = self.selected_voice_name()
        difficulty = self.difficulty_var.get().strip()

        self.translations_tree.delete(*self.translations_tree.get_children())
        self.translations_tree.insert("", tk.END, values=("Loading topic equivalents…", "", ""))
        self.play_all_button.config(state='disabled')
        self.current_topic = display

        threading.Thread(
            target=self.process_topic_selection,
            args=(topic_id, topic, language, voice_name, difficulty),
            daemon=True,
        ).start()

    def process_topic_selection(self, topic_id, topic, language, voice_name, difficulty):
        try:
            entries = topic.get("entries", [])
            english_entries = [eng for eng, _ in entries]
            if language == "es":
                target_entries = [esp for _, esp in entries]
            else:
                logger.info(
                    "Translating %d entry(ies) for topic '%s' into %s.",
                    len(english_entries),
                    topic.get("name_en", topic_id),
                    self.get_language_display(language),
                )
                target_entries = self.tutor.translate_entries(
                    topic_id=topic_id,
                    topic_name_en=topic.get("name_en", topic_id),
                    english_entries=english_entries,
                    target_language=language,
                    difficulty=difficulty,
                )

            bilingual = BilingualDict(untranslated_words=english_entries, translated_words=target_entries)
            self.last_bilingual_content = bilingual

            def update_table():
                self.translations_tree.delete(*self.translations_tree.get_children())
                self.translations_tree.heading("Target Language", text=self.get_language_display(language))
                for eng, target in zip(english_entries, target_entries):
                    self.translations_tree.insert("", tk.END, values=(eng, target, "▶"))
                self.play_all_button.config(state='normal')

            self.root.after(0, update_table)

            if language == "es":
                self.prepare_audio_from_disk(topic.get("name_es", topic_id), list(zip(english_entries, target_entries)), voice_name)
            else:
                self.prepare_audio_files(bilingual, language, voice_name)

        except Exception as e:
            logger.error("Error loading topic equivalents: %s", e)
            self.root.after(0, lambda: self.display_message("Could not load topic equivalents. Check logs for details."))

    def prepare_audio_from_disk(self, topic, entries, voice_name=None):
        """
        Point each word to its pre-generated .mp3 under BASE_DIR/<topic>/<word>.mp3,
        generating fallback audio if a file is missing.
        """
        self.audio_files_target.clear()
        topic_dir = BASE_DIR / topic

        missing = []
        for _eng, esp in entries:
            path = topic_dir / f"{esp}.mp3"
            if path.is_file():
                self.audio_files_target[esp] = str(path)
            else:
                missing.append(esp)

        if missing:
            logger.warning(
                "No cached Spanish audio found for %d item(s); generating fallback audio.",
                len(missing),
            )
            generated = self.tutor.text_to_speech_batch(missing, "es", voice_name)
            self.audio_files_target.update(generated)

    def save_test_results(self, test_mode: str):
        """
        Saves the results of the completed test (orthographic or phonologic)
        in a human-readable report, including:
         - timestamp
         - test type
         - source of items (vocab topic or LLM prompt)
         - raw score / total
         - percentage
         - table of incorrect items
        """
        filename = "test1_results.txt" if test_mode == "verbal" else "test2_results.txt"
        filepath = TEST_RESULTS_DIR / filename
        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # capture source: either a selected vocab topic or the LLM prompt
        source = getattr(self, "current_topic", None)
        if not source and hasattr(self, "last_concept_prompt"):
            source = f"LLM prompt: \"{self.last_concept_prompt}\""
        elif source:
            source = f"Vocabulary topic: \"{source}\""
        else:
            source = "Unknown"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_label = "Orthographic (verbal)" if test_mode == "verbal" else "Phonologic (audio)"
        pct = (self.score / self.total_questions * 100) if self.total_questions else 0

        with open(filepath, "a", encoding="utf-8") as f:
            f.write("===== Test Results =====\n")
            f.write(f"Timestamp       : {timestamp}\n")
            f.write(f"Test Type       : {test_label}\n")
            f.write(f"Source          : {source}\n")
            f.write(f"Raw Score       : {self.score} / {self.total_questions}\n")
            f.write(f"Percentage      : {pct:.2f}%\n")
            f.write("Incorrect Items :\n")

            if not self.incorrect_items:
                f.write("  None\n")
            else:
                f.write("  Order | Prompt                         | Correct Answer   | Your Answer\n")
                f.write("  ------|--------------------------------|------------------|-------------\n")
                for order, prompt, correct, user_ans in sorted(self.incorrect_items, key=lambda x: x[0]):
                    f.write(f"   {order:<4} | {prompt[:30]:<30} | {correct:<16} | {user_ans}\n")

            f.write("\n")
        logger.info("Test results written.")


# Usage Example with GUI
def main():
    # Launch GUI
    root = tk.Tk()
    app = TutorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# for later:
# ADD CONVERSATION (with chatting lay-out)
# ADD relevant GUI control (playback speed, tts gender, colors, context separate from user_query, goals, etc)
# ADD method to manually input sentences for translation
# FIX method to save test 2 results; now it saves duplicate name but should be their respective translated version
# FIX method to play the sound; only plays sound correctly after concept is generated
