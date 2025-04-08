import openai
from gtts import gTTS
import pygame
import tempfile
import os
import logging
from pydantic import BaseModel
from typing import List
from colorama import init, Fore, Style
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, colorchooser
import threading
from dotenv import load_dotenv
import random

# Initialize colorama
init(autoreset=True)

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

# Load environment variables
load_dotenv()

# Access the OpenAI API key from the environment variable (do NOT upload to GitHub!)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("No OpenAI API key found in environment variables.")
    raise ValueError("No OpenAI API key found in environment variables.")

openai.api_key = api_key
logger.info("OpenAI API key loaded successfully.")


class BilingualDict(BaseModel):
    untranslated_words: List[str]
    translated_words: List[str]


def call_GPT(
        system_prompt: str,
        user_query: str,
        pydantic_model: BaseModel,
        model: str = "gpt-4o",
) -> BaseModel:
    """
    Calls the OpenAI GPT model with the given system prompt and user query,
    and parses the response using the provided Pydantic model.

    Args:
        system_prompt (str): The system prompt for GPT.
        user_query (str): The user query to send to GPT.
        pydantic_model (BaseModel): The Pydantic model to parse the response.
        model (str): The GPT model to use.

    Returns:
        BaseModel: The parsed response as a Pydantic model.
    """
    try:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=pydantic_model,
        )
        print(f"LLM response generated successfully: {response.choices[0].message.parsed}")
        logger.info("LLM response generated successfully")
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise


class Tutor:
    """
    A language tutor class that integrates OpenAI's GPT-4o for generating bilingual
    content and utilizes text-to-speech for auditory learning in selected languages.
    Note on voice gender: gTTS does not support explicitly selecting male/female voices.
    If a male voice is required, consider using a different TTS service that supports
    voice gender selection (e.g., Amazon Polly, Azure TTS, etc).
    """

    def __init__(self):
        """
        Initializes the Tutor class and pygame mixer.
        """
        logger.info("Tutor initialized.")
        self.audio_files_en = {}
        self.audio_files_target = {}
        self.target_language = 'es'  # Default target language

        # Initialize pygame mixer once
        try:
            pygame.mixer.init()
            logger.info("Pygame mixer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing pygame mixer: {e}")
            messagebox.showerror("Audio Initialization Error",
                                 "Failed to initialize audio system. Check logs for details.")

    def request_concept(self, concept: str, num_items: int, target_language: str) -> BilingualDict:
        """
        Requests a bilingual dictionary based on the given concept from OpenAI's GPT-4.

        Parameters:
        - concept (str): The concept to generate content for.
        - num_items (int): Number of translation pairs to generate.
        - target_language (str): The target language code ('es' for Spanish, 'ru' for Russian).

        Returns:
        - BilingualDict: A Pydantic model containing the translations.
        """
        system_prompt = (
            "You are a multilingual assistant proficient in English and multiple other languages. "
            "Provide responses in a JSON format compatible with the given schema."
        )
        user_query = (
            f"Provide two lists (original untranslated - translated) to teach the language related to the concept '{concept}' which is an explicit request from the user. "
            f"Generate {num_items} items for each list: English items (untranslated) and their translations into the target language. "
            f"The items can be words, phrases, or sentences ; depending on the request! (so read carefully) Just anything typical to tutor the user. "
            f"The items are either in-first-person saying or general words/phrases/sentences. "
            f"Provide letters as items if alphabet is requested."
            f"Difficulty level: '{self.map_difficulty_to_level(target_language)}', adjust your response accordingly! ; There are five difficulty levels: 1. Beginner, 2. Elementary, 3. Intermediate, 4. Advanced, 5. Expert. "
            f"Target language: '{target_language}'. "
        )

        try:
            bilingual_content = call_GPT(
                system_prompt=system_prompt,
                user_query=user_query,
                pydantic_model=BilingualDict,
            )
            logger.info(f"Received bilingual content for concept '{concept}'.")
            return bilingual_content
        except Exception as e:
            logger.error(f"Failed to retrieve bilingual content: {e}")
            return BilingualDict(untranslated_words=[], translated_words=[])

    def map_difficulty_to_level(self, difficulty: str) -> str:
        """
        Maps difficulty level to a descriptive level for the GPT prompt.

        Parameters:
        - difficulty (str): The difficulty level selected by the user.

        Returns:
        - str: A descriptive difficulty level.
        """
        mapping = {
            'beginner': 'Beginner',
            'elementary': 'Elementary',
            'intermediate': 'Intermediate',
            'advanced': 'Advanced',
            'expert': 'Expert',
        }
        return mapping.get(difficulty.lower(), 'Intermediate')

    def text_to_speech(self, text: str, language: str = 'en') -> str:
        """
        Converts text to speech (default TTS voice) and saves it as a temporary MP3 file.
        NOTE: gTTS doesn't allow specifying male/female voice. If you require a specifically
        male/female voice, use another TTS provider that supports voice gender.
        """
        try:
            tts = gTTS(text=text, lang=language)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            logger.info(f"Text-to-speech conversion successful for language '{language}'.")
            return temp_file.name
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return ""

    def play_audio(self, file_path: str):
        """
        Plays the audio file at the specified path using pygame.

        Parameters:
        - file_path (str): The path to the audio file to play.
        """
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            logger.info(f"Playing audio file '{file_path}'.")

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            logger.info(f"Finished playing audio file '{file_path}'.")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            print(Fore.RED + "Error playing audio. Check logs for details.")


class TutorGUI:
    """
    A graphical user interface for the Language Tutor using Tkinter.
    """

    def __init__(self, root):
        self.tutor = Tutor()
        self.root = root
        self.root.title("Language Tutor")
        self.root.geometry("1200x800")  # Increased window size for better readability
        self.root.resizable(True, True)  # Allow window to be resizable
        self.current_bg_color = "#f0f0f0"  # Default background color
        self.current_fg_color = "#000000"  # Default foreground (text) color
        self.current_font_size = 14  # Increased default font size
        self.min_font_size = 12  # Minimum font size
        self.max_font_size = 24  # Maximum font size
        self.last_bilingual_content = None  # Store most recent bilingual content for testing
        self.create_widgets()
        self.create_menu()

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

    def create_widgets(self):
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
        style.configure("Treeview", font=("Helvetica", self.current_font_size), rowheight=self.current_font_size + 10)

        # Create custom styles for color changes
        style.configure("Custom.TLabelframe", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.TLabel", background=self.current_bg_color, foreground=self.current_fg_color)
        style.configure("Custom.Treeview", background=self.current_bg_color, foreground=self.current_fg_color,
                        fieldbackground=self.current_bg_color)
        style.map("Custom.Treeview", background=[('selected', '#ececec')], foreground=[('selected', '#000000')])

        # Frame for Concept Input
        input_frame = ttk.LabelFrame(self.root, text="Explore Russian or Spanish", padding=(20, 10),
                                     style="Custom.TLabelframe")
        input_frame.pack(fill="x", padx=20, pady=10)

        # Concept Entry
        ttk.Label(input_frame, text="Enter instruction or concept of choice:", style="Header.TLabel").grid(
            row=0, column=0, padx=5, pady=10, sticky="w")
        self.concept_entry = ttk.Entry(input_frame, width=50, font=("Helvetica", self.current_font_size))
        self.concept_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        # Number of Items Entry
        ttk.Label(input_frame, text="Number of items to generate:", style="Header.TLabel").grid(
            row=1, column=0, padx=5, pady=10, sticky="w")
        self.num_items_entry = ttk.Entry(input_frame, width=10, font=("Helvetica", self.current_font_size))
        self.num_items_entry.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        self.num_items_entry.insert(0, "10")  # Initialize with default value (10)

        # Difficulty Level Selection
        ttk.Label(input_frame, text="Select difficulty level:", style="Header.TLabel").grid(
            row=2, column=0, padx=5, pady=10, sticky="w")
        self.difficulty_var = tk.StringVar(value='intermediate')  # Default difficulty level
        difficulty_frame = ttk.Frame(input_frame, style="Custom.TLabelframe")
        difficulty_frame.grid(row=2, column=1, padx=5, pady=10, sticky="w")
        difficulty_levels = ['Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert']
        for level in difficulty_levels:
            ttk.Radiobutton(difficulty_frame, text=level, variable=self.difficulty_var,
                            value=level.lower().replace('-', ' '), style="Custom.TRadiobutton").pack(
                side="left", padx=5)

        # Language Selection
        ttk.Label(input_frame, text="Choose language for translation:", style="Header.TLabel").grid(
            row=3, column=0, padx=5, pady=10, sticky="w")
        self.language_var = tk.StringVar(value='es')
        language_frame = ttk.Frame(input_frame, style="Custom.TLabelframe")
        language_frame.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        ttk.Radiobutton(language_frame, text='Spanish (es)', variable=self.language_var,
                        value='es', style="Custom.TRadiobutton").pack(side="left", padx=5)
        ttk.Radiobutton(language_frame, text='Russian (ru)', variable=self.language_var,
                        value='ru', style="Custom.TRadiobutton").pack(side="left", padx=5)

        # Fix for the "Learn Concept" button color not showing on Windows (or certain themes).
        # Using a normal tk.Button can help if ttk styling is overridden by OS.
        self.learn_button = tk.Button(
            input_frame,
            text="Learn Concept",
            bg="#007BFF",  # Blue background color (not correctly implemented ; skip for now)
            fg="black",  # Black text
            activebackground="#0056b3",  # Darker blue when pressed
            activeforeground="white",
            font=("Helvetica", self.current_font_size),
            padx=10,
            pady=5,
            command=self.learn_concept
        )
        self.learn_button.grid(row=4, column=1, padx=5, pady=20, sticky="e")

        # Frame for Font Controls
        font_frame = ttk.LabelFrame(self.root, text="Font Controls", padding=(20, 10), style="Custom.TLabelframe")
        font_frame.pack(fill="x", padx=20, pady=10)

        self.decrease_font_button = ttk.Button(font_frame, text="--smaller--", command=self.decrease_font_size,
                                               style="Custom.TButton")
        self.decrease_font_button.pack(side="left", padx=5, pady=5)

        self.font_size_label = ttk.Label(font_frame, text=f"Font Size: {self.current_font_size}", style="Header.TLabel")
        self.font_size_label.pack(side="left", padx=10)

        self.increase_font_button = ttk.Button(font_frame, text="++bigger++", command=self.increase_font_size,
                                               style="Custom.TButton")
        self.increase_font_button.pack(side="left", padx=5, pady=5)

        # Frame for Displaying Translations
        display_frame = ttk.LabelFrame(self.root, text="Bilingual Translations", padding=(20, 10),
                                       style="Custom.TLabelframe")
        display_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Create Treeview for translations with adjusted column configurations
        columns = ("English", "Target Language", "Play")
        self.translations_tree = ttk.Treeview(display_frame, columns=columns,
                                              show='headings', selectmode="browse", style="Custom.Treeview")
        for col in columns:
            self.translations_tree.heading(col, text=col)
            if col == "Play":
                self.translations_tree.column(col, width=150, anchor="center", stretch=False)
            else:
                self.translations_tree.column(col, anchor="center", stretch=True)
        self.translations_tree.pack(fill="both", expand=True, side="left", padx=(0, 10), pady=10)

        # Add scrollbar to the treeview
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.translations_tree.yview)
        self.translations_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y", pady=10)

        # Frame for Audio Controls
        audio_frame = ttk.LabelFrame(self.root, text="Audio Controls", padding=(20, 10), style="Custom.TLabelframe")
        audio_frame.pack(fill="x", padx=20, pady=10)

        self.play_all_button = ttk.Button(audio_frame, text="Play All Audio", command=self.play_all_audio,
                                          state='disabled', style="Custom.TButton")
        self.play_all_button.pack(side="left", padx=5, pady=5)

        # Frame for Testing Controls - now with two separate buttons for Verbal and Audio testing.
        test_frame = ttk.LabelFrame(self.root, text="Testing Mode", padding=(20, 10), style="Custom.TLabelframe")
        test_frame.pack(fill="x", padx=20, pady=10)
        self.start_verbal_test_button = ttk.Button(test_frame, text="Start Test: type 1 - orthographic",
                                                   command=self.start_test_verbal, style="Custom.TButton")
        self.start_verbal_test_button.pack(side="left", padx=5, pady=5)
        self.start_audio_test_button = ttk.Button(test_frame, text="Start Test: type 2 - phonologic",
                                                  command=self.start_test_audio, style="Custom.TButton")
        self.start_audio_test_button.pack(side="left", padx=5, pady=5)

        # Frame for Viewing Logs
        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=(20, 10), style="Custom.TLabelframe")
        log_frame.pack(fill="x", padx=20, pady=10)

        self.view_logs_button = ttk.Button(log_frame, text="View Logs", command=self.view_logs, style="Custom.TButton")
        self.view_logs_button.pack(padx=5, pady=5, anchor="e")

    def increase_font_size(self):
        """
        Increases the font size of the Treeview and updates the row height.
        """
        if self.current_font_size < self.max_font_size:
            self.current_font_size += 2
            self.update_font_size()
        else:
            messagebox.showinfo("Font Size", f"Maximum font size of {self.max_font_size} reached.")

    def decrease_font_size(self):
        """
        Decreases the font size of the Treeview and updates the row height.
        """
        if self.current_font_size > self.min_font_size:
            self.current_font_size -= 2
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
        concept = self.concept_entry.get().strip()
        language = self.language_var.get().strip()
        num_items_str = self.num_items_entry.get().strip()
        difficulty_level = self.difficulty_var.get().strip()

        if not concept:
            messagebox.showwarning("Input Error", "Please enter a concept to learn about.")
            return

        if num_items_str:
            if not num_items_str.isdigit() or int(num_items_str) <= 0:
                messagebox.showwarning("Input Error", "Please enter a valid positive integer for the number of items.")
                return
            num_items = int(num_items_str)
        else:
            # Set a default number if not provided
            num_items = 10
            logger.info(f"Number of items not provided. Defaulting to {num_items}.")
            messagebox.showinfo("Default Number of Items", "Number of items not provided. Defaulting to 10.")

        # Disable the button to prevent multiple clicks
        self.learn_button.config(state='disabled')
        self.play_all_button.config(state='disabled')
        self.translations_tree.delete(*self.translations_tree.get_children())

        # Start a new thread to handle the learning process
        threading.Thread(target=self.process_learning, args=(concept, language, num_items, difficulty_level),
                         daemon=True).start()

    def process_learning(self, concept, language, num_items, difficulty_level):
        try:
            bilingual_content = self.tutor.request_concept(concept, num_items, language)
            if not bilingual_content.translated_words:
                self.display_message("Failed to retrieve content.")
                return

            # Save content for testing later
            self.last_bilingual_content = bilingual_content

            # Determine target language display name
            target_lang_display = self.get_language_display(language)

            # Update Treeview heading for target language
            self.translations_tree.heading("Target Language", text=target_lang_display)

            # Populate the Treeview with translations and a play button in the last column
            for eng, target in zip(bilingual_content.untranslated_words, bilingual_content.translated_words):
                self.translations_tree.insert('', tk.END, values=(eng, target, "▶"))

            # Bind the play button for the target language
            self.translations_tree.bind('<ButtonRelease-1>', self.on_tree_click)

            # Enable the Play All button
            self.play_all_button.config(state='normal')

            # Prepare audio for all words
            self.prepare_audio_files(bilingual_content, language)

        except Exception as e:
            logger.error(f"Error in process_learning: {e}")
            self.display_message("An error occurred. Check logs for details.")
        finally:
            self.learn_button.config(state='normal')

    def get_language_display(self, language_code):
        language_map = {'en': 'Spanish', 'es': 'Spanish', 'ru': 'Russian'}
        return language_map.get(language_code, 'Target Language')

    def prepare_audio_files(self, bilingual_content, language):
        """
        Pre-generates audio files for target language words based on the selected language.
        Stores the file paths in a dictionary for easy access during playback.
        """
        self.audio_files_en = {}  # English audio not generated intentionally.
        self.audio_files_target = {}
        for eng, target in zip(bilingual_content.untranslated_words, bilingual_content.translated_words):
            # Only generate audio for the target language word (in Spanish or Russian)
            audio_target = self.tutor.text_to_speech(target, language)
            if audio_target:
                self.audio_files_target[target] = audio_target

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

    def view_logs(self):
        try:
            with open("PGB_Model_Log.log", "r") as log_file:
                lines = log_file.readlines()
            log_window = tk.Toplevel(self.root)
            log_window.title("Log History")
            log_window.geometry("800x600")
            log_text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, state='normal', font=("Helvetica", 10))
            log_text.pack(fill="both", expand=True)
            log_text.insert(tk.END, ''.join(lines[-500:]))  # Display last 500 lines
            log_text.config(state='disabled')
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            messagebox.showerror("Log Error", "Unable to read log file.")

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
        For verbal tests the question is the English sentence and answer options are translations.
        For audio tests the question is the audio prompt (target) and answer options are English sentences.
        The question is displayed in italic.
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
            # Test finished: hide submit button and options; display only final feedback.
            percentage = (self.score / self.total_questions) * 100 if self.total_questions > 0 else 0
            feedback = "Excellent job!" if percentage >= 80 else "Good effort, keep practicing!" if percentage >= 50 else "Needs more practice."
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

        # Prepare options: include correct answer plus up to (max_display - 1) distractors.
        all_options.discard(correct_answer)
        distractors = random.sample(sorted(all_options), min(max_display - 1, len(all_options))) if all_options else []
        options = distractors + [correct_answer]
        random.shuffle(options)

        # Create radio buttons for options (packed to fill vertically).
        for opt in options:
            rb = ttk.Radiobutton(self.options_frame, text=opt, variable=self.selected_option, value=opt)
            rb.pack(fill="x", padx=5, pady=5, anchor="w")
            self.radio_buttons.append(rb)

    def check_answer(self):
        """
        Records the user's answer for the current question and moves to the next question.
        No immediate per-question feedback is displayed.
        Incorrect answers are tracked and will be displayed at the end.
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
        else:
            self.score += 1

        # Proceed to next question
        self.show_next_question()


# Usage Example with GUI
def main():
    # Initialize the Tutor
    tutor = Tutor()

    # Launch GUI
    root = tk.Tk()
    app = TutorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# for later:
# ADD TESTING (with latent disposition adaptation - and store the results in a file)
# ADD CONVERSATION (with chatting lay-out)
# ADD relevant GUI control (playback speed, tts gender, colors, context separate from user_query, goals, etc)
# ADD method to manually input sentences for translation
# ADD way to show handling of HTTP request(s)
