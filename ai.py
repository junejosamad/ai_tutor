import os
import tempfile
import pytesseract
import pdfplumber
from PIL import Image
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
import streamlit as st
import re
import nltk
import random
import torch

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# Determine device
device_pipeline = 0 if torch.cuda.is_available() else -1  # Use GPU if available for Transformers pipelines
if device_pipeline == -1:
    torch_device_embeddings = torch.device("cpu")
else:
    # For embeddings, explicitly use CPU if pipeline takes GPU, or manage GPU memory carefully.
    # Using CPU for embeddings by default to avoid OOM with multiple models on one GPU.
    torch_device_embeddings = torch.device("cpu")
    # Or if you want to try GPU for embeddings: torch_device_embeddings = torch.device(f"cuda:{device_pipeline}")

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device_pipeline)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device_pipeline)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': torch_device_embeddings})


# Removed module-level globals:
# vector_dbs = {}
# extracted_texts = {}
# current_doc_title = None
# All state will be managed via st.session_state

# ---------------------------------------------
# Extract text from PDF or Image
# ---------------------------------------------
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name
    suffix = file_name.lower()

    file_extension = os.path.splitext(suffix)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    text = ""
    try:
        if suffix.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif suffix.endswith((".png", ".jpg", ".jpeg")):
            try:
                text = pytesseract.image_to_string(Image.open(path))
            except Exception as ocr_error:
                st.error(f"Error during OCR for {file_name}: {ocr_error}")
                text = ""
        else:
            st.warning(f"Unsupported file type: {file_name}. Only PDF and images are supported.")
            text = ""
    except Exception as e:
        st.error(f"Error processing file {file_name}: {e}")
        text = ""
    finally:
        os.remove(path)
    return text.strip(), file_name


# ---------------------------------------------
# Store Embeddings in FAISS
# ---------------------------------------------
def store_vector_in_session(text, doc_title):
    # Uses st.session_state.vector_dbs_global
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_for_faiss = text_splitter.create_documents([text], metadatas=[{"title": doc_title}])

    if doc_title in st.session_state.vector_dbs_global:
        if docs_for_faiss:
            st.session_state.vector_dbs_global[doc_title].add_documents(docs_for_faiss)
        else:
            st.warning(f"No new text chunks to add to existing vector store for {doc_title}.")
    else:
        if docs_for_faiss:
            st.session_state.vector_dbs_global[doc_title] = FAISS.from_documents(docs_for_faiss, embedding_model)
        else:
            st.warning(f"No text chunks to create a new vector store for {doc_title}.")


# ---------------------------------------------
# Summarize Text
# ---------------------------------------------
def summarize_text(text_to_summarize):
    if not text_to_summarize.strip():
        return "Cannot summarize empty text."
    if len(text_to_summarize.split()) < 50:
        return "Text too short to summarize effectively."

    max_chunk_char_len = 1024 * 3
    chunks = [text_to_summarize[i: i + max_chunk_char_len] for i in
              range(0, len(text_to_summarize), max_chunk_char_len)]
    summaries = []

    for chunk in chunks:
        if not chunk.strip():
            continue
        num_words_chunk = len(chunk.split())
        min_len_summary = max(20, int(num_words_chunk * 0.1))
        max_len_summary = min(150, int(num_words_chunk * 0.5))

        if min_len_summary >= max_len_summary:
            min_len_summary = max(10, max_len_summary // 2)
        if max_len_summary < min_len_summary:
            if num_words_chunk > 10:
                summaries.append(chunk)
            continue
        try:
            summary_output = summarizer(chunk, max_length=max_len_summary, min_length=min_len_summary, do_sample=False)
            summaries.append(summary_output[0]["summary_text"])
        except Exception as e:
            st.error(f"Error during summarization of a chunk: {e}")

    if not summaries:
        return "Could not generate summary."
    return " ".join(summaries)


# ---------------------------------------------
# Question Answering
# ---------------------------------------------
def answer_question_from_session(question, doc_title=None):
    # Uses st.session_state.vector_dbs_global
    active_vector_dbs = st.session_state.vector_dbs_global

    if not active_vector_dbs:
        st.warning("Please upload and process a file first in the 'Upload & Extract' tab.")
        return ""
    if not question.strip():
        st.warning("Please enter a question.")
        return ""

    context_parts = []
    try:
        if doc_title and doc_title != "All Documents":
            if doc_title in active_vector_dbs and active_vector_dbs[doc_title].index.ntotal > 0:
                retriever = active_vector_dbs[doc_title].as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(question)
                if not relevant_docs:
                    return f"No relevant information found in '{doc_title}' for your question."
                context_parts.extend([doc.page_content for doc in relevant_docs])
            else:
                st.warning(f"No content available or document '{doc_title}' not found for searching.")
                return f"No content available or document '{doc_title}' not found for searching."

        elif doc_title == "All Documents" or not doc_title:
            for db_key, db in active_vector_dbs.items():
                if db.index.ntotal > 0:
                    # Consider relevance scores if combining from many sources
                    retrieved_from_db = db.similarity_search(question, k=2)
                    context_parts.extend([doc.page_content for doc in retrieved_from_db])
            if not context_parts:
                return "No relevant information found across any uploaded documents for your question."
        else:  # Should not happen if doc_title is from selectbox
            st.error(f"Invalid document selection: '{doc_title}'.")
            return ""

        if not context_parts:
            return "Could not retrieve any context for the question."

        context = "\n\n".join(context_parts)

        if len(context.strip()) < len(question.strip()) + 10:
            return "The context retrieved is too short or not informative enough to answer the question."

        qa_input = {'question': question, 'context': context}
        answer_output = qa_pipeline(qa_input)
        return answer_output["answer"].strip()

    except Exception as e:
        st.error(f"Error during question answering: {e}")
        return "An error occurred while trying to answer the question."


# ---------------------------------------------
# Flashcard Generation
# ---------------------------------------------
def generate_flashcards_from_text(text_content):
    flashcards = []
    seen_terms = set()
    if not text_content or not text_content.strip():
        return flashcards

    try:
        sentences = nltk.sent_tokenize(text_content)
    except Exception as e:
        st.error(f"Error tokenizing sentences for flashcards: {e}")
        return flashcards

    for i, sent in enumerate(sentences):
        words = nltk.word_tokenize(sent)
        tagged_words = nltk.pos_tag(words)
        potential_terms = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('NP')]

        for term in potential_terms:
            if term.lower() in seen_terms or len(term) < 3:
                continue
            defining_patterns = [
                r"\b" + re.escape(term) + r"\b\s+(?:is|are|was|were)\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s+refers?\s+to\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s+means?\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s*,\s*defined\s+as\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s*:\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
            ]
            found_definition = None
            for pattern in defining_patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match and len(match.groups()) >= 1:
                    definition = match.group(1).strip()
                    if 2 <= len(definition.split()) <= 40 and len(definition) > 5:
                        found_definition = definition
                        break
            if found_definition:
                flashcards.append({"term": term, "definition": found_definition})
                seen_terms.add(term.lower())
    return flashcards


# ---------------------------------------------
# Text to Speech
# ---------------------------------------------
def text_to_speech_gtts(text_to_read):
    if not text_to_read or not text_to_read.strip():
        st.warning("No text provided for audio.")
        return None
    try:
        tts = gTTS(text_to_read)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            audio_path = tmp_audio.name
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error during text-to-speech: {e}")
        return None


# ---------------------------------------------
# Quiz Generation and Handling
# ---------------------------------------------
def generate_quiz_questions_from_text(text_for_quiz, num_questions=5):
    flashcards = generate_flashcards_from_text(text_for_quiz)
    if not flashcards:
        st.info("Not enough content to generate flashcards for a quiz.")
        return []

    questions = []
    if len(flashcards) < 1:
        st.info("Not enough flashcards to generate a quiz. Try a different document or section.")
        return []

    num_questions = min(num_questions, len(flashcards))
    if num_questions == 0:
        return []

    selected_flashcards_indices = random.sample(range(len(flashcards)), num_questions)

    for index in selected_flashcards_indices:
        card = flashcards[index]
        correct_answer = card['term']
        definition = card['definition']
        other_terms = [fc['term'] for i, fc in enumerate(flashcards) if i != index and fc['term'] != correct_answer]

        num_distractors_to_sample = min(3, len(other_terms))
        incorrect_options = []
        if other_terms and num_distractors_to_sample > 0:
            incorrect_options = random.sample(other_terms, num_distractors_to_sample)

        options = [correct_answer] + incorrect_options
        random.shuffle(options)

        if len(options) > 1:  # Only add if there's at least one distractor
            questions.append({
                "question": f"What is the term for: \"{definition}\"",
                "options": options,
                "correct_answer": correct_answer,
                "user_answer": None,
                "id": f"q_{index}_{random.randint(1000, 9999)}"  # More robust unique ID
            })
    return questions


def display_quiz_interface(quiz_questions):
    if not quiz_questions:
        st.info("No quiz questions available to display.")
        return

    for i, q in enumerate(quiz_questions):
        st.subheader(f"Question {i + 1}:")
        st.write(q["question"])
        st.session_state.user_answers[q['id']] = st.radio(
            f"Select your answer for Question {i + 1}:",
            q["options"],
            key=q['id'],
            index=None
        )

    if st.button("Submit Quiz"):
        st.session_state.quiz_submitted = True
        st.rerun()


def grade_submitted_quiz():
    score = 0
    total = len(st.session_state.quiz_questions)
    results_display = []

    for q in st.session_state.quiz_questions:
        user_answer = st.session_state.user_answers.get(q['id'])
        correct = q["correct_answer"]
        is_correct = (user_answer == correct)
        if is_correct:
            score += 1
        results_display.append({
            "question": q["question"],
            "user_answer": user_answer if user_answer is not None else "Not Answered",
            "correct_answer": correct,
            "is_correct": is_correct
        })

    st.subheader(f"Quiz Results: Your Score: {score}/{total}")
    for res in results_display:
        st.markdown(f"---")
        st.markdown(f"**Q:** {res['question']}")
        st.markdown(f"- Your answer: `{res['user_answer']}`")
        st.markdown(f"- Correct answer: `{res['correct_answer']}`")
        if res["is_correct"]:
            st.success("Correct ✅")
        elif res['user_answer'] == "Not Answered":
            st.warning("Not Answered ⚠️")
        else:
            st.error("Incorrect ❌")
    st.markdown(f"---")
    if st.button("Try Another Quiz or Document"):
        keys_to_reset = ['quiz_questions', 'user_answers', 'quiz_submitted', 'current_quiz_doc', 'quiz_num_q',
                         'quiz_initialized_for_doc']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ---------------------------------------------
# Streamlit Interface with Tabs
# ---------------------------------------------
st.set_page_config(layout="wide")
st.title("📘 AI Study Assistant")

# Initialize session state variables
if 'quiz_submitted' not in st.session_state: st.session_state.quiz_submitted = False
if 'user_answers' not in st.session_state: st.session_state.user_answers = {}
if 'quiz_questions' not in st.session_state: st.session_state.quiz_questions = []
if 'current_quiz_doc' not in st.session_state: st.session_state.current_quiz_doc = None
if 'quiz_num_q' not in st.session_state: st.session_state.quiz_num_q = 5
if 'quiz_initialized_for_doc' not in st.session_state: st.session_state.quiz_initialized_for_doc = False
if 'extracted_texts_global' not in st.session_state: st.session_state.extracted_texts_global = {}
if 'vector_dbs_global' not in st.session_state: st.session_state.vector_dbs_global = {}
if 'current_flashcards' not in st.session_state: st.session_state.current_flashcards = []
if 'flashcard_idx' not in st.session_state: st.session_state.flashcard_idx = 0
if 'show_definition' not in st.session_state: st.session_state.show_definition = False

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Extract", "Summarize", "Question Answering", "Flashcards", "Quiz"])

with tab1:
    st.header("📤 Upload and Extract Text")
    uploaded_files = st.file_uploader("Upload PDF or Image files", type=["pdf", "png", "jpg", "jpeg"],
                                      accept_multiple_files=True)

    if uploaded_files:
        for file_obj in uploaded_files:  # Renamed to avoid conflict with 'file' module
            if file_obj.name not in st.session_state.extracted_texts_global:
                with st.spinner(f"Extracting text from {file_obj.name}..."):
                    extracted_text, doc_title = extract_text_from_file(file_obj)
                if extracted_text:
                    st.session_state.extracted_texts_global[doc_title] = extracted_text
                    st.success(f"Text Extracted Successfully from {doc_title}!")
                    with st.expander(f"View Extracted Text from {doc_title} (first 3000 chars)"):
                        st.text_area("Extracted Text", value=extracted_text[:3000], height=200,
                                     key=f"text_area_{doc_title}")

                    with st.spinner(f"Storing embeddings for {doc_title}..."):
                        store_vector_in_session(extracted_text, doc_title)
                    st.success(f"Embeddings stored for {doc_title}!")
                else:
                    st.warning(f"Could not extract any text from {file_obj.name}.")
            else:
                st.info(f"'{file_obj.name}' has already been processed.")

    if not st.session_state.extracted_texts_global:
        st.info("Upload files to begin.")
    else:
        st.subheader("Processed Documents:")
        for title_key in st.session_state.extracted_texts_global.keys():
            st.markdown(f"- {title_key}")

with tab2:
    st.header("📝 Summarize Text")
    doc_titles_available = list(st.session_state.extracted_texts_global.keys())
    if doc_titles_available:
        selected_doc_title_summary = st.selectbox("Select document to summarize:", doc_titles_available,
                                                  key="summarize_select", index=None)
        if selected_doc_title_summary and st.button("Generate Summary", key="summarize_btn"):
            if selected_doc_title_summary in st.session_state.extracted_texts_global:
                with st.spinner(f"Summarizing {selected_doc_title_summary}..."):
                    summary_text = summarize_text(st.session_state.extracted_texts_global[selected_doc_title_summary])
                st.subheader("Summary")
                st.markdown(summary_text)

                if summary_text and "too short" not in summary_text.lower() and "cannot summarize" not in summary_text.lower() and "could not generate" not in summary_text.lower():
                    with st.spinner("Generating audio for summary..."):
                        audio_path = text_to_speech_gtts(summary_text)
                    if audio_path:
                        st.audio(audio_path)
            else:
                st.warning(f"Original text for {selected_doc_title_summary} not found.")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")

with tab3:
    st.header("❓ Question Answering")
    # Ensure vector_dbs_global is populated before creating options
    doc_titles_qa_options = ["All Documents"] + list(st.session_state.vector_dbs_global.keys())

    if len(doc_titles_qa_options) > 1:  # Check if any documents are processed
        selected_doc_title_qa = st.selectbox("Search within document:", doc_titles_qa_options, key="qa_select_doc",
                                             index=0)
        question = st.text_input("Ask a question about the content:", key="qa_input")

        if st.button("Get Answer", key="qa_btn"):
            if question:
                with st.spinner("Searching for answer..."):
                    # answer_question_from_session directly uses st.session_state.vector_dbs_global
                    answer = answer_question_from_session(question, doc_title=selected_doc_title_qa)
                if answer:
                    st.subheader("Answer:")
                    st.markdown(answer)
                # else: The function answer_question_from_session already shows warnings/errors
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload and process at least one file in the 'Upload & Extract' tab for Q&A.")

with tab4:
    st.header("🧠 Interactive Learning: Flashcards")
    doc_titles_flashcard = list(st.session_state.extracted_texts_global.keys())
    if doc_titles_flashcard:
        selected_doc_flashcard = st.selectbox("Generate flashcards from document:", doc_titles_flashcard,
                                              key="flashcard_select", index=None)

        if selected_doc_flashcard and st.button("Generate Flashcards",
                                                key="flashcard_btn"):  # check selected_doc_flashcard is not None
            if selected_doc_flashcard in st.session_state.extracted_texts_global:
                with st.spinner(f"Generating flashcards from {selected_doc_flashcard}..."):
                    flashcards_list = generate_flashcards_from_text(
                        st.session_state.extracted_texts_global[selected_doc_flashcard])

                if flashcards_list:
                    st.subheader(f"Generated Flashcards for {selected_doc_flashcard} ({len(flashcards_list)} cards)")
                    st.session_state.current_flashcards = flashcards_list
                    st.session_state.flashcard_idx = 0
                    st.session_state.show_definition = False
                else:
                    st.info(
                        "No flashcards could be generated from this document. The document might be too short or lack clear term-definition patterns.")
                    st.session_state.current_flashcards = []  # Clear old flashcards
            else:
                st.warning(f"Original text for {selected_doc_flashcard} not found.")

        if st.session_state.current_flashcards:  # Check if list is not empty
            total_cards = len(st.session_state.current_flashcards)
            idx = st.session_state.flashcard_idx

            card = st.session_state.current_flashcards[idx]

            st.markdown(f"---")
            st.markdown(f"**Card {idx + 1} of {total_cards}**")
            st.subheader("Term:")
            st.markdown(f"### {card['term']}")

            if st.session_state.show_definition:
                st.subheader("Definition:")
                st.markdown(f"> {card['definition']}")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=(idx == 0), key="prev_flash_btn"):
                    st.session_state.flashcard_idx -= 1
                    st.session_state.show_definition = False
                    st.rerun()
            with col2:
                if st.button("Reveal/Hide Definition", type="primary", key="reveal_flash_btn"):
                    st.session_state.show_definition = not st.session_state.show_definition
                    st.rerun()
            with col3:
                if st.button("Next ➡️", disabled=(idx == total_cards - 1), key="next_flash_btn"):
                    st.session_state.flashcard_idx += 1
                    st.session_state.show_definition = False
                    st.rerun()
            st.markdown(f"---")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")

with tab5:
    st.header("📝 Quiz Yourself!")
    doc_titles_quiz = list(st.session_state.extracted_texts_global.keys())

    if not doc_titles_quiz:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab to generate a quiz.")
    else:
        selected_doc_for_quiz = st.selectbox("Select document for Quiz:", doc_titles_quiz, key="quiz_doc_select",
                                             index=None)

        if selected_doc_for_quiz:
            num_q_options = [5, 10, 15, 20]
            num_questions_to_gen = st.select_slider(
                "Number of questions for the quiz:",
                options=num_q_options,
                value=st.session_state.quiz_num_q,  # Persist selection
                key="num_quiz_questions_slider"
            )

            # Generate Quiz / Reset logic
            if st.button("Generate / Reset Quiz", key="start_quiz_btn"):
                with st.spinner(f"Generating {num_questions_to_gen} quiz questions for {selected_doc_for_quiz}..."):
                    quiz_text_content = st.session_state.extracted_texts_global.get(selected_doc_for_quiz)
                    if quiz_text_content:
                        st.session_state.quiz_questions = generate_quiz_questions_from_text(quiz_text_content,
                                                                                            num_questions_to_gen)
                        st.session_state.user_answers = {q['id']: None for q in st.session_state.quiz_questions}
                        st.session_state.quiz_submitted = False
                        st.session_state.current_quiz_doc = selected_doc_for_quiz
                        st.session_state.quiz_num_q = num_questions_to_gen
                        st.session_state.quiz_initialized_for_doc = True
                        if not st.session_state.quiz_questions:
                            st.warning(
                                "Could not generate any questions. The document might be too short or lack suitable content.")
                            st.session_state.quiz_initialized_for_doc = False
                    else:
                        st.error("Could not retrieve text for the selected document.")
                        st.session_state.quiz_initialized_for_doc = False
                st.rerun()

            # Display quiz or results only if initialized for the current selection
            if (st.session_state.quiz_initialized_for_doc and
                    st.session_state.current_quiz_doc == selected_doc_for_quiz and
                    st.session_state.quiz_num_q == num_questions_to_gen):  # Check if settings match current quiz

                if st.session_state.quiz_questions:  # Only display if questions exist
                    if not st.session_state.quiz_submitted:
                        display_quiz_interface(st.session_state.quiz_questions)
                    else:
                        grade_submitted_quiz()
                # else: (Covered by warning during generation)
            elif selected_doc_for_quiz:  # If doc is selected but quiz not matching/generated for it
                st.info("Click 'Generate / Reset Quiz' with the desired document and number of questions.")

        else:
            st.info("Select a document to start a quiz.")