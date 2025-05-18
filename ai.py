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
except Exception:  # CORRECTED
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except Exception:  # CORRECTED
    nltk.download('punkt', quiet=True)

device_pipeline = 0 if torch.cuda.is_available() else -1  # Use GPU if available for Transformers pipelines
if device_pipeline == -1:
    torch_device_embeddings = torch.device("cpu")
else:

    torch_device_embeddings = torch.device("cpu")
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device_pipeline)
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device_pipeline)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                            model_kwargs={'device': torch_device_embeddings})
except Exception as e:
    st.error(f"Error loading HuggingFace models: {e}. Please check your internet connection and model names.")

    st.stop()

def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name
    # It's good practice to make the suffix check case-insensitive from the start.
    file_extension = os.path.splitext(file_name)[1].lower()

    # Use the file extension for the temp file, not the whole suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    text = ""
    try:
        if file_extension == ".pdf":  # Use the extracted extension
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Ensure page_text is not None
                        text += page_text + "\n"
        elif file_extension in (".png", ".jpg", ".jpeg"):  # Use the extracted extension
            try:
                text = pytesseract.image_to_string(Image.open(path))
            except Exception as ocr_error:
                st.error(f"Error during OCR for {file_name}: {ocr_error}")
                # text remains ""
        else:
            st.warning(f"Unsupported file type: {file_name}. Only PDF and images (PNG, JPG, JPEG) are supported.")
            # text remains ""
    except Exception as e:
        st.error(f"Error processing file {file_name}: {e}")
        # text remains ""
    finally:
        if os.path.exists(path):  # Check if path exists before removing
            os.remove(path)
    return text.strip(), file_name


# ---------------------------------------------
# Store Embeddings in FAISS
# ---------------------------------------------
def store_vector_in_session(text, doc_title):
    if not text or not text.strip():
        st.warning(f"Cannot store embeddings for {doc_title} as the text is empty.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Ensure metadata is correctly formatted if you plan to use it for filtering later.
    docs_for_faiss = text_splitter.create_documents([text], metadatas=[
        {"source": doc_title}])  # Changed "title" to "source" for clarity, common in Langchain

    if not docs_for_faiss:
        st.warning(f"No text chunks were generated for {doc_title} after splitting. Cannot create vector store.")
        return

    try:
        if doc_title in st.session_state.vector_dbs_global:
            st.session_state.vector_dbs_global[doc_title].add_documents(docs_for_faiss)
            # st.success(f"Added new content from {doc_title} to existing vector store.") # Optional success message
        else:
            st.session_state.vector_dbs_global[doc_title] = FAISS.from_documents(docs_for_faiss, embedding_model)
            # st.success(f"Created new vector store for {doc_title}.") # Optional success message
    except Exception as e:
        st.error(f"Failed to store embeddings for {doc_title}: {e}")


# ---------------------------------------------
# Summarize Text
# ---------------------------------------------
def summarize_text(text_to_summarize):
    if not text_to_summarize or not text_to_summarize.strip():  # Check for None or empty string
        return "Cannot summarize empty text."
    if len(text_to_summarize.split()) < 50:  # Using split() is fine for a rough word count
        return "Text too short to summarize effectively (less than 50 words)."

    max_chunk_char_len = 1024 * 3
    chunks = [text_to_summarize[i: i + max_chunk_char_len] for i in
              range(0, len(text_to_summarize), max_chunk_char_len)]
    summaries = []
    processed_chunks = 0

    for chunk_idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        num_words_chunk = len(chunk.split())
        # Adjust min/max length carefully based on model constraints and desired output
        min_len_summary = max(20, int(num_words_chunk * 0.1))
        max_len_summary = min(150, int(num_words_chunk * 0.3))  # Reduced max ratio a bit for tighter summaries

        if min_len_summary >= max_len_summary:  # if text is very short
            min_len_summary = max(10, max_len_summary // 2)  # Ensure min_len is reasonable

        # If after adjustment, it's still problematic, or chunk is too small
        if max_len_summary < min_len_summary or num_words_chunk < 20:  # Arbitrary small chunk threshold
            if num_words_chunk > 5:  # If it has some content, just append the chunk itself
                summaries.append(chunk)
            processed_chunks += 1
            continue
        try:
            # It's good to let the user know what's happening for long processes
            # st.write(f"Summarizing chunk {chunk_idx + 1}/{len(chunks)}...") # This might be too verbose in a loop
            summary_output = summarizer(chunk, max_length=max_len_summary, min_length=min_len_summary, do_sample=False)
            summaries.append(summary_output[0]["summary_text"])
            processed_chunks += 1
        except Exception as e:
            st.error(f"Error during summarization of chunk {chunk_idx + 1}: {e}")
            # Optionally append the original chunk if summarization fails for it, or skip
            # summaries.append(f"[Error summarizing chunk: {chunk[:100]}...]")

    if not summaries and processed_chunks == 0:  # No processable chunks found
        return "No content suitable for summarization was found."
    if not summaries:  # Some chunks processed, but all failed to summarize
        return "Could not generate summary from the provided text."

    return " ".join(summaries)


# ---------------------------------------------
# Question Answering
# ---------------------------------------------
def answer_question_from_session(question, doc_title=None):
    active_vector_dbs = st.session_state.get('vector_dbs_global', {})  # Use .get for safety

    if not active_vector_dbs:
        st.warning("Please upload and process a file first in the 'Upload & Extract' tab.")
        return ""  # Return empty string, as st.warning already displayed
    if not question or not question.strip():  # Check for None or empty
        st.warning("Please enter a question.")
        return ""

    context_parts = []
    retrieved_something = False
    try:
        if doc_title and doc_title != "All Documents":
            db_to_search = active_vector_dbs.get(doc_title)
            if db_to_search and hasattr(db_to_search, 'index') and db_to_search.index.ntotal > 0:
                # k=3 might be too few for complex questions, k=5 is a common starting point
                retriever = db_to_search.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(question)
                if relevant_docs:
                    context_parts.extend([doc.page_content for doc in relevant_docs])
                    retrieved_something = True
                else:
                    return f"No relevant information found in '{doc_title}' for your question."
            else:
                return f"No content available or document '{doc_title}' not found/processed for searching."

        elif doc_title == "All Documents" or not doc_title:  # Search all if "All Documents" or None
            if not active_vector_dbs:  # Double check, though covered above
                return "No documents have been processed yet."
            for db_key, db in active_vector_dbs.items():
                if hasattr(db, 'index') and db.index.ntotal > 0:
                    # k=2 per document for "All Docs" is reasonable to avoid too much context
                    retrieved_from_db = db.similarity_search(question, k=2)
                    if retrieved_from_db:
                        context_parts.extend([doc.page_content for doc in retrieved_from_db])
                        retrieved_something = True
            if not retrieved_something:
                return "No relevant information found across any uploaded documents for your question."
        else:
            st.error(f"Invalid document selection: '{doc_title}'. This should not happen.")
            return ""

        if not context_parts:  # Should be covered by earlier returns, but as a safeguard
            return "Could not retrieve any context for the question."

        context = "\n\n".join(list(set(context_parts)))  # Use set to remove duplicate context parts if any

        # Check context length against question length, also ensure context is not too massive for the QA model
        # Distilbert has a max token limit (usually 512 including question + context)
        # This is a rough check; tokenizing would be more accurate.
        if len(context.strip()) < len(question.strip()) + 10:
            return "The context retrieved is too short or not informative enough to answer the question."

        # Rough truncation if context is too long (better to use model's tokenizer for precision)
        # QA model token limit is often 512. Question takes some, context takes the rest.
        # Let's assume question is ~50 tokens, context can be ~450 tokens.
        # Average token length ~4 chars. So, 450*4 = 1800 chars.
        max_context_char_len = 2000  # A bit generous
        if len(context) > max_context_char_len:
            context = context[:max_context_char_len]
            st.info("Retrieved context was truncated to fit the model's limits.")

        qa_input = {'question': question, 'context': context}
        answer_output = qa_pipeline(qa_input)

        if not answer_output or not answer_output.get("answer"):
            return "The model could not find an answer in the provided context."

        return answer_output["answer"].strip()

    except Exception as e:
        st.error(f"Error during question answering: {e}")
        # Consider logging the full traceback for debugging if using a logging library
        return "An error occurred while trying to answer the question."


# ---------------------------------------------
# Flashcard Generation
# ---------------------------------------------
def generate_flashcards_from_text(text_content):
    flashcards = []
    seen_terms = set()  # To avoid duplicate terms
    if not text_content or not text_content.strip():
        return flashcards

    try:
        # Consider splitting into smaller chunks if text_content is massive,
        # as sentence tokenization on very large strings can be slow.
        sentences = nltk.sent_tokenize(text_content)
    except Exception as e:  # Catch specific errors if known, e.g., from nltk.download issues
        st.error(f"Error tokenizing sentences for flashcards: {e}")
        return flashcards

    if not sentences:
        return flashcards

    for sent_idx, sent in enumerate(sentences):
        if not sent.strip():
            continue
        try:
            words = nltk.word_tokenize(sent)
            tagged_words = nltk.pos_tag(words)
        except Exception as e:
            st.warning(f"Could not tokenize/tag sentence for flashcards: {sent[:50]}... Error: {e}")
            continue

        # Improved term extraction: Look for Noun Phrases (e.g., using NLTK's regex chunker)
        # For simplicity, current 'NN' or 'NP' (Proper Noun) is okay for a hackathon.
        potential_terms = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('NP')]
        # You could also try to chain consecutive nouns/proper nouns to form multi-word terms.

        for term in potential_terms:
            term_lower = term.lower()
            if term_lower in seen_terms or len(term) < 3 or not term[
                0].isalnum():  # Avoid very short or punctuation-starting terms
                continue

            # Regex patterns for definitions. These are good.
            # Consider adding patterns for terms defined using parentheses, e.g., "An AI (Artificial Intelligence)..."
            defining_patterns = [
                r"\b" + re.escape(term) + r"\b\s+(?:is|are|was|were)\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s+refers?\s+to\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s+means?\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s*,\s*defined\s+as\s+(?:a|an|the|)\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                r"\b" + re.escape(term) + r"\b\s*:\s*([^.,;\n]+?)(?:\.|,|\n|;|$)",
                # r"([^.,;\n]+?)\s+\((?:often\s+called|also\s+known\s+as|aka)\s+" + re.escape(term) + r"\)", # Term in parenthesis
            ]
            found_definition = None
            for pattern in defining_patterns:
                try:
                    # Search in a slightly larger context if definition might span sentences (more complex)
                    # For now, searching within `sent` is fine.
                    match = re.search(pattern, sent, re.IGNORECASE)
                    if match and len(match.groups()) >= 1:
                        definition = match.group(1).strip()
                        # Add more conditions for a good definition
                        if 2 <= len(definition.split()) <= 40 and len(
                                definition) > 5 and term.lower() not in definition.lower():
                            found_definition = definition
                            break
                except re.error as re_err:  # Catch regex compilation errors if patterns are dynamic
                    st.warning(f"Regex error for flashcards: {re_err}")
                    continue  # Skip this pattern

            if found_definition:
                flashcards.append({"term": term.strip(), "definition": found_definition.strip()})
                seen_terms.add(term_lower)
                if len(flashcards) >= 50:  # Limit number of flashcards to avoid overwhelming UI / long processing
                    st.info("Reached maximum flashcard generation limit (50).")
                    return flashcards
    return flashcards


# ---------------------------------------------
# Text to Speech
# ---------------------------------------------
def text_to_speech_gtts(text_to_read):
    if not text_to_read or not text_to_read.strip():
        st.warning("No text provided for audio.")
        return None
    try:
        tts = gTTS(text_to_read, lang='en')  # Specify language
        # Use a more robust way to handle temp file path for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="audio_") as tmp_audio:
            audio_path = tmp_audio.name
        tts.save(audio_path)
        return audio_path
    except AssertionError as ae:  # gTTS can raise AssertionError for empty text after sanitization
        st.warning(f"Could not generate audio: {ae}. The text might be empty or unsuitable.")
        return None
    except Exception as e:
        st.error(f"Error during text-to-speech: {e}")
        return None


# ---------------------------------------------
# Quiz Generation and Handling
# ---------------------------------------------
def generate_quiz_questions_from_text(text_for_quiz, num_questions=5):
    if not text_for_quiz or not text_for_quiz.strip():
        st.info("Cannot generate quiz from empty text.")
        return []

    # Generate more flashcards than needed for quiz to have a good pool for distractors
    flashcards = generate_flashcards_from_text(text_for_quiz)  # generate_flashcards limits to 50

    if not flashcards:
        st.info("Not enough distinct terms and definitions found in the text to generate flashcards for a quiz.")
        return []

    questions = []
    # Ensure we have enough flashcards for the questions and for distractors
    # Each question needs 1 correct answer + at least 1 distractor (ideally 3)
    if len(flashcards) < 2:  # Need at least 2 flashcards to make one question with one distractor
        st.info(f"Not enough flashcards ({len(flashcards)}) to generate a meaningful quiz. Need at least 2.")
        return []

    actual_num_questions = min(num_questions, len(flashcards))
    if actual_num_questions == 0:
        return []

    # Ensure we have enough unique terms for distractors.
    # If we want N questions, and each needs 3 distractors, we ideally need N*4 unique terms.
    # Or, at least N + 3 unique terms if distractors can be reused carefully (not implemented here).
    # For simplicity, we'll sample from available flashcards.

    try:
        # Ensure we don't try to sample more questions than available flashcards
        selected_flashcards_indices = random.sample(range(len(flashcards)), actual_num_questions)
    except ValueError:  # Not enough flashcards to sample `actual_num_questions`
        st.warning(f"Could not sample {actual_num_questions} questions from {len(flashcards)} flashcards.")
        return []

    for q_idx, index in enumerate(selected_flashcards_indices):
        card = flashcards[index]
        correct_answer = card['term']
        definition = card['definition']

        # Get other terms for distractors, ensuring they are different from the correct answer
        other_terms = [fc['term'] for i, fc in enumerate(flashcards) if
                       i != index and fc['term'].lower() != correct_answer.lower()]

        # Ensure we have enough unique other terms for distractors
        num_distractors_to_sample = min(3, len(set(other_terms)))  # Use set to count unique distractors available

        incorrect_options = []
        if other_terms and num_distractors_to_sample > 0:
            try:
                incorrect_options = random.sample(list(set(other_terms)), num_distractors_to_sample)
            except ValueError:  # Not enough unique other_terms to sample
                # Fallback: if we need 3 but only have 1 or 2 unique distractors
                incorrect_options = list(set(other_terms))

        options = [correct_answer] + incorrect_options
        # Ensure options are unique (case-insensitive check might be good too)
        unique_options = []
        seen_options_lower = set()
        for opt in options:
            if opt.lower() not in seen_options_lower:
                unique_options.append(opt)
                seen_options_lower.add(opt.lower())
        options = unique_options
        random.shuffle(options)

        if len(options) > 1:  # Only add if there's at least one distractor, so at least two options
            questions.append({
                "question": f"What is the term for: \"{definition}\"",
                "options": options,
                "correct_answer": correct_answer,
                # user_answer is initialized when displaying quiz
                "id": f"q_{q_idx}_{random.randint(1000, 9999)}"  # q_idx makes it more unique
            })
        if len(questions) >= actual_num_questions:
            break  # Stop if we have enough questions

    if not questions:
        st.info("Could not generate any quiz questions with sufficient options.")
    return questions


def display_quiz_interface(quiz_questions):
    if not quiz_questions:
        st.info("No quiz questions available to display.")
        return

    # Initialize user_answers in session_state if not already done for these specific questions
    for q in quiz_questions:
        if q['id'] not in st.session_state.user_answers:
            st.session_state.user_answers[q['id']] = None

    for i, q in enumerate(quiz_questions):
        st.subheader(f"Question {i + 1}:")
        st.write(q["question"])
        # Ensure options are strings for st.radio
        str_options = [str(opt) for opt in q["options"]]

        # Retrieve current answer or default to None for st.radio's index
        current_user_answer = st.session_state.user_answers.get(q['id'])
        try:
            current_index = str_options.index(current_user_answer) if current_user_answer in str_options else None
        except ValueError:  # Should not happen if current_user_answer is from options
            current_index = None

        st.session_state.user_answers[q['id']] = st.radio(
            label=f"Select your answer for Question {i + 1}:",  # Label is required
            options=str_options,
            key=q['id'],  # Unique key for each radio button
            index=current_index,  # Pre-select if answer exists
            label_visibility="collapsed"  # Hide the "Select your answer..." label above radio
        )

    if st.button("Submit Quiz", type="primary"):
        st.session_state.quiz_submitted = True
        st.rerun()


def grade_submitted_quiz():
    score = 0
    if not st.session_state.get('quiz_questions'):  # Check if quiz_questions exists
        st.error("Quiz questions not found in session. Please generate the quiz again.")
        return

    total = len(st.session_state.quiz_questions)
    if total == 0:
        st.info("No questions were in the quiz to grade.")
        return

    results_display = []

    for q in st.session_state.quiz_questions:
        user_answer = st.session_state.user_answers.get(q['id'])
        correct_answer = q["correct_answer"]
        is_correct = (user_answer == correct_answer)  # Direct string comparison

        if is_correct:
            score += 1
        results_display.append({
            "question": q["question"],
            "user_answer": user_answer if user_answer is not None else "Not Answered",
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

    st.subheader(f"Quiz Results: Your Score: {score}/{total}")
    for res_idx, res in enumerate(results_display):
        st.markdown(f"---")
        st.markdown(f"**Q{res_idx + 1}:** {res['question']}")
        st.markdown(f"- Your answer: `{res['user_answer']}`")
        st.markdown(f"- Correct answer: `{res['correct_answer']}`")
        if res["is_correct"]:
            st.success("Correct ✅")
        elif res['user_answer'] == "Not Answered":
            st.warning("Not Answered ⚠️")
        else:
            st.error("Incorrect ❌")
    st.markdown(f"---")

    # Button to reset quiz-specific state and allow trying another
    if st.button("Try Another Quiz or Document", key="reset_quiz_button"):
        keys_to_reset = ['quiz_questions', 'user_answers', 'quiz_submitted',
                         'current_quiz_doc', 'quiz_num_q', 'quiz_initialized_for_doc']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        # Potentially also reset flashcard state if quiz is closely tied
        # if 'current_flashcards' in st.session_state: del st.session_state['current_flashcards']
        # if 'flashcard_idx' in st.session_state: del st.session_state['flashcard_idx']
        st.rerun()


# ---------------------------------------------
# Streamlit Interface with Tabs
# ---------------------------------------------
st.set_page_config(layout="wide", page_title="AI Study Assistant")  # Added page title
st.title("📘 AI Study Assistant")

# Initialize session state variables robustly using .get() or direct assignment
# It's good practice to initialize all session state keys you expect to use.
if 'quiz_submitted' not in st.session_state: st.session_state.quiz_submitted = False
if 'user_answers' not in st.session_state: st.session_state.user_answers = {}
if 'quiz_questions' not in st.session_state: st.session_state.quiz_questions = []
if 'current_quiz_doc' not in st.session_state: st.session_state.current_quiz_doc = None
if 'quiz_num_q' not in st.session_state: st.session_state.quiz_num_q = 5  # Default number of questions
if 'quiz_initialized_for_doc' not in st.session_state: st.session_state.quiz_initialized_for_doc = False

if 'extracted_texts_global' not in st.session_state: st.session_state.extracted_texts_global = {}
if 'vector_dbs_global' not in st.session_state: st.session_state.vector_dbs_global = {}

if 'current_flashcards' not in st.session_state: st.session_state.current_flashcards = []
if 'flashcard_idx' not in st.session_state: st.session_state.flashcard_idx = 0
if 'show_definition' not in st.session_state: st.session_state.show_definition = False

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📤 Upload & Extract", "📝 Summarize", "❓ Question Answering", "🧠 Flashcards", "🎯 Quiz"])

with tab1:
    st.header("📤 Upload and Extract Text")
    uploaded_files = st.file_uploader("Upload PDF or Image files", type=["pdf", "png", "jpg", "jpeg"],
                                      accept_multiple_files=True, key="file_uploader")

    if uploaded_files:
        all_files_processed_successfully = True
        for file_obj in uploaded_files:
            # Process each file only once by checking if its name is already a key
            if file_obj.name not in st.session_state.extracted_texts_global:
                with st.spinner(f"Processing {file_obj.name}..."):
                    extracted_text, doc_title = extract_text_from_file(file_obj)
                    if extracted_text:
                        st.session_state.extracted_texts_global[doc_title] = extracted_text
                        st.success(f"Text Extracted Successfully from {doc_title}!")
                        # Store embeddings immediately after successful extraction
                        store_vector_in_session(extracted_text, doc_title)
                        # st.success(f"Embeddings stored for {doc_title}!") # Optional: store_vector_in_session can show its own errors
                    else:
                        st.warning(f"Could not extract any text from {file_obj.name}. It might be empty or corrupted.")
                        all_files_processed_successfully = False
            # else:
            # st.info(f"'{file_obj.name}' has already been processed and its text is available.")

        # Display extracted text (optional, can make UI busy if many files)
        # if st.session_state.extracted_texts_global:
        #     st.subheader("Preview Extracted Texts (First 500 Chars):")
        #     for title, text_content in st.session_state.extracted_texts_global.items():
        #         with st.expander(f"{title}"):
        #             st.text_area("Preview", value=text_content[:500], height=100, disabled=True, key=f"preview_{title}")

    if not st.session_state.extracted_texts_global:
        st.info("Upload PDF or image files to begin your AI-assisted study session.")
    else:
        st.subheader("📚 Processed Documents:")
        for title_key in st.session_state.extracted_texts_global.keys():
            st.markdown(f"- {title_key}")
        if st.button("Clear All Processed Documents", key="clear_docs_btn"):
            st.session_state.extracted_texts_global = {}
            st.session_state.vector_dbs_global = {}
            st.session_state.current_flashcards = []
            st.session_state.quiz_questions = []
            # Reset other relevant states if needed
            st.rerun()

with tab2:
    st.header("📝 Summarize Text")
    doc_titles_available = list(st.session_state.extracted_texts_global.keys())

    if not doc_titles_available:
        st.info("Please upload and extract text from a file in the 'Upload & Extract' tab first.")
    else:
        selected_doc_title_summary = st.selectbox(
            "Select document to summarize:",
            doc_titles_available,
            key="summarize_select",
            index=None,  # No default selection
            placeholder="Choose a document..."
        )
        if selected_doc_title_summary:  # Only show button if a document is selected
            if st.button("Generate Summary", key="summarize_btn", type="primary"):
                text_to_summarize = st.session_state.extracted_texts_global.get(selected_doc_title_summary)
                if text_to_summarize:
                    with st.spinner(f"Summarizing {selected_doc_title_summary}... This may take a moment."):
                        summary_text = summarize_text(text_to_summarize)

                    st.subheader(f"Summary for {selected_doc_title_summary}")
                    st.markdown(summary_text)

                    # Generate TTS for the summary if it's valid
                    if summary_text and "too short" not in summary_text.lower() and \
                            "cannot summarize" not in summary_text.lower() and \
                            "could not generate" not in summary_text.lower() and \
                            "no content suitable" not in summary_text.lower():
                        with st.spinner("Generating audio for summary..."):
                            audio_path = text_to_speech_gtts(summary_text)
                        if audio_path:
                            try:
                                with open(audio_path, "rb") as audio_file:
                                    st.audio(audio_file.read(), format="audio/mp3")
                                os.remove(audio_path)  # Clean up the temp audio file
                            except FileNotFoundError:
                                st.error("Audio file not found. Could not play summary audio.")
                            except Exception as e:
                                st.error(f"Error playing audio: {e}")
                else:
                    st.warning(f"Could not find the text content for {selected_doc_title_summary} to summarize.")
        elif st.session_state.summarize_select is None and doc_titles_available:  # if placeholder is active
            st.info("Select a document from the list above to generate its summary.")

with tab3:
    st.header("❓ Question Answering")
    doc_titles_qa_options = ["All Processed Documents"] + list(st.session_state.vector_dbs_global.keys())

    if len(doc_titles_qa_options) <= 1:  # Only "All Processed Documents" means no actual docs
        st.info("Please upload and process at least one file in the 'Upload & Extract' tab for Q&A.")
    else:
        selected_doc_title_qa = st.selectbox(
            "Search within document:",
            doc_titles_qa_options,
            key="qa_select_doc",
            index=0  # Default to "All Processed Documents"
        )
        question = st.text_input("Ask a question about the content:", key="qa_input",
                                 placeholder="E.g., What is the main idea?")

        if st.button("Get Answer", key="qa_btn", type="primary", disabled=(not question.strip())):
            # Button disabled if question is empty; no need for explicit warning here if using disabled
            # if not question.strip():
            #     st.warning("Please enter a question.")
            # else:
            with st.spinner("Searching for answer... This might take a moment."):
                # Adjust doc_title if "All Processed Documents" is selected for the function logic
                search_title = None if selected_doc_title_qa == "All Processed Documents" else selected_doc_title_qa
                answer = answer_question_from_session(question, doc_title=search_title)

            if answer:
                st.subheader("Answer:")
                st.markdown(answer)
                # TTS for answer
                with st.spinner("Generating audio for answer..."):
                    audio_path = text_to_speech_gtts(answer)
                if audio_path:
                    try:
                        with open(audio_path, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                        os.remove(audio_path)
                    except Exception as e:
                        st.error(f"Error playing answer audio: {e}")

            # else: answer_question_from_session handles its own st.warning/st.error for no answer

with tab4:
    st.header("🧠 Interactive Learning: Flashcards")
    doc_titles_flashcard = list(st.session_state.extracted_texts_global.keys())

    if not doc_titles_flashcard:
        st.info(
            "Please upload and extract text from a file in the 'Upload & Extract' tab first to generate flashcards.")
    else:
        selected_doc_flashcard = st.selectbox(
            "Generate flashcards from document:",
            doc_titles_flashcard,
            key="flashcard_select",
            index=None,
            placeholder="Choose a document..."
        )

        if selected_doc_flashcard:
            # Button to generate/regenerate flashcards
            if st.button("Generate Flashcards", key="flashcard_btn", type="primary"):
                text_for_flashcards = st.session_state.extracted_texts_global.get(selected_doc_flashcard)
                if text_for_flashcards:
                    with st.spinner(f"Generating flashcards from {selected_doc_flashcard}..."):
                        flashcards_list = generate_flashcards_from_text(text_for_flashcards)

                    if flashcards_list:
                        st.success(f"Generated {len(flashcards_list)} flashcards for {selected_doc_flashcard}.")
                        st.session_state.current_flashcards = flashcards_list
                        st.session_state.flashcard_idx = 0
                        st.session_state.show_definition = False
                        st.session_state.current_flashcard_doc = selected_doc_flashcard  # Track current doc for flashcards
                    else:
                        st.info(
                            "No flashcards could be generated. The document might be too short, lack clear term-definition patterns, or the content was unsuitable.")
                        st.session_state.current_flashcards = []  # Clear old flashcards
                else:
                    st.warning(f"Could not find text for {selected_doc_flashcard}.")

            # Display flashcards if they exist for the selected document or a previously selected one
            # This logic ensures flashcards remain visible even if user changes selectbox unless "Generate" is pressed for new doc
            # A better UX might be to clear flashcards if the selected_doc_flashcard changes and no new generate button is pressed.
            # For now, we check if current_flashcards exist.
            if st.session_state.current_flashcards and \
                    st.session_state.get('current_flashcard_doc') == selected_doc_flashcard:

                total_cards = len(st.session_state.current_flashcards)
                idx = st.session_state.flashcard_idx
                card = st.session_state.current_flashcards[idx]

                st.markdown(f"---")
                st.markdown(f"**Card {idx + 1} of {total_cards}** (from *{selected_doc_flashcard}*)")

                # Use columns for a cleaner layout
                col_term, col_def = st.columns(2)
                with col_term:
                    st.subheader("Term:")
                    st.markdown(f"### {card['term']}")

                with col_def:
                    if st.session_state.show_definition:
                        st.subheader("Definition:")
                        st.markdown(f"> {card['definition']}")
                    else:
                        st.subheader("Definition:")
                        st.markdown("> (Click 'Reveal Definition' to see)")

                # Navigation buttons
                nav_cols = st.columns([1, 2, 1])  # Previous, Reveal/Hide, Next
                with nav_cols[0]:  # Previous
                    if st.button("⬅️ Previous", disabled=(idx == 0), key="prev_flash_btn", use_container_width=True):
                        st.session_state.flashcard_idx -= 1
                        st.session_state.show_definition = False
                        st.rerun()
                with nav_cols[1]:  # Reveal/Hide
                    reveal_text = "Hide Definition" if st.session_state.show_definition else "Reveal Definition"
                    if st.button(reveal_text, type="secondary", key="reveal_flash_btn", use_container_width=True):
                        st.session_state.show_definition = not st.session_state.show_definition
                        st.rerun()
                with nav_cols[2]:  # Next
                    if st.button("Next ➡️", disabled=(idx == total_cards - 1), key="next_flash_btn",
                                 use_container_width=True):
                        st.session_state.flashcard_idx += 1
                        st.session_state.show_definition = False
                        st.rerun()
                st.markdown(f"---")
            elif st.session_state.get(
                    'current_flashcard_doc') != selected_doc_flashcard and selected_doc_flashcard is not None:
                st.info(f"Click 'Generate Flashcards' for '{selected_doc_flashcard}' to view its flashcards.")

        elif st.session_state.flashcard_select is None and doc_titles_flashcard:
            st.info("Select a document from the list above to generate flashcards.")

with tab5:
    st.header("🎯 Quiz Yourself!")
    doc_titles_quiz = list(st.session_state.extracted_texts_global.keys())

    if not doc_titles_quiz:
        st.info("Please upload and extract text from a file in the 'Upload & Extract' tab to generate a quiz.")
    else:
        selected_doc_for_quiz = st.selectbox(
            "Select document for Quiz:",
            doc_titles_quiz,
            key="quiz_doc_select",
            index=None,
            placeholder="Choose a document for the quiz..."
        )

        if selected_doc_for_quiz:
            num_q_options = [3, 5, 10, 15]  # Adjusted options, 3 might be good for small docs
            # Retrieve current slider value or default
            current_slider_val = st.session_state.get('quiz_num_q_slider_val', 5)

            num_questions_to_gen = st.select_slider(
                "Number of questions for the quiz:",
                options=num_q_options,
                value=current_slider_val,
                key="num_quiz_questions_slider_key"  # Ensure a unique key
            )
            st.session_state.quiz_num_q_slider_val = num_questions_to_gen  # Store the slider's current value

            # Generate Quiz / Reset logic
            if st.button("Generate / Reset Quiz", key="start_quiz_btn", type="primary"):
                with st.spinner(f"Generating {num_questions_to_gen} quiz questions for {selected_doc_for_quiz}..."):
                    quiz_text_content = st.session_state.extracted_texts_global.get(selected_doc_for_quiz)
                    if quiz_text_content:
                        generated_questions = generate_quiz_questions_from_text(quiz_text_content, num_questions_to_gen)

                        if generated_questions:
                            st.session_state.quiz_questions = generated_questions
                            # Initialize user_answers for the new set of questions
                            st.session_state.user_answers = {q['id']: None for q in st.session_state.quiz_questions}
                            st.session_state.quiz_submitted = False
                            st.session_state.current_quiz_doc = selected_doc_for_quiz
                            st.session_state.quiz_num_q = num_questions_to_gen  # Store the number of questions intended for this quiz
                            st.session_state.quiz_initialized_for_doc = True
                            st.success(
                                f"Quiz generated with {len(st.session_state.quiz_questions)} questions for {selected_doc_for_quiz}.")
                        else:
                            st.warning(
                                "Could not generate any quiz questions. The document might be too short, lack suitable content for flashcards, or fail to produce enough distinct options.")
                            st.session_state.quiz_initialized_for_doc = False
                            st.session_state.quiz_questions = []  # Ensure it's empty
                    else:
                        st.error(f"Could not retrieve text for {selected_doc_for_quiz}.")
                        st.session_state.quiz_initialized_for_doc = False
                        st.session_state.quiz_questions = []
                st.rerun()  # Rerun to reflect changes, especially if new questions are generated

            # Display quiz or results
            # Check if a quiz has been initialized for the *currently selected document and number of questions*
            quiz_is_relevant = (
                    st.session_state.quiz_initialized_for_doc and
                    st.session_state.current_quiz_doc == selected_doc_for_quiz and
                    st.session_state.quiz_num_q == num_questions_to_gen
            # Check if quiz params match current UI selections
            )

            if quiz_is_relevant and st.session_state.quiz_questions:
                if not st.session_state.quiz_submitted:
                    display_quiz_interface(st.session_state.quiz_questions)
                else:
                    grade_submitted_quiz()
            elif selected_doc_for_quiz:  # If a document is selected but quiz isn't relevant/generated
                # This condition helps guide the user if they change doc/num_q after generating a quiz
                if st.session_state.quiz_questions and not quiz_is_relevant:
                    st.info(
                        f"A quiz for '{st.session_state.current_quiz_doc}' with {st.session_state.quiz_num_q} questions is loaded. Click 'Generate / Reset Quiz' to create one for '{selected_doc_for_quiz}' with {num_questions_to_gen} questions.")
                elif not st.session_state.quiz_questions:  # No quiz loaded at all
                    st.info("Click 'Generate / Reset Quiz' with the desired document and number of questions.")


        elif st.session_state.quiz_doc_select is None and doc_titles_quiz:  # If placeholder active
            st.info("Select a document and number of questions to start a quiz.")