import numpy as np
import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences


def extract_questions_from_dataframe(questions_dataframe, config, word2idx,
                                     prediction_mode=False):
    if prediction_mode:
        loaded_file = _load_dataframe_from_file('extracted_questions_test.npz')
        if loaded_file:
            return loaded_file['questions_A'], loaded_file['questions_B'], None
    else:
        loaded_file = _load_dataframe_from_file('extracted_questions_train.npz')
        if loaded_file:
            return loaded_file['questions_A'], loaded_file['questions_B'], \
                   loaded_file['labels']

    questions_A = []
    questions_B = []
    labels = []
    for i, row in questions_dataframe.iterrows():
        question_A = str(row['question1'])
        question_B = str(row['question2'])
        # if not prediction_mode and \
        #         config.MAX_SENTENCE_LENGTH and \
        #         (len(question_A) < config.MAX_SENTENCE_LENGTH or
        #          len(question_B) < config.MAX_SENTENCE_LENGTH):
        #     continue
        questions_A.append(question_A)
        questions_B.append(question_B)
        if 'is_duplicate' in row:
            labels.append(row['is_duplicate'])

    questions_A, questions_B = clean(questions_A), clean(questions_B)
    questions_A = cast_to_word_indices(questions_A, word2idx, config)
    questions_B = cast_to_word_indices(questions_B, word2idx, config)
    questions_A = pad_sequences(
        questions_A,
        maxlen=config.MAX_SENTENCE_LENGTH,
        value=config.PAD_TOKEN
    )
    questions_B = pad_sequences(
        questions_B,
        maxlen=config.MAX_SENTENCE_LENGTH,
        value=config.PAD_TOKEN
    )
    labels = np.array(labels)

    if prediction_mode:
        np.savez(
            'extracted_questions_test',
            questions_A=questions_A,
            questions_B=questions_B
        )
    else:
        np.savez(
            'extracted_questions_train',
            questions_A=questions_A,
            questions_B=questions_B,
            labels=labels
        )
    print('{num} questions preprocessed'.format(num=len(questions_A)))
    return questions_A, questions_B, labels


def _load_dataframe_from_file(file_path):
    try:
        # Try to load previously preprocessed data
        loaded_file = np.load(file_path)
        print('{num} preprocessed questions loaded from disk'.format(
            num=len(loaded_file['questions_A'])))
        return loaded_file
    except IOError:
        print('No saved file, preprocessing from scratch')
        return None


def clean(questions):
    cleaned_questions = []
    for text in questions:
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        cleaned_questions.append(text.split())
    return cleaned_questions


def cast_to_word_indices(questions, word2idx, config):
    result = []
    for question in questions:
        result.append([word2idx[word] if word in word2idx else config.OOV_TOKEN
                       for word in question] + [config.EOS_TOKEN])
    return result


def load_embeddings(file_path, config):
    word_vectors = np.zeros(
        shape=(config.VOCABULARY_SIZE + config.OFFSET,
               config.EMBEDDING_DIMENSION)
    )
    word2idx = {}
    i = config.OFFSET
    for line in open(file_path, encoding='utf-8', mode='r'):
        items = line.replace('\r', '').replace('\n', '').split(' ')
        if len(items) < 10:
            continue
        word = items[0]
        word_vectors[i, :] = np.array([float(j) for j in items[1:]])
        word2idx[word] = i
        i += 1
    return word_vectors, word2idx


def save_submission(predictions, config):
    predictions = predictions.ravel()
    submission = pd.DataFrame({'is_duplicate': predictions})
    submission.to_csv(
        path_or_buf=config.stamp(comment='1') + '.csv',
        index_label='test_id'
    )
    return submission
