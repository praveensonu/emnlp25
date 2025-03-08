import spacy
from nltk.metrics.distance import edit_distance

def print_similarities(embeddings_1, embeddings_2, similarities):
    """
    Prints each sentence from embeddings_1 followed by each sentence from embeddings_2
    along with their corresponding similarity score from the similarities matrix.

    Parameters:
        embeddings_1 (list): A list of sentences (or embeddings) for the first set.
        embeddings_2 (list): A list of sentences (or embeddings) for the second set.
        similarities (list of list of float): A 2D list where each element similarities[i][j]
                                              is the similarity score between embeddings_1[i]
                                              and embeddings_2[j].
    """
    for idx_i, sentence1 in enumerate(embeddings_1):
        print(sentence1)
        for idx_j, sentence2 in enumerate(embeddings_2):
            print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")


def get_pos_sequence(sentence):
    """
    Parse the sentence and return its sequence of POS tags.
    """
    doc = nlp(sentence)
    return [token.pos_ for token in doc]

def syntactic_similarity(sentence1, sentence2):
    """
    Compute a syntactic similarity score based on the edit distance
    between the sequences of POS tags from two sentences.

    The score is normalized between 0 and 1, where 1 indicates identical structure.
    """
    pos_seq1 = get_pos_sequence(sentence1)
    pos_seq2 = get_pos_sequence(sentence2)

    # Compute the edit distance between the two POS tag sequences.
    distance = edit_distance(pos_seq1, pos_seq2)

    # Normalize the distance by the length of the longer sequence.
    max_len = max(len(pos_seq1), len(pos_seq2))
    normalized_distance = distance / max_len if max_len != 0 else 0

    # A lower normalized distance means higher similarity.
    similarity = 1 - normalized_distance
    return similarity

def compare_sentence_lists(sentences_1, sentences_2):
    """
    Compare each sentence in sentences_1 to each sentence in sentences_2 and return
    a list of tuples containing the two sentences and their syntactic similarity score.
    """
    results = []
    for sent1 in sentences_1:
        for sent2 in sentences_2:
            score = syntactic_similarity(sent1, sent2)
            results.append((sent1, sent2, score))
    return results