import os
import sys
from typing import Optional
from datasets import load_dataset
import dacy



from collections import defaultdict
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc


sys.path.append(".")
sys.path.append('../..')
import dfm
from description_pattern_lists import (danish_pornographic_terms,)


def terms_to_lowercase_match_patterns(
    term_list: list, label: Optional[str] = None, label_prefix: str = ""
) -> list:
    """
    Takes a list of terms and creates a list of SpaCy patterns in the shape {"label": [{"LOWER": "term"}]}
    """
    out_list = []

    for term in term_list:
        if label is None:
            cur_label = label_prefix + term
            out_list.append({cur_label: [{"LOWER": term}]})
        else:
            cur_label = label_prefix + label
            out_list.append({cur_label: [{"LOWER": term}]})

    return out_list



def remove_irrelevant_columns(ds):
    return ds.remove_columns(["LICENSE", "uri", "date_built"])


def gen_matcher_object_from_pattern_list(
    pattern_container_list: list, nlp: Language
) -> Matcher:
    """
    Generates matcher objects from a list of dictionaries with {matcher_label (str): pattern (list)}
    Pattern must conform to SpaCy pattern standards:

    Example:
        >>> pattern_container_list = [
        >>>    {"atheism": [{"LOWER": {"REGEX": "athei.+"}}]},
        >>>    {"atheism": [{"LOWER": {"REGEX": "atei.+"}}]},
        >>>    {"skøde": [{"LOWER": "skøde"}]},
        >>> ]
    """
    matcher_object = Matcher(nlp.vocab)

    for pattern_container in pattern_container_list:
        pattern_label, subpattern_list = list(*pattern_container.items())

        matcher_object.add(pattern_label, [subpattern_list])

    return matcher_object


def get_match_counts_from_doc(doc: Doc, matcher_object: Matcher, nlp: Language) -> dict:
    """
    Get match counts for a list of SpaCy matcher-objects

    args:
        doc (Doc)
        pattern_container_list (list): A list of dictionaries fitting SpaCy patterns
        nlp: Language

    returns:
        A dictionary of the format {pattern_label (str): count (int)}.
    """

    counts = defaultdict(int)

    # Make sure that all elements are represented in the dict
    for pattern in matcher_object._patterns:
        pattern_label = nlp.vocab.strings[pattern]

        counts[pattern_label] = 0

    for match_id, start, end in matcher_object(doc):
        counts[nlp.vocab.strings[match_id]] += 1

    return dict(counts)


def get_match_counts_from_batch(batch, matcher_object: Matcher, nlp: Language) -> dict:
    """
    Takes a spacy batch of docs and processes them into a dictionary with
    {match_label (str): match_counts (list of ints)}
    """

    docs = nlp.pipe(batch["text"])

    batch_match_counts = defaultdict(list)

    for doc in docs:
        doc_match_counts = get_match_counts_from_doc(doc, matcher_object, nlp)

        for pattern_label in doc_match_counts.keys():
            pattern_match_count = doc_match_counts.get(pattern_label, 0)

            batch_match_counts[pattern_label].append(pattern_match_count)

    return dict(batch_match_counts)


if __name__ == "__main__":
    any_token_pattern = [{"tokens": [{"TEXT": {"REGEX": ".+"}}]}]

    
    ############################
    # Danish pornographic words #
    ###########################



    pornographic_patterns = terms_to_lowercase_match_patterns(
            danish_pornographic_terms, label_prefix="porn_"
    )


    combined_patterns = (
        any_token_pattern
        + pornographic_patterns
    )

    nlp = spacy.blank("da")
    nlp.max_length = 50000000

    matcher_objects = gen_matcher_object_from_pattern_list(combined_patterns, nlp)
    
    ds = dfm.load_dfm_dataset('dagw')

    ds_sharded = ds.shuffle().shard(num_shards=100, index=0)  # Work on 1/10th of DGW

    dgw_processed = ds_sharded.map(
        lambda batch: get_match_counts_from_batch(batch, matcher_objects, nlp),
        batched=True,
        batch_size=50,
        num_proc=16,
    )

    if not os.path.exists("csv"):
        os.makedirs("csv")

    remove_irrelevant_columns(dgw_processed).to_csv("csv/dagw2.csv")
