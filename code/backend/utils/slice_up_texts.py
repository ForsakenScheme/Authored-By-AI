import os
from backend.utils.log import get_logger

logger = get_logger(__name__)

def slice_up_local_texts(data_language = "english"):
    """
    For folder in raw folder,
    for file in ai/human,
    get text from file and slice up according to paragraphs,
    output the individual paragraphs as new files in ./backend/data/processed/name_of_folder_it_was_from (ai, human)/index.txt

    Example for paragraphs from "1.txt" from ai :

    Should be stored as

    - ./backend/data/processed/ai/1_1.txt
    - ./backend/data/processed/ai/1_2.txt
    - ./backend/data/processed/ai/1_3.txt

    and so forth for each paragraph contained in the first essay.

    """
    data_path = os.path.join(os.getcwd(), "code", "backend", "data", data_language)
    raw_path = os.path.join(data_path, "raw")
    processed_path = os.path.join(data_path, "processed")
    valid_directories = ["human", "ai"]

    for dir in valid_directories:
        try:
            subdir = os.path.join(raw_path, dir)
            if os.path.exists(subdir) and os.path.isdir(subdir):
                for root, _, files in os.walk(subdir):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        split_file_name = os.path.splitext(filename)
                        with open(filepath, "r", encoding="utf-8") as file_in:
                            paragraphs = file_in.read().split("\n")
                            paragraph_nb = 1
                            # get rid of empty paragraphs
                            paragraphs = [
                                paragraph for paragraph in paragraphs if paragraph
                            ]
                            for paragraph in paragraphs:
                                with open(
                                    os.path.join(
                                        processed_path, dir, split_file_name[0]
                                    )
                                    + "_"
                                    + str(paragraph_nb)
                                    + split_file_name[1],
                                    "w",
                                    encoding="utf-8",
                                ) as file_out:
                                    file_out.write(paragraph)
                                paragraph_nb += 1
        except OSError as e:
            print(
                f"Error {e} :  an error occured while trying to read the directory {subdir}."
            )
    logger.info("Texts have been sliced up.")


def slice_up_raw_db_texts(texts, labels):
    """
    For each text in texts, slice up according to paragraphs,
    output the individual paragraphs as new files entries with their corresponding labels as list of sliced texts and their labels.

    Parameters:

    texts: list of strings representing texts extracted from the raw database.
    labels: list of strings representing the labels of the texts extracted from the raw database.

    Returns:

    sliced_texts: list of strings representing sliced texts.
    sliced_labels: list of strings representing sliced_texts labels.
    """
    sliced_texts = []
    sliced_labels = []
    for text, label in zip(texts, labels):
        paragraphs = text.split("\n")
        # Get rid of empty paragraphs
        paragraphs = [
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        ]
        sliced_texts.extend(paragraphs)
        sliced_labels.extend([label] * len(paragraphs))
    logger.info("Texts from raw database table have been sliced up.")
    return sliced_texts, sliced_labels
