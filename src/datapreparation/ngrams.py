import re


def ngrams(string: str, n=3):
    """
    Generate a full list of ngrams from a string.

        Parameters:
            string (str): Strings to generate ngrams from.
            n (int): Maximum length of the n-gram. Defaults to 3.

        Returns:
            ngrams_lst (list): Returns a list of ngrams generated from the input string.
    """

    # Remove Punctuation from the string
    pattern = r'[,-./]|\s'
    string = re.sub(pattern, r'', string)

    # Generate zip of ngrams (n defined in function argument)
    n_grams = zip(*[string[i:] for i in range(n)])

    # Return ngram list
    return [''.join(n_gram) for n_gram in n_grams]
