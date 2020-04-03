#!/usr/bin/env python3


def preprocDF(df, negations=False):
    '''Data preprocessing των δεδομένων. Δέχεται κείμενο και κάνει:

        - Μετατροπή των κεφαλαίων χαρακτήρων σε πεζούς
        - Αφαίρεση των μη αλφαριθμητικών χαρακτήρων
        - Αφαίρεση βασικών σημεών στήξης
        - Διαχωρισμό σε λέξεις
        - Χειρισμό αρνήσεων (προαιρετικά)
    Παράμετροι:
        - df: Τα δεδομένα εισόδου (κείμενο) σε δομή DataFrame
        - negations: boolean, optional (default=False). Κατά πόσο θα χειριστεί τις αρνήσεις.
        
    :Example:
    >>> from commonModules import preprocDF
    >>> preprocDF(dataDF, negations=True)
    >>> 
    '''
    import pandas as pd
    import re
    from nltk.tokenize import word_tokenize

    # Απενεργοποίηση κάποιων συχνών προειδοποιητικών μηνυμάτων των pandas
    pd.options.mode.chained_assignment = None

    for i in range(len(df)):
        df['Sentences'][i] = df['Sentences'][i].lower()
#       Deal with Negations. Χρησιμοποιώ μερικές κλασικές λέξεις άρνησης
        if(negations):
            negation_words = ['no','not',"don't","didn't",'never']
            stop_negation_words = ['\.', ',', '!', ';', '\?']
            for word in negation_words:
                df['Sentences'][i] = re.sub(word,' NEG_WORD ',df['Sentences'][i])
            for word in stop_negation_words:
                df['Sentences'][i] = re.sub(word,' STOP_NEG ',df['Sentences'][i])

        df['Sentences'][i] = re.sub(r'\W+',' ',df['Sentences'][i])
        df['Sentences'][i] = re.sub(r'[\(\)\[\]~\.\?\',/\\]',' ',df['Sentences'][i])
        df['Sentences'][i] = re.sub(r'\s+',' ',df['Sentences'][i])
        df['Sentences'][i] = word_tokenize(df['Sentences'][i])
#       Deal with Negations.
        if(negations):
            for k in range(len(df['Sentences'][i])):
                if df['Sentences'][i][k] == 'NEG_WORD' and k < (len(df['Sentences'][i])-1):
                    j = k + 1
                    while j < len(df['Sentences'][i]) and df['Sentences'][i][j] not in ['NEG_WORD', 'STOP_NEG']:
                        df['Sentences'][i][j] = 'NOT_'+df['Sentences'][i][j]
                        j += 1
            df['Sentences'][i] = [w for w in df['Sentences'][i] if w not in ['NEG_WORD', 'STOP_NEG']]

