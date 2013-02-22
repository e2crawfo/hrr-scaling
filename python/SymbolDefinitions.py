isA_symbols = ['@', '@i']
holonym_symbols = ['#p', '#s', '#m']
inv_isA_symbols = ['~', '~i']
#vocab_symbols = isA_symbols + holonym_symbols + inv_isA_symbols

reflexive = {
    '@' : '~',
    '@i': '~i',
    '~' : '@',
    '~i': '@i',
    '#p': '%p',
    '#s': '%s',
    '#m': '%m',
    '%p': '#p',
    '%s': '#s',
    '%m': '#m'
    }

entails= ['*']                      # ENTAILS
ignore= ['^', '\\', '$', '<', '&'] # IGNORE
hyponymy = ['~', '~i']                # HYPONYMY
attribute = ['=']                      # ATTRIBUTE
antonym = ['!']                      # ANTONYM
causes = ['>']                      # CAUSES
meronymy = ['%s','%m','%p']           # MERONYMY

relationSymbols = [isA_symbols,                # HYPERNYMY
                   holonym_symbols,            # HOLONYMY
                   entails,
                   ignore,
                   hyponymy,
                   attribute,
                   antonym,
                   causes,
                   meronymy]

vocab_symbols = [r for i in relationSymbols for r in i]

sentence_symbols = {    # (frequency of occurrence, part of speech)
    'ss_subject': (1.0, 'n'),
    'ss_object' : (0.8, 'n'),
    'ss_verb'   : (1.0, 'v'),
    'ss_adverb' : (0.6, 'r'),
    'ss_sadj'   : (0.3, 'a'), # subject adjective
    'ss_oadj'   : (0.3, 'a')  # object adjective
    }
