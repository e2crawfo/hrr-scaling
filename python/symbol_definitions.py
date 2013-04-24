_isA_symbols = ['@']
#_isA_symbols = ['@', '@i']
holonym_symbols = ['#p', '#s', '#m']
inv_isA_symbols = ['~', '~i']
_uni_symbols = _isA_symbols + holonym_symbols 

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

relationSymbols = [_isA_symbols,                # HYPERNYMY
                   holonym_symbols,            # HOLONYMY
                   entails,
                   ignore,
                   hyponymy,
                   attribute,
                   antonym,
                   causes,
                   meronymy]

_bi_symbols = [r for i in relationSymbols for r in i]

_sentence_role_symbols = {    # (frequency of occurrence, part of speech)
    'ss_subject': (1.0, 'n'),
    'ss_object' : (0.8, 'n'),
    'ss_verb'   : (1.0, 'v'),
    'ss_adverb' : (0.6, 'r'),
    'ss_sadj'   : (0.3, 'a'), # subject adjective
    'ss_oadj'   : (0.3, 'a')  # object adjective
    }

def isA_symbols():
  """Return isA relation symbols"""
  return _isA_symbols

def uni_relation_symbols():
  """Return unidirectional relation symbols"""
  return _uni_symbols

def bi_relation_symbols():
  """Return bidirectional relation symbols"""
  return _bi_symbols

def sentence_role_symbols():
  return _sentence_role_symbols

