# Regular Expressions

import re

text = "1959 I was born in 1969"

# Συνάρτηση match()
# Local αναζήτηση. Ψάχνει να βρει το pattern από την ΑΡΧΗ της πρότασης
re.match(r"1959", text)  # Το βρίσκει
re.match(r"1969", text)  # Δεν το βρίσκει
# Επίσης ψάχνει ολόκληρες λέξεις αν δεν χρησιμοποιηθούν wildcards
re.match(r"959", text)  # Δεν το βρίσκει
re.match(r".959", text)  # Το βρίσκει

# Συνάρτηση search()
# Global αναζήτηση. Ψάχνει να βρει το pattern ΠΑΝΤΟΥ στην πρόταση. ΔΕΝ χρειάζεται
# να είναι ολοκληρη λέξη
re.search(r'[or]+', text)  # Το βρίσκει

# Starts with.
# Αν όλο το κείμενο ξεκινάει με ένα συγκεκριμένο pattern.
# Χρησιμοποιεί το σύμβολο ^. Συνδυάζεται με match() και search(). Global αναζήτηση
re.match(r"^195", text)  # Το βρίσκει
re.search(r"^195", text)  # Το βρίσκει
re.match(r"^1969", text)  # Δεν το βρίσκει

# Ends with.
# Αν όλο το κείμενο τελειώνει με ένα συγκεκριμένο pattern.
# Χρησιμοποιεί το σύμβολο $. Συνδυάζεται ΜΟΝΟ με search(). Global αναζήτηση
re.search(r"1969$", text)  # Το βρίσκει
re.match(r"1969$", text)  # Δεν το βρίσκει
re.search(r"69", text)  # Το βρίσκει

# Συνάρτηση sub()
# Αντικαθιστά ΟΛΕΣ τις εμφανίσεις ενός pattern με ένα άλλο. Δεν μεταβάλλει το
# αρχικό κείμενο (πρέπει να γίνει εκ νέου ανάθεση)
re.sub("19", "20", text, count=1)
# Με την παράμετρο counts μπορώ να καθορίσω πόσες εμφανίσεις αντικαθίστανται
e.sub("19", "20", text, count=1)
# Με την παράμετρο flags=re.I δηλωνω την μετατροπή ως case insensitive
re.sub("[a-z]", "0", text)
re.sub("[a-z]", "0", text, flags=re.I)
# Τροποποιείται το αρχικό κείμενο
text = re.sub("1969", "1979", text)


#  Regex Cheat Sheet
#  
#  Basics
#  *    Match preceding character 0 or more times
#  +    Match preceding character 1 or more times
#  .    Match any single character
#  x|y  Match either 'x' or 'y'
#      Escape a special character
#  b    The character b
#  abc  The string abc
#  
#  Character Classes I
#  d    Match a digit character
#  D    Match a non-digit character
#  s    Match a single white space character (space, tab, form feed, or line feed)
#  S    Match a single character other than white space
#  w    Match any alphanumeric character (including underscore)
#  W    Match any non-word character
#  
#  Character Classes II
#  [abc] Match any one of the characters in the set 'abc'
#  [^abc] Match anything not in character set 'abc'
#  [b]  Match a backspace
#  
#  Assertions
#  ^    Match beginning of input
#  $    Match end of input
#  b   Match a word boundary
#  B   Match a non-word boundary
#  ?=   Lookahead
#  ?!   Negative lookahead
#  
#  Assertions II
#  ?<=  Lookbehind
#  ?<!  Negative lookbehind
#  ?>   Once-only subexpression
#  ?()  If then condition
#  ?()| If then else condition
#  ?#   Comment
#  
#  Quantifiers
#  {n}  Match exactly n occurrences of preceding character
#  {n,m} Match at least n and at most m occurrences of the preceding character
#  ?    Match 0 or 1
#  
#  Special Characters I
#  cX  Match control character X in a string
#  \n   Match a line feed
#  \r   Match a carriage return
#  \t   Match a tab
#  0   Match a NULL
#  
#  Special Characters II
#  \f   Match a form feed
#  v   Match a vertical tab
#  xhh Match character with code hh (2 hex digits)
#  uhhhh Match character with code hhhh (4 hex digits)
#  
#  Flags
#  g    Global search
#  i    Case-insensitive search
#  m    Multi-line search
#  y    "sticky" search match starting at current position in target string
#  
#  Groups
#  (x)   Match 'x' and remember the match
#  (?:x) Match 'x' but do not remember the match
#  \n    A back reference to the last substring matching the n parenthetical in the regex 


