import re
import string

def cleanup_string(sentence):
    sentence = sentence.encode('ascii', 'ignore').decode('ascii')
    sentence = re.sub(r'https\:\/\/\S+', '', sentence)
    
    #sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def cleanup_object(input):
    if isinstance(input, str):
        return cleanup_string(input)
    elif isinstance(input, list):
        return [cleanup_object(item) for item in input]
    else:
        return input

def cleanup_dataframe(df):
    for col in df.columns:
        df[col] = df[col].apply(cleanup_object)

# FORMAT CHECKER:
_LINE_PATTERN_A = re.compile('^\d+\t.*\t\w+$') # <id> <TAB> <text> <TAB> <class_label>

def check_format(input_file): # Kanskje endre fra 'input_file' til 'datasets' (type: dict)...
    with open(input_file, encoding='utf-8') as f:
        next(f)
        file_content = f.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            if _LINE_PATTERN_A.match(line):
                id, text, class_label = line.split('\t') # Usikker om dette trengs... kan bruke 'class_label' til å sjekke med kalkulert/output label
            else: 
                print(f"Wrong line format: {line}")
                return False
    
    print(f"File '{input_file}' is in the correct format.")
    return True


# CLAIM CHECKER (TODO):
def is_claim(text):
    text_lower = text.lower()
    if is_claim:
        return 'Yes'
    else:
        return 'No'