
import re
import random
import string
import os
from datasets import Dataset, DatasetDict


""" #### Functions for Generating Data"""


def number_to_word(n):
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return words[n]

def tokenize_formula(formula):
    tokens = re.findall(r'\b[A-Za-z_0-9]+\b', formula)
    exclude = {'F', 'G', 'U', 'X', 'R', 'W', 'M'}  # Exclude LTL operators
    return set(token for token in tokens if token not in exclude)

def generate_random_word(characters, min_length, max_length, seed=None):
    if seed is not None:
        random.seed(seed)
    word_length = random.randint(min_length, max_length)
    return ''.join(random.choice(characters) for _ in range(word_length))

def replace_phrases_with_random_words(data, random_words=None, characters=string.ascii_letters + "_%#", min_length=5, max_length=20,  seed="yes"):
    # random.seed(seed)
    
    # if random_words is None:
    #     random_words = [generate_random_word(characters, min_length, max_length, seed=seed) for _ in range(len(data["ltl"]))] # Generate a big pool
    # print(len(data["ltl"]))
    updated_ltl = []
    updated_eng = []
    
    for index, (formula, sentence) in enumerate(zip(data["ltl"], data["en"])):
        if seed == "yes":
            seed = index
        else:
            seed = None
        ltl_phrases = tokenize_formula(formula)
        # print(ltl_phrases)
        eng_phrases = {}
        for token in ltl_phrases:
            if "_" in token:
                parts = token.split("_")
                if parts[1].isdigit():
                    eng_phrase = parts[0] + " " + number_to_word(int(parts[1]))
                else:
                    eng_phrase = " ".join(parts)
            else:
                eng_phrase = token  # For tokens without underscores
            eng_phrases[token] = eng_phrase

        word_mappings = {}
        if random_words:
             unused_words = list(random_words)  # Reset the list of unused words for each pair

        for eng_phrase in eng_phrases.values():
            # If we have a random list
            if random_words:
                if seed is not None:
                    random.seed(seed)
                # random.seed(index)
                word = random.choice(unused_words)
                unused_words.remove(word)  # Remove the word from the unused list
            # Else if we want to generate random words
            else:  
                word = generate_random_word(characters, min_length, max_length, seed=seed)

            word_mappings[eng_phrase] = word


        for ltl_phrase, eng_phrase in eng_phrases.items():
            formula = re.sub(r'\b' + ltl_phrase + r'\b', word_mappings[eng_phrase], formula)
            sentence = sentence.replace(eng_phrase, word_mappings[eng_phrase])

        updated_ltl.append(formula)
        updated_eng.append(sentence)

    return Dataset.from_dict({"ltl": updated_ltl, "en": updated_eng})


""" #### Basic print function to Compare new data"""

def printCompare(dataset, modset):
    print(dataset["ltl"][1])
    print(dataset["en"][1])
    print(modset["ltl"][1])
    print(modset["en"][1])

""" #### Write generated datasets to a text file """

def write_array_to_file(data, filename="output.txt"):
    
    with open(filename, "w") as file:
        for item in data:
            file.write(item + "\n")

def write_datasetDict_to_txt(dataset_dict, output_dir, dir_name, dtype):

    test_dir = os.path.join(output_dir, dir_name)
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for key, dataset in dataset_dict.items():
        write_array_to_file(dataset["en"], f"{test_dir}/{dtype}-{key}-eng.txt")
        print(f"created {output_dir}/{dtype}-{key}-eng.txt")
        write_array_to_file(dataset["ltl"], f"{test_dir}/{dtype}-{key}-ltl.txt")
        print(f"created {output_dir}/{dtype}-{key}-ltl.txt")

def write_tok_datasetDict(dataset_dict, output_dir):

    test_dir = os.path.join(output_dir, "tokenized_datasets")
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for key, dataset in dataset_dict.items():
        write_array_to_file(dataset["en"], f"{test_dir}/test-{key}-eng.txt")
        print(f"created {output_dir}/test-{key}-eng.txt")
        write_array_to_file(dataset["ltl"], f"{test_dir}/test-{key}-ltl.txt")
        print(f"created {output_dir}/test-{key}-ltl.txt")
        write_array_to_file(dataset["input_ids"], f"{test_dir}/test-{key}-input_ids.txt")
        print(f"created {output_dir}/test-{key}-input_ids.txt")
        write_array_to_file(dataset["attention_mask"], f"{test_dir}/test-{key}-attn.txt")
        print(f"created {output_dir}/test-{key}-attn.txt")
        write_array_to_file(dataset["labels"], f"{test_dir}/test-{key}-labels.txt")
        print(f"created {output_dir}/test-{key}-labels.txt")

def generate_test_dataset(og_data, random_words, unique_characters, polysemous_words):
    
    print("Rare words \n")
    rare_test = replace_phrases_with_random_words(og_data, random_words=random_words)
    printCompare(og_data, rare_test)

    print("\n\n")

    print("Random words (Gibberish) \n")
    random_test = replace_phrases_with_random_words(og_data, characters=string.ascii_letters, min_length=5, max_length=8)
    printCompare(og_data, random_test)


    print("\n\n")

    print("Non-standard characters \n")
    nonstd_test = replace_phrases_with_random_words(og_data, characters=string.ascii_letters + unique_characters, min_length=5, max_length=8)
    printCompare(og_data, nonstd_test)

    print("\n\n")

    print("Contexualized (polysemous) words \n")
    poly_test = replace_phrases_with_random_words(og_data, random_words=polysemous_words)
    printCompare(og_data, poly_test)

    test_dataset = DatasetDict({
        'original': og_data,
        'rare': rare_test,
        'random': random_test,
        'nonstd': nonstd_test,
        'poly': poly_test,
    })

    # print(test_dataset)

    return test_dataset


""" #### List of inputs for generating Test data """

unique_characters = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØŒßÞÙÚÛÜÝÿŠŽĆČŁŃŇŘŤŮŻąćęłńśźżđħĳĸŀħċğĕėęĝĞĠġĥĦƒĄĒĘĚĔĖƐƏƒǺǻǼǽǾǿȀȁȂȃȄȅȆȇȈȉȊȋȌȍȎȏȐȑȒȓȔȕȖȗɂɃɄɅɆɇɈɉɊɋɌɍɎɏẀẁẂẃẄẅẆẇẈẉẊẋẌẍẎẏẐẑẒẓẔẕẖẗẘẙẚẛẜẝẞẟẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"

polysemous_words = [
    "address",
    "arm",
    "back",
    "bank",
    "bar",
    "bear",
    "bolt",
    "book",
    "bounce",
    "brush",
    "bug",
    "bust",
    "call",
    "can",
    "chair",
    "charge",
    "check",
    "clip",
    "close",
    "date",
    "deck",
    "draft",
    "drill",
    "drop",
    "file",
    "flag",
    "float",
    "fly",
    "fold",
    "frame",
    "grill",
    "hammer",
    "head",
    "hold",
    "jam",
    "jerk",
    "juggle",
    "jump",
    "kick",
    "knot",
    "lap",
    "light",
    "match",
    "mind",
    "mint",
    "nail",
    "note",
    "park",
    "pass",
    "peel",
    "pilot",
    "pin",
    "pitch",
    "plunge",
    "point",
    "post",
    "pound",
    "press",
    "pump",
    "race",
    "rain",
    "ring",
    "rock",
    "roll",
    "roof",
    "rush",
    "sack",
    "sail",
    "scale",
    "seal",
    "shake",
    "ship",
    "shower",
    "skirt",
    "slip",
    "sound",
    "spring",
    "stamp",
    "steer",
    "stick",
    "string",
    "stroke",
    "switch",
    "tap",
    "tear",
    "tire",
    "toast",
    "top",
    "trip",
    "vent",
    "wave",
    "whip",
    "wound",
    "wrap",
    "yard",
    "zoom"
]


random_words = [
    "absquatulate",   # To leave hurriedly.
    "jentacular",     # Pertaining to breakfast.
    "cachinnate",     # To laugh loudly.
    "yarborough",     # A hand of cards with no card above a nine.
    "floccinaucinihilipilification",   # The act of describing or regarding something as unimportant.
    "nudiustertian",  # Relating to the day before yesterday.
    "limerence",      # The state of being infatuated with another person.
    "susurrus",       # Whispering or rustling sound.
    "defenestration", # The act of throwing someone out a window.
    "agastopia",      # Admiration of a particular part of someone's body.
    "ultracrepidarian",  # A person who gives opinions on subjects they know nothing about.
    "bibliopole",     # A person who buys and sells books.
    "cynosure",       # A person or thing that is the center of attention or admiration.
    "digerati",       # People with expertise or professional involvement in information technology.
    "esemplastic",    # Having the ability to shape diverse elements or concepts into a unified whole.
    "fugacious",      # Transient or fleeting.
    "gargalesthesia", # The sensation caused by tickling.
    "heterodox",      # Not conforming with accepted standards or beliefs.
    "illecebrous",    # Alluring.
    "jumentous",      # Smelling like horse urine.
    "kenspeckle",     # Conspicuous or easily recognizable.
    "logomachy",      # An argument about words.
    "mellifluous",    # A sound that is sweet and smooth.
    "nepenthe",       # Something that can make you forget pain or sorrow.
    "obambulate",     # To walk around aimlessly.
    "peregrinate",    # To travel or wander around.
    "quixotic",       # Extremely idealistic, unrealistic, or impractical.
    "rumbustious",    # Boisterous or unruly.
    "slubberdegullion", # A slovenly or slobbish person.
    "triskaidekaphobia", # Fear of the number 13.
    "uxorious",       # Excessively fond of or submissive to one's wife.
    "ventriloquy",    # The art of speaking in such a manner that the voice appears to come from elsewhere.
    "widdershins",    # In a left-handed, wrong, or contrary direction.
    "xenoglossy",     # The ability to speak a language without having learned it.
    "yonderly",       # Mentally or emotionally distant; absent-minded.
    "zeugma",         # The use of a word to modify two or more words when it should only modify one.
    "ailurophile",    # A person who loves cats.
    "borborygm",      # The sound of gas traveling through the intestines.
    "catawampus",     # Askew or awry.
    "doodle-sack",    # Old English word for bagpipe.
    "ephemeral",      # Lasting for a very short time.
    "farraginous",    # Consisting of a confused mixture; formed of various materials in no fixed order or arrangement.
    "gobbledygook",   # Language that is meaningless or hard to understand.
    "hobbledehoy",    # An awkward adolescent boy.
    "illuminati",     # People claiming to possess special enlightenment or knowledge of something.
    "jiggumbob",      # A thingamajig or whatchamacallit.
    "kludge",         # A workaround or quick-and-dirty solution that is clumsy yet effective.
    "lollygag",       # To spend time aimlessly; to dawdle.
    "mugwump",        # An independent politician who does not follow any party.
    "noodle",         # A person's head.
    "oxter",          # Old word meaning armpit.
    "pauciloquent",   # Uttering few words; brief in speech.
    "quockerwodger",  # A wooden puppet controlled by strings.
    "ratoon",         # A small shoot growing from the root of a plant.
    "sialoquent",     # Spitting while speaking.
    "tittynope",      # A small quantity of something left over.
    "ulotrichous",    # Having woolly or crispy hair.
    "vex",            # To irritate or annoy.
    "wabbit",         # Exhausted or slightly unwell.
    "xertz",          # To gulp something down quickly and greedily.
    "yex",            # A hiccup or belch.
    "zoanthropy",     # Delusion of a person who believes they've changed into an animal.
    "abibliophobia",  # The fear of running out of reading material.
    "bloviate",       # To speak at length in a pompous or boastful manner.
    "callypygian",    # Having a beautiful backside.
    "dactylonomy",    # Counting using one's fingers.
    "ergophobia",     # Fear of work.
    "frugivorous",    # Fruit-eating.
    "gallivant",      # To go around from one place to another in the pursuit of pleasure or entertainment.
    "hullabaloo",     # Uproar or commotion.
    "inaniloquent",   # Pertaining to idle talk.
    "juxtapose",      # To place side by side for comparison.
    "kakorrhaphiophobia", # Fear of failure.
    "ludibrious",     # Apt to be a subject of jest or mockery.
    "mumpsimus",      # A traditional custom or notion adhered to although shown to be unreasonable.
    "niggle",         # To spend excessive time on minor details.
    "onomatomania",   # Vexation at having difficulty in finding the right word.
    "philoprogenitive",  # Fond of children.
    "quidnunc",       # One who always wants to know what is going on.
    "ragamuffin",     # A ragged often disreputable person.
    "snickersnee",    # A large knife.
    "tatterdemalion", # A person wearing tattered clothing.
    "unputdownable",  # Impossible to put down or stop reading or watching.
    "vomitory",       # An exit or outlet.
    "whippersnapper", # A young, impertinent person.
    "xylography",     # The art of engraving on wood.
    "yare",           # Ready or prepared.
    "zephyr"          # A gentle, mild breeze.
]