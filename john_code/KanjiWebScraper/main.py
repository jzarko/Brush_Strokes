import requests
from bs4 import BeautifulSoup
import json
import os
import shutil

json_filename1 = 'kanji.json'
json_filename2 = 'kanji_with_english.json'
csv_filename1 = 'labeled_kanji_format.csv'
csv_filename2 = 'labeled_kanji.csv'


def main():
    # Create a json file from the kanji unicode html file if not already made.
    if not os.path.exists(json_filename1):
        read_from_import_list()
    else:
        print(f"loading in {json_filename1}...")

    kanji_dictionary = {}
    # Read from json file and input into kanji_dictionary
    with open(json_filename1, 'r') as infile:
        kanji_dictionary = json.load(infile)

    if not os.path.exists(json_filename2):
        print('Translating...')
        limit = -1  # Limits amount of kanji looked up before stopping. Use -1 for all kanji.
        for kanji in kanji_dictionary:
            if limit == 0:
                break
            print(kanji_dictionary[kanji])
            limit -= 1
            # Replace previous dictionary item with a list containing it, and the english meaning.
            new_item = [kanji_dictionary[kanji], scrape_jisho_kanji_to_english(kanji_dictionary[kanji])]
            kanji_dictionary[kanji] = new_item
        print(kanji_dictionary)
        create_json_file(json_filename2, kanji_dictionary)
    else:
        print(f"{json_filename2} found!")
        with open(json_filename2, 'r') as infile:
            kanji_dictionary = json.load(infile)

    add_kanji_and_english_to_txt(kanji_dictionary)
    label_datasets_with_csv()


def add_kanji_and_english_to_txt(kanji_dictionary):
    if not os.path.exists(csv_filename2):
        print("Creating kanji csv file")
        with open(csv_filename1, 'r') as infile, open(csv_filename2, 'w', encoding='utf-8') as outfile:
            for line in infile.readlines():
                line_items = line.split(sep=',')
                # Get the key to input to kanji_dictionary. Then, adjust key to match format.
                # [0] grabs the unicode. [2:] removes the 'U+'. [:-1] removes the '/'.
                kanji_key = line_items[0][2:][:-1]
                kanji_key = '0x' + kanji_key
                kanji_key = kanji_key.lower()
                new_line = line.strip()
                try:
                    result = kanji_dictionary[kanji_key]
                    print(result)
                    new_line += f",{result[0]},{result[1]}\n"
                except KeyError:
                    print(f"Kanji for {kanji_key} not found!")
                    new_line += ',NA\n'
                outfile.write(new_line)
    else:
        print("Labeled kanji file exists!")


def create_json_file(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def label_datasets_with_csv():
    print("Copying...")
    # Easily get n-numbers of classes to use labeled correctly.
    num_classes = -1  # Number of kuzushiji kanji folders to bring in.
    folder1 = "../full_dataset"  # Folder where all kanji imgs and labels are.
    folder2 = "../final_dataset"  # Folder to copy the top n classes.
    # 'utf-8' is needed so Japanese can be inputted.
    with open(csv_filename2, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            # Early break if set for it.
            if num_classes == 0:
                break
            if num_classes != -1:
                num_classes -= 1

            line = line.split(',')
            folder_name = line[0][:-1]  # Folder name without the '/'.
            kanji_symbol = line[2]
            kanji_meaning = line[3].replace(' ', '_')  # Replace all spaces with underscore.
            # Removes newline character if exists.
            if '\n' in kanji_meaning:
                kanji_meaning = kanji_meaning[:-1]
            kanji_folder_name = f"{kanji_symbol}({kanji_meaning})"
            src_folder = os.path.join(folder1, folder_name)
            dst_folder = os.path.join(folder2, kanji_folder_name)
            if not os.path.exists(dst_folder):
                shutil.copytree(src_folder, dst_folder)
                print(folder_name, "copied to:", kanji_symbol, kanji_meaning)
            else:
                print(kanji_symbol, kanji_meaning, 'exists')


def read_from_import_list():
    """
    Read from the 'import_kanji.txt' and creates a dictionary input of each kanji's unicode value.
    Then, creates 'kanji.json' file to avoid repeating the code.
    """
    # Set all unicode to output their kanji counterpart.
    kanji_dictionary = {}
    filename = 'import_kanji.txt'
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = line.split()
            # Take the unicode out of the list.
            unicode_num = line[0]
            line.pop(0)

            for i in range(len(line)):
                # Convert string to int, add i, then back to hex.
                adjusted_unicode_num = hex(int(unicode_num, 16) + i)
                kanji_dictionary[adjusted_unicode_num] = line[i]
    create_json_file(json_filename1, kanji_dictionary)


def scrape_jisho_kanji_to_english(kanji):
    """
    Using jisho.org, lookup the English meaning of the kanji and return it.

    :param kanji: in Japanese form
    :return: english meaning of the kanji
    """
    url = f'https://jisho.org/search/{kanji} %23kanji'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}
    response = requests.get(url, headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    result = soup.findAll('div', {'class': 'kanji-details__main-meanings'})
    text = 'NA'
    if result:
        text = result[0].text.strip()
    return text


if __name__ == '__main__':
    main()
