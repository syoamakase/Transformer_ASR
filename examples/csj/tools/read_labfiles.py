import argparse
import re
import pyopenjtalk

vocab_id_txt = '/n/work1/ueno/data/jsut/data/train/vocab.id'
vocab_id = {}

with open(vocab_id_txt) as f:
    for line in f:
        phone_id, phone = line.strip().split(' ')
        vocab_id[phone] = phone_id

def text_to_sequence(text):
    results = []
    for t in text:
        results.append(vocab_id[t])

    return results

def parse_label(meta_data):
    with open(meta_data) as f:
        for line in f:
            file_id, text = line.strip().split('|', 1)
            
            phone = pyopenjtalk.g2p(text)
            phone = [str(i) for i in phone.split(' ')]

            phone = list(filter(lambda p: p != 'pau', phone))
            #phone = list(filter(lambda p: p != ' ', phone))
            phone = list(filter(lambda p: p != ' ', phone))
            phone = list(filter(lambda p: p != '', phone))
            texts = ' '.join([str(i) for i in text_to_sequence(phone)]) + ' 1'

            print(f'{file_id}|{texts}') 
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_data')
    args = parser.parse_args()
    meta_data = args.meta_data

    results = parse_label(meta_data)
