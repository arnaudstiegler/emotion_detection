import csv
import pandas as pd

def process_raw_isear_data(data_path='data/isear.csv', save_path='data/isear_processed.csv'):
    texts=[]
    emotions=[]
    print("Reading from {}".format(data_path))
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                text_id = row.index('SIT')
                emotion_id = row.index('Field1')
            else:
                texts.append(row[text_id])
                emotions.append(row[emotion_id])
                line_count += 1
        print(f'Processed {line_count} lines.')

    df = pd.DataFrame(list(zip(texts,emotions)))
    print("Writing to {}".format(save_path))
    df.to_csv(save_path)
    return

if __name__=='__main__':
    process_raw_isear_data()