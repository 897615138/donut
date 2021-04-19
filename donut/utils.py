import csv

def split_csv(file_name, begin, num):
    header = ["timestamp", "value", "label", "KPI ID"]
    rows = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        count = 1
        for i in reader:
            if count in range(begin, begin + num):
                rows.append([int(i[0]), float(i[1]), int(i[2]), str(i[3])])
            count = count + 1
    with open(file_name + 'new.csv', 'w', newline='')as f:
        ff = csv.writer(f)
        ff.writerow(header)
        ff.writerows(rows)

# split_csv('../sample_data/real.csv',1,65536)
