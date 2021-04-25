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


def is_in(num, lis, range_1=-200, range_2=200):
    if len(lis) == 0:
        return False
    for i in range(range_1, range_2):
        if num + i in lis:
            return True
    return False

# split_csv('../../sample_data/real.csv',2048,1024)
