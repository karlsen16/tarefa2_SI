import csv

with open('com_label.txt', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(data)