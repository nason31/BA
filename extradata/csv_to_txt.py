import csv

def csv_to_txt(input_csv, output_txt, csv_delimiter=",", txt_delimiter="\t"):
    print("start")
    with open(input_csv, "r") as csv_file, open(output_txt, "w") as txt_file:
        reader = csv.reader(csv_file, delimiter=csv_delimiter)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                label, text = row[0], row[1]
                txt_file.write(f"{label}{txt_delimiter}{text}\n")
    print(f"Converted {input_csv} to {output_txt}")

# Paths to input and output files
csv_to_txt("Ohsumed/test.csv", "Ohsumed/test.txt")
csv_to_txt("Ohsumed/train.csv", "Ohsumed/train.txt")

