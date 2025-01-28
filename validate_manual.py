##################################################################################
##################################################################################
##                                                                              ##
##    This file is a part of codes used in the Research Paper titled:           ##
##    "COVID-19 and Digital Health Communication on Rheumatoid Arthritis:       ##
##    A Natural Language Processing Analysis of Reddit Discussion"              ##
##    Copyright (C) 2025  Naisarg Patel (github:naisarg14)                      ##
##                                                                              ##
##    This program is free software: you can redistribute it and/or modify      ##
##    it under the terms of the GNU General Public License as published by      ##
##    the Free Software Foundation, either version 3 of the License, or         ##
##    (at your option) any later version.                                       ##
##                                                                              ##
##    This program is distributed in the hope that it will be useful,           ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##    GNU General Public License for more details.                              ##
##                                                                              ##
##    You should have received a copy of the GNU General Public License         ##
##    along with this program.  If not, see <https://www.gnu.org/licenses/>.    ##
##                                                                              ##
##################################################################################
##################################################################################
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    manual_file = "manually_annotated.csv"
    true_labels = []
    predicted_labels = []

    with open(manual_file, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        total = 0
        for row in reader:
            total += 1
            manual = row["Manual_Sentiment"]
            if not manual:
                continue

            if str(manual) == "0.0":
                true_labels.append("Negative")
            elif str(manual) == "1.0":
                true_labels.append("Neutral")
            elif str(manual) == "2.0":
                true_labels.append("Positive")

            predicted_labels.append(row["Overall Sentiment"])

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )

    print(f"Total Labeled: {len(true_labels)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"Total in file: {total}")
    print(f"Total Remaining: {1000-len(true_labels)}")

if __name__ == "__main__":
    main()