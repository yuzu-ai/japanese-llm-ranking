import json


def process_txt_to_jsonl(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        category = ""
        question_id = 1
        for line in f_in:
            line = line.strip()  # remove leading/trailing white space
            if line.startswith("#"):  # this is a category line
                category = line[1:].strip()  # remove '#'
            elif line:  # this is a question line
                data = {
                    "category": category,
                    "question_id": question_id,
                    "text": line.strip().strip("「」"),
                }

                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                question_id += 1


# specify input and output files
input_file = "rakuda_koukou_v0.txt"
output_file = input_file.split(".")[0] + ".jsonl"
process_txt_to_jsonl(input_file, output_file)
