import json


# def process_txt_to_jsonl(input_file, output_file):
#     with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
#         category = ""
#         question_id = 1
#         for line in f_in:
#             line = line.strip()  # remove leading/trailing white space
#             if line.startswith("#"):  # this is a category line
#                 category = line[1:].strip()  # remove '#'
#             elif line:  # this is a question line
#                 data = {
#                     "category": category,
#                     "question_id": question_id,
#                     "text": line.strip().strip("「」"),
#                 }

#                 f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 question_id += 1


# # specify input and output files
# input_file = "rakuda_koukou_v0.txt"
# output_file = input_file.split(".")[0] + ".jsonl"
# process_txt_to_jsonl(input_file, output_file)


# def sort_jsonl(input_file, output_file):
#     # Load jsonl file into memory as list of dictionaries
#     with open(input_file, 'r') as f:
#         data = [json.loads(line) for line in f]

#     # Sort list of dictionaries
#     data.sort(key=lambda d: (d['category'], d['question_id']))

#     # Write sorted data to jsonl file
#     with open(output_file, 'w') as f:
#         for item in data:
#             f.write(json.dumps(item,ensure_ascii=False) + '\n')

# # specify input and output files
# input_file = "rakuda_v1.jsonl"
# output_file = "sorted_rakuda_v1.jsonl"
# sort_jsonl(input_file, output_file)


import json
import shortuuid

def replace_question_id_with_uuid(input_file, output_file):
    # Load jsonl file into memory as list of dictionaries
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Replace 'question_id' with a short UUID in each dictionary
    for item in data:
        item['question_id'] = shortuuid.uuid()

    # Write modified data to jsonl file
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item,ensure_ascii=False) + '\n')

input_file = "sorted_rakuda_v1.jsonl"
output_file = "rakuda_v1.jsonl"
replace_question_id_with_uuid(input_file, output_file)