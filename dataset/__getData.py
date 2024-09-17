import pandas as pd

# Đường dẫn đến tệp .parquet
file_path = "./dataset/train-00001-of-00082.parquet"

# Đọc tệp .parquet bằng pandas
df = pd.read_parquet(file_path)

q = df["query"]
a = df['answer']

print(list(q))

# for i in range(len(a)):
#     with open("source_sequences_1.txt", "a", encoding='utf-8') as f:
#             f.write(sentence['content']+"\n")

# Hiển thị thông tin của dataframe
# for batch in df['messages']:
#     for sentence in batch:
#         if sentence['role'] == "user":
#             with open("source_sequences_1.txt", "a", encoding='utf-8') as f:
#                 f.write(sentence['content']+"\n")
#         else:
#             with open("target_sequences_1.txt", "a", encoding='utf-8') as f:
#                 f.write(sentence['content']+"\n")
