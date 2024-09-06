import pandas as pd

# Đường dẫn đến tệp .parquet
file_path = "train_sft-00000-of-00001.parquet"

# Đọc tệp .parquet bằng pandas
df = pd.read_parquet(file_path)

# Hiển thị thông tin của dataframe
for batch in df['messages']:
    for sentence in batch:
        if sentence['role'] == "user":
            with open("source_sequences.txt", "a", encoding='utf-8') as f:
                f.write(sentence['content']+"\n")
        else:
            with open("target_sequences.txt", "a", encoding='utf-8') as f:
                f.write(sentence['content']+"\n")
