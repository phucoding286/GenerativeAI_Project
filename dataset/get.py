import pandas as pd

df = pd.read_parquet("D:\\Desktop\\GenerativeAI_Projects\\dataset\\grt2_inference.parquet")

print(df["response_message"])

with open("./dataset/a.txt", "a", encoding="utf-8") as file:
    for ans in df['response_message']:
        try:
            file.write(ans.replace("\n", " ")+"\n")
        except: continue