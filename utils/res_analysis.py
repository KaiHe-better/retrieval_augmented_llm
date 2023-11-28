import re

file = "/raid/hpc/hekai/WorkShop/My_project/RA_LLM/results/output/[]/2023-11-19_result.log"

with open(file, "r") as f:
    data = f.readlines()

total_cnt = 0
neg_one_cnt = 0
for int_str in data:
    if int_str != "\n":
        match = re.search(r'pred (-?\d+)', int_str)
        if match:
            # 获取匹配到的数字部分
            result = match.group(1)
            if str(result)=="-1":
                neg_one_cnt+=1
            total_cnt+=1

print("total_cnt", total_cnt)
print("neg_one_cnt", neg_one_cnt)
   

