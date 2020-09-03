# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from vector import  dis_matrix, graph, cluster, ConcatinateClusterTexts

df = pd.read_csv('dataset.csv', encoding="utf8", index_col="id")


content = df['content']
print(len(content))
content_list = content.values.astype('U')

Dis_matrix = dis_matrix(content_list)

res = graph(Dis_matrix)
res_cluster = cluster(res)
for i in range(0,len(res_cluster)):
    print ("Cluster " + str(i))
    x =  ConcatinateClusterTexts(content_list, res_cluster, i)
    # for idx in res_cluster[i]:
    #     print(content[idx]) # Cần get index theo data dầu vào của các bài.

## """
# TO DO: Cần làm queue thời gian load data. Hiện đang chưa làm đk. thời gian cluster bài báo t time thời gian dịch là delta time. ví dụ cluster các bài trong khoảng t = 30 phút delta time = 15.
#  14:30 - 15:00 -> 14:45 - 15:15, các bài sẽ bị xóa từ 14:30 - 14:45 cập nhật các bài từ 15:00 - 15:15.
