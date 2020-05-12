from pyspark import SparkContext
import sys
import json
import time
import copy
import itertools
import re
import math
from collections import Counter

# spark-submit task2predict.py test_review.json task2.model task2.predict
start_time = time.time()

test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('ERROR')

with open(model_file) as f:
    model = json.load(f)

b_tfidf = model['business']
u_tfidf = model['user']

RDD0 = sc.textFile(test_file)\
    .map(lambda line: (json.loads(line)['user_id'], json.loads(line)['business_id']))\
    .collect()

# bs = list(b_tfidf.keys())
# us = list(u_tfidf.keys())


def cosinefunc(x):
    # print(0)
    u = x[0]
    b = x[1]
    # print(1)
    # #if (b in bs) & (u in us):
    try:
        u_words = u_tfidf[u]
        b_words = b_tfidf[b]
        #if type(u_words) == list and type(b_words) == list:
        # print(2)
        u_words = set(u_words)
        b_words = set(b_words)
        # print(2)
        nume = len(u_words.intersection(b_words))
        deno = math.sqrt(len(u_words)) * math.sqrt(len(b_words))
        # print(3)
        sim = nume / deno
        # print(4)
        if (sim > 0.01) | (sim == 0.01):
            # print(5)
            return [u, b, sim]
        else:
            return 0
    except:
        return 0

output = open(output_file, 'w')
for line in RDD0:
    pair = cosinefunc(line)
    if type(pair) == list:
        output.write('{"user_id": "' + str(pair[0]) + '", "business_id": "' + str(pair[1]) + '", "sim": ' + str(pair[2]) + '}')
        output.write("\n")

# t = cosinefunc(("pA0Ke_97Qn8ROyZbEMtVWg", "tA5mqEP68qjbTFK9c14D7A"))
# print(t)

# RDD1 = RDD0.map(lambda line: (json.loads(line)['user_id'], json.loads(line)['business_id'])) \
#     .map(cosinefunc)\
#     .filter(lambda x: type(x) == list)\
#     .collect()
#

print('Duration: ' + str(time.time() - start_time))
