import json, random
import numpy as np
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial import ConvexHull
from collections import defaultdict
from surprise import BaselineOnly, Reader, Dataset

def get_primitive(data, n=10): 
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'business_id', 'stars']], reader)
    trainset = data.build_full_trainset()
    algo = BaselineOnly()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    with open('primitive.json','w') as f:
        json.dump(top_n,f)
    return top_n

def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = str(i)
        idx2term[i] = terms[i]
    return term2idx, idx2term

def random_walk(G, walk_length, start_node):
	'''
	Simulate a random walk starting from start node.
	'''
	walk = [start_node]
	while len(walk) < walk_length:
		cur = walk[-1]
		cur_nbrs = sorted(G.neighbors(cur))
		if len(cur_nbrs) > 0:
			walk.append(random.choice(cur_nbrs))
		else:
			break
	return walk

def height(pnt, start, end):
    s1 = np.linalg.norm(pnt - start)
    s2 = np.linalg.norm(start - end)
    s3 = np.linalg.norm(pnt - end)
    s = (s1 + s2 + s3) / 2
    area = (s*(s-s1)*(s-s2)*(s-s3)) ** 0.5
    return 2 * area / s2

# Generate Heterogenous Information Network 
data = pd.read_csv('test.csv')
data = eval(data.to_json(orient='records'))
user_id = list(set([x['user_id'] for x in data]))
business_id = list(set([x['business_id'] for x in data]))
entity = list(set([y for x in data for y in eval(x['entity'])]))
G = nx.Graph()
G.add_nodes_from(entity)
G.add_nodes_from(user_id)
G.add_nodes_from(business_id)      
for x in data:
    G.add_edge(x['user_id'], x['business_id'])
    for e in eval(x['entity']):
        G.add_edge(x['user_id'], e)
        G.add_edge(x['business_id'], e)

# Conduct Random Walk on HIN
walk = []
term2idx, idx2term = dictionary(user_id+business_id+entity)
for user in user_id:
    walk.append(random_walk(G, 10, user))
for ent in entity:
    walk.append(random_walk(G, 10, ent))
for business in business_id:
    walk.append(random_walk(G, 10, business))
walk = [[term2idx[x] for x in y] for y in walk]
model = Word2Vec(walk, size=128, window=10, min_count=0, sg=1, workers=0, iter=100)
X = np.array([model.wv[str(x)] for x in range(len(term2idx))])
X = TSNE(n_jobs=1).fit_transform(X)
embedding = {}
for i in range(len(term2idx)):
    embedding[idx2term[str(i)]] = X[i]

#Calculate Unexpectedness & Relevance
relevance, unexpectedness = {}, {}
for user in user_id:
    relevance[user], unexpectedness[user] = {}, {}
    neighbor = list(G.neighbors(user)) + [user]
    points = np.array([embedding[x] for x in neighbor])
    hull = ConvexHull(points)
    for business in business_id:
        newpoint = embedding[business]
        hull_dist = 0
        for v_idx in range(len(hull.vertices)):
            hull_dist += height(newpoint, points[hull.vertices[v_idx-1]], points[hull.vertices[v_idx]])
        inside = ConvexHull(points+[newpoint]).volume == ConvexHull(points).volume
        if inside == True:
            unexpectedness[user][business] = 0
        else:
            unexpectedness[user][business] = hull_dist / len(hull.vertices)        
        relevance[user][business] = np.linalg.norm(embedding[user]-embedding[business])
    max_unexpectedness = max(unexpectedness[user].values())
    max_relevance = max(relevance[user].values())
    for business in business_id:
        unexpectedness[user][business] = unexpectedness[user][business] / max_unexpectedness
        relevance[user][business] = relevance[user][business] / max_relevance
with open('unexpectedness.json','w') as f:
    json.dump(unexpectedness,f)
with open('relevance.json','w') as f:
    json.dump(relevance,f)

#Unexpected Recommendation
data = pd.read_csv('test.csv')
data = data[['user_id','business_id','stars']] 
primitive = get_primitive(data)

from surprise import KNNBaseline
from surprise.model_selection import cross_validate
reader = Reader(rating_scale=(1, 5))
subdata = Dataset.load_from_df(data[['user_id', 'business_id', 'stars']], reader)
prediction = cross_validate(KNNBaseline(), subdata, measures=['RMSE_LC','MAE_LC','PRECISION_LC','RECALL_LC','UNEXP_LC','SERENDIPITY_LC','DIVERSITY_LC','COVERAGE_LC'], cv=5, verbose=True)
#prediction = cross_validate(KNNBaseline(), subdata, measures=['RMSE','MAE','PRECISION','RECALL','UNEXP','SERENDIPITY','DIVERSITY','COVERAGE'], cv=5, verbose=True)