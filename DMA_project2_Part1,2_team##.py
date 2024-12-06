# TODO: CHANGE THIS FILE NAME TO DMA_project2_Part1,2_team##.py
# EX. TEAM 1 --> DMA_project2_Part1,2_team01.py

# TODO: IMPORT LIBRARIES NEEDED FOR PROJECT 2
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import association_rules, apriori

np.random.seed(0)

team = 00



# PART 1: Association analysis
def part1():
  
    # TODO: Requirement 1-1. MAKE HORIZONTAL VIEW
    # file name: DMA_project2_team##_part2_horizontal.pkl
    # use to_pickle(): df.to_pickle(filename)


    # TODO: Requirement 1-2. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part2_association.pkl (pandas dataframe)



# TODO: Requirement 2-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n, user_based=True):
    results = defaultdict(list)
    if user_based:
        # TODO: testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장
        # Hint: testset은 (user_id, post_id, interest_degree)의 tuple을 요소로 갖는 list
        testset_id = []

        predictions = algo.test(testset_id)
        for uid, bname, true_r, est, _ in predictions:
            # TODO: results는 user_id를 key로, [(post_id, estimated_degree)의 tuple이 모인 list]를 value로 갖는 dictionary
            pass
    else:
        # TODO: testset의 데이터 중 post id이 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, post_id, interest_degree)의 tuple을 요소로 갖는 list
        testset_id = []
        
        predictions = algo.test(testset_id)
        for uid, bname, true_r, est, _ in predictions:
            # TODO: results는 post_id를 key로, [(user_id, estimated_degree)의 tuple이 모인 list]를 value로 갖는 dictionary
            pass
    for id_, ratings in results.items():
        # TODO: degree 순서대로 정렬하고 top-n개만 유지
        pass

    return results


# PART 2. Requirement 2-2, 2-3, 2-4
def part2():
    file_path = 'DMA_project2_UPI.csv'
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 30), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    # TODO: Requirement 3-2. User-based Recommendation
    uid_list = ['1496', '2061', '2324', '4041', '4706']
    # TODO: set algorithm for 2-2-1
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-2-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-2-2
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-2-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: 2-2-3. Best Model
    best_algo_ub = None

    # TODO: Requirement 2-3. Item-based Recommendation
    bname_list = [‘20’, ‘45’, ‘48’, ‘139’, ‘162’]
    # TODO - set algorithm for 2-3-1
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, bname_list, n=10, user_based=False)
    with open('2-3-1.txt', 'w') as f:
        for bname, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Post ID %s top-10 results\n' % bname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-3-2
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, bname_list, n=10, user_based=False)
    with open('2-3-2.txt', 'w') as f:
        for bname, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Post ID %s top-10 results\n' % bname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO: 2-3-3. Best Model
    best_algo_ib = None

    # TODO: Requirement 2-4. Matrix-factorization Recommendation
    # TODO: set algorithm for 2-4-1
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-4-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-2
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-4-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-3
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-4-3.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-4
    algo = None
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('2-4-4.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: 2-4-5. Best Model
    best_algo_mf = None
    

if __name__ == '__main__':
    part1()
    part2()



