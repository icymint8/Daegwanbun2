# TODO: CHANGE THIS FILE NAME TO DMA_project2_Part1,2_team##.py
# EX. TEAM 1 --> DMA_project2_Part1,2_team04.py

# TODO: IMPORT LIBRARIES NEEDED FOR PROJECT 2
from surprise import Dataset, KNNBasic, KNNWithMeans
from surprise import Reader
from surprise import SVD, SVDpp
from surprise import KNNWithZScore, KNNBaseline
from surprise.model_selection import KFold
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import association_rules, apriori
from surprise.model_selection import cross_validate

np.random.seed(0)

team = 4



# PART 1: Association analysis
def part1():
  
    # TODO: Requirement 1-1. MAKE HORIZONTAL VIEW
    # file name: DMA_project2_team##_part2_horizontal.pkl
    # use to_pickle(): df.to_pickle(filename)
    df=pd.read_csv('DMA_project_UPI.csv')
    pivot_df = df.pivot(index='user', columns='post', values='IntDegree')
    pivot_df = pivot_df.notna().astype(int)
    print(pivot_df)
    pivot_df.to_pickle('DMA_project2_team04_part2_horizontal.pkl')
    # TODO: Requirement 1-2. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part2_association.pkl (pandas dataframe)
    fl=apriori(pivot_df, min_support=0.2, use_colnames=True)
    print(fl)
    rules=association_rules(fl,1,metric='lift',min_threshold=2)
    print(rules)
    rules.to_pickle('DMA_project2_team04_part2_association.pkl')

#Part2: recommendation system
# TODO: Requirement 2-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n, user_based=True):
    results = defaultdict(list)
    if user_based:
        # TODO: testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장
        # Hint: testset은 (user_id, post_id, interest_degree)의 tuple을 요소로 갖는 list
        testset_id = [entry for entry in testset if entry[0] in id_list]

        predictions = algo.test(testset_id)
        for uid, bname, true_r, est, _ in predictions:
            # TODO: results는 user_id를 key로, [(post_id, estimated_degree)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[uid].append((bname,est))

    else:
        # TODO: testset의 데이터 중 post id이 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, post_id, interest_degree)의 tuple을 요소로 갖는 list
        testset_id = [entry for entry in testset if entry[1] in id_list]
        
        predictions = algo.test(testset_id)
        for uid, bname, true_r, est, _ in predictions:
            # TODO: results는 post_id를 key로, [(user_id, estimated_degree)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[bname].append((uid,est))

    for id_, ratings in results.items():
        # TODO: degree 순서대로 정렬하고 top-n개만 유지
        results[id_] = sorted(ratings, key=lambda x: x[1], reverse = True)[:n]

    return results


# PART 2. Requirement 2-2, 2-3, 2-4
def part2():
    file_path = 'DMA_project_UPI.csv'
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 30), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    # TODO: Requirement 3-2. User-based Recommendation
    uid_list = ['1496', '2061', '2324', '4041', '4706']
    # TODO: set algorithm for 2-2-1
    sim_options_1 = {'name': 'cosine','user_based': True}
    algo_1 = KNNBasic(sim_options= sim_options_1)
    algo_1.fit(trainset)

    results_1 = get_top_n(algo_1, testset, uid_list, n=5, user_based=True)
    with open('2-2-1.txt', 'w') as f:
        for uid, ratings in sorted(results_1.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-2-2
    sim_options_2 = {'name': 'pearson', 'user_based': True}
    algo_2 = KNNWithMeans(sim_options= sim_options_2)
    algo_2.fit(trainset)

    results_2 = get_top_n(algo_2, testset, uid_list, n=5, user_based=True)
    with open('2-2-2.txt', 'w') as f:
        for uid, ratings in sorted(results_2.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: 2-2-3. Best Model
    best_algo_ub = None
    #chat gpt가 써주기는 했는데 좀 손볼필요는 있을 듯. KNNBasic이랑 KNNWithMeans말고 쓸 수 있는 방법도 포함해야 될 수도 있음
    # Requirement 2-2-3: Determine the best user-based model
    print("Evaluating models with cross-validation...")
    scores_1 = cross_validate(algo_1, data, measures=['rmse', 'mae'], cv=5, verbose=False)
    scores_2 = cross_validate(algo_2, data, measures=['rmse', 'mae'], cv=5, verbose=False)

    # Compare RMSE scores to determine the best algorithm
    rmse_1 = scores_1['test_rmse'].mean()
    rmse_2 = scores_2['test_rmse'].mean()
    best_algo_ub = algo_1 if rmse_1 < rmse_2 else algo_2
    print(f"Best user-based model is: {'KNNBasic (Cosine)' if best_algo_ub == algo_1 else 'KNNWithMeans (Pearson)'}")

    # Save the best model to file
    with open('best_user_based_model.txt', 'w') as f:
        f.write(f"Best user-based model: {'KNNBasic (Cosine)' if best_algo_ub == algo_1 else 'KNNWithMeans (Pearson)'}\n")
        f.write(f"KNNBasic RMSE: {rmse_1:.4f}\n")
        f.write(f"KNNWithMeans RMSE: {rmse_2:.4f}\n")



    # TODO: Requirement 2-3. Item-based Recommendation
    bname_list = ['20', '45', '48', '139', '162']
    # TODO - set algorithm for 2-3-1
    sim_options_1 = {'name': 'cosine', 'user_based': False}
    algo_1 = KNNBasic(sim_options= sim_options_1)
    algo_1.fit(trainset)

    results_1 = get_top_n(algo_1, testset, bname_list, n=10, user_based=False)
    with open('2-3-1.txt', 'w') as f:
        for bname, ratings in sorted(results_1.items(), key=lambda x: x[0]):
            f.write('Post ID %s top-10 results\n' % bname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-3-2
    sim_options_2 = {'name': 'pearson', 'user_based': False}
    algo_2 = KNNWithMeans(sim_options= sim_options_2)
    algo_2.fit(trainset)
    results_2 = get_top_n(algo_2, testset, bname_list, n=10, user_based=False)
    with open('2-3-2.txt', 'w') as f:
        for bname, ratings in sorted(results_2.items(), key=lambda x: x[0]):
            f.write('Post ID %s top-10 results\n' % bname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO: 2-3-3. Best Model
    best_algo_ib = None
    # 얘도 마찬가지로 수정 필요할듯
    # Requirement 2-3-3: Determine the best item-based model
    print("Evaluating item-based models with cross-validation...")
    scores_1 = cross_validate(algo_1, data, measures=['rmse', 'mae'], cv=5, verbose=False)
    scores_2 = cross_validate(algo_2, data, measures=['rmse', 'mae'], cv=5, verbose=False)

    # Compare RMSE scores to determine the best algorithm
    rmse_1 = scores_1['test_rmse'].mean()
    rmse_2 = scores_2['test_rmse'].mean()
    best_algo_ib = algo_1 if rmse_1 < rmse_2 else algo_2
    print(f"Best item-based model is: {'KNNBasic (Cosine)' if best_algo_ib == algo_1 else 'KNNWithMeans (Pearson)'}")

    # Save the best model details
    with open('best_item_based_model.txt', 'w') as f:
        f.write(f"Best item-based model: {'KNNBasic (Cosine)' if best_algo_ib == algo_1 else 'KNNWithMeans (Pearson)'}\n")
        f.write(f"KNNBasic RMSE: {rmse_1:.4f}\n")
        f.write(f"KNNWithMeans RMSE: {rmse_2:.4f}\n")


    # TODO: Requirement 2-4. Matrix-factorization Recommendation
    # TODO: set algorithm for 2-4-1
    algo_1 = SVD(n_factors=100, n_epochs=50, biased=False)
    algo_1.fit(trainset)

    results = get_top_n(algo_1, testset, uid_list, n=5, user_based=True)
    with open('2-4-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-2
    algo_2 = SVD(n_factors=200, n_epochs=50,biased=True)
    algo_2.fit(trainset)
    results = get_top_n(algo_2, testset, uid_list, n=5, user_based=True)
    with open('2-4-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-3
    algo_3 = SVDpp(n_factors=100,n_epochs=50)
    algo_3.fit(trainset)
    results = get_top_n(algo_3, testset, uid_list, n=5, user_based=True)
    with open('2-4-3.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: set algorithm for 2-4-4
    algo_4 = SVDpp(n_factors=100,n_epochs=100)
    algo_4.fit(trainset)
    results = get_top_n(algo_4, testset, uid_list, n=5, user_based=True)
    with open('2-4-4.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for bname, score in ratings:
                f.write('Post ID %s\n\tscore %s\n' % (bname, str(score)))
            f.write('\n')

    # TODO: 2-4-5. Best Model
    best_algo_mf = None

    # Requirement 2-4-5: Determine the best model
    print("Evaluating matrix-factorization models with cross-validation...")
    scores_1 = cross_validate(algo_1, data, measures=['rmse', 'mae'], cv=5, verbose=False)
    scores_2 = cross_validate(algo_2, data, measures=['rmse', 'mae'], cv=5, verbose=False)
    scores_3 = cross_validate(algo_3, data, measures=['rmse', 'mae'], cv=5, verbose=False)
    scores_4 = cross_validate(algo_4, data, measures=['rmse', 'mae'], cv=5, verbose=False)

    # Calculate average RMSE for each model
    rmse_1 = scores_1['test_rmse'].mean()
    rmse_2 = scores_2['test_rmse'].mean()
    rmse_3 = scores_3['test_rmse'].mean()
    rmse_4 = scores_4['test_rmse'].mean()

    # Determine the best algorithm
    rmse_scores = [rmse_1, rmse_2, rmse_3, rmse_4]
    best_index = rmse_scores.index(min(rmse_scores))
    best_algo_mf = [algo_1, algo_2, algo_3, algo_4][best_index]
    best_algo_name = ['SVD (default)', 'SVD (custom)', 'SVDpp (default)', 'SVDpp (custom)'][best_index]

    print(f"Best matrix-factorization model is: {best_algo_name}")

    # Save the best model details
    with open('best_matrix_factorization_model.txt', 'w') as f:
        f.write(f"Best matrix-factorization model: {best_algo_name}\n")
        f.write(f"SVD (default) RMSE: {rmse_1:.4f}\n")
        f.write(f"SVD (custom) RMSE: {rmse_2:.4f}\n")
        f.write(f"SVDpp (default) RMSE: {rmse_3:.4f}\n")
        f.write(f"SVDpp (custom) RMSE: {rmse_4:.4f}\n")



if __name__ == '__main__':
    part1()
    part2()



