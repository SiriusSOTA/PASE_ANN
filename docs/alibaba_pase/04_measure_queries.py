import psycopg2
import csv
import numpy as np

conn = psycopg2.connect(dbname='***', user='***', port=1921,
                        password='***', host='***')
vector_count = 1
total_time = 0
recall_sum = 0
cluster_numbers_to_select = [1, 2, 5, 10, 20, 30, 50, 75, 100]


def fvecs_read(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    fv = fv.copy()
    return fv


def ivecs_read(filename):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    fv = fv.copy()
    return fv


def query_nearest_neighbors(csv_writer, vector, ground_truth, cluster_number_to_select):
    assert vector_count == len(ground_truth)
    cursor.execute("drop table if exists public.results;")

    cursor.execute(f"""
        EXPLAIN ANALYZE SELECT
        array(select id FROM public.gist_vectors
        ORDER BY
        vector <#> '{vector}:{cluster_number_to_select}:0'::pase
        ASC LIMIT {vector_count}) as result into public.results;
    """)
    answers = cursor.fetchall()
    spent_time = float(answers[-1][0][16:-3])

    cursor.execute("select (select result from public.results);")

    answers = cursor.fetchall()
    global total_time
    total_time += spent_time

    correct_answers = 0
    for a in answers[0][0]:
        if a in ground_truth:
            correct_answers += 1
    recall = correct_answers / vector_count
    global recall_sum
    recall_sum += recall

    csv_writer.writerow([spent_time, recall])


cursor = conn.cursor()
cursor.execute("set search_path to 'information_schema';")

with open(f'measurement_gist_alibaba_{vector_count}.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["average_query_time", "average_recall"])
    for cluster_number_to_select in cluster_numbers_to_select:
        idx = 0
        total_time = 0
        recall_sum = 0
        with open(f'measurement_gist_alibaba_{vector_count}_full_{cluster_number_to_select}.csv', 'w',
                  newline='') as csvfilefull:
            csv_writer_full = csv.writer(csvfilefull, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            ground_truth = ivecs_read("../../test/test_data/gist/gist_groundtruth.ivecs")
            for vec in fvecs_read("../../test/test_data/gist/gist_query.fvecs"):
                query_nearest_neighbors(csv_writer_full, ",".join(vec.astype(str)), ground_truth[idx][:vector_count],
                                        cluster_number_to_select)
                idx += 1
        print(f"Cluster count selected for one query: {cluster_number_to_select}.")
        print(f"Average query time: {total_time / idx}.")
        print(f"Average recall1@{vector_count}: {recall_sum / idx}.")
        csv_writer.writerow([total_time / idx, recall_sum / idx])
