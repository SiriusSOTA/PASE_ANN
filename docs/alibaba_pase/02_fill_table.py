import psycopg2
import numpy as np

conn = psycopg2.connect(dbname='***', user='***', port=1921,
                        password='***', host='***')

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


def fill_table_query():
    query = "INSERT INTO public.gist_vectors (id, vector) VALUES "
    i = 0
    for vec in fvecs_read("../../test/test_data/gist/gist_base.fvecs"):
        query += (',' if i % 10000 != 0 else "") + f"({i},'" + "{" + ",".join(vec.astype(str)) + "}')"
        i += 1
        if i % 10000 == 0 and i != 0:
            query += ";"
            cursor.execute(query)
            conn.commit()
            query = "INSERT INTO public.gist_vectors (id, vector) VALUES "
            print(f"Added {i} vectors")


cursor = conn.cursor()
cursor.execute("set search_path to 'information_schema';")

fill_table_query()
conn.commit()
