import psycopg2
import time

conn = psycopg2.connect(dbname='***', user='***', port=1921,
                        password='***', host='***')

cursor = conn.cursor()
start = time.time()
cursor.execute(
    """
        create index
        on public.sift_vectors
        using pase_ivfflat (vector)
        with (
            clustering_type = 1,
            distance_type = 0,
            dimension = 128,
            base64_encoded = 0,
            clustering_params = "10,1000");
    """)
conn.commit()
end = time.time()
print(f"Index build took {end - start} seconds.")
