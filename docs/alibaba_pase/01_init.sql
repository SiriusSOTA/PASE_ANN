create extension pase;
set search_path to 'information_schema';
create table public.sift_vectors (id integer, vector float4[]);
