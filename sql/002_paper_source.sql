-- Milestone 4: record which source each chunk came from.
-- Run this in the Supabase SQL Editor, after 001_paper_chunks.sql.
--
-- Chunks outlive a single run, so the in-memory paper->source mapping from the
-- merge node is gone by the time a later run retrieves those chunks. Storing
-- the source on the row is what keeps a citation traceable across runs.
--
-- Safe to run against an existing table: rows written before Milestone 4 came
-- from arXiv, which is exactly what the default backfills them to.

alter table paper_chunks
    add column if not exists source text not null default 'arxiv';

-- The similarity-search function has to return the new column. Postgres will
-- not let `create or replace` change a function's return type, so the old
-- definition must be dropped first.
drop function if exists match_paper_chunks(vector(1536), int, float);

create or replace function match_paper_chunks(
    query_embedding vector(1536),
    match_count int default 5,
    match_threshold float default 0.7
)
returns table (
    id bigint,
    paper_id text,
    title text,
    authors text,
    source text,
    chunk_index int,
    content text,
    similarity float
)
language sql stable
as $$
    select
        id,
        paper_id,
        title,
        authors,
        source,
        chunk_index,
        content,
        1 - (embedding <=> query_embedding) as similarity
    from paper_chunks
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;
