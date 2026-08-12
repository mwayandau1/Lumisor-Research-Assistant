-- Milestone 2: pgvector embeddings table for research papers
-- Run this in Supabase SQL Editor

-- Enable pgvector extension (if not already enabled)
create extension if not exists vector;

-- Paper chunks with embeddings
create table if not exists paper_chunks (
    id bigserial primary key,
    paper_id text not null,              -- arXiv ID (e.g., "2401.12345")
    title text not null,
    authors text,
    chunk_index int not null,
    content text not null,
    embedding vector(1536),              -- text-embedding-3-small dimension
    created_at timestamptz default now(),
    
    unique(paper_id, chunk_index)
);

-- Index for similarity search
create index if not exists paper_chunks_embedding_idx 
on paper_chunks using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Function for similarity search
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
        chunk_index,
        content,
        1 - (embedding <=> query_embedding) as similarity
    from paper_chunks
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;
