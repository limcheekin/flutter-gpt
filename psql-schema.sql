-- REF: https://github.com/langchain-ai/langchain-template-supabase/blob/main/supabase/migrations/20230311201746_setup-vector-store.sql
create table documents (
  id varchar(36) primary key,
  name varchar(255) not null,
  content text,
  checksum varchar(128),
  token_count integer, 
  embedding vector(1536),  -- 1536 works for OpenAI embeddings, change if needed
  metadata jsonb
);

-- Create a function to search for documents
create function match_documents(query_embedding vector(1536), match_count int)
returns table(id varchar(36), content text, metadata jsonb, token_count integer, similarity float)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    token_count,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$
;