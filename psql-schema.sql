-- Create a function to search for documents
create function match_documents(embedding vector(768), match_count int)
returns table(uuid uuid, document text, cmetadata json, custom_id text, similarity float)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    uuid,
    document,
    cmetadata,
    custom_id,
    1 - (documents.embedding <=> query_embedding) as similarity
  from langchain_pg_embedding documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$
;