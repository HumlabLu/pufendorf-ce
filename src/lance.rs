use std::sync::Arc;
use anyhow::{Result, bail};

// use arrow_array::{RecordBatch, RecordBatchIterator, ArrayRef, Int32Array, StringArray};
// use arrow_schema::{Schema, Field, DataType};

use std::io::{self, BufRead};
use std::path::Path;
use std::fs::File;

use log::{debug, info, trace, error};

use arrow_array::{
  ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray, Float32Array
};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::index::Index;
use lancedb::database::CreateTableMode;
use lancedb::query::{QueryBase, ExecutableQuery};

use iced::futures::TryStreamExt;

use crate::embedder::{chunk_file_pdf, chunk_file_prefix_txt, chunk_file_txt, get_embedding_dim};
use uuid::Uuid;
use std::collections::HashMap;

fn build_schema(dim: i32) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("abstract", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]))
}

fn schema_index(schema: &Schema, name: &str) -> Result<usize> {
    schema.index_of(name).map_err(|e| anyhow::anyhow!("Missing column '{name}' in schema: {e}"))
}

pub async fn get_row_count(db_name: &str, table_name: &str) -> usize {
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB");

    // Create empty table if not exist?
    // Disadvantage is that once we have a table, we cannot use create anymore because
    // the table exists...(if we don't replace it).
    let table = db.open_table(table_name).execute().await.expect("Cannot open table");
    let rc = table.count_rows(None).await.expect("?");

    rc
}

pub async fn append_documents<P>(filename: P, db_name: &str, table_name: &str, chunk_size: usize) -> Result<(), anyhow::Error>
where
    P: AsRef<Path>,
{
    // Embedding model (downloads once, then runs locally).
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let dim = get_embedding_dim(&EmbeddingModel::AllMiniLML6V2)? as i32;
    info!("Embedding dim {}", dim);

    // let db_name = config.db_path.clone(); //data/lancedb_fastembed";
    // let table_name = config.table_name.clone(); //"docs".to_string();
    let db = lancedb::connect(db_name).execute().await.expect("Cannot connect to DB.");

    if let Ok(ref table) = db.open_table(table_name).execute().await {
        info!("Row count {:?}", table.count_rows(None).await.expect("?"));
    }

    info!("Database: {db_name}");
    info!("Table name: {table_name}");

    // Better top use chunker with a really large chunk value instead?
    // That gives the same effect?
    if false {
        let path = filename.as_ref();
        let chunks = if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                match ext {
                    "txt" => chunk_file_txt(&filename, chunk_size),
                    "pdf" => chunk_file_pdf(&filename, chunk_size),
                    _ => Err(anyhow::anyhow!("Unsupported file extension: {:?}", ext)),
                }
            } else {
                Err(anyhow::anyhow!("No file extension found"))
            }
        } else {
            Err(anyhow::anyhow!("Not a file: {:?}", path))
        };
        let new_docs = match chunks {
            Ok(d) => d,
            Err(e) => {
                error!("Aborting: {e}");
                return Err(e);
            }
        };
        info!("New docs {}", new_docs.len());
        let _embeddings = embedder.embed(new_docs.clone(), None)?;
    }
    
    // let (v1, v2) = read_file_to_vecs(&filename);
    let (v1, v2) = chunk_file_prefix_txt(&filename, chunk_size).expect("No file");
    
    let doc_embeddings = embedder.embed(v2.clone(), None).unwrap();
    let _starting_id = 0;

    // let ids = Arc::new(Int32Array::from_iter_values(starting_id..starting_id + v1.len() as i32));
    let ids = Arc::new(StringArray::from_iter_values(
        (0..v1.len()).map(|_| Uuid::now_v7().to_string()),
    ));
    let abstracts = Arc::new(arrow_array::StringArray::from_iter_values(v1.iter().cloned()));
    let texts = Arc::new(arrow_array::StringArray::from_iter_values(v2.iter().cloned()));
    let vectors = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        doc_embeddings
            .iter()
            .map(|v| Some(v.iter().copied().map(Some).collect::<Vec<_>>())),
        dim,
    ));

    if let Ok(ref table) = db.open_table(table_name).execute().await {
        let schema: Arc<Schema> = table.schema().await.expect("No schema?");
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
        let mut column_map: HashMap<&str, ArrayRef> = HashMap::new();
        column_map.insert("id", ids);
        column_map.insert("abstract", abstracts);
        column_map.insert("text", texts);
        column_map.insert("vector", vectors);

        for field in schema.fields() {
            if let Some(array) = column_map.get(field.name().as_str()) {
                columns.push(array.clone());
            } else {
                return Err(anyhow::anyhow!("No data provided for column '{}'", field.name()));
            }
        }

        info!("Preparing batches.");
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        /*
        table.add(Box::new(batches))
            .mode(AddDataMode::Append)
            .execute()
            .await?;
        */
        
        let merge_insert = db
            .open_table(table_name)
            .execute()
            .await
            .expect("Failed to open table")
            .merge_insert(&["text"]) // vector?
            // .when_matched_update_all(None)
            .when_not_matched_insert_all()
            .clone();

        info!("Executed.");

        merge_insert.execute(Box::new(batches)).await.expect("Merge insert failed.");

        // Not updated?
        info!("Row count now {:?}", table.count_rows(None).await.expect("Cannot count rows!"));
    } else {
        info!("Cannot open table?");
    }

    Ok(())
}

pub async fn _open_existing_table(db_uri: &str, table_name: &str) -> Result<lancedb::table::Table> {
    debug!("open_existing_table(...).");

    let db = lancedb::connect(db_uri).execute().await?;

    let names = db.table_names().execute().await?;
    if !names.iter().any(|n| n == table_name) {
        bail!("Table '{table_name}' does not exist in {db_uri}");
    }

    Ok(db.open_table(table_name).execute().await?)
}

fn _read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    debug!("read_lines(...).");
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// Returns an empty vec if file not found.
// Superceded by read_file_to_vecs(...).
pub fn _read_file_to_vec<P>(filename: P) -> Vec<String>
where
    P: AsRef<Path>,
{
    debug!("read_file_to_vec(...).");
    let mut docs = Vec::new();
    if let Ok(lines) = _read_lines(&filename) {
        for line in lines {
            if let Ok(content) = line {
                if content.len() > 12 {
                    trace!("{}", &content);
                    docs.push(content);
                }
            }
        }
    } else {
        eprintln!("Could not read file: {}", filename.as_ref().display());
    }
    info!("Doc count: {}", docs.len());
    docs
}

// Read file prepared by prepare_writings.py.
// Expected format: meta-info <TAB> info-text
// The ifo-text is vectorised/searchable.
pub fn _read_file_to_vecs<P>(filename: P) -> (Vec<String>, Vec<String>)
where
    P: AsRef<Path>,
{
    debug!("read_file_to_vecs(...).");
    let mut col1 = Vec::new();
    let mut col2 = Vec::new();

    if let Ok(lines) = _read_lines(&filename) {
        for line in lines {
            if let Ok(content) = line {
                // Arbitrary limit...
                if content.len() <= 12 {
                    continue;
                }
                let mut parts = content.splitn(3, '\t');
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    trace!("{}\t{}", a, b);
                    col1.push(a.to_owned());
                    col2.push(b.to_owned());
                }
            }
        }
    } else {
        error!("Could not read file: {}", filename.as_ref().display());
    }

    info!("Row count: {}", col1.len());
    (col1, col2)
}

// The db_name and table_name are hardcoded!
// Filename argument reads the data for a new database.
pub async fn create_database<P>(filename: P, db_name: &str, table_name: &str, embedder: &mut TextEmbedding, chunk_size: usize) -> Result<(), anyhow::Error>
where
    P: AsRef<Path>,
{
    info!("Creating database.");

    // Embedder, plus determine dimension.
    let dim = get_embedding_dim(&EmbeddingModel::AllMiniLML6V2)? as i32;
    info!("Embedding dim {}", dim);

    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");
    let schema = build_schema(dim);

    info!("Database: {db_name}");
    info!("Table name: {table_name}");

    // Return the table? Overwrite?
    // We can overwrite, but return here anyway.
    if let Ok(ref _table) = db.open_table(table_name).execute().await {
        info!("Table {} already exists, skipping.", &table_name);
        return Ok(());
    };

    // Create tabel/data etc.
    // let docs = read_file_to_vec(&filename);
    /*
    let chunks = chunk_file_txt(&filename, 512);
    let docs = match chunks {
        Ok(d) => d,
        Err(e) => {
            error!("{e}");
            return Err(e);
        }
    };*/

    // v1 is the meta-data, v2 the information.
    // let (v1, v2) = read_file_to_vecs(&filename);
    let (v1, v2) = chunk_file_prefix_txt(&filename, chunk_size).expect("No file");
    
    // let doc_embeddings = embedder.embed(docs.clone(), None).unwrap();
    let doc_embeddings = embedder.embed(v2.clone(), None).unwrap();

    // let ids = Arc::new(Int32Array::from_iter_values(0..(v1.len() as i32)));
    let ids = Arc::new(StringArray::from_iter_values(
        (0..v1.len()).map(|_| Uuid::now_v7().to_string()),
    ));
    let abstracts = Arc::new(arrow_array::StringArray::from_iter_values(v1.iter().cloned()));
    let texts = Arc::new(arrow_array::StringArray::from_iter_values(v2.iter().cloned()));
    let vectors = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        doc_embeddings
            .iter()
            .map(|v| Some(v.iter().copied().map(Some).collect::<Vec<_>>())),
        dim,
    ));

    let batch = RecordBatch::try_new(schema.clone(), vec![ids, abstracts, texts, vectors]).unwrap();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let t = db.create_table(table_name, Box::new(batches))
        .mode(CreateTableMode::Overwrite)
        .execute().await.unwrap();

    let n = v2.len();
    if n >= 256 {
        info!("Creating vector index.");
        t.create_index(&["vector"], Index::Auto).
            execute().await.unwrap();
    } else {
        info!("Skipping vector index: only {n} rows (need >= 256 for PQ training)");
    }

    // BM25indexer
    t.create_index(&["abstract"], Index::FTS(Default::default()))
        .execute()
        .await?;

    t.create_index(&["text"], Index::FTS(Default::default()))
        .execute()
        .await?;

    Ok(())
}

// Creates if not exists.
pub async fn create_empty_table(db_name: &str, table_name: &str, dim: i32) -> Result<(), anyhow::Error> {
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");
    let schema = build_schema(dim);

    debug!("Database: {db_name}");
    debug!("Table name: {table_name}");

    // Return the table? Overwrite?
    // We can overwrite, but return here anyway.
    if let Ok(ref _table) = db.open_table(table_name).execute().await {
        debug!("Table {} already exists, skipping.", &table_name);
        return Ok(());
    };

    let batch = create_empty_batch(dim);
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let _t = db.create_table(table_name, Box::new(batches))
        .execute().await.unwrap();

    Ok(())
}

fn create_empty_batch(dim: i32) -> RecordBatch {
    let schema = build_schema(dim);

    // let id_col = Arc::new(Int32Array::from(Vec::<i32>::new()));
    let id_col = Arc::new(StringArray::from(Vec::<String>::new()));
    let abstract_col = Arc::new(StringArray::from(Vec::<String>::new()));
    let text_col = Arc::new(StringArray::from(Vec::<String>::new()));

    let values = Arc::new(Float32Array::from(Vec::<f32>::new()));
    let vector_col = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            values,
            None,
        ).expect("Cannot create empty vector")
    );

    let empty_batch = RecordBatch::try_new(
        schema.clone(),
        vec![id_col, abstract_col, text_col, vector_col],
    ).expect("Cannot create empty batch");

    /* db.create_table("docs", empty_batch)
      .mode(CreateTableMode::ExistOk)
      .execute()
      .await?; */

    empty_batch
}

pub async fn dump_table(db_name: &str, table_name: &str, lim: usize) -> Result<(), anyhow::Error> {
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB");
    let table = db.open_table(table_name).execute().await.expect("Cannot open table");
    let schema = table.schema().await.expect("No schema?");
    let id_idx = schema_index(&schema, "id")?;
    let abstract_idx = schema_index(&schema, "abstract")?;
    let text_idx = schema_index(&schema, "text")?;
    let vector_idx = schema_index(&schema, "vector")?;

    let results: Vec<RecordBatch> = table
        .query()
        .limit(lim)
        .execute()
        .await.expect("err")
        .try_collect()
        .await.expect("err");

    // FIXME use schema, as we do in stream_oai().
    for batch in &results {
        let ids = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let abstracts = batch
            .column(abstract_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let texts = batch
            .column(text_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let vectors = batch
            .column(vector_idx)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let id = ids.value(i);
            let astract = abstracts.value(i);
            let text = texts.value(i);

            let vec_values = vectors.value(i);
            let vec = vec_values
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values();

            info!("{}|{}|{}|{:6.3?}", &id[24.min(id.len())..], &astract[..12.min(astract.len())], &text[..24.min(text.len())], &vec[..3.min(vec.len())]);
            debug!("id={id}");
            debug!("abstract={astract}");
            debug!("text={text}");
            debug!("vector[..3]={:?}", &vec[..3.min(vec.len())]);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use lancedb::index::scalar::FullTextSearchQuery;
    use tokio::runtime::Runtime;

    fn temp_db_path() -> String {
        let mut dir = std::env::temp_dir();
        dir.push(format!("lancedb_test_{}", Uuid::now_v7()));
        std::fs::create_dir_all(&dir).expect("create temp db dir");
        dir.to_string_lossy().to_string()
    }

    async fn create_test_table(db_name: &str, table_name: &str) -> Result<lancedb::table::Table> {
        let db = lancedb::connect(db_name).execute().await?;
        let dim = 2;
        let schema = build_schema(dim);

        let ids = Arc::new(StringArray::from_iter_values([
            Uuid::now_v7().to_string(),
            Uuid::now_v7().to_string(),
            Uuid::now_v7().to_string(),
        ]));
        let abstracts = Arc::new(StringArray::from_iter_values([
            "a1".to_string(),
            "a2".to_string(),
            "a3".to_string(),
        ]));
        let texts = Arc::new(StringArray::from_iter_values([
            "alpha cat".to_string(),
            "beta dog".to_string(),
            "gamma fish".to_string(),
        ]));
        let vectors = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            [
                Some(vec![Some(1.0), Some(0.0)]),
                Some(vec![Some(0.0), Some(1.0)]),
                Some(vec![Some(0.7), Some(0.7)]),
            ],
            dim,
        ));

        let batch = RecordBatch::try_new(schema.clone(), vec![ids, abstracts, texts, vectors])?;
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let t = db
            .create_table(table_name, Box::new(batches))
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;

        t.create_index(&["text"], Index::FTS(Default::default()))
            .execute()
            .await?;

        Ok(t)
    }

    #[test]
    fn vector_retrieval_returns_expected_row() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let db_name = temp_db_path();
            let table = create_test_table(&db_name, "docs").await.unwrap();
            let schema = table.schema().await.unwrap();
            let text_idx = schema_index(&schema, "text").unwrap();

            let batches: Vec<RecordBatch> = table
                .query()
                .nearest_to(&[1.0_f32, 0.0_f32])
                .expect("nearest_to")
                .limit(1)
                .execute()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();

            let texts = batches[0]
                .column(text_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(texts.value(0), "alpha cat");
        });
    }

    #[test]
    fn full_text_retrieval_returns_expected_row() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let db_name = temp_db_path();
            let table = create_test_table(&db_name, "docs").await.unwrap();
            let schema = table.schema().await.unwrap();
            let text_idx = schema_index(&schema, "text").unwrap();

            let fts = FullTextSearchQuery::new("beta".to_string());
            let batches: Vec<RecordBatch> = table
                .query()
                .full_text_search(fts)
                .limit(1)
                .execute()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();

            let texts = batches[0]
                .column(text_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(texts.value(0), "beta dog");
        });
    }
}
