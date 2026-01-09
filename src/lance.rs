use std::sync::Arc;
use anyhow::{Result, bail};

// use arrow_array::{RecordBatch, RecordBatchIterator, ArrayRef, Int32Array, StringArray};
// use arrow_schema::{Schema, Field, DataType};

use lancedb::table::{AddDataMode};

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use log::{debug, info, trace, error};

use arrow_array::{
  ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray
  
};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb::index::Index;

use crate::embedder::{chunk_file_pdf, chunk_file_txt};

pub async fn get_row_count(table_name: &str) -> usize {
    let db_name = "data/lancedb_fastembed";
    let table_name = "docs";
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");

    let table = db.open_table(table_name).execute().await.expect("Cannot open table.");
    let rc = table.count_rows(None).await.expect("?");
    rc
}

pub async fn append_documents<P>(filename: P) -> Result<(), anyhow::Error>
where
    P: AsRef<Path>,
{
    // Embedding model (downloads once, then runs locally).
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);
    let dim = model_info.unwrap().dim as i32;
    info!("Embedding dim {}", dim);

    let db_name = "data/lancedb_fastembed";
    let table_name = "docs".to_string();
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");

    if let Ok(ref table) = db.open_table(&table_name).execute().await {
        info!("Row count {:?}", table.count_rows(None).await.expect("?"));
    }

    info!("Database: {db_name}");
    info!("Table name: {table_name}");

    let starting_id = 0;
    let path = filename.as_ref();
    let chunks = if path.is_file() {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext {
                "txt" => chunk_file_txt(&filename, 128),
                "pdf" => chunk_file_pdf(&filename, 128),
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
    let embeddings = embedder.embed(new_docs.clone(), None)?;
    let dim = embeddings[0].len() as i32;

    if let Ok(ref table) = db.open_table(&table_name).execute().await {
        let schema: Arc<Schema> = table.schema().await.expect("No schema?");

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

        /*
        info!("Schema: {:?}", schema);
        for f in schema.fields() {
            info!("col={} type={:?} nullable={}", f.name(), f.data_type(), f.is_nullable());
        }
        */

        for field in schema.fields() {
            match field.name().as_str() {
                "id" => {
                    columns.push(Arc::new(
                        Int32Array::from_iter_values(starting_id..starting_id + new_docs.len() as i32),
                    ) as ArrayRef);
                }
                "abstract" => {
                    columns.push(Arc::new(
                        StringArray::from_iter_values(new_docs.iter().cloned()),
                    ) as ArrayRef);
                }
                "text" => {
                    columns.push(Arc::new(
                        StringArray::from_iter_values(new_docs.iter().cloned()),
                    ) as ArrayRef);
                }
                "vector" => {
                    columns.push(Arc::new(
                        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                            embeddings.iter().map(|v| {
                                Some(v.iter().copied().map(Some).collect::<Vec<_>>())
                            }),
                            dim,
                        ),
                    ) as ArrayRef);
                }
                _ if field.is_nullable() => {
                    columns.push(arrow_array::new_null_array(field.data_type(), new_docs.len()));
                }
                other => bail!("Unhandled non-nullable column in append: {other}"),
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
            .open_table(&table_name)
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
        info!("Row count {:?}", table.count_rows(None).await.expect("Cannot count rows!"));
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

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    debug!("read_lines(...).");
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// Returns an empty vec if file not found.
pub fn read_file_to_vec<P>(filename: P) -> Vec<String>
where
    P: AsRef<Path>,
{
    debug!("read_file_to_vec(...).");
    let mut docs = Vec::new();
    if let Ok(lines) = read_lines(&filename) {
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

// The db_name and table_name are hardcoded!
// Filename argument reads the data for a new database.
pub async fn create_database<P>(filename: P) -> Result<(), anyhow::Error>
where
    P: AsRef<Path>,
{
    info!("Creating database.");

    // Embedding model (downloads once, then runs locally).
    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    ).expect("No embedding model.");

    // Embedder, plus determine dimension.
    let model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);
    let dim = model_info.unwrap().dim as i32;
    info!("Embedding dim {}", dim);

    let db_name = "data/lancedb_fastembed";
    let table_name = "docs".to_string();
    let db = lancedb::connect(&db_name).execute().await.expect("Cannot connect to DB.");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("abstract", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    info!("Database: {db_name}");
    info!("Table name: {table_name}");

    // Return the table?
    if let Ok(ref _table) = db.open_table(&table_name).execute().await {
        info!("Table {} already exists, skipping.", &table_name);
        return Ok(());
    };

    // Create tabel/data etc.
    // let docs = read_file_to_vec(&filename);
    let chunks = chunk_file_txt(&filename, 128);
    let docs = match chunks {
        Ok(d) => d,
        Err(e) => {
            error!("{e}");
            return Err(e);
        }
    };

    let doc_embeddings = embedder.embed(docs.clone(), None).unwrap();

    let ids = Arc::new(Int32Array::from_iter_values(0..(docs.len() as i32)));
    let abstracts = Arc::new(arrow_array::StringArray::from_iter_values(docs.iter().cloned()));
    let texts = Arc::new(arrow_array::StringArray::from_iter_values(docs.iter().cloned()));
    let vectors = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        doc_embeddings
            .iter()
            .map(|v| Some(v.iter().copied().map(Some).collect::<Vec<_>>())),
        dim,
    ));

    let batch = RecordBatch::try_new(schema.clone(), vec![ids, abstracts, texts, vectors]).unwrap();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    db.create_table(&table_name, Box::new(batches)).execute().await.unwrap();

    let t = db.open_table(&table_name).execute().await.unwrap();

    let n = docs.len();
    if n >= 256 {
        info!("Creating vector index.");
        t.create_index(&["vector"], Index::Auto).execute().await.unwrap();
    } else {
        info!("Skipping vector index: only {n} rows (need >= 256 for PQ training)");
    }

    Ok(())
}

