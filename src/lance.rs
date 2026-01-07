use std::sync::Arc;
use anyhow::{Result, bail};

// use arrow_array::{RecordBatch, RecordBatchIterator, ArrayRef, Int32Array, StringArray};
// use arrow_schema::{Schema, Field, DataType};

use lancedb::table::{AddDataMode, NewColumnTransform, ColumnAlteration};

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use log::{debug, error, info, trace};

use arrow_array::{
  ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
  new_null_array,
};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};

pub async fn append_documents(
    table: &lancedb::table::Table,
    embedder: &mut fastembed::TextEmbedding,
    new_docs: Vec<String>,
    starting_id: i32,
) -> Result<()> {
    debug!("append_documents(...).");
    let schema: Arc<Schema> = table.schema().await.expect("No schema?");

    let embeddings = embedder.embed(new_docs.clone(), None)?;
    let dim = embeddings[0].len() as i32;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

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

    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    table.add(Box::new(batches))
        .mode(AddDataMode::Append)
        .execute()
        .await?;

    Ok(())
}

pub async fn open_existing_table(db_uri: &str, table_name: &str) -> Result<lancedb::table::Table> {
    debug!("open_existing_table(...).");

    let db = lancedb::connect(db_uri).execute().await?;

    let names = db.table_names().execute().await?;
    if !names.iter().any(|n| n == table_name) {
        bail!("Table '{table_name}' does not exist in {db_uri}");
    }

    Ok(db.open_table(table_name).execute().await?)
}


fn make_append_batch(next_id0: i32, rows: &[(&str, &str)]) -> Result<(Arc<Schema>, RecordBatch)> {
    // This schema must match the *current* table schema (names + types + nullability).
    // In real code, fetch table.schema().await? and build arrays in that order.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, true),
    ]));

    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(next_id0..next_id0 + rows.len() as i32));
    let texts: ArrayRef = Arc::new(StringArray::from_iter_values(rows.iter().map(|(t, _src)| *t)));
    let sources: ArrayRef = Arc::new(StringArray::from_iter(rows.iter().map(|(_t, src)| Some(*src))));

    let batch = RecordBatch::try_new(schema.clone(), vec![ids, texts, sources])?;
    Ok((schema, batch))
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

