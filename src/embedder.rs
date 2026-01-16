use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use std::fs;
use std::fs::read_dir;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::path::PathBuf;
use text_splitter::TextSplitter;

// let model = parse_embedding_model(cli.emodel).expect("bleh");
pub fn parse_embedding_model(model: &str) -> Result<EmbeddingModel, String> {
    match model.to_lowercase().as_str() {
        "all-minilm-l6-v2" | "allminilml6v2" => Ok(EmbeddingModel::AllMiniLML6V2),
        "all-MiniLM-L12-v2" => Ok(EmbeddingModel::AllMiniLML12V2),
        "bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
        _ => Err(format!("Unknown embedding model {model}")),
    }
}

// Use textsplitter-rs.
/*
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
)
*/
pub fn chunk_string(text: &str, max_len: usize) -> Vec<String> {
    // Maximum number of characters in a chunk
    let max_characters = max_len - 25..max_len + 25; //225..275;
    let splitter = TextSplitter::new(max_characters);

    splitter.chunks(text).map(|v| v.to_string()).collect()
}

// Does this function make sense? We woud need to feed it each line in one
// of the "prepared" documents, prefix is the first part, the text the rest.
pub fn chunk_string_prefix(text: &str, prefix: &str, max_len: usize) -> (Vec<String>, Vec<String>) {
    let max_characters = max_len - 25..max_len + 25;
    let splitter = TextSplitter::new(max_characters);

    let mut prefixes = Vec::new();
    let mut chunks = Vec::new();

    for (i, chunk) in splitter.chunks(text).enumerate() {
        prefixes.push(format!("{prefix}{}", i));
        chunks.push(chunk.to_string());
    }

    (prefixes, chunks)
}

// Return a vector with filenames with correct extension.
pub fn read_dir_contents<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<PathBuf>> {
    // Read the directory
    let mut file_paths = Vec::new();

    for entry in read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "xml" || ext == "txt" || ext == "md" {
                    file_paths.push(path);
                }
            }
        } else if path.is_dir() {
            // Meander down into sub-directories.
            println!("Dir {:?}", path);
            for fp in read_dir_contents(path).unwrap() {
                file_paths.push(fp);
            }
        }
    }
    Ok(file_paths)
}

// Shouldn't all this embed, it only chunks.
pub fn chunk_file_txt<P: AsRef<Path>>(path: P, chunk_size: usize) -> anyhow::Result<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    Ok(chunk_string(&contents, chunk_size))
}

pub fn chunk_file_prefix_txt<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let file = File::open(&path)?;
    let mut col1 = Vec::new();
    let mut col2 = Vec::new();

    let lines = io::BufReader::new(file).lines();
    for (i, line) in lines.enumerate() {
        if let Ok(content) = line {
            // Arbitrary limit...
            if content.len() <= 12 {
                continue;
            }
            let mut parts = content.splitn(3, '\t');
            if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                // trace!("{}\t{}", a, b);
                let prefix = format!("{}-{}-", a.trim(), i); // chunker adds an index too.
                println!("{}", &prefix);
                let (prefixes, chunks) = chunk_string_prefix(b.trim(), &prefix, chunk_size); // text, prefix, len
                col1.extend(prefixes);
                col2.extend(chunks);
            }
        }
    }

    Ok((col1, col2))
}

// This one also only chunks.
pub fn chunk_file_pdf<P: AsRef<Path>>(path: P, chunk_size: usize) -> anyhow::Result<Vec<String>> {
    let bytes = std::fs::read(path).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
    Ok(chunk_string(&out, chunk_size))
}

pub fn embeddings<S: AsRef<str> + Send + Sync>(texts: Vec<S>) -> anyhow::Result<Vec<Embedding>> {
    // Instantiate the model using the builder pattern.
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )
    .expect("Cannot initialise model.");

    // Generate embeddings.
    let embeddings = model.embed(texts, None).expect("Cannot create embeddings.");
    Ok(embeddings)
}

pub fn get_embedding_dim() -> anyhow::Result<usize> {
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2);
    Ok(test_model_info.unwrap().dim)
}

// =====================================================================
// Tests.
// Use
//   cargo test --release -- --nocapture
// to see dnbg/println output in tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn make_test_path(file_name: &str) -> PathBuf {
        let project_root = env!("CARGO_MANIFEST_DIR");
        let data_path = Path::new(project_root)
            .join("assets")
            // .join("data")
            .join(file_name);
        data_path
    }

    #[test]
    fn chunk_ml100() {
        let max_len = 100;
        let max_characters = max_len - 25..max_len + 25;
        let splitter = TextSplitter::new(max_characters);
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = splitter.chunks(&text).collect();
        assert!(
            result[0]
                == "the quick brown fox jumps over the lazy dog. And another sentence. Seven!"
        );
    }

    #[test]
    fn chunk_ml48() {
        let max_len = 48;
        let max_characters = max_len - 12..max_len + 12;
        let splitter = TextSplitter::new(max_characters);
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = splitter.chunks(&text).collect();
        assert!(result[0] == "the quick brown fox jumps over the lazy dog.");
        assert!(result[1] == "And another sentence. Seven!");
    }

    #[test]
    fn chunk_ml28() {
        let max_len = 28;
        let max_characters = max_len - 12..max_len + 12;
        let splitter = TextSplitter::new(max_characters);
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = splitter.chunks(&text).collect();
        assert!(result[0] == "the quick brown fox");
        assert!(result[1] == "jumps over the lazy dog.");
        assert!(result[2] == "And another sentence.");
        assert!(result[3] == "Seven!");
    }

    #[test]
    fn chunk_ml12() {
        let max_len = 12;
        let max_characters = max_len - 4..max_len + 4;
        let splitter = TextSplitter::new(max_characters);
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = splitter.chunks(&text).collect();
        //dbg!("{:?}", &result);
        assert!(result[0] == "the quick");
        assert!(result[1] == "brown fox");
        assert!(result[2] == "jumps over");
        assert!(result[3] == "the lazy dog.");
        assert!(result[4] == "And another");
        assert!(result[5] == "sentence.");
        assert!(result[6] == "Seven!");
    }

    #[test]
    fn chunk_a_string() {
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = chunk_string(&text, 28);
        dbg!("{:?}", &result);
        assert!(result[0] == "the quick brown fox jumps over the lazy dog.");
        assert!(result[1] == "And another sentence.");
        assert!(result[2] == "Seven!");
    }

    #[test]
    fn large_chunk() {
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let result: Vec<_> = chunk_string(&text, 1024);
        dbg!("{:?}", &result);
        assert!(
            result[0]
                == "the quick brown fox jumps over the lazy dog. And another sentence. Seven!"
        );
    }

    #[test]
    fn prefix_chunk_a_string() {
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let (prefixes, results): (Vec<_>, Vec<_>) = chunk_string_prefix(&text, "PREFIX", 28);
        dbg!("{:?}", &prefixes);
        dbg!("{:?}", &results);
        assert!(prefixes[0] == "PREFIX0");
        assert!(results[0] == "the quick brown fox jumps over the lazy dog.");
        // assert!(result[1] == "And another sentence.");
        // assert!(result[2] == "Seven!");
    }

    #[test]
    fn prefix_large_chunk() {
        let text =
            "the quick brown fox jumps over the lazy dog. And another sentence. Seven!".to_string();
        let (prefixes, results): (Vec<_>, Vec<_>) = chunk_string_prefix(&text, "PREFIX", 1024);
        dbg!("{:?}", &prefixes);
        dbg!("{:?}", &results);
        assert!(prefixes[0] == "PREFIX0");
        assert!(
            results[0]
                == "the quick brown fox jumps over the lazy dog. And another sentence. Seven!"
        );
    }

    #[test]
    fn chunk_prefix_file() {
        let data_path = make_test_path("extras_prepared.txt");
        let (v1, v2) = chunk_file_prefix_txt(data_path, 128).expect("No file");
        dbg!("{:?}", &v1);
        dbg!("{:?}", &v2);
        assert!(v1[1] == "BOOK EXTRA/CHAPTER STANFORD/Encyclopedia-0-1");
        assert!(
            v2[1]
                == "Your approach was secular, non-metaphysical, and anti-authoritarian; it eschewed religious appeals, scholastic"
        );
    }
}
