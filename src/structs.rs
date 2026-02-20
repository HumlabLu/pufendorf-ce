use fastembed::{TextEmbedding, TextRerank};
use std::str::FromStr;
use std::{
    fmt,
    sync::{Arc, Mutex},
};

// Struct for model options like temp &c.
#[derive(Debug, Default, Clone)]
pub struct ModelOptions {
    pub temperature: f32,
    pub num_predict: u32,
}
impl ModelOptions {
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn num_predict(mut self, num_predict: u32) -> Self {
        self.num_predict = num_predict;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    OpenAI,
    Ollama,
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::OpenAI => write!(f, "OpenAI"),
            Mode::Ollama => write!(f, "Ollama"),
        }
    }
}
impl FromStr for Mode {
    type Err = String;

    fn from_str(input: &str) -> Result<Mode, Self::Err> {
        match input {
            "openai" => Ok(Mode::OpenAI),
            "ollama" => Ok(Mode::Ollama),
            _ => Err(input.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Vector,
    FullText,
    Both,
}
impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Vector => write!(f, "Vector"),
            SearchMode::FullText => write!(f, "FullText"),
            SearchMode::Both => write!(f, "Both"),
        }
    }
}
impl FromStr for SearchMode {
    type Err = String;

    fn from_str(input: &str) -> Result<SearchMode, Self::Err> {
        match input {
            "vector" => Ok(SearchMode::Vector),
            "fulltext" => Ok(SearchMode::FullText),
            "fts" => Ok(SearchMode::FullText),
            "both" => Ok(SearchMode::Both),
            _ => Err(input.to_string()),
        }
    }
}
// Full-text query. (Also for text field?)

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
}
#[derive(Debug, Clone)]
pub struct Line {
    pub role: Role,
    pub content: String,
}

// Event messages for iced GUI.
#[derive(Debug, Clone)]
pub enum Message {
    DraftChanged(String),
    Submit,
    TemperatureChanged(f32),
    CutOffChanged(f32),
    MaxContextChanged(u32),

    ClearAll,

    LlmChunk(String),
    LlmDone,
    LlmErr(String),
}

// "Global" data for the iced app.
pub struct App {
    pub table_name: String,
    pub model_str: String,
    pub searchmode: SearchMode,
    pub fontsize: u32,
    pub fontname: String,
    pub cut_off: f32,
    pub max_context: u32,
    pub db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
    pub embedder: Arc<Mutex<TextEmbedding>>,
    pub reranker: Arc<Mutex<TextRerank>>,

    pub mode: Mode,

    pub temperature: f32,
    pub num_predict: u32,

    pub draft: String, // User input
    pub lines: Vec<Line>,
    pub waiting: bool,

    pub history: Arc<Mutex<Vec<Line>>>,

    pub system_prompt: String,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: String,
    pub text: String,
    pub astract: String,
    pub vec_dist: Option<f32>,
    pub fts_score: Option<f32>,
}

fn extract_score(vec: Option<f32>, fts: Option<f32>) -> (f32, String) {
    match (vec, fts) {
        (Some(v), None) => (v, "vec".to_string()),
        (None, Some(v)) => (v, "fts".to_string()),
        (None, None) => panic!("Neither vec nor FTS score present"),
        (Some(v), Some(f)) => (v, format!("both {}/{}", v, f)), //panic!("Both vec and FTS scores present"),
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (score, label) = extract_score(self.vec_dist, self.fts_score);
        write!(f, "({}={}) {}\t{}", label, score, self.astract, self.text)
    }
}
