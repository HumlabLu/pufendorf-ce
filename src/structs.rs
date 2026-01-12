use std::{
    fmt,
    sync::{Arc, Mutex},
};

// Struct for model options like temp &c.
#[derive(Debug, Default, Clone)]
pub struct ModelOptions {
    pub temperature: f32,
    pub num_predict: i32,
}
impl ModelOptions {
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = num_predict;
        self
    }
}

// Not used anymore... make this OpenAI/Ollama?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Chat,
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Chat => write!(f, "Chat (history)"),
        }
    }
}

#[derive(Debug, Clone)]
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
    ModeChanged(Mode),

    TemperatureChanged(f32),
    CutOffChanged(f32),
    MaxContextChanged(u32),
    NumPredictChanged(i32),

    ResetParams,
    ClearAll,

    LlmChunk(String),
    LlmDone,
    LlmErr(String),
}

// "Global" data for the iced app.
pub struct App {
    pub config: AppConfig,

    pub model: String,
    pub mode: Mode,

    pub temperature: f32,
    pub num_predict: i32,

    pub draft: String, // User input
    pub lines: Vec<Line>,
    pub waiting: bool,

    pub history: Arc<Mutex<Vec<Line>>>,

    pub system_prompt: String,
    pub extra_info: String,

    pub db_connexion: Arc<Mutex<Option<lancedb::Connection>>>,
}

#[derive(Clone)]
pub struct AppConfig {
    pub db_path: String,
    pub model: String,
    pub fontsize: u32,
    pub cut_off: f32,
    pub max_context: u32,
}
